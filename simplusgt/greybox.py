"""Standalone greybox impedance participation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig

from .dss import DescriptorStateSpace, feedback, inverse
from .pipeline import RunResult, run_case


APPARATUS_LAYER1 = "app-l1"
APPARATUS_LAYER2 = "app-l2"
APPARATUS_LAYER3 = "app-l3"
SENSITIVITY_LAYER12 = "sens-l12"
SENSITIVITY_LAYER3 = "sens-l3"
ADMITTANCE_ONLY = "admittance-only"
ALL_LAYERS = "all"

SELECTABLE_LAYERS = {
    APPARATUS_LAYER1,
    APPARATUS_LAYER2,
    APPARATUS_LAYER3,
    SENSITIVITY_LAYER12,
    SENSITIVITY_LAYER3,
}


@dataclass(frozen=True)
class FrequencyGrid:
    min_hz: float = 0.1
    max_hz: float = 1000.0
    count: int = 80
    values_hz: tuple[float, ...] | None = None

    def frequencies(self) -> np.ndarray:
        if self.values_hz is not None:
            values = np.asarray(self.values_hz, dtype=float)
        else:
            if self.min_hz <= 0 or self.max_hz <= 0:
                raise ValueError("Frequency grid bounds must be positive for logarithmic sampling")
            if self.max_hz < self.min_hz:
                raise ValueError("frequency_grid.max_hz must be greater than or equal to min_hz")
            values = np.logspace(np.log10(self.min_hz), np.log10(self.max_hz), int(self.count))
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Frequency grid must contain non-negative finite values")
        return values


@dataclass(frozen=True)
class GreyboxLayerSelection:
    apparatus_layer1: bool = False
    apparatus_layer2: bool = False
    apparatus_layer3: bool = False
    sensitivity_layer12: bool = False
    sensitivity_layer3: bool = False

    @classmethod
    def from_names(cls, names: list[str] | tuple[str, ...] | str | None) -> "GreyboxLayerSelection":
        if names is None:
            names = [ADMITTANCE_ONLY]
        if isinstance(names, str):
            names = [item.strip() for item in names.split(",") if item.strip()]
        normalized = {name.strip().lower() for name in names}
        if not normalized or ADMITTANCE_ONLY in normalized:
            normalized.discard(ADMITTANCE_ONLY)
        if ALL_LAYERS in normalized:
            normalized = set(SELECTABLE_LAYERS)
        unknown = normalized - SELECTABLE_LAYERS
        if unknown:
            raise ValueError(f"Unknown greybox layer(s): {', '.join(sorted(unknown))}")
        return cls(
            apparatus_layer1=APPARATUS_LAYER1 in normalized,
            apparatus_layer2=APPARATUS_LAYER2 in normalized,
            apparatus_layer3=APPARATUS_LAYER3 in normalized,
            sensitivity_layer12=SENSITIVITY_LAYER12 in normalized,
            sensitivity_layer3=SENSITIVITY_LAYER3 in normalized,
        )

    def names(self) -> list[str]:
        names = []
        if self.apparatus_layer1:
            names.append(APPARATUS_LAYER1)
        if self.apparatus_layer2:
            names.append(APPARATUS_LAYER2)
        if self.apparatus_layer3:
            names.append(APPARATUS_LAYER3)
        if self.sensitivity_layer12:
            names.append(SENSITIVITY_LAYER12)
        if self.sensitivity_layer3:
            names.append(SENSITIVITY_LAYER3)
        return names or [ADMITTANCE_ONLY]

    @property
    def needs_modes(self) -> bool:
        return any((self.apparatus_layer1, self.apparatus_layer2, self.apparatus_layer3))

    @property
    def needs_sensitivity(self) -> bool:
        return self.sensitivity_layer12 or self.sensitivity_layer3


@dataclass(frozen=True)
class GreyboxConfig:
    case_path: Path
    output_path: Path | None = None
    layers: GreyboxLayerSelection = field(default_factory=GreyboxLayerSelection)
    modes: tuple[int, ...] = (0,)
    apparatus: tuple[int, ...] | None = None
    layer3_apparatus: tuple[int, ...] | None = None
    sensitivity_lines: tuple[int, ...] = ()
    perturbation_factor: float = 1e-5
    frequency_grid: FrequencyGrid = field(default_factory=FrequencyGrid)
    max_admittance_samples: int = 80
    max_matrix_size: int = 80


@dataclass(frozen=True)
class WholeSystemAdmittanceResult:
    frequencies_hz: np.ndarray
    input_labels: list[str]
    output_labels: list[str]
    values: np.ndarray
    sampled: bool = False


@dataclass(frozen=True)
class GreyboxModelBundle:
    gm_trim: DescriptorStateSpace
    zm: DescriptorStateSpace
    ybus: DescriptorStateSpace


@dataclass(frozen=True)
class GreyboxModeResult:
    mode_index: int
    eigenvalue: complex
    residues: list[np.ndarray | None] = field(default_factory=list)
    apparatus_impedances: list[np.ndarray | None] = field(default_factory=list)
    layer1: list[dict[str, Any]] = field(default_factory=list)
    layer2: list[dict[str, Any]] = field(default_factory=list)
    layer3: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GreyboxSensitivityResult:
    mode_index: int
    eigenvalue: complex
    matrix: np.ndarray
    layer12: list[dict[str, Any]] = field(default_factory=list)
    layer3: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GreyboxResult:
    config: GreyboxConfig
    run_result: RunResult
    admittance: WholeSystemAdmittanceResult
    impedance: WholeSystemAdmittanceResult
    modes: list[GreyboxModeResult] = field(default_factory=list)
    sensitivity: list[GreyboxSensitivityResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evalfr(model: DescriptorStateSpace, s: complex) -> np.ndarray:
    """Evaluate a descriptor transfer matrix at complex frequency ``s``."""

    if model.nx == 0:
        return model.D.astype(complex)
    lhs = s * model.E - model.A
    try:
        dynamic = model.C @ np.linalg.solve(lhs, model.B)
    except np.linalg.LinAlgError:
        dynamic = model.C @ np.linalg.pinv(lhs) @ model.B
    return dynamic + model.D


def load_greybox_config(path: str | Path) -> GreyboxConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    base_dir = path.parent
    return greybox_config_from_dict(raw, base_dir=base_dir)


def greybox_config_from_dict(raw: dict[str, Any], *, base_dir: Path | None = None) -> GreyboxConfig:
    base_dir = base_dir or Path.cwd()
    if "case_path" not in raw:
        raise ValueError("Greybox config requires case_path")
    case_path = _resolve_path(raw["case_path"], base_dir)
    output_path = _resolve_path(raw["output_path"], base_dir) if raw.get("output_path") else None
    frequency_grid = _frequency_grid_from_dict(raw.get("frequency_grid", {}))
    return GreyboxConfig(
        case_path=case_path,
        output_path=output_path,
        layers=GreyboxLayerSelection.from_names(raw.get("layers")),
        modes=tuple(int(idx) for idx in raw.get("modes", [0])),
        apparatus=_optional_int_tuple(raw.get("apparatus")),
        layer3_apparatus=_optional_int_tuple(raw.get("layer3_apparatus")),
        sensitivity_lines=tuple(int(idx) for idx in raw.get("sensitivity_lines", [])),
        perturbation_factor=float(raw.get("perturbation_factor", 1e-5)),
        frequency_grid=frequency_grid,
        max_admittance_samples=int(raw.get("max_admittance_samples", 80)),
        max_matrix_size=int(raw.get("max_matrix_size", 80)),
    )


def run_greybox_from_config(path: str | Path) -> GreyboxResult:
    config = load_greybox_config(path)
    return run_greybox(config)


def run_greybox_case(case_path: str | Path, config: GreyboxConfig | None = None, **overrides: Any) -> GreyboxResult:
    base = config or GreyboxConfig(case_path=Path(case_path))
    merged = GreyboxConfig(
        case_path=Path(case_path),
        output_path=overrides.get("output_path", base.output_path),
        layers=overrides.get("layers", base.layers),
        modes=tuple(overrides.get("modes", base.modes)),
        apparatus=overrides.get("apparatus", base.apparatus),
        layer3_apparatus=overrides.get("layer3_apparatus", base.layer3_apparatus),
        sensitivity_lines=tuple(overrides.get("sensitivity_lines", base.sensitivity_lines)),
        perturbation_factor=float(overrides.get("perturbation_factor", base.perturbation_factor)),
        frequency_grid=overrides.get("frequency_grid", base.frequency_grid),
        max_admittance_samples=int(overrides.get("max_admittance_samples", base.max_admittance_samples)),
        max_matrix_size=int(overrides.get("max_matrix_size", base.max_matrix_size)),
    )
    return run_greybox(merged)


def run_greybox(config: GreyboxConfig) -> GreyboxResult:
    run_result = run_case(config.case_path)
    warnings = list(run_result.warnings)
    admittance = _sample_whole_system_admittance(run_result.whole_system_dss, config)
    impedance_bundle = whole_system_impedance_bundle(run_result)
    impedance = _sample_whole_system_impedance(impedance_bundle, config)
    modes = _mode_results(run_result, config, warnings) if config.layers.needs_modes else []
    sensitivity = _sensitivity_results(run_result, config, warnings) if config.layers.needs_sensitivity else []
    return GreyboxResult(
        config=config,
        run_result=run_result,
        admittance=admittance,
        impedance=impedance,
        modes=modes,
        sensitivity=sensitivity,
        warnings=warnings,
    )


def whole_system_impedance_model(run_result: RunResult) -> DescriptorStateSpace:
    """Recreate MATLAB WholeSysZ_cal: Zsys = feedback(inv(Gm(PortI, PortV)), Ybus)."""

    bundle = whole_system_impedance_bundle(run_result)
    return feedback(bundle.zm, bundle.ybus, list(range(bundle.zm.nu)), list(range(bundle.zm.ny)))


def whole_system_impedance_bundle(run_result: RunResult) -> GreyboxModelBundle:
    """Build the intermediate models used by MATLAB WholeSysZ_cal."""

    gm = run_result.apparatus_block
    port_v, port_i = _voltage_current_ports(gm, int(np.max(run_result.buses_after_load[:, 0])))
    gm_trim = gm.truncate(port_i, port_v)
    zm = inverse(gm_trim)
    return GreyboxModelBundle(gm_trim=gm_trim, zm=zm, ybus=run_result.network.ybus)


def _sample_whole_system_admittance(model: DescriptorStateSpace, config: GreyboxConfig) -> WholeSystemAdmittanceResult:
    frequencies = config.frequency_grid.frequencies()
    sampled = False
    if frequencies.size > config.max_admittance_samples:
        idx = np.unique(np.linspace(0, frequencies.size - 1, config.max_admittance_samples, dtype=int))
        frequencies = frequencies[idx]
        sampled = True
    values = np.asarray([evalfr(model, 2j * np.pi * freq) for freq in frequencies], dtype=complex)
    return WholeSystemAdmittanceResult(
        frequencies_hz=frequencies,
        input_labels=list(model.inputs),
        output_labels=list(model.outputs),
        values=values,
        sampled=sampled,
    )


def _sample_whole_system_impedance(bundle: GreyboxModelBundle, config: GreyboxConfig) -> WholeSystemAdmittanceResult:
    frequencies = config.frequency_grid.frequencies()
    sampled = False
    if frequencies.size > config.max_admittance_samples:
        idx = np.unique(np.linspace(0, frequencies.size - 1, config.max_admittance_samples, dtype=int))
        frequencies = frequencies[idx]
        sampled = True
    values = []
    for freq in frequencies:
        s = 2j * np.pi * freq
        zm = evalfr(bundle.zm, s)
        ybus = evalfr(bundle.ybus, s)
        values.append(np.linalg.solve(np.eye(zm.shape[0]) + zm @ ybus, zm))
    return WholeSystemAdmittanceResult(
        frequencies_hz=frequencies,
        input_labels=list(bundle.zm.inputs),
        output_labels=list(bundle.zm.outputs),
        values=np.asarray(values, dtype=complex),
        sampled=sampled,
    )


def _voltage_current_ports(gm: DescriptorStateSpace, num_bus: int) -> tuple[list[int], list[int]]:
    port_v: list[int] = []
    port_i: list[int] = []
    for bus in range(1, num_bus + 1):
        if f"v_d{bus}" in gm.inputs:
            in1 = gm.inputs.index(f"v_d{bus}")
            out1 = gm.outputs.index(f"i_d{bus}")
            port_v.extend([in1, in1 + 1])
            port_i.extend([out1, out1 + 1])
        elif f"v{bus}" in gm.inputs:
            port_v.append(gm.inputs.index(f"v{bus}"))
            port_i.append(gm.outputs.index(f"i{bus}"))
        else:
            raise ValueError(f"Unable to find voltage/current ports for bus {bus}")
    return port_v, port_i


def _mode_results(run_result: RunResult, config: GreyboxConfig, warnings: list[str]) -> list[GreyboxModeResult]:
    model = run_result.whole_system_dss
    if model.nx == 0:
        return []
    try:
        standard_a, standard_b, standard_c, _ = model.to_state_space()
        eigenvalues, right = np.linalg.eig(standard_a)
        left = np.linalg.inv(right)
    except np.linalg.LinAlgError:
        eigenvalues, left, right = eig(model.A, model.E, left=True, right=True)
        left = left.conj().T
        standard_b = model.B
        standard_c = model.C
        warnings.append("Greybox modal residues used generalized eigenvectors because E is singular")
    order = [idx for idx, value in enumerate(eigenvalues) if np.isfinite(value)]
    selected = _selected_mode_indices(order, config.modes, len(eigenvalues))
    apparatus_indices = _selected_apparatus_indices(run_result, config.apparatus)
    results: list[GreyboxModeResult] = []
    port_slices = _apparatus_port_slices(run_result)
    for mode_index in selected:
        residues: list[np.ndarray | None] = []
        impedances: list[np.ndarray | None] = []
        for app_idx, (out_slice, in_slice) in enumerate(port_slices):
            if app_idx not in apparatus_indices:
                residues.append(None)
                impedances.append(None)
                continue
            phi = right[:, mode_index:mode_index + 1]
            psi = left[mode_index:mode_index + 1, :]
            residues.append(standard_c[out_slice, :] @ phi @ psi @ standard_b[:, in_slice])
            impedances.append(apparatus_impedance(run_result.apparatus_models[app_idx], eigenvalues[mode_index]))
        layer1, layer2 = _apparatus_layer12(run_result, residues, impedances, apparatus_indices, config.layers)
        layer3 = _apparatus_layer3(run_result, residues, impedances, apparatus_indices, eigenvalues[mode_index], config, warnings)
        results.append(
            GreyboxModeResult(
                mode_index=int(mode_index),
                eigenvalue=complex(eigenvalues[mode_index]),
                residues=residues,
                apparatus_impedances=impedances,
                layer1=layer1,
                layer2=layer2,
                layer3=layer3,
            )
        )
    return results


def apparatus_impedance(model: DescriptorStateSpace, s: complex) -> np.ndarray | None:
    size = _leading_square_size(model)
    if size == 0:
        return None
    admittance = evalfr(model.truncate(list(range(size)), list(range(size))), s)
    try:
        return np.linalg.inv(admittance)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(admittance)


def _apparatus_layer12(
    run_result: RunResult,
    residues: list[np.ndarray | None],
    impedances: list[np.ndarray | None],
    apparatus_indices: set[int],
    layers: GreyboxLayerSelection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer1_all: dict[int, float] = {}
    layer2_all: dict[int, complex] = {}
    for idx in apparatus_indices:
        residue = residues[idx]
        impedance = impedances[idx]
        if residue is None or impedance is None:
            continue
        if layers.apparatus_layer1:
            layer1_all[idx] = float(np.linalg.norm(residue, "fro") * np.linalg.norm(impedance, "fro"))
        if layers.apparatus_layer2:
            layer2_all[idx] = -np.vdot(impedance, residue)
    layer1_sum = sum(layer1_all.values()) or 1.0
    layer2_real_sum = sum(abs(np.real(value)) for value in layer2_all.values()) or 1.0
    layer2_imag_sum = sum(abs(np.imag(value)) for value in layer2_all.values()) or 1.0
    layer1 = [
        {
            "apparatus_index": idx,
            "label": _apparatus_label(run_result, idx),
            "value": value,
            "normalized": value / layer1_sum,
        }
        for idx, value in sorted(layer1_all.items())
    ]
    layer2 = [
        {
            "apparatus_index": idx,
            "label": _apparatus_label(run_result, idx),
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
            "real_normalized": float(np.real(value) / layer2_real_sum),
            "imag_normalized": float(np.imag(value) / layer2_imag_sum),
        }
        for idx, value in sorted(layer2_all.items())
    ]
    return layer1, layer2


def _apparatus_layer3(
    run_result: RunResult,
    residues: list[np.ndarray | None],
    impedances: list[np.ndarray | None],
    apparatus_indices: set[int],
    eigenvalue: complex,
    config: GreyboxConfig,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not config.layers.apparatus_layer3:
        return []
    layer3_indices = _selected_apparatus_indices(run_result, config.layer3_apparatus) & apparatus_indices
    results: list[dict[str, Any]] = []
    for idx in layer3_indices:
        residue = residues[idx]
        original_impedance = impedances[idx]
        if residue is None or original_impedance is None:
            continue
        params = run_result.netlists.apparatus_params[idx]
        numeric_params = {key: value for key, value in params.items() if isinstance(value, (int, float))}
        for name, value in numeric_params.items():
            delta = config.perturbation_factor * (1 + abs(float(value)))
            perturbed = dict(params)
            perturbed[name] = float(value) + delta
            try:
                app = run_result.case.Apparatus[idx]
                model = __import__("simplusgt.models", fromlist=["create_apparatus_model"]).create_apparatus_model(
                    app.BusNo,
                    app.Type,
                    _apparatus_power_flow(run_result.power_flow_after_load, app.BusNo),
                    perturbed,
                    1 / run_result.case.Basic.Fs,
                )
                new_impedance = apparatus_impedance(model, eigenvalue)
            except Exception as exc:  # pragma: no cover - warning path depends on apparatus equations
                warnings.append(f"Layer 3 skipped {name} for apparatus {idx + 1}: {exc}")
                continue
            if new_impedance is None:
                continue
            delta_z = (new_impedance - original_impedance) / delta
            d_lambda_rad = -np.vdot(delta_z, residue)
            results.append(
                {
                    "apparatus_index": idx,
                    "label": _apparatus_label(run_result, idx),
                    "parameter": name,
                    "d_lambda_rad": d_lambda_rad,
                    "d_lambda_hz": d_lambda_rad / (2 * np.pi),
                    "d_lambda_pu_hz": d_lambda_rad * float(value) / (2 * np.pi),
                }
            )
    return results


def _sensitivity_results(run_result: RunResult, config: GreyboxConfig, warnings: list[str]) -> list[GreyboxSensitivityResult]:
    if np.any(run_result.netlists.buses[:, 11] != 1):
        warnings.append("Greybox sensitivity layers currently support AC dq networks only")
        return []
    model = run_result.whole_system_dss
    try:
        a, b, c, _ = model.to_state_space()
        eigenvalues, right = np.linalg.eig(a)
        left = np.linalg.inv(right)
    except np.linalg.LinAlgError:
        warnings.append("Greybox sensitivity skipped because whole-system E matrix is singular")
        return []
    selected = _selected_mode_indices([idx for idx, value in enumerate(eigenvalues) if np.isfinite(value)], config.modes, len(eigenvalues))
    results = []
    for mode_index in selected:
        phi = right[:, mode_index:mode_index + 1]
        psi = left[mode_index:mode_index + 1, :]
        matrix = -c @ phi @ psi @ b
        layer12 = _sensitivity_layer12(matrix, run_result) if config.layers.sensitivity_layer12 else []
        layer3 = _sensitivity_layer3(matrix, run_result, config, warnings) if config.layers.sensitivity_layer3 else []
        results.append(GreyboxSensitivityResult(mode_index=int(mode_index), eigenvalue=complex(eigenvalues[mode_index]), matrix=matrix, layer12=layer12, layer3=layer3))
    return results


def _sensitivity_layer12(matrix: np.ndarray, run_result: RunResult) -> list[dict[str, Any]]:
    n_bus = run_result.netlists.buses.shape[0]
    records: list[dict[str, Any]] = []
    for i in range(n_bus):
        block = matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2]
        records.append({"component": f"Node {i + 1}", "kind": "node", "layer1": float(np.linalg.norm(block, "fro")), "layer2_real": float(np.real(np.trace(block))), "layer2_imag": float(np.imag(np.trace(block)))})
    for row in run_result.lines_after_load:
        f_bus, t_bus = int(row[0]), int(row[1])
        if f_bus == t_bus:
            continue
        i, j = f_bus - 1, t_bus - 1
        block = (
            matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2]
            + matrix[2 * j:2 * j + 2, 2 * j:2 * j + 2]
            - matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            - matrix[2 * j:2 * j + 2, 2 * i:2 * i + 2]
        )
        records.append({"component": f"Branch {f_bus}-{t_bus}", "kind": "branch", "layer1": float(np.linalg.norm(block, "fro")), "layer2_real": float(np.real(np.trace(block))), "layer2_imag": float(np.imag(np.trace(block)))})
    layer1_sum = sum(record["layer1"] for record in records) or 1.0
    real_sum = sum(abs(record["layer2_real"]) for record in records) or 1.0
    imag_sum = sum(abs(record["layer2_imag"]) for record in records) or 1.0
    for record in records:
        record["layer1_normalized"] = record["layer1"] / layer1_sum
        record["layer2_real_normalized"] = record["layer2_real"] / real_sum
        record["layer2_imag_normalized"] = record["layer2_imag"] / imag_sum
    return records


def _sensitivity_layer3(matrix: np.ndarray, run_result: RunResult, config: GreyboxConfig, warnings: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = set(config.sensitivity_lines)
    for line_idx, row in enumerate(run_result.lines_after_load):
        if selected and line_idx not in selected and line_idx + 1 not in selected:
            continue
        f_bus, t_bus = int(row[0]), int(row[1])
        if f_bus == t_bus:
            continue
        i, j = f_bus - 1, t_bus - 1
        block = (
            matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2]
            + matrix[2 * j:2 * j + 2, 2 * j:2 * j + 2]
            - matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            - matrix[2 * j:2 * j + 2, 2 * i:2 * i + 2]
        )
        for col, name in ((2, "R"), (3, "X")):
            value = row[col]
            if not np.isfinite(value) or value == 0:
                continue
            records.append(
                {
                    "component": f"Branch {f_bus}-{t_bus}.{name}",
                    "line_index": line_idx,
                    "parameter": name,
                    "d_lambda_rad": complex(np.trace(block) / value),
                    "d_lambda_pu_hz": complex(np.trace(block) * value / (2 * np.pi)),
                }
            )
    if not records:
        warnings.append("Sensitivity Layer 3 found no supported selected R/X branch parameters")
    return records


def _apparatus_port_slices(run_result: RunResult) -> list[tuple[slice, slice]]:
    slices = []
    in_start = 0
    out_start = 0
    for model in run_result.apparatus_models:
        size = _leading_square_size(model)
        in_stop = in_start + model.nu
        out_stop = out_start + model.ny
        slices.append((slice(out_start, out_start + size), slice(in_start, in_start + size)))
        in_start = in_stop
        out_start = out_stop
    return slices


def _leading_square_size(model: DescriptorStateSpace) -> int:
    return min(model.nu, model.ny)


def _selected_mode_indices(finite_indices: list[int], requested: tuple[int, ...], total: int) -> list[int]:
    selected = []
    for item in requested:
        idx = int(item)
        if idx < 0:
            continue
        if idx in finite_indices:
            selected.append(idx)
        elif idx < len(finite_indices):
            selected.append(finite_indices[idx])
        elif idx < total:
            selected.append(idx)
    return list(dict.fromkeys(selected))


def _selected_apparatus_indices(run_result: RunResult, requested: tuple[int, ...] | None) -> set[int]:
    count = len(run_result.apparatus_models)
    if requested is None:
        return set(range(count))
    indices = set()
    for item in requested:
        idx = int(item)
        if 1 <= idx <= count:
            indices.add(idx - 1)
        elif 0 <= idx < count:
            indices.add(idx)
    return indices


def _apparatus_label(run_result: RunResult, index: int) -> str:
    apparatus = run_result.case.Apparatus[index]
    buses = "-".join(str(bus) for bus in apparatus.BusNo)
    return f"Apparatus {index + 1} type {apparatus.Type} at bus {buses}"


def _apparatus_power_flow(power_flow: list[np.ndarray], buses: tuple[int, ...]) -> np.ndarray:
    if len(buses) == 1:
        return power_flow[buses[0] - 1]
    return np.concatenate([power_flow[bus - 1] for bus in buses])


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _frequency_grid_from_dict(raw: dict[str, Any]) -> FrequencyGrid:
    if "values_hz" in raw:
        return FrequencyGrid(values_hz=tuple(float(value) for value in raw["values_hz"]))
    return FrequencyGrid(
        min_hz=float(raw.get("min_hz", 0.1)),
        max_hz=float(raw.get("max_hz", 1000.0)),
        count=int(raw.get("count", 80)),
    )
