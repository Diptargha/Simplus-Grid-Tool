"""Export run results to dashboard-ready JSON and greybox Excel workbooks."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import descriptor_modes, stability_report
from .greybox import GreyboxResult, load_greybox_config, run_greybox
from .pipeline import RunResult, run_case


def _complex_dict(value: complex) -> dict[str, float | None]:
    return {"real": _json_number(np.real(value)), "imag": _json_number(np.imag(value))}


def _json_number(value: float | int | np.floating) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _array_rows(array: np.ndarray) -> list[list[float | None]]:
    return [[_json_number(value) for value in row] for row in np.asarray(array)]


def _complex_array_rows(array: np.ndarray) -> list[list[dict[str, float | None]]]:
    return [[_complex_dict(value) for value in row] for row in np.asarray(array)]


APPARATUS_TYPE_NAMES = {
    0: "SynchronousMachine",
    1: "SynchronousMachine",
    10: "GridFollowingVSI",
    11: "GridFollowingVSI",
    12: "GridFollowingVSI",
    19: "GridFollowingInverterStationary",
    20: "GridFormingVSI",
    30: "Battery",
    40: "PhotovoltaicGFM",
    41: "PhotovoltaicGFL",
    50: "WindTurbineGFM",
    51: "WindTurbineGFL",
    90: "InfiniteBusAc",
    100: "FloatingBusAc",
    1010: "GridFeedingBuck",
    1090: "InfiniteBusDc",
    1100: "FloatingBusDc",
    2000: "InterlinkAcDc",
    2001: "InterlinkAcDc",
    2002: "InterlinkAcDc",
}


# Base state symbols (bus/branch suffixes stripped) → short physical/control meaning.
# Drawn from MATLAB/Python SignalList names; not present as a glossary in MATLAB.
STATE_SYMBOL_DESCRIPTIONS: dict[str, str] = {
    # Synchronous machine / common dq
    "i_d": "d-axis current",
    "i_q": "q-axis current",
    "v_d": "d-axis voltage",
    "v_q": "q-axis voltage",
    "w": "Angular frequency",
    "theta": "dq-frame / rotor angle",
    "epsilon": "Integrated angle (global-frame embedding)",
    # Current / voltage controller integrators
    "i_d_i": "d-axis current-controller integrator",
    "i_q_i": "q-axis current-controller integrator",
    "v_d_i": "d-axis voltage-controller integrator",
    "v_q_i": "q-axis voltage-controller integrator",
    "v_od_i": "d-axis output-voltage controller integrator",
    "v_oq_i": "q-axis output-voltage controller integrator",
    "i_ld_i": "d-axis inductor-current controller integrator",
    "i_lq_i": "q-axis inductor-current controller integrator",
    # PLL
    "w_pll_i": "PLL frequency integrator",
    "pll_i": "PLL integrator",
    # DC-link / battery / PV
    "v_dc": "DC-link voltage",
    "v_dc_i": "DC-voltage controller integrator",
    "i": "DC-side current",
    "i_i": "DC-current controller integrator",
    "i_bat": "Battery current",
    "i_bat_ref": "Battery current reference",
    "duty_cycle": "DC/DC converter duty cycle",
    "v_pv": "PV array voltage",
    "v_pv_i": "PV voltage-controller integrator",
    "i_l": "PV / boost inductor current",
    "i_l_i": "Inductor-current controller integrator",
    "v_i": "Voltage-controller integrator (PV GFM)",
    # Grid-forming LC filter
    "i_ld": "d-axis filter inductor current",
    "i_lq": "q-axis filter inductor current",
    "v_od": "d-axis filter capacitor / output voltage",
    "v_oq": "q-axis filter capacitor / output voltage",
    "i_od": "d-axis output current",
    "i_oq": "q-axis output current",
    "v_d_ref": "d-axis voltage reference (droop)",
    "v_od_r": "d-axis output-voltage reference",
    # Machine / wind
    "i_sd": "d-axis machine-side current",
    "i_sq": "q-axis machine-side current",
    "i_sd_i": "d-axis machine-current controller integrator",
    "i_sq_i": "q-axis machine-current controller integrator",
    "w_m": "Mechanical / rotor speed",
    "w_m_i": "Speed-controller integrator",
    "theta_m": "Mechanical rotor angle",
    "i_a": "Phase-a current (stationary frame)",
    "i_b": "Phase-b current (stationary frame)",
    "i_al": "Alpha-axis current",
    "i_be": "Beta-axis current",
    "i_al_i": "Alpha-axis current-controller integrator",
    "i_be_i": "Beta-axis current-controller integrator",
    "i_al_ii": "Alpha-axis current-controller double integrator",
    "i_be_ii": "Beta-axis current-controller double integrator",
    "i_gd": "d-axis grid-side current",
    "i_gq": "q-axis grid-side current",
    "v_o": "Output voltage magnitude",
    "i_Ld": "d-axis load inductor current",
    "i_Lq": "q-axis load inductor current",
    # Network DSS branch states (compact naming)
    "id": "Network branch d-axis current",
    "iq": "Network branch q-axis current",
    "vd": "Network branch / shunt d-axis voltage",
    "vq": "Network branch / shunt q-axis voltage",
    "v": "Network DC branch / bus voltage state",
    # Interconnection algebraic states
    "xi": "Algebraic interconnection variable",
}


def _state_base_symbol(state: str) -> str:
    """Strip bus / branch suffixes from a whole-system state label."""

    if re.fullmatch(r"xi_\d+", state):
        return "xi"
    # Compact network labels: id1-2, iq3-4, vd1-1, i2-3, v5-5
    compact = re.fullmatch(r"(id|iq|vd|vq|i|v)(\d+)-(\d+)", state)
    if compact:
        return compact.group(1)
    # Apparatus labels: i_d2, w_pll_i3, theta1-2 (multi-bus apparatus)
    suffixed = re.fullmatch(r"(.+?)(\d+(?:-\d+)?)", state)
    if suffixed and suffixed.group(1):
        return suffixed.group(1)
    return state


def _state_description(state: str) -> str:
    """Human-readable meaning of a state symbol (independent of which apparatus)."""

    base = _state_base_symbol(state)
    if base in STATE_SYMBOL_DESCRIPTIONS:
        return STATE_SYMBOL_DESCRIPTIONS[base]
    if base.startswith("inv_u"):
        return "Inverted algebraic input (descriptor reduction)"
    return f"State symbol '{base}' (no glossary entry)"


def apparatus_name_by_bus(result: RunResult) -> dict[int, str]:
    """Map each bus number to a human-readable connected apparatus label."""

    return _apparatus_component_map(result)


def _apparatus_component_map(result: RunResult) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for apparatus in result.case.Apparatus:
        type_name = APPARATUS_TYPE_NAMES.get(apparatus.Type, f"ApparatusType{apparatus.Type}")
        buses = "-".join(str(bus) for bus in apparatus.BusNo)
        label = f"{type_name} (type {apparatus.Type}) at bus {buses}"
        for bus in apparatus.BusNo:
            mapping[bus] = label
    return mapping


def _state_component(state: str, apparatus_by_bus: dict[int, str] | None = None) -> str:
    apparatus_by_bus = apparatus_by_bus or {}
    branch_match = re.search(r"(\d+)-(\d+)$", state)
    if branch_match:
        from_bus, to_bus = branch_match.groups()
        if from_bus == to_bus:
            return f"Network self branch bus {from_bus}"
        return f"Network branch {from_bus}-{to_bus}"
    bus_match = re.search(r"(\d+)$", state)
    if bus_match:
        bus = int(bus_match.group(1))
        return apparatus_by_bus.get(bus, f"Apparatus at bus {bus}")
    if state.startswith("xi_"):
        return "Algebraic interconnection"
    return "Whole-system interconnection"


def _state_records(labels: list[str], apparatus_by_bus: dict[int, str] | None = None) -> list[dict[str, Any]]:
    return [
        {"index": idx, "name": state, "component": _state_component(state, apparatus_by_bus)}
        for idx, state in enumerate(labels)
    ]


def _component_lookup(labels: list[str], apparatus_by_bus: dict[int, str]) -> dict[str, str]:
    lookup = {_record["name"]: _record["component"] for _record in _state_records(labels, apparatus_by_bus)}
    return lookup


def _matrix_payload(matrix: np.ndarray, labels: list[str], max_size: int, apparatus_by_bus: dict[int, str]) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = matrix.shape
    if max(rows, cols) <= max_size:
        row_idx = list(range(rows))
        col_idx = list(range(cols))
        sampled = matrix
    else:
        row_idx = np.unique(np.linspace(0, rows - 1, max_size, dtype=int)).tolist()
        col_idx = np.unique(np.linspace(0, cols - 1, max_size, dtype=int)).tolist()
        sampled = matrix[np.ix_(row_idx, col_idx)]
    finite_sample = sampled[np.isfinite(sampled)]
    max_abs = float(np.max(np.abs(finite_sample))) if finite_sample.size else 0.0
    return {
        "name": "whole_system_dss_A",
        "rows": rows,
        "cols": cols,
        "sampled": max(rows, cols) > max_size,
        "row_indices": row_idx,
        "col_indices": col_idx,
        "row_labels": [labels[idx] if idx < len(labels) else f"x{idx + 1}" for idx in row_idx],
        "col_labels": [labels[idx] if idx < len(labels) else f"x{idx + 1}" for idx in col_idx],
        "row_components": [_state_component(labels[idx], apparatus_by_bus) if idx < len(labels) else "Unknown" for idx in row_idx],
        "col_components": [_state_component(labels[idx], apparatus_by_bus) if idx < len(labels) else "Unknown" for idx in col_idx],
        "max_abs": max_abs,
        "values": _array_rows(sampled),
    }


def _network_graph_payload(result: RunResult) -> dict[str, Any]:
    apparatus_by_bus: dict[int, list[dict[str, Any]]] = {}
    for apparatus in result.case.Apparatus:
        name = APPARATUS_TYPE_NAMES.get(apparatus.Type, f"ApparatusType{apparatus.Type}")
        item = {"name": name, "type": apparatus.Type, "buses": list(apparatus.BusNo)}
        for bus in apparatus.BusNo:
            apparatus_by_bus.setdefault(bus, []).append(item)

    shunts_by_bus: dict[int, list[dict[str, Any]]] = {}
    original_self = {
        int(row[0]): row
        for row in result.netlists.lines
        if int(row[0]) == int(row[1])
    }
    for idx, row in enumerate(result.lines_after_load):
        from_bus = int(row[0])
        to_bus = int(row[1])
        if from_bus != to_bus:
            continue
        original = original_self.get(from_bus)
        source = "network"
        if original is None:
            source = "load-converted"
        elif not np.allclose(original[2:6], row[2:6], equal_nan=True):
            source = "network + load-converted"
        shunts_by_bus.setdefault(from_bus, []).append(
            {
                "id": idx,
                "bus": from_bus,
                "source": source,
                "r": _json_number(row[2]),
                "x": _json_number(row[3]),
                "b": _json_number(row[4]),
                "g": _json_number(row[5]),
                "tap": _json_number(row[6]),
                "ac_dc": "AC" if int(row[7]) == 1 else "DC",
            }
        )

    nodes = []
    for row in result.netlists.buses:
        bus = int(row[0])
        nodes.append(
            {
                "id": bus,
                "label": f"Bus {bus}",
                "bus_type": int(row[1]),
                "area": int(row[10]),
                "ac_dc": "AC" if int(row[11]) == 1 else "DC",
                "voltage": _json_number(row[2]),
                "angle_rad": _json_number(row[3]),
                "apparatus": apparatus_by_bus.get(bus, []),
                "shunts": shunts_by_bus.get(bus, []),
            }
        )

    edges = []
    seen: set[tuple[int, int, int]] = set()
    for idx, row in enumerate(result.netlists.lines):
        from_bus = int(row[0])
        to_bus = int(row[1])
        if from_bus == to_bus:
            continue
        key = (from_bus, to_bus, idx)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "id": idx,
                "from": from_bus,
                "to": to_bus,
                "r": _json_number(row[2]),
                "x": _json_number(row[3]),
                "b": _json_number(row[4]),
                "g": _json_number(row[5]),
                "tap": _json_number(row[6]),
                "ac_dc": "AC" if int(row[7]) == 1 else "DC",
            }
        )

    shunts = [shunt for values in shunts_by_bus.values() for shunt in values]
    return {"nodes": nodes, "edges": edges, "shunts": shunts}


def result_to_dashboard_data(
    result: RunResult,
    *,
    case_path: str | None = None,
    top_states: int = 12,
    max_matrix_size: int = 160,
) -> dict[str, Any]:
    finite_eigs = result.eigenvalues[np.isfinite(result.eigenvalues)]
    report = stability_report(result.eigenvalues)
    modes = descriptor_modes(result.whole_system_dss, max_states_per_mode=top_states)
    state_labels = list(result.whole_system_dss.states)
    apparatus_by_bus = _apparatus_component_map(result)
    components_by_state = _component_lookup(state_labels, apparatus_by_bus)
    return {
        "schema_version": 1,
        "case": {
            "path": case_path,
            "bus_count": int(result.netlists.buses.shape[0]),
            "line_count": int(result.netlists.lines.shape[0]),
            "state_count": int(result.whole_system_dss.nx),
            "input_count": int(result.whole_system_dss.nu),
            "output_count": int(result.whole_system_dss.ny),
        },
        "stability": {
            "stable": bool(report.stable),
            "finite_mode_count": int(len(finite_eigs)),
            "raw_eigenvalue_count": int(len(result.eigenvalues)),
            "unstable_mode_count": int(len(report.unstable_modes)),
        },
        "warnings": list(result.warnings),
        "apparatus": [
            {
                "buses": list(apparatus.BusNo),
                "type": apparatus.Type,
                "name": APPARATUS_TYPE_NAMES.get(apparatus.Type, f"ApparatusType{apparatus.Type}"),
            }
            for apparatus in result.case.Apparatus
        ],
        "states": _state_records(state_labels, apparatus_by_bus),
        "eigenvalues": [
            {
                "index": idx,
                "rad_per_sec": _complex_dict(value),
                "hz": _complex_dict(value / (2 * np.pi)),
                "frequency_hz": _json_number(abs(np.imag(value)) / (2 * np.pi)),
                "real_hz": _json_number(np.real(value) / (2 * np.pi)),
                "imag_hz": _json_number(np.imag(value) / (2 * np.pi)),
            }
            for idx, value in enumerate(finite_eigs)
        ],
        "modes": [
            {
                "index": mode.mode_index,
                "eigenvalue": _complex_dict(mode.eigenvalue),
                "eigenvalue_hz": _complex_dict(mode.eigenvalue_hz),
                "frequency_hz": _json_number(mode.frequency_hz),
                "damping_ratio": _json_number(mode.damping_ratio) if mode.damping_ratio is not None else None,
                "participation": [
                    {"state": state, "component": components_by_state.get(state, _state_component(state, apparatus_by_bus)), "factor": _json_number(factor)}
                    for state, factor in mode.state_participation
                ],
            }
            for mode in modes
        ],
        "power_flow": {
            "rows": _array_rows(np.vstack(result.power_flow.power_flow)),
            "columns": ["P_load_convention", "Q_load_convention", "V", "angle_rad", "omega_rad_s"],
        },
        "network_graph": _network_graph_payload(result),
        "matrices": {
            "A": _matrix_payload(result.whole_system_dss.A, state_labels, max_matrix_size, apparatus_by_bus),
        },
    }


def export_dashboard_json(
    case_path: str | Path,
    output_path: str | Path,
    *,
    top_states: int = 12,
    max_matrix_size: int = 160,
) -> Path:
    case_path = Path(case_path)
    output_path = Path(output_path)
    result = run_case(case_path)
    data = result_to_dashboard_data(result, case_path=str(case_path), top_states=top_states, max_matrix_size=max_matrix_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    return output_path


def greybox_result_to_dashboard_data(result: GreyboxResult) -> dict[str, Any]:
    run_result = result.run_result
    config = result.config
    apparatus_by_bus = _apparatus_component_map(run_result)
    return {
        "schema_version": 1,
        "analysis_type": "greybox",
        "case": {
            "path": str(config.case_path),
            "bus_count": int(run_result.netlists.buses.shape[0]),
            "line_count": int(run_result.netlists.lines.shape[0]),
            "apparatus_count": int(len(run_result.case.Apparatus)),
        },
        "config": {
            "layers": config.layers.names(),
            "modes": list(config.modes),
            "apparatus": list(config.apparatus) if config.apparatus is not None else None,
            "layer3_apparatus": list(config.layer3_apparatus) if config.layer3_apparatus is not None else None,
            "sensitivity_lines": list(config.sensitivity_lines),
            "frequency_grid": {
                "min_hz": _json_number(float(np.min(result.admittance.frequencies_hz))) if result.admittance.frequencies_hz.size else None,
                "max_hz": _json_number(float(np.max(result.admittance.frequencies_hz))) if result.admittance.frequencies_hz.size else None,
                "count": int(result.admittance.frequencies_hz.size),
            },
        },
        "warnings": list(result.warnings),
        "whole_system_admittance": _transfer_payload(result.admittance, apparatus_by_bus),
        "whole_system_impedance": _transfer_payload(result.impedance, apparatus_by_bus),
        "modes": [
            {
                "index": mode.mode_index,
                "eigenvalue": _complex_dict(mode.eigenvalue),
                "eigenvalue_hz": _complex_dict(mode.eigenvalue / (2 * np.pi)),
                "layer1": mode.layer1,
                "layer2": mode.layer2,
                "layer3": _json_safe_records(mode.layer3),
            }
            for mode in result.modes
        ],
        "sensitivity": [
            {
                "mode_index": item.mode_index,
                "eigenvalue": _complex_dict(item.eigenvalue),
                "layer12": item.layer12,
                "layer3": _json_safe_records(item.layer3),
            }
            for item in result.sensitivity
        ],
    }


def _transfer_payload(transfer: Any, apparatus_by_bus: dict[int, str]) -> dict[str, Any]:
    input_components = [_state_component(label, apparatus_by_bus) for label in transfer.input_labels]
    output_components = [_state_component(label, apparatus_by_bus) for label in transfer.output_labels]
    return {
        "sampled": bool(transfer.sampled),
        "frequencies_hz": [_json_number(value) for value in transfer.frequencies_hz],
        "inputs": list(transfer.input_labels),
        "outputs": list(transfer.output_labels),
        "input_components": input_components,
        "output_components": output_components,
        "channels": [
            {
                "row": row,
                "col": col,
                "output": output,
                "input": input_label,
                "output_component": output_components[row],
                "input_component": input_components[col],
                "label": f"{output} ({output_components[row]}) / {input_label} ({input_components[col]})",
            }
            for row, output in enumerate(transfer.output_labels)
            for col, input_label in enumerate(transfer.input_labels)
        ],
        "values": [_complex_array_rows(matrix) for matrix in transfer.values],
    }


def export_greybox_json(
    case_path: str | Path | None,
    output_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> Path:
    if config_path is not None:
        config = load_greybox_config(config_path)
        if output_path is not None:
            config = replace(config, output_path=Path(output_path))
        if overrides:
            config = replace(config, **overrides)
        result = run_greybox(config)
    elif case_path is not None:
        from .greybox import GreyboxConfig, GreyboxLayerSelection, run_greybox_case

        layers = overrides.pop("layers", None)
        if layers is not None and not isinstance(layers, GreyboxLayerSelection):
            layers = GreyboxLayerSelection.from_names(layers)
        config = GreyboxConfig(case_path=Path(case_path), output_path=Path(output_path) if output_path else None, layers=layers or GreyboxLayerSelection())
        result = run_greybox_case(case_path, config=config, **overrides)
    else:
        raise ValueError("Either case_path or config_path is required for greybox export")
    final_output = Path(output_path or result.config.output_path or "Results/greybox_results.json")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    data = greybox_result_to_dashboard_data(result)
    with final_output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    return final_output


def export_greybox_excel(result: GreyboxResult, output_path: str | Path) -> Path:
    """Write greybox Ysys/Zsys samples, eigenvalues, State-PF, and Layer tables.

    Sheet order (Layer tabs first so they are easy to find in Excel):
    - ``Summary`` — case / grid / mode metadata
    - ``Eigenvalues`` / ``StatePF``
    - ``Layer1`` / ``Layer2`` / ``Layer3`` — apparatus modal layers (if computed)
    - ``Sens_Layer12`` — sensitivity Layer 1/2 (if computed)
    - ``Channels`` / ``Channels_Zsys``
    - ``Ysys`` / ``Zsys`` — long-form frequency response
    - ``Ysys_MagPhase`` / ``Zsys_MagPhase`` / ``Ysys_RealImag`` / ``Zsys_RealImag`` when they fit
    """

    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {"Property": "case_path", "Value": str(result.config.case_path)},
            {"Property": "bus_count", "Value": int(result.run_result.netlists.buses.shape[0])},
            {"Property": "line_count", "Value": int(result.run_result.netlists.lines.shape[0])},
            {"Property": "apparatus_count", "Value": int(len(result.run_result.case.Apparatus))},
            {"Property": "layers", "Value": ",".join(result.config.layers.names())},
            {"Property": "modes", "Value": ",".join(str(m) for m in result.config.modes)},
            {"Property": "freq_count", "Value": int(result.impedance.frequencies_hz.size)},
            {
                "Property": "freq_min_hz",
                "Value": float(np.min(result.impedance.frequencies_hz)) if result.impedance.frequencies_hz.size else None,
            },
            {
                "Property": "freq_max_hz",
                "Value": float(np.max(result.impedance.frequencies_hz)) if result.impedance.frequencies_hz.size else None,
            },
            {"Property": "ysys_sampled", "Value": bool(result.admittance.sampled)},
            {"Property": "zsys_sampled", "Value": bool(result.impedance.sampled)},
            {"Property": "warnings", "Value": " | ".join(result.warnings)},
        ]
    )

    layer1_rows: list[dict[str, Any]] = []
    layer2_rows: list[dict[str, Any]] = []
    layer3_rows: list[dict[str, Any]] = []
    for mode in result.modes:
        eig = complex(mode.eigenvalue)
        eig_hz = eig / (2 * np.pi)
        eig_cols = {
            "mode_index": mode.mode_index,
            "eigenvalue_real_rad_s": float(np.real(eig)),
            "eigenvalue_imag_rad_s": float(np.imag(eig)),
            "eigenvalue_real_hz": float(np.real(eig_hz)),
            "eigenvalue_imag_hz": float(np.imag(eig_hz)),
        }
        for item in mode.layer1:
            layer1_rows.append(
                {
                    **eig_cols,
                    **{k: item.get(k) for k in ("apparatus_index", "label", "value", "normalized")},
                }
            )
        for item in mode.layer2:
            layer2_rows.append(
                {
                    **eig_cols,
                    **{
                        k: item.get(k)
                        for k in (
                            "apparatus_index",
                            "label",
                            "real",
                            "imag",
                            "real_normalized",
                            "imag_normalized",
                        )
                    },
                }
            )
        for item in mode.layer3:
            d_rad = complex(item.get("d_lambda_rad", 0.0))
            d_hz = complex(item.get("d_lambda_hz", 0.0))
            d_pu = complex(item.get("d_lambda_pu_hz", 0.0))
            layer3_rows.append(
                {
                    **eig_cols,
                    "apparatus_index": item.get("apparatus_index"),
                    "label": item.get("label"),
                    "parameter": item.get("parameter"),
                    "d_lambda_rad_real": float(np.real(d_rad)),
                    "d_lambda_rad_imag": float(np.imag(d_rad)),
                    "d_lambda_hz_real": float(np.real(d_hz)),
                    "d_lambda_hz_imag": float(np.imag(d_hz)),
                    "d_lambda_pu_hz_real": float(np.real(d_pu)),
                    "d_lambda_pu_hz_imag": float(np.imag(d_pu)),
                }
            )

    sens_rows: list[dict[str, Any]] = []
    for sens in result.sensitivity:
        eig = complex(sens.eigenvalue)
        for item in sens.layer12:
            sens_rows.append(
                {
                    "mode_index": sens.mode_index,
                    "eigenvalue_real_rad_s": float(np.real(eig)),
                    "eigenvalue_imag_rad_s": float(np.imag(eig)),
                    **item,
                }
            )

    written: dict[str, int] = {}
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        written["Summary"] = len(summary)

        eig_df = _eigenvalues_dataframe(result.run_result.eigenvalues)
        if not eig_df.empty:
            eig_df.to_excel(writer, sheet_name="Eigenvalues", index=False)
            written["Eigenvalues"] = len(eig_df)

        state_pf_df = _state_pf_dataframe(result.run_result)
        if not state_pf_df.empty:
            state_pf_df.to_excel(writer, sheet_name="StatePF", index=False)
            written["StatePF"] = len(state_pf_df)

        # Layer sheets immediately after modal tables (before wide FRF tabs).
        if layer1_rows:
            pd.DataFrame(layer1_rows).to_excel(writer, sheet_name="Layer1", index=False)
            written["Layer1"] = len(layer1_rows)
        if layer2_rows:
            pd.DataFrame(layer2_rows).to_excel(writer, sheet_name="Layer2", index=False)
            written["Layer2"] = len(layer2_rows)
        if layer3_rows:
            pd.DataFrame(layer3_rows).to_excel(writer, sheet_name="Layer3", index=False)
            written["Layer3"] = len(layer3_rows)
        if sens_rows:
            pd.DataFrame(sens_rows).to_excel(writer, sheet_name="Sens_Layer12", index=False)
            written["Sens_Layer12"] = len(sens_rows)

        ch_y = _transfer_channel_table(result.admittance, "Ysys")
        ch_y.to_excel(writer, sheet_name="Channels", index=False)
        written["Channels"] = len(ch_y)
        ch_z = _transfer_channel_table(result.impedance, "Zsys")
        ch_z.to_excel(writer, sheet_name="Channels_Zsys", index=False)
        written["Channels_Zsys"] = len(ch_z)

        for name, transfer in (("Ysys", result.admittance), ("Zsys", result.impedance)):
            long_df = _transfer_long_table(transfer)
            long_df.to_excel(writer, sheet_name=name, index=False)
            written[name] = len(long_df)
            wide_mag = _transfer_wide_mag_phase(transfer)
            wide_ri = _transfer_wide_real_imag(transfer)
            if wide_mag is not None:
                sheet = _sheet_name(f"{name}_MagPhase")
                wide_mag.to_excel(writer, sheet_name=sheet, index=False)
                written[sheet] = len(wide_mag)
            if wide_ri is not None:
                sheet = _sheet_name(f"{name}_RealImag")
                wide_ri.to_excel(writer, sheet_name=sheet, index=False)
                written[sheet] = len(wide_ri)

    # #region agent log
    try:
        import json as _json
        import time as _time

        with open(
            r"C:\Users\z004z29x\PycharmProjects\Simplus-Grid-Tool\debug-e9b6f6.log",
            "a",
            encoding="utf-8",
        ) as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "e9b6f6",
                        "runId": "post-fix",
                        "hypothesisId": "C",
                        "location": "export.py:export_greybox_excel",
                        "message": "greybox excel sheets written",
                        "data": {
                            "output": str(output_path),
                            "n_modes": len(result.modes),
                            "layer_flags": result.config.layers.names(),
                            "layer1_rows": len(layer1_rows),
                            "layer2_rows": len(layer2_rows),
                            "layer3_rows": len(layer3_rows),
                            "sens_l12_rows": len(sens_rows),
                            "sheets": written,
                        },
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    return output_path


def _eigenvalues_dataframe(eigenvalues: np.ndarray):
    import pandas as pd

    values = np.asarray(eigenvalues, dtype=complex).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return pd.DataFrame()
    order = np.lexsort((np.abs(np.imag(values)), np.real(values)))
    values = values[order]
    rows = []
    for idx, lam in enumerate(values):
        denom = abs(lam)
        zeta = None if denom == 0 else float(-np.real(lam) / denom)
        lam_hz = lam / (2 * np.pi)
        rows.append(
            {
                "mode_index": idx,
                "real_rad_s": float(np.real(lam)),
                "imag_rad_s": float(np.imag(lam)),
                "real_hz": float(np.real(lam_hz)),
                "imag_hz": float(np.imag(lam_hz)),
                "frequency_hz": float(abs(np.imag(lam_hz))),
                "damping_ratio": zeta,
            }
        )
    return pd.DataFrame(rows)


def _state_pf_dataframe(run_result: RunResult):
    import pandas as pd

    apparatus_by_bus = _apparatus_component_map(run_result)
    modes = descriptor_modes(run_result.whole_system_dss, max_states_per_mode=None)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        eig = complex(mode.eigenvalue)
        eig_hz = complex(mode.eigenvalue_hz)
        for state_index, (state, factor) in enumerate(mode.state_participation):
            rows.append(
                {
                    "mode_index": mode.mode_index,
                    "real_rad_s": float(np.real(eig)),
                    "imag_rad_s": float(np.imag(eig)),
                    "real_hz": float(np.real(eig_hz)),
                    "imag_hz": float(np.imag(eig_hz)),
                    "frequency_hz": float(mode.frequency_hz),
                    "damping_ratio": mode.damping_ratio,
                    "state_index": state_index,
                    "state": state,
                    "description": _state_description(state),
                    "apparatus": _state_component(state, apparatus_by_bus),
                    "pf_abs": float(factor),
                }
            )
    return pd.DataFrame(rows)


def _sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]\*\:\?\/\\]", "_", name)
    return cleaned[:31]


def _safe_header(text: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(text)).strip("_")
    return cleaned[:40] or "ch"


def _transfer_channel_table(transfer: Any, prefix: str):
    import pandas as pd

    rows = []
    for row, output in enumerate(transfer.output_labels):
        for col, input_label in enumerate(transfer.input_labels):
            rows.append(
                {
                    "transfer": prefix,
                    "row": row,
                    "col": col,
                    "output": output,
                    "input": input_label,
                    "channel": f"{output}/{input_label}",
                }
            )
    return pd.DataFrame(rows)


def _transfer_long_table(transfer: Any):
    import pandas as pd

    freq = np.asarray(transfer.frequencies_hz, dtype=float).ravel()
    values = np.asarray(transfer.values, dtype=complex)
    rows: list[dict[str, Any]] = []
    for f_idx, f_hz in enumerate(freq):
        matrix = values[f_idx]
        for row, output in enumerate(transfer.output_labels):
            for col, input_label in enumerate(transfer.input_labels):
                z = complex(matrix[row, col])
                rows.append(
                    {
                        "Frequency_Hz": float(f_hz),
                        "Output": output,
                        "Input": input_label,
                        "Row": row,
                        "Col": col,
                        "Mag": float(np.abs(z)),
                        "Phase_deg": float(np.angle(z) * 180.0 / np.pi),
                        "Real": float(np.real(z)),
                        "Imag": float(np.imag(z)),
                    }
                )
    return pd.DataFrame(rows)


def _transfer_wide_mag_phase(transfer: Any):
    import pandas as pd

    n_out = len(transfer.output_labels)
    n_in = len(transfer.input_labels)
    # Excel column limit is 16384; reserve one for frequency.
    if 1 + 2 * n_out * n_in > 16384:
        return None
    freq = np.asarray(transfer.frequencies_hz, dtype=float).ravel()
    values = np.asarray(transfer.values, dtype=complex)
    data: dict[str, Any] = {"Frequency_Hz": freq}
    for row, output in enumerate(transfer.output_labels):
        for col, input_label in enumerate(transfer.input_labels):
            channel = values[:, row, col]
            tag = _safe_header(f"{output}__{input_label}")
            data[f"Mag_{tag}"] = np.abs(channel)
            data[f"Phase_{tag}"] = np.angle(channel) * 180.0 / np.pi
    return pd.DataFrame(data)


def _transfer_wide_real_imag(transfer: Any):
    import pandas as pd

    n_out = len(transfer.output_labels)
    n_in = len(transfer.input_labels)
    if 1 + 2 * n_out * n_in > 16384:
        return None
    freq = np.asarray(transfer.frequencies_hz, dtype=float).ravel()
    values = np.asarray(transfer.values, dtype=complex)
    data: dict[str, Any] = {"Frequency_Hz": freq}
    for row, output in enumerate(transfer.output_labels):
        for col, input_label in enumerate(transfer.input_labels):
            channel = values[:, row, col]
            tag = _safe_header(f"{output}__{input_label}")
            data[f"Real_{tag}"] = np.real(channel)
            data[f"Imag_{tag}"] = np.imag(channel)
    return pd.DataFrame(data)


def _json_safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for record in records:
        item = {}
        for key, value in record.items():
            if isinstance(value, complex):
                item[key] = _complex_dict(value)
            elif isinstance(value, (float, int, np.floating)):
                item[key] = _json_number(value)
            elif isinstance(value, np.ndarray):
                item[key] = _complex_array_rows(value) if np.iscomplexobj(value) else _array_rows(value)
            else:
                item[key] = value
        safe.append(item)
    return safe
