"""Export run results to dashboard-ready JSON."""

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
