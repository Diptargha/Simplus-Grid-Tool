"""Dynamic network descriptor models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dss import DescriptorStateSpace, arrange, parallel_sum, switch_inputs_outputs


@dataclass(frozen=True)
class NetworkDSS:
    ybus: DescriptorStateSpace
    zbus: DescriptorStateSpace
    cells: list[list[DescriptorStateSpace]]


def _static(d: np.ndarray, inputs: list[str] | None = None, outputs: list[str] | None = None) -> DescriptorStateSpace:
    return DescriptorStateSpace.static(np.asarray(d, dtype=float), inputs=inputs, outputs=outputs)


def _ac_zero() -> DescriptorStateSpace:
    return _static(np.zeros((2, 2)))


def _dc_zero() -> DescriptorStateSpace:
    return _static(np.zeros((1, 1)))


def _ac_dc_zero() -> DescriptorStateSpace:
    return _static(np.zeros((1, 2)))


def _dc_ac_zero() -> DescriptorStateSpace:
    return _static(np.zeros((2, 1)))


def _branch_model(from_bus: int, to_bus: int, r: float, x: float, b: float, g: float, tap: float, area: int, w: float) -> DescriptorStateSpace:
    if area == 1:
        return _ac_branch(from_bus, to_bus, r, x, b, g, w)
    if area == 2:
        return _dc_branch(from_bus, to_bus, r, x, b, g, w)
    raise ValueError(f"Unknown line area type {area}")


def _ac_branch(from_bus: int, to_bus: int, r: float, x: float, b: float, g: float, w: float) -> DescriptorStateSpace:
    if (np.isinf(r) or np.isinf(x)) and g == 0 and b == 0:
        return _ac_zero()
    if (r == 0 and x == 0) or np.isinf(g) or np.isinf(b):
        raise ValueError(f"short circuit, ac branch from {from_bus} to {to_bus}")
    if from_bus != to_bus:
        if not (g == 0 and b == 0):
            raise ValueError(f"Ac mutual branch {from_bus}{to_bus} contains C and/or G")
        if x == 0:
            return _static(np.linalg.inv(np.array([[r, 0.0], [0.0, r]])))
        inductance = x / w
        return DescriptorStateSpace(
            np.array([[-r, x], [-x, -r]], dtype=float) / inductance,
            np.eye(2) / inductance,
            np.eye(2),
            np.zeros((2, 2)),
            np.eye(2),
            [f"id{from_bus}-{to_bus}", f"iq{from_bus}-{to_bus}"],
        )
    if r != 0 and not np.isinf(r):
        raise ValueError(f"Ac self branch {from_bus}{to_bus} contains R")
    if b == 0:
        base = _static(np.linalg.inv(np.array([[g, 0.0], [0.0, g]])))
    else:
        capacitance = b / w
        z_gc = DescriptorStateSpace(
            np.array([[-g, b], [-b, -g]], dtype=float) / capacitance,
            np.eye(2) / capacitance,
            np.eye(2),
            np.zeros((2, 2)),
            np.eye(2),
            [f"vd{from_bus}-{to_bus}", f"vq{from_bus}-{to_bus}"],
        )
        base = switch_inputs_outputs(z_gc, 2)
    if np.isinf(x):
        return base
    if x == 0:
        raise ValueError("The inductive load is short-circuit")
    inductance = x / w
    y_x = DescriptorStateSpace(
        -np.array([[0.0, -x], [x, 0.0]]) / inductance,
        np.eye(2) / inductance,
        np.eye(2),
        np.zeros((2, 2)),
        np.eye(2),
        [f"id{from_bus}-{to_bus}", f"iq{from_bus}-{to_bus}"],
    )
    return parallel_sum(base, y_x)


def _dc_branch(from_bus: int, to_bus: int, r: float, x: float, b: float, g: float, w: float) -> DescriptorStateSpace:
    if (np.isinf(r) or np.isinf(x)) and g == 0 and b == 0:
        return _dc_zero()
    if (r == 0 and x == 0) or np.isinf(g) or np.isinf(b):
        raise ValueError(f"short circuit, dc branch from {from_bus} to {to_bus}")
    if from_bus != to_bus:
        if not (g == 0 and b == 0):
            raise ValueError(f"Dc mutual branch {from_bus}-{to_bus} contains C and/or G")
        if x == 0:
            return _static([[1 / r]])
        inductance = x / w
        return DescriptorStateSpace(
            np.array([[-r / inductance]]),
            np.array([[1 / inductance]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[1.0]]),
            [f"i{from_bus}-{to_bus}"],
        )
    if not (np.isinf(r) or np.isinf(x)):
        raise ValueError(f"Dc self branch {from_bus}{to_bus} contains R and/or L")
    if b == 0:
        return _static([[1 / g]])
    capacitance = b / w
    z_gc = DescriptorStateSpace(
        np.array([[-g / capacitance]]),
        np.array([[1 / capacitance]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[1.0]]),
        [f"v{from_bus}-{to_bus}"],
    )
    return switch_inputs_outputs(z_gc, 1)


def ybus_calc_dss(list_bus: np.ndarray, list_line: np.ndarray, w: float) -> tuple[DescriptorStateSpace, list[list[DescriptorStateSpace]]]:
    if list_line.size == 0:
        raise ValueError("Network DSS requires at least one line")
    fb = list_line[:, 0].astype(int)
    tb = list_line[:, 1].astype(int)
    num_bus = int(max(np.max(fb), np.max(tb)))
    area_bus = list_bus[:, 11].astype(int)
    cells: list[list[DescriptorStateSpace]] = []
    for i in range(num_bus):
        row = []
        for j in range(num_bus):
            if area_bus[i] == 1 and area_bus[j] == 1:
                row.append(_ac_zero())
            elif area_bus[i] == 2 and area_bus[j] == 2:
                row.append(_dc_zero())
            elif area_bus[i] == 1 and area_bus[j] == 2:
                row.append(_dc_ac_zero())
            elif area_bus[i] == 2 and area_bus[j] == 1:
                row.append(_ac_dc_zero())
            else:
                raise ValueError("Unknown bus area type")
        cells.append(row)
    for idx in range(len(fb)):
        f = fb[idx] - 1
        t = tb[idx] - 1
        branch = _branch_model(
            int(fb[idx]), int(tb[idx]), list_line[idx, 2], list_line[idx, 3],
            list_line[idx, 4], list_line[idx, 5], list_line[idx, 6], int(list_line[idx, 7]), w
        )
        tap = list_line[idx, 6]
        if f != t:
            cells[f][t] = parallel_sum(cells[f][t], branch.scaled(-1 / tap))
            cells[t][f] = cells[f][t].copy()
            cells[f][f] = parallel_sum(cells[f][f], branch.scaled(1 / tap ** 2))
            cells[t][t] = parallel_sum(cells[t][t], branch)
        else:
            cells[f][t] = parallel_sum(cells[f][t], branch)
    ybus = arrange(cells)
    input_names: list[str] = []
    output_names: list[str] = []
    for idx, area in enumerate(area_bus[:num_bus], start=1):
        if area == 1:
            input_names.extend([f"v_d{idx}", f"v_q{idx}"])
            output_names.extend([f"i_d{idx}", f"i_q{idx}"])
        else:
            input_names.append(f"v{idx}")
            output_names.append(f"i{idx}")
    ybus.inputs = input_names
    ybus.outputs = output_names
    ybus.check_dimensions()
    return ybus, cells


def network_dss(list_bus: np.ndarray, list_line: np.ndarray, w: float) -> NetworkDSS:
    ybus, cells = ybus_calc_dss(list_bus, list_line, w)
    zbus = switch_inputs_outputs(ybus, ybus.ny)
    return NetworkDSS(ybus=ybus, zbus=zbus, cells=cells)
