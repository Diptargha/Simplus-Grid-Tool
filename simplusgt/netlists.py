"""MATLAB-compatible bus, line, and apparatus netlist normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Apparatus, CaseData


@dataclass(frozen=True)
class NormalizedNetlists:
    buses: np.ndarray
    lines: np.ndarray
    apparatus_buses: list[tuple[int, ...]]
    apparatus_types: list[int]
    apparatus_params: list[dict]


def rearrange_buses(case: CaseData) -> tuple[np.ndarray, int]:
    rows = [
        [
            bus.BusNo,
            bus.BusType,
            bus.Voltage,
            bus.Theta,
            bus.PGi,
            bus.QGi,
            bus.PLi,
            bus.QLi,
            bus.Qmin,
            bus.Qmax,
            bus.AreaNo,
            bus.AcDc,
        ]
        for bus in case.Bus
    ]
    if not rows:
        raise ValueError("Case has no buses")
    list_bus = np.array(rows, dtype=float)
    area_ids = np.unique(list_bus[:, 10].astype(int))
    if len(area_ids) != int(np.max(area_ids)):
        raise ValueError("The total area number is different from the maximum area index")
    for area in area_ids:
        area_rows = list_bus[list_bus[:, 10] == area]
        area_rows = area_rows[np.argsort(area_rows[:, 0])]
        area_types = np.unique(area_rows[:, 11].astype(int))
        if len(area_types) != 1:
            raise ValueError("In an area, the area type has to be the same for all buses")
        slack = np.flatnonzero(area_rows[:, 1] == 1)
        if len(slack) == 0:
            raise ValueError(f"The system has no slack bus in area {area}")
        if slack[0] != 0:
            raise ValueError("The first bus in each area has to be the slack bus")
        if len(slack) > 1:
            raise ValueError(f"The system has more than one slack bus in area {area}")
    if np.any((list_bus[:, 11] == 2) & (list_bus[:, 1] == 2)):
        bad = int(list_bus[np.flatnonzero((list_bus[:, 11] == 2) & (list_bus[:, 1] == 2))[0], 0])
        raise ValueError(f"Bus {bad} is a dc bus, whose type can not be 2")
    numbers = list_bus[:, 0].astype(int)
    if len(numbers) != int(np.max(numbers)):
        raise ValueError("The total bus number is different from the maximum bus index")
    unique, counts = np.unique(numbers, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"Bus {unique[np.argmax(counts)]} is defined multiple times")
    order = np.argsort(list_bus[:, 0], kind="stable")
    update_bus = list_bus[order]
    return update_bus, update_bus.shape[0]


def _line_rows(case: CaseData) -> np.ndarray:
    rows = [
        [line.FromBus, line.ToBus, line.R, line.wL, line.wC, line.G, line.TurnsRatio]
        for line in case.NetworkLine
    ]
    list_line = np.array(rows, dtype=float) if rows else np.empty((0, 7), dtype=float)
    ieee_rows = [
        [line.FromBus, line.ToBus, line.R, line.X, line.B, line.G, line.TurnsRatio]
        for line in case.NetworkLineIEEE
    ]
    if not ieee_rows:
        return list_line
    ieee = np.array(ieee_rows, dtype=float)
    ieee[np.isnan(ieee)] = np.inf
    fb, tb, r, x, b, g, t = (ieee[:, idx].copy() for idx in range(7))
    n_bus = int(max(np.max(fb), np.max(tb)))
    fb_self = np.arange(1, n_bus + 1, dtype=float)
    b_self = np.zeros(n_bus)
    g_self = np.zeros(n_bus)
    for idx in range(len(fb)):
        if fb[idx] == tb[idx]:
            if not (np.isinf(r[idx]) and np.isinf(x[idx])):
                raise ValueError(f"Branch {idx + 1} is a self branch. Its R and X should be inf")
            if t[idx] != 1:
                raise ValueError(f"Branch {idx + 1} is a self branch. Its turn ratio should be 1")
        b_self[int(fb[idx]) - 1] += b[idx] / 2
        b_self[int(tb[idx]) - 1] += b[idx] / 2
        g_self[int(fb[idx]) - 1] += g[idx] / 2
        g_self[int(tb[idx]) - 1] += g[idx] / 2
    update = np.column_stack(
        [
            np.concatenate([fb, fb_self]),
            np.concatenate([tb, fb_self]),
            np.concatenate([r, np.full(n_bus, np.inf)]),
            np.concatenate([x, np.full(n_bus, np.inf)]),
            np.concatenate([np.zeros_like(b), b_self]),
            np.concatenate([np.zeros_like(g), g_self]),
            np.concatenate([t, np.ones(n_bus)]),
        ]
    )
    keep = ~(((np.isinf(update[:, 2])) | (np.isinf(update[:, 3]))) & (update[:, 4] == 0) & (update[:, 5] == 0))
    return update[keep]


def rearrange_lines(case: CaseData, list_bus: np.ndarray) -> tuple[np.ndarray, int, int]:
    list_line = _line_rows(case)
    if list_line.size == 0:
        return np.empty((0, 8), dtype=float), 0, int(np.max(list_bus[:, 0]))
    list_line = list_line.copy()
    list_line[np.isnan(list_line)] = np.inf
    if list_line.shape[1] > 7:
        raise ValueError("Line data overflow")
    fb, tb = list_line[:, 0].astype(int), list_line[:, 1].astype(int)
    r, x, b, g, t = (list_line[:, idx] for idx in range(2, 7))
    for idx in range(list_line.shape[0]):
        if ((np.isinf(r[idx]) or np.isinf(x[idx])) and b[idx] == 0 and g[idx] == 0):
            raise ValueError(f"Branch{fb[idx]}{tb[idx]} is open circuit")
        if ((r[idx] == 0 and x[idx] == 0) or np.isinf(b[idx]) or np.isinf(g[idx])):
            raise ValueError(f"Branch{fb[idx]}{tb[idx]} is short circuit")
        if r[idx] < 0 or x[idx] < 0 or b[idx] < 0 or g[idx] < 0:
            raise ValueError("Negative line paramters")
        if t[idx] <= 0:
            raise ValueError("Turns ratio can not be less than or equal to 0")
        if fb[idx] == tb[idx] and r[idx] != 0 and not np.isinf(r[idx]):
            raise ValueError("The self branch has to be a LCG parallel branch with R = 0 or a CG parallel branch with R = inf")
    area_by_bus = {int(row[0]): int(row[11]) for row in list_bus}
    area_col = []
    for from_bus, to_bus in zip(fb, tb):
        if area_by_bus[from_bus] != area_by_bus[to_bus]:
            raise ValueError(f"The branch {from_bus}-{to_bus} is a hybrid branch")
        area_col.append(area_by_bus[from_bus])
    list_line = np.column_stack([list_line, np.array(area_col, dtype=float)])
    for idx in range(list_line.shape[0]):
        if list_line[idx, 0] > list_line[idx, 1]:
            list_line[idx, [0, 1]] = list_line[idx, [1, 0]]
            list_line[idx, 2] *= list_line[idx, 6] ** 2
            list_line[idx, 3] *= list_line[idx, 6] ** 2
            list_line[idx, 6] = 1 / list_line[idx, 6]
    order = np.lexsort((list_line[:, 1], list_line[:, 0]))
    list_line = list_line[order]
    return list_line, list_line.shape[0], int(max(np.max(list_line[:, 0]), np.max(list_line[:, 1])))


def rearrange_apparatus(case: CaseData) -> tuple[list[tuple[int, ...]], list[int], list[dict]]:
    apparatus: list[Apparatus] = case.Apparatus
    return [item.BusNo for item in apparatus], [item.Type for item in apparatus], [item.Para for item in apparatus]


def normalize_case(case: CaseData) -> NormalizedNetlists:
    buses, _ = rearrange_buses(case)
    lines, _, _ = rearrange_lines(case, buses)
    apparatus_buses, apparatus_types, apparatus_params = rearrange_apparatus(case)
    return NormalizedNetlists(buses, lines, apparatus_buses, apparatus_types, apparatus_params)


def find_branch(lines: np.ndarray, from_bus: int, to_bus: int) -> int | None:
    matches = np.flatnonzero((lines[:, 0] == from_bus) & (lines[:, 1] == to_bus))
    return int(matches[0]) if len(matches) else None
