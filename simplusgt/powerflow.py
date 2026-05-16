"""Power-flow routines ported from `+SimplusGT/+PowerFlow`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .netlists import find_branch


@dataclass(frozen=True)
class PowerFlowResult:
    power_flow: list[np.ndarray]
    ybus: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    angle: np.ndarray
    p: np.ndarray
    q: np.ndarray
    vm: np.ndarray


def pol2rect(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * angle)


def ybus_calc(lines: np.ndarray) -> np.ndarray:
    if lines.size == 0:
        return np.zeros((0, 0), dtype=complex)
    fb = lines[:, 0].astype(int)
    tb = lines[:, 1].astype(int)
    r, x, b, g, tap, area = (lines[:, idx] for idx in range(2, 8))
    n_bus = int(max(np.max(fb), np.max(tb)))
    y = np.zeros(len(fb), dtype=complex)
    for idx in range(len(fb)):
        if int(area[idx]) == 1:
            yp = g[idx] + 1j * b[idx]
            series = 0.0 if (np.isinf(r[idx]) or np.isinf(x[idx])) else 1 / (r[idx] + 1j * x[idx])
            y[idx] = yp + series
        elif int(area[idx]) == 2:
            if r[idx] == 0 and np.isinf(g[idx]):
                raise ValueError(f"Branch {fb[idx]}-{tb[idx]} is a DC branch, whose resistance can NOT be zero")
            series = 0.0 if np.isinf(r[idx]) else 1 / r[idx]
            y[idx] = g[idx] + series
        else:
            raise ValueError(f"Unknown area type {area[idx]}")
    ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for idx in range(len(fb)):
        f = fb[idx] - 1
        t = tb[idx] - 1
        if f != t:
            ybus[f, t] -= y[idx] / tap[idx]
            ybus[t, f] = ybus[f, t]
            ybus[f, f] += y[idx] / (tap[idx] ** 2)
            ybus[t, t] += y[idx]
        else:
            ybus[f, t] += y[idx]
    return ybus


def power_flow_gs(list_bus: np.ndarray, list_line: np.ndarray, w0: float) -> PowerFlowResult:
    ybus = ybus_calc(list_line)
    n_bus = int(np.max(list_bus[:, 0]))
    bus_type = list_bus[:, 1].astype(int).copy()
    slack = np.flatnonzero(bus_type == 1)
    v0 = list_bus[:, 2]
    th0 = list_bus[:, 3]
    p = list_bus[:, 4] - list_bus[:, 6]
    q = list_bus[:, 5] - list_bus[:, 7]
    qmin = list_bus[:, 8]
    qmax = list_bus[:, 9]
    voltage = pol2rect(v0, th0)
    previous = voltage.copy()
    tolerance = 1.0
    iteration = 0
    current = ybus @ voltage
    while tolerance > 1e-8 and iteration <= 100_000:
        for idx in range(n_bus):
            if bus_type[idx] == 1:
                continue
            sum_yv = sum(ybus[idx, col] * voltage[col] for col in range(n_bus) if col != idx)
            if bus_type[idx] == 2:
                q[idx] = -np.imag(np.conj(voltage[idx]) * (sum_yv + ybus[idx, idx] * voltage[idx]))
                if q[idx] > qmax[idx] or q[idx] < qmin[idx]:
                    q[idx] = qmin[idx] if q[idx] < qmin[idx] else qmax[idx]
                    bus_type[idx] = 3
            voltage[idx] = ((p[idx] - 1j * q[idx]) / np.conj(voltage[idx]) - sum_yv) / ybus[idx, idx]
            if bus_type[idx] == 2:
                voltage[idx] = abs(previous[idx]) * np.exp(1j * np.angle(voltage[idx]))
        iteration += 1
        if iteration == 100_000:
            raise RuntimeError("The PowerFlow does not converge or needs more steps to converge")
        current = ybus @ voltage
        apparent = voltage * np.conj(current)
        toler_v = np.max(np.abs(np.abs(voltage) - np.abs(previous)))
        toler_p = np.abs(np.real(apparent) - p)
        toler_p[slack] = 0
        tolerance = max(toler_v, float(np.max(toler_p)), 0.0)
        previous = voltage.copy()
    angle = np.angle(voltage)
    vm = np.abs(voltage)
    apparent = voltage * np.conj(current)
    p_final = np.real(apparent)
    q_final = np.imag(apparent)
    power_flow = [np.array([-p_final[idx], -q_final[idx], vm[idx], angle[idx], w0], dtype=float) for idx in range(n_bus)]
    return PowerFlowResult(power_flow, ybus, voltage, current, angle, p_final, q_final, vm)


def load_flow(nbus: int, v: np.ndarray, delta: np.ndarray, ybus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vm = pol2rect(v, delta)
    si = np.zeros(nbus, dtype=complex)
    for idx in range(nbus):
        for col in range(nbus):
            si[idx] += np.conj(vm[idx]) * vm[col] * ybus[idx, col]
    return np.real(si), -np.imag(si)


def power_flow_nr(list_bus: np.ndarray, list_line: np.ndarray, w0: float) -> PowerFlowResult:
    y = ybus_calc(list_line)
    nbus = len(list_bus)
    bus_type = list_bus[:, 1].astype(int)
    voltage_mag = list_bus[:, 2].copy()
    delta = list_bus[:, 3].copy()
    p_spec = list_bus[:, 4] - list_bus[:, 6]
    q_spec = list_bus[:, 5] - list_bus[:, 7]
    qmin = list_bus[:, 8]
    qmax = list_bus[:, 9]
    conductance = np.real(y)
    susceptance = np.imag(y)
    pq = np.flatnonzero(bus_type == 3)
    tol = 1.0
    iteration = 1
    while tol > 1e-5:
        p = np.zeros(nbus)
        q = np.zeros(nbus)
        for idx in range(nbus):
            for col in range(nbus):
                p[idx] += voltage_mag[idx] * voltage_mag[col] * (
                    conductance[idx, col] * np.cos(delta[idx] - delta[col])
                    + susceptance[idx, col] * np.sin(delta[idx] - delta[col])
                )
                q[idx] += voltage_mag[idx] * voltage_mag[col] * (
                    conductance[idx, col] * np.sin(delta[idx] - delta[col])
                    - susceptance[idx, col] * np.cos(delta[idx] - delta[col])
                )
        if 2 < iteration <= 7:
            for idx in range(1, nbus):
                if bus_type[idx] == 2:
                    qg = q[idx] + list_bus[idx, 7]
                    if qg < qmin[idx]:
                        voltage_mag[idx] += 0.01
                    elif qg > qmax[idx]:
                        voltage_mag[idx] -= 0.01
        dp = p_spec - p
        dq_all = q_spec - q
        mismatch = np.concatenate([dp[1:], dq_all[pq]])
        j1 = np.zeros((nbus - 1, nbus - 1))
        for i in range(nbus - 1):
            m = i + 1
            for k in range(nbus - 1):
                n = k + 1
                if n == m:
                    for n2 in range(nbus):
                        j1[i, k] += voltage_mag[m] * voltage_mag[n2] * (
                            -conductance[m, n2] * np.sin(delta[m] - delta[n2])
                            + susceptance[m, n2] * np.cos(delta[m] - delta[n2])
                        )
                    j1[i, k] -= voltage_mag[m] ** 2 * susceptance[m, m]
                else:
                    j1[i, k] = voltage_mag[m] * voltage_mag[n] * (
                        conductance[m, n] * np.sin(delta[m] - delta[n])
                        - susceptance[m, n] * np.cos(delta[m] - delta[n])
                    )
        j2 = np.zeros((nbus - 1, len(pq)))
        for i in range(nbus - 1):
            m = i + 1
            for k, n in enumerate(pq):
                if n == m:
                    for n2 in range(nbus):
                        j2[i, k] += voltage_mag[n2] * (
                            conductance[m, n2] * np.cos(delta[m] - delta[n2])
                            + susceptance[m, n2] * np.sin(delta[m] - delta[n2])
                        )
                    j2[i, k] += voltage_mag[m] * conductance[m, m]
                else:
                    j2[i, k] = voltage_mag[m] * (
                        conductance[m, n] * np.cos(delta[m] - delta[n])
                        + susceptance[m, n] * np.sin(delta[m] - delta[n])
                    )
        j3 = np.zeros((len(pq), nbus - 1))
        for i, m in enumerate(pq):
            for k in range(nbus - 1):
                n = k + 1
                if n == m:
                    for n2 in range(nbus):
                        j3[i, k] += voltage_mag[m] * voltage_mag[n2] * (
                            conductance[m, n2] * np.cos(delta[m] - delta[n2])
                            + susceptance[m, n2] * np.sin(delta[m] - delta[n2])
                        )
                    j3[i, k] -= voltage_mag[m] ** 2 * conductance[m, m]
                else:
                    j3[i, k] = voltage_mag[m] * voltage_mag[n] * (
                        -conductance[m, n] * np.cos(delta[m] - delta[n])
                        - susceptance[m, n] * np.sin(delta[m] - delta[n])
                    )
        j4 = np.zeros((len(pq), len(pq)))
        for i, m in enumerate(pq):
            for k, n in enumerate(pq):
                if n == m:
                    for n2 in range(nbus):
                        j4[i, k] += voltage_mag[n2] * (
                            conductance[m, n2] * np.sin(delta[m] - delta[n2])
                            - susceptance[m, n2] * np.cos(delta[m] - delta[n2])
                        )
                    j4[i, k] -= voltage_mag[m] * susceptance[m, m]
                else:
                    j4[i, k] = voltage_mag[m] * (
                        conductance[m, n] * np.sin(delta[m] - delta[n])
                        - susceptance[m, n] * np.cos(delta[m] - delta[n])
                    )
        jacobian = np.block([[j1, j2], [j3, j4]])
        correction = np.linalg.solve(jacobian, mismatch)
        delta[1:] += correction[: nbus - 1]
        dv = correction[nbus - 1 :]
        for k, idx in enumerate(pq):
            voltage_mag[idx] += dv[k]
        iteration += 1
        tol = float(np.max(np.abs(mismatch))) if mismatch.size else 0.0
        if iteration > 100:
            raise RuntimeError("Newton-Raphson power flow did not converge")
    pi, qi = load_flow(nbus, voltage_mag, delta, y)
    voltage = pol2rect(voltage_mag, delta)
    current = y @ voltage
    power_flow = [np.array([-pi[idx], -qi[idx], voltage_mag[idx], delta[idx], w0], dtype=float) for idx in range(nbus)]
    return PowerFlowResult(power_flow, y, voltage, current, delta, pi, qi, voltage_mag)


def load_to_self_branch(list_bus: np.ndarray, list_line: np.ndarray, power_flow: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    bus_index = list_bus[:, 0].astype(int)
    pl = list_bus[:, 6]
    ql = list_bus[:, 7]
    area_type = list_bus[:, 11]
    update_bus = list_bus.copy()
    update_bus[:, 6] = 0
    update_bus[:, 7] = 0
    update_pf = [item.copy() for item in power_flow]
    for idx, bus_no in enumerate(bus_index):
        if pl[idx] < 0:
            raise ValueError(f"Passive load at bus {bus_no} can not generate active power")
        update_pf[idx][0] = power_flow[idx][0] - pl[idx]
        update_pf[idx][1] = power_flow[idx][1] - ql[idx]
    update_line = list_line.copy()
    for idx, bus_no in enumerate(bus_index):
        voltage = power_flow[idx][2]
        gl = pl[idx] / (voltage ** 2)
        if ql[idx] > 0:
            xl = voltage ** 2 / ql[idx]
            bl = 0.0
        else:
            xl = np.inf
            bl = -ql[idx] / (voltage ** 2)
        if np.isinf(gl) or xl == 0 or np.isinf(bl):
            raise ValueError(f"The passive load at bus {bus_no} is short-circuit")
        if gl != 0 or not np.isinf(xl) or bl != 0:
            branch_idx = find_branch(update_line, int(bus_no), int(bus_no))
            if branch_idx is not None:
                if update_line[branch_idx, 2] != 0 and not np.isinf(update_line[branch_idx, 2]):
                    raise ValueError("A self branch can not have R")
                if not np.isinf(xl):
                    update_line[branch_idx, 2] = 0
                old_x = update_line[branch_idx, 3]
                if old_x == 0 or xl == 0:
                    update_line[branch_idx, 3] = 0
                elif np.isinf(old_x):
                    update_line[branch_idx, 3] = xl
                elif np.isinf(xl):
                    update_line[branch_idx, 3] = old_x
                else:
                    update_line[branch_idx, 3] = 1 / (1 / old_x + 1 / xl)
                update_line[branch_idx, 4] += bl
                update_line[branch_idx, 5] += gl
            else:
                update_line = np.vstack([update_line, [bus_no, bus_no, 0, xl, bl, gl, 1, area_type[idx]]])
    order = np.lexsort((update_line[:, 1], update_line[:, 0]))
    return update_bus, update_line[order], update_pf


def run_power_flow(list_bus: np.ndarray, list_line: np.ndarray, w0: float, algorithm: int) -> PowerFlowResult:
    if np.any(list_bus[:, 11] == 2):
        algorithm = 1
    if algorithm == 1:
        return power_flow_gs(list_bus, list_line, w0)
    if algorithm == 2:
        return power_flow_nr(list_bus, list_line, w0)
    raise ValueError("Wrong setting for power flow algorithm")
