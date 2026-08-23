"""Analysis helpers for Python SimplusGT results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dss import DescriptorStateSpace
from .powerflow import ybus_calc


@dataclass(frozen=True)
class StabilityReport:
    eigenvalues_hz: np.ndarray
    stable: bool
    unstable_modes: np.ndarray


@dataclass(frozen=True)
class ModeParticipation:
    mode_index: int
    eigenvalue: complex
    eigenvalue_hz: complex
    frequency_hz: float
    damping_ratio: float | None
    state_participation: list[tuple[str, float]]


def stability_report(eigenvalues: np.ndarray, tolerance_hz: float = 1e-4) -> StabilityReport:
    finite_values = eigenvalues[np.isfinite(eigenvalues)]
    finite = finite_values / (2 * np.pi)
    unstable = finite[np.real(finite) > tolerance_hz]
    return StabilityReport(eigenvalues_hz=finite, stable=len(unstable) == 0, unstable_modes=unstable)


def descriptor_modes(model: DescriptorStateSpace, max_states_per_mode: int | None = None) -> list[ModeParticipation]:
    """Compute finite generalized modes and state participation factors."""

    if model.nx == 0:
        return []
    from scipy.linalg import eig

    eigenvalues, left, right = eig(model.A, model.E, left=True, right=True)
    modes: list[ModeParticipation] = []
    state_names = model.states or [f"x{idx + 1}" for idx in range(model.nx)]
    finite_indices = [idx for idx, value in enumerate(eigenvalues) if np.isfinite(value)]
    finite_indices.sort(key=lambda idx: (np.real(eigenvalues[idx]), abs(np.imag(eigenvalues[idx]))))
    for output_index, eig_index in enumerate(finite_indices):
        value = eigenvalues[eig_index]
        left_vec = left[:, eig_index]
        right_vec = right[:, eig_index]
        scale = np.vdot(left_vec, model.E @ right_vec)
        if abs(scale) > 1e-12:
            left_vec = left_vec / np.conj(scale)
        factors = np.abs(right_vec * np.conj(left_vec))
        if np.max(factors) > 0:
            factors = factors / np.max(factors)
        pairs = sorted(zip(state_names, factors.astype(float)), key=lambda item: item[1], reverse=True)
        if max_states_per_mode is not None:
            pairs = pairs[:max_states_per_mode]
        sigma = float(np.real(value))
        omega = float(abs(np.imag(value)))
        denom = float(abs(value))
        damping_ratio = None if denom == 0 else -sigma / denom
        modes.append(
            ModeParticipation(
                mode_index=output_index,
                eigenvalue=value,
                eigenvalue_hz=value / (2 * np.pi),
                frequency_hz=omega / (2 * np.pi),
                damping_ratio=damping_ratio,
                state_participation=pairs,
            )
        )
    return modes


def frequency_response(model: DescriptorStateSpace, omega: np.ndarray) -> np.ndarray:
    """Evaluate C (jwE - A)^-1 B + D for a descriptor model."""

    responses = np.zeros((len(omega), model.ny, model.nu), dtype=complex)
    for idx, w in enumerate(omega):
        if model.nx:
            responses[idx] = model.C @ np.linalg.solve(1j * w * model.E - model.A, model.B) + model.D
        else:
            responses[idx] = model.D
    return responses


def plot_pole_map(eigenvalues: np.ndarray, *, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    eig_hz = eigenvalues[np.isfinite(eigenvalues)] / (2 * np.pi)
    ax.scatter(np.real(eig_hz), np.imag(eig_hz), marker="x")
    ax.set_xlabel("Real Part (Hz)")
    ax.set_ylabel("Imaginary Part (Hz)")
    ax.set_title("Global pole map")
    ax.grid(True)
    return ax


def bus_type_vif(apparatus_types: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """MATLAB ``BusTypeVIF`` — 1-based bus indices as voltage / current / floating."""

    vbus: list[int] = []
    ibus: list[int] = []
    fbus: list[int] = []
    for idx, app_type in enumerate(apparatus_types, start=1):
        app_type = int(app_type)
        if (0 <= app_type <= 9) or app_type == 90 or (20 <= app_type <= 40) or app_type == 50:
            vbus.append(idx)
        elif (10 <= app_type <= 19) or app_type in {41, 51}:
            ibus.append(idx)
        elif app_type == 100:
            fbus.append(idx)
        else:
            raise ValueError(f"Unknown apparatus type {app_type} at index {idx}")
    return vbus, ibus, fbus


def bus_type_name(bus: int, vbus: Sequence[int], ibus: Sequence[int], fbus: Sequence[int]) -> str:
    """Human-readable V/I/F bus type for grid-strength hovers and labels."""

    if bus in vbus:
        return "Voltage bus (V)"
    if bus in ibus:
        return "Current bus (I)"
    if bus in fbus:
        return "Floating bus (F)"
    return "Unknown"


def matrix_partition(
    matrix: np.ndarray, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB ``MatrixPartition``."""

    matrix = np.asarray(matrix)
    return (
        matrix[:rows, :cols],
        matrix[:rows, cols:],
        matrix[rows:, :cols],
        matrix[rows:, cols:],
    )


def matrix_twist(matrix: np.ndarray, index: int) -> np.ndarray:
    """MATLAB ``MatrixTwist`` — hybrid form that swaps ports after ``index``.

    Original: ``[y1; y2] = G [u1; u2]``.
    Twisted:  ``[y1; u2] = H [u1; y2]``.
    """

    g = np.asarray(matrix, dtype=complex)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("MatrixTwist requires a square matrix")
    n = g.shape[0]
    if index < 0 or index > n:
        raise ValueError(f"MatrixTwist index {index} out of range for size {n}")
    if index == 0:
        return np.linalg.inv(g)
    if index == n:
        return g.copy()
    g11, g12, g21, g22 = matrix_partition(g, index, index)
    inv_g22 = np.linalg.inv(g22)
    h11 = g11 - g12 @ inv_g22 @ g21
    h12 = g12 @ inv_g22
    h21 = -inv_g22 @ g21
    h22 = inv_g22
    return np.block([[h11, h12], [h21, h22]])


def bus_strength(list_line: np.ndarray, apparatus_types: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``BusStrength`` — hybrid/Thevenin-style bus strength diagonals.

    Voltage buses use ``H_ii``; current and floating buses use ``1/H_ii``.
    ``graph_matrix`` holds branch ``|y|`` weights for network layout plots.
    """

    lines = np.asarray(list_line, dtype=float)
    ybus = ybus_calc(lines)
    n_bus = int(ybus.shape[0]) if ybus.size else len(apparatus_types)
    if n_bus == 0:
        return np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=float)

    vbus, ibus, fbus = bus_type_vif(apparatus_types)
    order = vbus + ibus + fbus
    if len(order) != n_bus or sorted(order) != list(range(1, n_bus + 1)):
        raise ValueError(
            "BusStrength requires one apparatus type per bus covering 1..N "
            f"(got {len(apparatus_types)} types for {n_bus} buses)"
        )

    order0 = [idx - 1 for idx in order]
    ybus_sort = ybus[np.ix_(order0, order0)]
    hbus_sort = matrix_twist(ybus_sort, len(vbus))
    hbus = np.zeros((n_bus, n_bus), dtype=complex)
    hbus[np.ix_(order0, order0)] = hbus_sort

    vbus_set = set(vbus)
    ydiag = np.zeros(n_bus, dtype=complex)
    for bus in range(1, n_bus + 1):
        hii = hbus[bus - 1, bus - 1]
        if bus in vbus_set:
            ydiag[bus - 1] = hii
        elif abs(hii) < 1e-18:
            ydiag[bus - 1] = np.inf
        else:
            ydiag[bus - 1] = 1.0 / hii

    graph_matrix = np.zeros((n_bus, n_bus), dtype=float)
    if lines.size:
        for row in lines:
            f, t = int(row[0]) - 1, int(row[1]) - 1
            r, x, b, g, _tap, area = row[2], row[3], row[4], row[5], row[6], int(row[7])
            if area == 1:
                series = 0.0 if (np.isinf(r) or np.isinf(x)) else 1 / (r + 1j * x)
                y = g + 1j * b + series
            else:
                series = 0.0 if np.isinf(r) else 1 / r
                y = g + series
            if f != t:
                graph_matrix[f, t] = graph_matrix[t, f] = float(abs(y))
    return ydiag, graph_matrix


def topology_graph(list_line: np.ndarray) -> nx.Graph:
    import networkx as nx

    graph = nx.Graph()
    for row in list_line:
        f, t = int(row[0]), int(row[1])
        graph.add_node(f)
        graph.add_node(t)
        if f != t:
            graph.add_edge(f, t)
    return graph


def participation_factors(a: np.ndarray) -> np.ndarray:
    """Return normalized state participation factors for a standard A matrix."""

    eigvals, right = np.linalg.eig(a)
    left = np.linalg.inv(right)
    factors = np.abs(right * left.T)
    column_sums = factors.sum(axis=0)
    column_sums[column_sums == 0] = 1
    return factors / column_sums
