"""Descriptor state-space utilities.

The core model is:

    E xdot = A x + B u
    y      = C x + D u
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag, eig


def _matrix(value: np.ndarray | list | float | int | None, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    if value is None:
        if rows is None or cols is None:
            return np.empty((0, 0), dtype=float)
        return np.zeros((rows, cols), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


@dataclass
class DescriptorStateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray
    states: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.A = _matrix(self.A)
        self.B = _matrix(self.B)
        self.C = _matrix(self.C)
        self.D = _matrix(self.D)
        self.E = _matrix(self.E)
        if self.A.size == 0:
            nx = 0
            ny, nu = self.D.shape
            self.A = np.zeros((0, 0))
            self.B = np.zeros((0, nu))
            self.C = np.zeros((ny, 0))
            self.E = np.zeros((0, 0))
        else:
            nx = self.A.shape[0]
            if self.E.size == 0:
                self.E = np.eye(nx)
        self.check_dimensions()

    @classmethod
    def static(cls, D: np.ndarray, *, inputs: list[str] | None = None, outputs: list[str] | None = None) -> "DescriptorStateSpace":
        d = _matrix(D)
        return cls(np.zeros((0, 0)), np.zeros((0, d.shape[1])), np.zeros((d.shape[0], 0)), d, np.zeros((0, 0)),
                   [], inputs or [], outputs or [])

    @classmethod
    def from_state_space(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        *,
        states: list[str] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
    ) -> "DescriptorStateSpace":
        a = _matrix(A)
        return cls(a, B, C, D, np.eye(a.shape[0]), states or [], inputs or [], outputs or [])

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.D.shape[1]

    @property
    def ny(self) -> int:
        return self.D.shape[0]

    def copy(self) -> "DescriptorStateSpace":
        return DescriptorStateSpace(
            self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy(), self.E.copy(),
            list(self.states), list(self.inputs), list(self.outputs)
        )

    def check_dimensions(self) -> None:
        nx = self.A.shape[0]
        if self.A.shape != (nx, nx):
            raise ValueError("A must be square")
        if self.E.shape != (nx, nx):
            raise ValueError("E dimension mismatch")
        if self.B.shape[0] != nx:
            raise ValueError("B dimension mismatch")
        if self.C.shape[1] != nx:
            raise ValueError("C dimension mismatch")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D dimension mismatch")
        if self.states and len(self.states) != nx:
            raise ValueError("State string dimension mismatch")
        if self.inputs and len(self.inputs) != self.nu:
            raise ValueError("Input string dimension mismatch")
        if self.outputs and len(self.outputs) != self.ny:
            raise ValueError("Output string dimension mismatch")

    def scaled(self, factor: float) -> "DescriptorStateSpace":
        return DescriptorStateSpace(
            self.A, self.B, factor * self.C, factor * self.D, self.E,
            list(self.states), list(self.inputs), list(self.outputs)
        )

    def truncate(self, outputs: list[int], inputs: list[int]) -> "DescriptorStateSpace":
        return DescriptorStateSpace(
            self.A.copy(),
            self.B[:, inputs].copy(),
            self.C[outputs, :].copy(),
            self.D[np.ix_(outputs, inputs)].copy(),
            self.E.copy(),
            list(self.states),
            [self.inputs[i] for i in inputs] if self.inputs else [],
            [self.outputs[i] for i in outputs] if self.outputs else [],
        )

    def to_state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.nx == 0:
            return self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy()
        return np.linalg.solve(self.E, self.A), np.linalg.solve(self.E, self.B), self.C.copy(), self.D.copy()

    def eigenvalues(self) -> np.ndarray:
        if self.nx == 0:
            return np.array([], dtype=complex)
        return eig(self.A, self.E, right=False)


def empty(inputs: int = 0, outputs: int = 0) -> DescriptorStateSpace:
    return DescriptorStateSpace.static(np.zeros((outputs, inputs)))


def append(g1: DescriptorStateSpace, g2: DescriptorStateSpace) -> DescriptorStateSpace:
    return DescriptorStateSpace(
        block_diag(g1.A, g2.A),
        block_diag(g1.B, g2.B),
        block_diag(g1.C, g2.C),
        block_diag(g1.D, g2.D),
        block_diag(g1.E, g2.E),
        list(g1.states) + list(g2.states),
        list(g1.inputs) + list(g2.inputs),
        list(g1.outputs) + list(g2.outputs),
    )


def parallel_sum(g1: DescriptorStateSpace, g2: DescriptorStateSpace) -> DescriptorStateSpace:
    if g1.nu != g2.nu or g1.ny != g2.ny:
        raise ValueError("Parallel sum requires equal input and output dimensions")
    return DescriptorStateSpace(
        block_diag(g1.A, g2.A),
        np.vstack([g1.B, g2.B]),
        np.hstack([g1.C, g2.C]),
        g1.D + g2.D,
        block_diag(g1.E, g2.E),
        list(g1.states) + list(g2.states),
        list(g1.inputs),
        list(g1.outputs),
    )


def arrange(cells: list[list[DescriptorStateSpace]]) -> DescriptorStateSpace:
    rows: list[DescriptorStateSpace] = []
    for row in cells:
        rows.append(_horizontal(row))
    return _vertical(rows)


def _horizontal(row: list[DescriptorStateSpace]) -> DescriptorStateSpace:
    if not row:
        return empty()
    ny = row[0].ny
    if any(item.ny != ny for item in row):
        raise ValueError("Horizontal DSS arrangement requires equal output dimensions")
    A = block_diag(*[item.A for item in row])
    E = block_diag(*[item.E for item in row])
    B = block_diag(*[item.B for item in row])
    C = np.hstack([item.C for item in row]) if A.size else np.zeros((ny, 0))
    D = np.hstack([item.D for item in row])
    return DescriptorStateSpace(A, B, C, D, E, sum((item.states for item in row), []),
                                sum((item.inputs for item in row), []), list(row[0].outputs))


def _vertical(rows: list[DescriptorStateSpace]) -> DescriptorStateSpace:
    if not rows:
        return empty()
    nu = rows[0].nu
    if any(item.nu != nu for item in rows):
        raise ValueError("Vertical DSS arrangement requires equal input dimensions")
    A = block_diag(*[item.A for item in rows])
    E = block_diag(*[item.E for item in rows])
    B = np.vstack([item.B for item in rows]) if A.size else np.zeros((0, nu))
    C = block_diag(*[item.C for item in rows])
    D = np.vstack([item.D for item in rows])
    return DescriptorStateSpace(A, B, C, D, E, sum((item.states for item in rows), []),
                                list(rows[0].inputs), sum((item.outputs for item in rows), []))


def switch_inputs_outputs(g: DescriptorStateSpace, length: int) -> DescriptorStateSpace:
    if length > min(g.nu, g.ny):
        raise ValueError("Switch length exceeds input/output dimensions")
    # Algebraic state xi represents the old output now exposed as an input.
    nx, nu, ny = g.nx, g.nu, g.ny
    l = length
    a = np.block([
        [g.A, np.zeros((nx, l))],
        [-g.C[:l, :], np.zeros((l, l))],
    ])
    b = np.vstack([
        np.hstack([g.B[:, :l], g.B[:, l:]]),
        np.hstack([-g.D[:l, :l], -g.D[:l, l:]]),
    ])
    c = np.block([
        [-g.D[:l, :l] @ g.C[:l, :] if nx else np.zeros((l, nx)), np.eye(l)],
        [g.C[l:, :], np.zeros((ny - l, l))],
    ])
    d_top = np.hstack([np.eye(l) - g.D[:l, :l] @ g.D[:l, :l], -g.D[:l, l:]]) if l else np.zeros((0, nu))
    d = np.vstack([d_top, np.hstack([g.D[l:, :l], g.D[l:, l:]])])
    e = block_diag(g.E, np.zeros((l, l)))
    states = list(g.states) + [f"xi_{idx + 1}" for idx in range(l)]
    inputs = list(g.outputs[:l]) + list(g.inputs[l:]) if g.inputs and g.outputs else []
    outputs = list(g.inputs[:l]) + list(g.outputs[l:]) if g.inputs and g.outputs else []
    return DescriptorStateSpace(a, b, c, d, e, states, inputs, outputs)


def inverse(g: DescriptorStateSpace) -> DescriptorStateSpace:
    """Return a descriptor realization of the inverse transfer matrix."""

    if g.nu != g.ny:
        raise ValueError("Descriptor inverse requires a square transfer matrix")
    nx, n = g.nx, g.nu
    a = np.block([
        [g.A, g.B],
        [g.C, g.D],
    ])
    b = np.vstack([np.zeros((nx, n)), -np.eye(n)])
    c = np.hstack([np.zeros((n, nx)), np.eye(n)])
    d = np.zeros((n, n))
    e = block_diag(g.E, np.zeros((n, n)))
    states = list(g.states) + [f"inv_u_{idx + 1}" for idx in range(n)]
    return DescriptorStateSpace(a, b, c, d, e, states, list(g.outputs), list(g.inputs))


def feedback(
    g1: DescriptorStateSpace,
    g2: DescriptorStateSpace,
    feedin: list[int],
    feedout: list[int],
    sign: float = -1.0,
) -> DescriptorStateSpace:
    """Close a selected-channel feedback loop.

    This mirrors MATLAB's selected-channel `feedback` at the matrix level and
    keeps all external inputs/outputs of `g1`.
    """

    feedin = list(feedin)
    feedout = list(feedout)
    if len(feedin) != g2.ny or len(feedout) != g2.nu:
        raise ValueError("Feedback channel dimensions do not match g2")
    s_in = np.zeros((g1.nu, len(feedin)))
    for col, idx in enumerate(feedin):
        s_in[idx, col] = 1.0
    s_out = np.zeros((len(feedout), g1.ny))
    for row, idx in enumerate(feedout):
        s_out[row, idx] = 1.0
    # Solve direct-feedthrough loop:
    # u2 = S_out(C1 x1 + D1 u1), u1 = u_ext + sign*S_in(C2 x2 + D2 u2)
    m = np.eye(g2.nu) - sign * s_out @ g1.D @ s_in @ g2.D
    inv_m = np.linalg.inv(m)
    x_to_u2 = inv_m @ np.hstack([s_out @ g1.C, sign * s_out @ g1.D @ s_in @ g2.C])
    uext_to_u2 = inv_m @ (s_out @ g1.D)
    x_to_u1 = np.hstack([np.zeros((g1.nu, g1.nx)), sign * s_in @ g2.C]) + sign * s_in @ g2.D @ x_to_u2
    uext_to_u1 = np.eye(g1.nu) + sign * s_in @ g2.D @ uext_to_u2
    a = np.block([
        [g1.A, np.zeros((g1.nx, g2.nx))],
        [np.zeros((g2.nx, g1.nx)), g2.A],
    ]) + np.vstack([g1.B @ x_to_u1, g2.B @ x_to_u2])
    b = np.vstack([g1.B @ uext_to_u1, g2.B @ uext_to_u2])
    c = np.hstack([g1.C, np.zeros((g1.ny, g2.nx))]) + g1.D @ x_to_u1
    d = g1.D @ uext_to_u1
    e = block_diag(g1.E, g2.E)
    return DescriptorStateSpace(a, b, c, d, e, list(g1.states) + list(g2.states), list(g1.inputs), list(g1.outputs))
