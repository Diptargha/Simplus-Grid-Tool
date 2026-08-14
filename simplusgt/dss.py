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
        return dss2ss(self)

    def eigenvalues(self) -> np.ndarray:
        if self.nx == 0:
            return np.array([], dtype=complex)
        try:
            a, _, _, _ = dss2ss(self)
            return np.linalg.eigvals(a)
        except (np.linalg.LinAlgError, ValueError):
            return eig(self.A, self.E, right=False)


def _left_inverse(matrix: np.ndarray, toler: float = 1e-14) -> np.ndarray:
    """Left inverse matching MATLAB ``lft_inv`` via QR."""

    q, r = np.linalg.qr(matrix, mode="complete")
    cols = matrix.shape[1]
    rank = int(np.linalg.matrix_rank(r, tol=toler))
    if rank < cols:
        raise np.linalg.LinAlgError("Matrix is not left-invertible")
    p = r[:cols, :]
    s = np.hstack([np.linalg.inv(p), np.zeros((cols, matrix.shape[0] - cols))])
    return s @ np.linalg.inv(q)


def dss2ss(gdss: DescriptorStateSpace, toler: float = 1e-14) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a descriptor model to explicit state space.

    Mirrors MATLAB ``SimplusGT.dss2ss`` for diagonal ``E``. Dynamic states are
    those with ``|E_ii| >= 1e-12``; algebraic states are eliminated.
    """

    if gdss.nx == 0:
        return gdss.A.copy(), gdss.B.copy(), gdss.C.copy(), gdss.D.copy()

    e = np.asarray(gdss.E, dtype=float)
    if e.shape != (gdss.nx, gdss.nx) or not np.allclose(e, np.diag(np.diag(e)), atol=toler):
        # Fall back to a dense solve when E is nonsingular and non-diagonal.
        return (
            np.linalg.solve(gdss.E, gdss.A),
            np.linalg.solve(gdss.E, gdss.B),
            gdss.C.copy(),
            gdss.D.copy(),
        )

    diag_e = np.diag(e)
    index1 = np.flatnonzero(np.abs(diag_e) >= 1e-12)
    index2 = np.flatnonzero(np.abs(diag_e) < 1e-12)
    a = np.asarray(gdss.A, dtype=float)
    b = np.asarray(gdss.B, dtype=float)
    c = np.asarray(gdss.C, dtype=float)
    d = np.asarray(gdss.D, dtype=float)

    if index1.size == 0:
        # Purely algebraic descriptor system: solve 0 = A x + B u for the map u->y.
        if index2.size == 0:
            return a.copy(), b.copy(), c.copy(), d.copy()
        a22 = a[np.ix_(index2, index2)]
        inv_a22 = np.linalg.inv(a22)
        return (
            np.zeros((0, 0)),
            np.zeros((0, gdss.nu)),
            np.zeros((gdss.ny, 0)),
            d - c[:, index2] @ inv_a22 @ b[index2, :],
        )

    a11 = a[np.ix_(index1, index1)]
    a12 = a[np.ix_(index1, index2)]
    a21 = a[np.ix_(index2, index1)]
    a22 = a[np.ix_(index2, index2)]
    b1 = b[index1, :]
    b2 = b[index2, :]
    c1 = c[:, index1]
    c2 = c[:, index2]
    e1 = np.diag(diag_e[index1])

    if index2.size == 0:
        inv_e1 = np.linalg.inv(e1)
        return inv_e1 @ a11, inv_e1 @ b1, c1.copy(), d.copy()

    if np.linalg.matrix_rank(a22, tol=toler) == a22.shape[0]:
        inv_a22 = np.linalg.inv(a22)
        a_ = a11 - a12 @ inv_a22 @ a21
        b_ = b1 - a12 @ inv_a22 @ b2
        c_ = c1 - c2 @ inv_a22 @ a21
        d_ = d - c2 @ inv_a22 @ b2
        inv_e1 = np.linalg.inv(e1)
        return inv_e1 @ a_, inv_e1 @ b_, c_, d_

    # Singular A22 path (MATLAB null-space reduction).
    from scipy.linalg import null_space

    n = null_space(a22.T).T
    if n.size == 0:
        raise np.linalg.LinAlgError("Unable to reduce singular algebraic block in dss2ss")
    k = n @ a21 @ np.linalg.inv(e1)
    a_21 = np.vstack([a21, k @ a11])
    a_22 = np.vstack([a22, k @ a12])
    b_2 = np.vstack([b2, k @ b1])
    f = np.vstack([np.zeros((a22.shape[0], b2.shape[1])), n @ b2])
    w = _left_inverse(a_22, toler)
    ei = np.linalg.inv(e1)
    a_ = ei @ (a11 - a12 @ w @ a_21)
    b_ = ei @ (b1 - a12 @ w @ b_2)
    bd = -ei @ a12 @ w @ f
    c_ = c1 - c2 @ w @ a_21
    d_ = d - c2 @ w @ b_2
    dd = -c2 @ w @ f
    a_[np.abs(a_) < toler] = 0.0
    b_[np.abs(b_) < toler] = 0.0
    c_[np.abs(c_) < toler] = 0.0
    d_[np.abs(d_) < toler] = 0.0
    if np.max(np.abs(bd)) >= toler or np.max(np.abs(dd)) >= toler:
        raise ValueError("Descriptor system is improper and cannot be converted to state space")
    return a_, b_, c_, d_


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
    """Switch the leading ``length`` inputs/outputs, matching MATLAB ``DssSwitchInOut``."""

    if length > min(g.nu, g.ny):
        raise ValueError("Switch length exceeds input/output dimensions")
    nx, nu, ny = g.nx, g.nu, g.ny
    l = length
    b1 = g.B[:, :l] if nx else np.zeros((0, l))
    b2 = g.B[:, l:] if nx else np.zeros((0, nu - l))
    c1 = g.C[:l, :] if nx else np.zeros((l, 0))
    c2 = g.C[l:, :] if nx else np.zeros((ny - l, 0))
    d11 = g.D[:l, :l]
    d12 = g.D[:l, l:]
    d21 = g.D[l:, :l]
    d22 = g.D[l:, l:]

    a = np.block([
        [g.A if nx else np.zeros((0, 0)), b1],
        [-c1, -d11],
    ])
    b = np.vstack([
        np.hstack([np.zeros((nx, l)), b2]),
        np.hstack([np.eye(l), -d12]),
    ])
    c = np.block([
        [np.zeros((l, nx)), np.eye(l)],
        [c2, d21],
    ])
    d = np.block([
        [np.zeros((l, l)), np.zeros((l, nu - l))],
        [np.zeros((ny - l, l)), d22],
    ])
    e = block_diag(g.E if nx else np.zeros((0, 0)), np.zeros((l, l)))
    base_states = list(g.states) if g.states else [f"x{idx + 1}" for idx in range(nx)]
    states = base_states + [f"xi_{idx + 1}" for idx in range(l)]
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
    states = list(g.states) + [f"inv_u_{idx + 1}" for idx in range(n)] if g.states else []
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
