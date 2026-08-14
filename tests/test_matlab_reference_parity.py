"""MATLAB vs Python parity checks for the IEEE 14-bus greybox reference.

Requires ``Results/matlab_reference.mat``. Tests skip cleanly when the file is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simplusgt.greybox import FrequencyGrid, GreyboxConfig, GreyboxLayerSelection, run_greybox


ROOT = Path(__file__).resolve().parents[1]
MATLAB_REFERENCE_PATH = ROOT / "Results" / "matlab_reference.mat"
IEEE14_CASE = ROOT / "Examples" / "AcPowerSystem" / "IEEE_14Bus" / "IEEE_14Bus.json"

# Frequency-response tolerances (complex entries).
YSYS_RTOL = 1e-4
YSYS_ATOL = 1e-4
ZSYS_RTOL = 1e-4
ZSYS_ATOL = 1e-4

# Eigenvalue nearest-neighbor tolerances (rad/s).
# Near-zero modes from DSS reduction can differ by ~1e-4 between MATLAB/Python.
EIG_RTOL = 1e-4
EIG_ATOL = 5e-4


def _require_scipy_io():
    scipy = pytest.importorskip("scipy")
    return scipy.io


def load_matlab_reference(path: Path = MATLAB_REFERENCE_PATH) -> dict[str, Any]:
    """Load a MATLAB ``.mat`` reference with structs as objects."""

    sio = _require_scipy_io()
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def normalize_matlab_array(value: Any, *, dtype=None) -> np.ndarray:
    """Normalize MATLAB scalar / cell / struct-squeezed values into a NumPy array."""

    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, np.ndarray) and value.dtype == object:
        flat = [normalize_matlab_array(item, dtype=dtype) for item in value.ravel(order="F")]
        try:
            stacked = np.stack(flat)
        except ValueError:
            return np.asarray(flat, dtype=object)
        return stacked.reshape(value.shape + flat[0].shape) if flat else np.asarray([], dtype=dtype)
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def as_complex_vector(value: Any) -> np.ndarray:
    """Return a 1-D complex vector of finite eigenvalues."""

    vector = normalize_matlab_array(value, dtype=complex).ravel()
    return vector[np.isfinite(vector)]


def align_matlab_freq_response(values: Any) -> np.ndarray:
    """Convert MATLAB ``(output, input, frequency)`` to Python ``(frequency, output, input)``."""

    array = normalize_matlab_array(values, dtype=complex)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D frequency response, got shape {array.shape}")
    return np.transpose(array, (2, 0, 1))


def complex_error_stats(actual: np.ndarray, desired: np.ndarray) -> tuple[float, float]:
    """Return ``(max_abs_error, max_rel_error)`` for complex arrays."""

    diff = np.abs(np.asarray(actual, dtype=complex) - np.asarray(desired, dtype=complex))
    scale = np.maximum(np.abs(actual), np.abs(desired))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(scale > 0, diff / scale, np.where(diff == 0, 0.0, np.inf))
    finite_diff = diff[np.isfinite(diff)]
    finite_rel = rel[np.isfinite(rel)]
    max_abs = float(np.max(finite_diff)) if finite_diff.size else float("nan")
    max_rel = float(np.max(finite_rel)) if finite_rel.size else float("nan")
    return max_abs, max_rel


def assert_complex_allclose(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    rtol: float,
    atol: float,
    name: str,
) -> None:
    """``np.testing.assert_allclose`` with max abs/rel diagnostics on failure."""

    actual = np.asarray(actual, dtype=complex)
    desired = np.asarray(desired, dtype=complex)
    if actual.shape != desired.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual {actual.shape} vs desired {desired.shape}. "
            "Check frequency-grid length and channel (output/input) ordering."
        )
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"{name}: actual array contains non-finite values")
    if not np.all(np.isfinite(desired)):
        n_bad = int(np.size(desired) - np.sum(np.isfinite(desired)))
        raise AssertionError(
            f"{name}: reference array contains {n_bad} non-finite value(s); "
            "check the MATLAB export of this quantity"
        )

    max_abs, max_rel = complex_error_stats(actual, desired)
    message = (
        f"{name} mismatch: max_abs={max_abs:.6e}, max_rel={max_rel:.6e} "
        f"(rtol={rtol}, atol={atol})"
    )
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=message)


def match_eigenvalues_nearest(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy nearest-neighbor match of ``candidate`` into ``reference`` (no reuse).

    Returns ``(matched_reference, matched_candidate, abs_errors)`` sorted by ascending error.
    """

    ref = np.asarray(reference, dtype=complex).ravel()
    cand = np.asarray(candidate, dtype=complex).ravel()
    if ref.size == 0 or cand.size == 0:
        empty = np.asarray([], dtype=complex)
        return empty, empty, np.asarray([], dtype=float)

    used = np.zeros(ref.size, dtype=bool)
    pairs: list[tuple[complex, complex, float]] = []
    for value in cand:
        distances = np.abs(ref - value)
        distances[used] = np.inf
        index = int(np.argmin(distances))
        if not np.isfinite(distances[index]):
            continue
        used[index] = True
        pairs.append((ref[index], value, float(distances[index])))

    pairs.sort(key=lambda item: item[2])
    matched_ref = np.asarray([item[0] for item in pairs], dtype=complex)
    matched_cand = np.asarray([item[1] for item in pairs], dtype=complex)
    errors = np.asarray([item[2] for item in pairs], dtype=float)
    return matched_ref, matched_cand, errors


def matlab_eigenvalues_rad_per_s(ref: dict[str, Any]) -> np.ndarray:
    """Prefer ``LambdaRad``; otherwise treat ``Mode`` as Hz and convert to rad/s."""

    if "LambdaRad" in ref:
        return as_complex_vector(ref["LambdaRad"])
    if "Mode" not in ref:
        raise KeyError("MATLAB reference is missing both LambdaRad and Mode")
    return as_complex_vector(ref["Mode"]) * (2.0 * np.pi)


def matlab_zsys_reference(ref: dict[str, Any], frequency_hz: np.ndarray) -> np.ndarray:
    """Return MATLAB Zsys on ``(frequency, output, input)``.

    Prefer exported ``Zsys_values``. If those are non-finite (known exporter issue),
    reconstruct from ``GmDss`` / ``YbusDss`` / ``PortI`` / ``PortV`` using the
    WholeSysZ_cal algebraic formula.
    """

    from simplusgt.dss import DescriptorStateSpace
    from simplusgt.greybox import evalfr

    if "Zsys_values" in ref:
        aligned = align_matlab_freq_response(ref["Zsys_values"])
        if np.all(np.isfinite(aligned)):
            return aligned

    required = ("GmDss", "YbusDss", "PortI", "PortV")
    missing = [name for name in required if name not in ref]
    if missing:
        raise AssertionError(
            "Zsys_values are non-finite and cannot reconstruct from MATLAB models; "
            f"missing {missing}. Re-export with the updated ExportMatlabReference.m."
        )

    def as_dss(struct_obj: Any) -> DescriptorStateSpace:
        a = np.asarray(getattr(struct_obj, "A"), dtype=float)
        b = np.asarray(getattr(struct_obj, "B"), dtype=float)
        c = np.asarray(getattr(struct_obj, "C"), dtype=float)
        d = np.asarray(getattr(struct_obj, "D"), dtype=float)
        e = np.asarray(getattr(struct_obj, "E"), dtype=float)
        a = np.atleast_2d(a) if np.size(a) else np.zeros((0, 0))
        if a.size == 0:
            d = np.atleast_2d(d)
            b = np.zeros((0, d.shape[1]))
            c = np.zeros((d.shape[0], 0))
            e = np.zeros((0, 0))
        else:
            b = np.atleast_2d(b)
            c = np.atleast_2d(c)
            d = np.atleast_2d(d)
            e = np.atleast_2d(e)
        return DescriptorStateSpace(a, b, c, d, e)

    gm = as_dss(ref["GmDss"])
    ybus = as_dss(ref["YbusDss"])
    port_i = (np.asarray(ref["PortI"]).astype(int) - 1).tolist()
    port_v = (np.asarray(ref["PortV"]).astype(int) - 1).tolist()
    gm_trim = gm.truncate(port_i, port_v)
    values = []
    for freq in frequency_hz:
        s = 2j * np.pi * float(freq)
        gm_s = evalfr(gm_trim, s)
        ybus_s = evalfr(ybus, s)
        values.append(np.linalg.solve(gm_s + ybus_s, np.eye(gm_s.shape[0], dtype=complex)))
    return np.asarray(values, dtype=complex)


@pytest.fixture(scope="module")
def matlab_ref() -> dict[str, Any]:
    if not MATLAB_REFERENCE_PATH.is_file():
        pytest.skip(f"MATLAB reference not found: {MATLAB_REFERENCE_PATH}")
    return load_matlab_reference(MATLAB_REFERENCE_PATH)


@pytest.fixture(scope="module")
def frequency_hz(matlab_ref: dict[str, Any]) -> np.ndarray:
    if "FrequencyHz" not in matlab_ref:
        pytest.fail("MATLAB reference is missing FrequencyHz")
    freq = normalize_matlab_array(matlab_ref["FrequencyHz"], dtype=float).ravel()
    if freq.size == 0:
        pytest.fail("FrequencyHz is empty")
    if not np.all(np.isfinite(freq)):
        pytest.fail("FrequencyHz contains non-finite values")
    if np.any(freq <= 0):
        pytest.fail("FrequencyHz must be strictly positive")
    return freq


@pytest.fixture(scope="module")
def greybox_on_matlab_grid(frequency_hz: np.ndarray):
    if not IEEE14_CASE.is_file():
        pytest.skip(f"IEEE 14-bus case not found: {IEEE14_CASE}")
    config = GreyboxConfig(
        case_path=IEEE14_CASE,
        layers=GreyboxLayerSelection.from_names("admittance-only"),
        frequency_grid=FrequencyGrid(values_hz=tuple(float(value) for value in frequency_hz)),
        max_admittance_samples=int(frequency_hz.size),
    )
    return run_greybox(config)


def test_matlab_reference_loads_and_normalizes(matlab_ref: dict[str, Any], frequency_hz: np.ndarray):
    assert frequency_hz.ndim == 1
    assert np.all(np.isfinite(frequency_hz))
    assert np.all(frequency_hz > 0)

    for key in ("Ysys_values", "Zsys_values"):
        assert key in matlab_ref, f"MATLAB reference is missing {key}"
    aligned_y = align_matlab_freq_response(matlab_ref["Ysys_values"])
    assert aligned_y.shape[0] == frequency_hz.size, (
        f"Ysys_values frequency axis length {aligned_y.shape[0]} != FrequencyHz length {frequency_hz.size}"
    )
    # Zsys may be all-NaN in older exports; reconstruction is validated in the Zsys parity test.
    z_raw = normalize_matlab_array(matlab_ref["Zsys_values"])
    if np.all(np.isfinite(z_raw)):
        aligned_z = align_matlab_freq_response(matlab_ref["Zsys_values"])
        assert aligned_z.shape[0] == frequency_hz.size


def test_ysys_frequency_response_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid):
    y_matlab = align_matlab_freq_response(matlab_ref["Ysys_values"])
    y_python = np.asarray(greybox_on_matlab_grid.admittance.values, dtype=complex)
    assert_complex_allclose(
        y_python,
        y_matlab,
        rtol=YSYS_RTOL,
        atol=YSYS_ATOL,
        name="Ysys_values vs result.admittance.values",
    )


def test_zsys_frequency_response_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid, frequency_hz: np.ndarray):
    z_matlab = matlab_zsys_reference(matlab_ref, frequency_hz)
    z_python = np.asarray(greybox_on_matlab_grid.impedance.values, dtype=complex)
    assert_complex_allclose(
        z_python,
        z_matlab,
        rtol=ZSYS_RTOL,
        atol=ZSYS_ATOL,
        name="Zsys_values vs result.impedance.values",
    )


def test_eigenvalue_nearest_neighbor_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid):
    matlab_eigs = matlab_eigenvalues_rad_per_s(matlab_ref)
    python_eigs = as_complex_vector(greybox_on_matlab_grid.run_result.eigenvalues)
    assert matlab_eigs.size > 0, "MATLAB eigenvalue vector is empty"
    assert python_eigs.size > 0, "Python eigenvalue vector is empty"

    # Match the smaller set into the larger set so ordering differences do not dominate.
    if python_eigs.size <= matlab_eigs.size:
        matched_ref, matched_cand, errors = match_eigenvalues_nearest(matlab_eigs, python_eigs)
        ref_name, cand_name = "MATLAB", "Python"
    else:
        matched_ref, matched_cand, errors = match_eigenvalues_nearest(python_eigs, matlab_eigs)
        ref_name, cand_name = "Python", "MATLAB"

    assert matched_ref.size > 0, "Nearest-neighbor matching produced no eigenvalue pairs"
    max_abs = float(np.max(errors))
    median_abs = float(np.median(errors))
    message = (
        f"Eigenvalue nearest-neighbor mismatch ({cand_name} -> {ref_name}): "
        f"pairs={matched_ref.size}, max_abs={max_abs:.6e}, median_abs={median_abs:.6e}, "
        f"matlab_count={matlab_eigs.size}, python_count={python_eigs.size} "
        f"(rtol={EIG_RTOL}, atol={EIG_ATOL}). "
        "Eigenvectors (Phi/Psi) are compared only up to nearest-mode matching of eigenvalues."
    )
    np.testing.assert_allclose(
        matched_cand,
        matched_ref,
        rtol=EIG_RTOL,
        atol=EIG_ATOL,
        err_msg=message,
    )


def _as_dss_struct(struct_obj: Any):
    from simplusgt.dss import DescriptorStateSpace

    a = np.asarray(getattr(struct_obj, "A"), dtype=float)
    b = np.asarray(getattr(struct_obj, "B"), dtype=float)
    c = np.asarray(getattr(struct_obj, "C"), dtype=float)
    d = np.asarray(getattr(struct_obj, "D"), dtype=float)
    e = np.asarray(getattr(struct_obj, "E"), dtype=float)
    a = np.atleast_2d(a) if np.size(a) else np.zeros((0, 0))
    if a.size == 0:
        d = np.atleast_2d(d)
        b = np.zeros((0, d.shape[1]))
        c = np.zeros((d.shape[0], 0))
        e = np.zeros((0, 0))
    else:
        b = np.atleast_2d(b)
        c = np.atleast_2d(c)
        d = np.atleast_2d(d)
        e = np.atleast_2d(e)
    return DescriptorStateSpace(a, b, c, d, e)


def test_ybus_static_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid):
    from simplusgt.greybox import evalfr

    assert "Ybus" in matlab_ref, "MATLAB reference is missing Ybus"
    y_matlab = normalize_matlab_array(matlab_ref["Ybus"], dtype=complex)
    y_python = evalfr(greybox_on_matlab_grid.run_result.network.ybus, 0.0)
    assert_complex_allclose(
        y_python,
        y_matlab,
        rtol=1e-9,
        atol=1e-9,
        name="static Ybus vs network.ybus@s=0",
    )


def test_gm_and_gsys_dimensions_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid):
    run_result = greybox_on_matlab_grid.run_result
    assert "GmDss" in matlab_ref and "GsysSs" in matlab_ref
    gm_m = _as_dss_struct(matlab_ref["GmDss"])
    assert run_result.apparatus_block.nx == gm_m.nx
    assert run_result.apparatus_block.nu == gm_m.nu
    assert run_result.apparatus_block.ny == gm_m.ny
    gsys_a = np.asarray(matlab_ref["GsysSs"].A)
    assert run_result.whole_system_ss is not None
    assert run_result.whole_system_ss[0].shape == gsys_a.shape


def test_gm_frequency_response_parity(matlab_ref: dict[str, Any], greybox_on_matlab_grid):
    from simplusgt.greybox import evalfr

    gm_m = _as_dss_struct(matlab_ref["GmDss"])
    gm_p = greybox_on_matlab_grid.run_result.apparatus_block
    for freq in (0.1, 1.0, 50.0, 100.0):
        s = 2j * np.pi * freq
        assert_complex_allclose(
            evalfr(gm_p, s),
            evalfr(gm_m, s),
            rtol=1e-6,
            atol=1e-6,
            name=f"GmDss frequency response at {freq} Hz",
        )


@pytest.fixture(scope="module")
def greybox_modal_layers(matlab_ref: dict[str, Any]):
    if not IEEE14_CASE.is_file():
        pytest.skip(f"IEEE 14-bus case not found: {IEEE14_CASE}")
    # Prefer MATLAB ModeSelect when available; otherwise pick an oscillatory mode.
    modes: tuple[int, ...]
    if "ModeSelectPythonZeroBased" in matlab_ref:
        modes_arr = normalize_matlab_array(matlab_ref["ModeSelectPythonZeroBased"], dtype=int).ravel()
        modes = tuple(int(v) for v in modes_arr[:3])
    else:
        modes = (5,)
    config = GreyboxConfig(
        case_path=IEEE14_CASE,
        layers=GreyboxLayerSelection.from_names(["app-l1", "app-l2", "sens-l12"]),
        modes=modes,
        frequency_grid=FrequencyGrid(values_hz=(1.0, 10.0, 50.0)),
    )
    return run_greybox(config)


def test_apparatus_layer12_shapes(greybox_modal_layers):
    assert greybox_modal_layers.modes, "Expected at least one modal result"
    # Prefer a mode with nonzero Layer1 (ModeSelect may include trivial poles).
    mode = next(
        (item for item in greybox_modal_layers.modes if sum(abs(x["value"]) for x in item.layer1) > 0),
        greybox_modal_layers.modes[0],
    )
    assert mode.layer1, "Apparatus Layer 1 is empty"
    assert mode.layer2, "Apparatus Layer 2 is empty"
    total = sum(item["normalized"] for item in mode.layer1)
    if sum(abs(item["value"]) for item in mode.layer1) > 0:
        assert abs(total - 1.0) < 1e-9


def test_sensitivity_layer12_shapes(greybox_modal_layers):
    assert greybox_modal_layers.sensitivity, "Expected at least one sensitivity result"
    sens = greybox_modal_layers.sensitivity[0]
    assert sens.layer12, "Sensitivity Layer 1/2 is empty"
    assert abs(sum(item["layer1_normalized"] for item in sens.layer12) - 1.0) < 1e-9


def test_matlab_layer1_export_when_present(matlab_ref: dict[str, Any], greybox_modal_layers):
    """Soft parity: compare normalized Layer1 shares when MATLAB export is finite."""

    if "Layer1" not in matlab_ref:
        pytest.skip("MATLAB reference has no Layer1 (re-export with modal layers enabled)")
    layer1_cell = matlab_ref["Layer1"]
    if layer1_cell is None:
        pytest.skip("MATLAB Layer1 is empty")
    candidates = layer1_cell.ravel() if isinstance(layer1_cell, np.ndarray) else [layer1_cell]
    matlab_first = None
    matlab_mode_pos = None
    for pos, item in enumerate(candidates):
        arr = normalize_matlab_array(item, dtype=float).ravel()
        if arr.size and np.all(np.isfinite(arr)) and float(np.sum(np.abs(arr))) > 0:
            matlab_first = np.abs(arr)
            matlab_mode_pos = pos
            break
    if matlab_first is None:
        pytest.skip("MATLAB Layer1 values are missing or non-finite")

    # Align Python mode with the same ModeSelect slot when possible.
    py_mode = greybox_modal_layers.modes[min(matlab_mode_pos or 0, len(greybox_modal_layers.modes) - 1)]
    py_vals = np.asarray([abs(item["value"]) for item in py_mode.layer1], dtype=float)
    if float(np.sum(py_vals)) <= 0:
        pytest.skip(f"Python Layer1 is zero for mode {py_mode.mode_index}")

    m_norm = matlab_first / np.sum(matlab_first)
    p_norm = py_vals / np.sum(py_vals)
    m_sorted = np.sort(m_norm)[::-1]
    p_sorted = np.sort(p_norm)[::-1]
    n = min(m_sorted.size, p_sorted.size, 5)
    np.testing.assert_allclose(p_sorted[:n], m_sorted[:n], rtol=0.35, atol=0.08)


def test_matlab_sensitivity_export_when_present(matlab_ref: dict[str, Any]):
    """Document sensitivity export availability; skip when WholeSysZ_cal failed."""

    for key in ("SensLayer1", "SensLayer2", "Layer12", "SensMatrix"):
        if key not in matlab_ref:
            pytest.skip(f"MATLAB reference missing {key} (sensitivity export skipped)")
    sens = normalize_matlab_array(matlab_ref["SensMatrix"], dtype=complex)
    if sens.size == 0 or not np.any(np.isfinite(sens)):
        pytest.skip("MATLAB SensMatrix is empty or non-finite")
    assert np.any(np.isfinite(sens))
