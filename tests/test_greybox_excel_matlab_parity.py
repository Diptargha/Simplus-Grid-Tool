"""Compare Python vs MATLAB greybox Excel Zsys sheets when both files exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON_XLSX = ROOT / "Results" / "IEEE_14Bus_greybox.xlsx"
MATLAB_XLSX = ROOT / "Results" / "IEEE_14Bus_greybox_matlab.xlsx"

ZSYS_RTOL = 1e-4
ZSYS_ATOL = 1e-4
KEYS = ["Frequency_Hz", "Row", "Col"]
VALUE_COLS = ["Mag", "Phase_deg", "Real", "Imag"]


def _require_workbook(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Greybox Excel not found: {path}")


def _load_zsys(path: Path) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name="Zsys")
    missing = set(KEYS + VALUE_COLS) - set(table.columns)
    if missing:
        raise AssertionError(f"{path.name} Zsys sheet is missing columns: {sorted(missing)}")
    table = table.copy()
    table["Frequency_Hz"] = np.round(table["Frequency_Hz"].to_numpy(dtype=float), decimals=10)
    return table.sort_values(KEYS).reset_index(drop=True)


def test_zsys_excel_python_vs_matlab():
    _require_workbook(PYTHON_XLSX)
    _require_workbook(MATLAB_XLSX)
    py = _load_zsys(PYTHON_XLSX)
    ml = _load_zsys(MATLAB_XLSX)
    merged = py.merge(ml, on=KEYS, suffixes=("_py", "_ml"), how="inner")
    assert not merged.empty, "No overlapping (Frequency_Hz, Row, Col) keys between Python and MATLAB Zsys sheets"
    assert len(merged) == len(py) == len(ml), (
        f"Row-count mismatch after aligning keys: python={len(py)}, matlab={len(ml)}, inner={len(merged)}"
    )
    for col in VALUE_COLS:
        actual = merged[f"{col}_py"].to_numpy(dtype=float)
        desired = merged[f"{col}_ml"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=ZSYS_RTOL,
            atol=ZSYS_ATOL,
            err_msg=f"Zsys column {col} mismatch between Python and MATLAB Excel exports",
        )
