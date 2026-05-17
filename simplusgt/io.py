"""Input loading for SimplusGT cases."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .schema import CaseData


def _replace_nonfinite(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_nonfinite(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_nonfinite(value) for value in obj]
    if isinstance(obj, str):
        lowered = obj.strip().lower()
        if lowered == "nan":
            return math.nan
        if lowered in {"inf", "+inf"}:
            return math.inf
        if lowered == "-inf":
            return -math.inf
    return obj


def load_json(path: str | Path) -> CaseData:
    """Load a SimplusGT JSON case.

    MATLAB writes some non-finite values as strings. The schema layer also
    normalizes them, but doing it here keeps the raw case data useful.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return CaseData.from_dict(_replace_nonfinite(data))


def _sheet_records(workbook: Any, sheet_name: str, *, skiprows: int = 3) -> list[dict[str, Any]]:
    import pandas as pd

    if sheet_name not in workbook.sheet_names:
        return []
    frame = pd.read_excel(workbook, sheet_name=sheet_name, skiprows=skiprows)
    frame = frame.dropna(how="all")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame.to_dict(orient="records")


def load_excel(path: str | Path) -> CaseData:
    """Load an Excel case using the same sheet names as `Excel2Json.m`.

    Apparatus parameter expansion in MATLAB is spreadsheet-specific. This
    loader supports workbooks already laid out with JSON-like sheets and is
    intentionally conservative for migration fixtures.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Excel loading requires the optional pandas/openpyxl dependencies") from exc

    workbook = pd.ExcelFile(path)
    basic_rows = _sheet_records(workbook, "Basic")
    advance_rows = _sheet_records(workbook, "Advance")
    if not basic_rows:
        raise ValueError("Excel case is missing a Basic sheet")
    data = {
        "Basic": basic_rows[0],
        "Advance": advance_rows[0] if advance_rows else {},
        "Bus": _sheet_records(workbook, "Bus"),
        "NetworkLine": _sheet_records(workbook, "NetworkLine"),
        "NetworkLineIEEE": _sheet_records(workbook, "NetworkLine_IEEE"),
        "Apparatus": _sheet_records(workbook, "Apparatus"),
    }
    return CaseData.from_dict(_replace_nonfinite(data))


def load_case(path: str | Path, input_format: str | None = None) -> CaseData:
    """Load a case from JSON or Excel."""

    case_path = Path(path)
    fmt = (input_format or case_path.suffix.lstrip(".")).lower()
    if fmt == "json":
        return load_json(case_path)
    if fmt in {"xlsx", "xlsm", "xls"}:
        return load_excel(case_path)
    raise ValueError(f"Unsupported input format: {fmt!r}")
