"""UserMain.py — single entry point for the Python SimplusGT pipeline.

Edit the USER SETTINGS block below, then run this file (PyCharm / VS Code /
``python UserMain.py``). No CLI is required.

Mirrors MATLAB ``UserMain.m``: pick a case, toggle which analyses and plots
to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# ---------------------------------------------------------------------------
# USER SETTINGS — customise these
# ---------------------------------------------------------------------------

# Case name (short name, as in MATLAB UserDataName) or a full/relative path.
# Short names are resolved under Examples/** and the repo root.
#
# Ac:
#   "SgInfiniteBus", "GflInverterInfiniteBus", "GfmInverterInfiniteBus",
#   "BessInfiniteBus", "PV_GfmInfiniteBus", "PV_GflInfiniteBus",
#   "WtGfmInfiniteBus", "WtGflInfiniteBus",
#   "IEEE_14Bus", "IEEE_30Bus", "IEEE_57Bus", "AU14Gen_59Bus", "NETS_NYPS_68Bus",
#   "BESS_Plant_10Bus", "PV_Plant_10Bus", "PV_BESS_Hybrid_Plant_10Bus",
#   "WSCC_9Bus",
# Dc:
#   "GfdBuckInfiniteBus", "TwoBusGfdBuck",
# Hybrid:
#   "Hybrid_4Bus", "HVDC_Infbus_4Bus", "HVDC_SG_4Bus", "MTDC_Infbus_4Bus",
# Default 4-bus:
#   "UserData"
USER_DATA_NAME = "IEEE_14Bus"

# Preferred file type when resolving a short name: "json" or "excel" (xlsx/xlsm).
USER_DATA_TYPE = "json"

# --- Core pipeline (always runs) -------------------------------------------
# Power flow + apparatus/network DSS + eigenvalues.

# --- Fundamental plots (MATLAB Main.m) -------------------------------------
ENABLE_PLOT_POLE = True
ENABLE_PLOT_ADMITTANCE = False          # Ydd + complex-vector Ydq+
ENABLE_PLOT_ADMITTANCE_DQ_AXES = False  # dd / dq / qd / qq (Modal BodeDraw)
ENABLE_PLOT_GRID_STRENGTH = True

# --- Greybox / modal analysis ----------------------------------------------
ENABLE_GREYBOX = True
ENABLE_GREYBOX_APP_LAYER1 = True   # apparatus Layer 1
ENABLE_GREYBOX_APP_LAYER2 = True   # apparatus Layer 2
ENABLE_GREYBOX_APP_LAYER3 = False  # apparatus Layer 3 (also adds Layer3 sheet to Excel export)
ENABLE_GREYBOX_SENS_LAYER12 = True   # sensitivity Layer 1/2
ENABLE_GREYBOX_SENS_LAYER3 = False   # sensitivity Layer 3
# Mode indices (0-based). Use "auto" to pick oscillatory modes.
GREYBOX_MODES = "auto"
GREYBOX_FREQ_MIN_HZ = 1
GREYBOX_FREQ_MAX_HZ = 1000.0
GREYBOX_FREQ_COUNT = 100

# --- Greybox plots ---------------------------------------------------------
ENABLE_PLOT_GREYBOX = True  # Ysys/Zsys Bode + Layer 1/2 charts

# --- Interactive HTML dashboard (Plotly) -----------------------------------
# Single browser file with pole / strength / greybox Bode / Layer 1–2 charts.
# Prefer this over matplotlib for Layer 1/2 (labels stay readable via hover).
ENABLE_HTML_DASHBOARD = True
HTML_DASHBOARD_DIR = "Results"  # writes <case>_dashboard.html

# --- Exports ---------------------------------------------------------------
ENABLE_EXPORT_DASHBOARD = False       # JSON dashboard
ENABLE_EXPORT_GREYBOX_JSON = False    # JSON greybox (includes Zsys)
ENABLE_EXPORT_GREYBOX_EXCEL = True    # Excel greybox (Ysys/Zsys, Eigenvalues, StatePF, Layers when enabled)
EXPORT_DIR = "Results"

# --- Display / save --------------------------------------------------------
SHOW_PLOTS = True          # open HTML dashboard (and matplotlib only if HTML is off)
SAVE_PLOTS = False         # also write matplotlib PNGs under PLOT_DIR
PLOT_DIR = "Results/plots"

# Non-interactive backend when not showing matplotlib windows.
if not SHOW_PLOTS or ENABLE_HTML_DASHBOARD:
    matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Implementation — usually no need to edit below
# ---------------------------------------------------------------------------

from simplusgt.analysis import stability_report
from simplusgt.export import export_greybox_excel, greybox_result_to_dashboard_data, result_to_dashboard_data
from simplusgt.greybox import FrequencyGrid, GreyboxConfig, GreyboxLayerSelection, run_greybox
from simplusgt.html_dashboard import write_analysis_dashboard
from simplusgt.pipeline import run_case
from simplusgt.plotting import plot_case_fundamentals, plot_greybox_summary

ROOT = Path(__file__).resolve().parent


def resolve_case_path(name: str, preferred: str = "json") -> Path:
    """Resolve a short case name or path to an existing case file."""

    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    rooted = ROOT / name
    if rooted.is_file():
        return rooted.resolve()

    stem = name
    for suffix in (".json", ".xlsx", ".xlsm", ".xls"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    preferred = preferred.lower().strip()
    if preferred in {"excel", "xlsx", "xlsm", "1"}:
        order = (".xlsx", ".xlsm", ".xls", ".json")
    else:
        order = (".json", ".xlsx", ".xlsm", ".xls")

    search_roots = [ROOT, ROOT / "Examples"]
    matches: list[Path] = []
    for ext in order:
        pattern = f"**/{stem}{ext}"
        for base in search_roots:
            matches.extend(sorted(base.glob(pattern)))
        if matches:
            break
    # Also allow bare name in repo root (UserData.json).
    for ext in order:
        direct = ROOT / f"{stem}{ext}"
        if direct.is_file():
            return direct.resolve()

    if not matches:
        raise FileNotFoundError(
            f'Could not find case "{name}" under {ROOT} or Examples/. '
            "Use a short name (e.g. IEEE_14Bus) or a full path to a .json/.xlsx file."
        )
    return matches[0].resolve()


def parse_modes(spec: str | tuple[int, ...] | list[int], eigenvalues) -> tuple[int, ...]:
    import numpy as np

    if isinstance(spec, (tuple, list)):
        return tuple(int(v) for v in spec)
    text = str(spec).strip().lower()
    if text == "auto":
        values = np.asarray(eigenvalues, dtype=complex).ravel()
        scored = [
            (idx, abs(float(np.imag(value))))
            for idx, value in enumerate(values)
            if np.isfinite(value) and abs(float(np.imag(value))) > 1e-6
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored:
            return tuple(idx for idx, _ in scored[:3])
        finite = [idx for idx, value in enumerate(values) if np.isfinite(value)]
        return tuple(finite[:3]) if finite else (0,)
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    case_path = resolve_case_path(USER_DATA_NAME, USER_DATA_TYPE)
    print("=" * 50)
    print("SimplusGT (Python)")
    print("=" * 50)
    print(f"Case: {case_path}")

    result = run_case(case_path)
    report = stability_report(result.eigenvalues)
    print(f"Buses: {result.netlists.buses.shape[0]}")
    print(f"Lines: {result.netlists.lines.shape[0]}")
    print(f"States: {result.whole_system_dss.nx}")
    print(f"Stable: {report.stable}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    plot_dir = ROOT / PLOT_DIR if SAVE_PLOTS else None
    # When the HTML dashboard is enabled, matplotlib is PNG-only (no SciView spam).
    show_mpl = bool(SHOW_PLOTS and not ENABLE_HTML_DASHBOARD)
    if any(
        (
            ENABLE_PLOT_POLE,
            ENABLE_PLOT_ADMITTANCE,
            ENABLE_PLOT_ADMITTANCE_DQ_AXES,
            ENABLE_PLOT_GRID_STRENGTH,
        )
    ) and (SAVE_PLOTS or show_mpl):
        print("\nFundamental plots...")
        saved = plot_case_fundamentals(
            result,
            output_dir=plot_dir,
            show=show_mpl,
            include_pole=ENABLE_PLOT_POLE,
            include_admittance=ENABLE_PLOT_ADMITTANCE,
            include_dq_axes=ENABLE_PLOT_ADMITTANCE_DQ_AXES,
            include_strength=ENABLE_PLOT_GRID_STRENGTH,
        )
        for key, path in saved.items():
            if path is not None:
                print(f"  {key}: {path}")

    greybox = None
    need_greybox = (
        ENABLE_GREYBOX
        or ENABLE_PLOT_GREYBOX
        or ENABLE_EXPORT_GREYBOX_JSON
        or ENABLE_EXPORT_GREYBOX_EXCEL
        or (ENABLE_HTML_DASHBOARD and ENABLE_PLOT_GREYBOX)
    )
    if need_greybox:
        print("\nGreybox analysis...")
        modes = parse_modes(GREYBOX_MODES, result.eigenvalues)
        greybox = run_greybox(
            GreyboxConfig(
                case_path=case_path,
                layers=GreyboxLayerSelection(
                    apparatus_layer1=ENABLE_GREYBOX_APP_LAYER1,
                    apparatus_layer2=ENABLE_GREYBOX_APP_LAYER2,
                    apparatus_layer3=ENABLE_GREYBOX_APP_LAYER3,
                    sensitivity_layer12=ENABLE_GREYBOX_SENS_LAYER12,
                    sensitivity_layer3=ENABLE_GREYBOX_SENS_LAYER3,
                ),
                modes=modes,
                frequency_grid=FrequencyGrid(
                    min_hz=GREYBOX_FREQ_MIN_HZ,
                    max_hz=GREYBOX_FREQ_MAX_HZ,
                    count=GREYBOX_FREQ_COUNT,
                ),
            )
        )
        print(f"  Modes: {modes}")
        print(f"  Apparatus Layer-1 modes: {len(greybox.modes)}")
        print(f"  Sensitivity results: {len(greybox.sensitivity)}")
        if greybox.warnings:
            for warning in greybox.warnings:
                if warning not in result.warnings:
                    print(f"  - {warning}")

        # Matplotlib greybox PNGs only when saving; interactive Layer charts use HTML.
        if ENABLE_PLOT_GREYBOX and SAVE_PLOTS:
            print("Greybox plots (PNG)...")
            saved_gb = plot_greybox_summary(greybox, output_dir=plot_dir, show=False)
            for key, path in saved_gb.items():
                if path is not None:
                    print(f"  {key}: {path}")
        elif ENABLE_PLOT_GREYBOX and show_mpl:
            print("Greybox plots...")
            saved_gb = plot_greybox_summary(greybox, output_dir=None, show=True)
            for key, path in saved_gb.items():
                if path is not None:
                    print(f"  {key}: {path}")

    if ENABLE_HTML_DASHBOARD:
        html_dir = ROOT / HTML_DASHBOARD_DIR
        html_path = html_dir / f"{case_path.stem}_dashboard.html"
        print("\nHTML dashboard...")
        written = write_analysis_dashboard(
            result,
            greybox,
            output_path=html_path,
            include_pole=ENABLE_PLOT_POLE,
            include_strength=ENABLE_PLOT_GRID_STRENGTH,
            include_admittance=ENABLE_PLOT_ADMITTANCE,
            include_dq_axes=ENABLE_PLOT_ADMITTANCE_DQ_AXES,
            include_greybox=ENABLE_PLOT_GREYBOX,
            case_label=case_path.stem,
            open_browser=SHOW_PLOTS,
        )
        print(f"  {written}")

    export_dir = ROOT / EXPORT_DIR
    if ENABLE_EXPORT_DASHBOARD:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{case_path.stem}_dashboard.json"
        payload = result_to_dashboard_data(result, case_path=str(case_path))
        out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\nDashboard export: {out}")

    if ENABLE_EXPORT_GREYBOX_JSON:
        if greybox is None:
            raise RuntimeError("ENABLE_EXPORT_GREYBOX_JSON requires greybox analysis")
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{case_path.stem}_greybox.json"
        payload = greybox_result_to_dashboard_data(greybox)
        out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Greybox JSON export: {out}")

    if ENABLE_EXPORT_GREYBOX_EXCEL:
        if greybox is None:
            raise RuntimeError("ENABLE_EXPORT_GREYBOX_EXCEL requires greybox analysis")
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{case_path.stem}_greybox.xlsx"
        export_greybox_excel(greybox, out)
        print(f"Greybox Excel export: {out}")
        # Surface Layer sheet presence in the console (tabs can be easy to miss in Excel).
        try:
            import openpyxl

            wb = openpyxl.load_workbook(out, read_only=True)
            layer_tabs = [n for n in wb.sheetnames if n.startswith("Layer") or n.startswith("Sens_")]
            print(f"  Sheets: {', '.join(wb.sheetnames)}")
            print(f"  Layer tabs: {', '.join(layer_tabs) if layer_tabs else '(none — enable ENABLE_GREYBOX_*_LAYER*)'}")
            wb.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not list sheets: {exc})")

    print("\nDone.")


if __name__ == "__main__":
    main()
