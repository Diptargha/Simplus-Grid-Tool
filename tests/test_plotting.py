"""Smoke tests for MATLAB-style fundamental plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from simplusgt.greybox import FrequencyGrid, GreyboxConfig, GreyboxLayerSelection, run_greybox
from simplusgt.pipeline import run_case
from simplusgt.plotting import plot_case_fundamentals, plot_greybox_summary


ROOT = Path(__file__).resolve().parents[1]
SG_CASE = ROOT / "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json"


def test_plot_case_fundamentals_writes_pngs(tmp_path):
    result = run_case(SG_CASE)
    saved = plot_case_fundamentals(result, output_dir=tmp_path, show=False)
    assert saved.get("pole_map") is not None
    assert Path(saved["pole_map"]).is_file()
    assert saved.get("admittance_dq") is not None
    assert Path(saved["admittance_dq"]).is_file()
    assert saved.get("admittance_dd") is not None
    assert Path(saved["admittance_dd"]).is_file()


def test_plot_greybox_summary_writes_pngs(tmp_path):
    config = GreyboxConfig(
        case_path=SG_CASE,
        layers=GreyboxLayerSelection.from_names(["app-l1", "app-l2"]),
        modes=(0,),
        frequency_grid=FrequencyGrid(values_hz=(1.0, 10.0)),
    )
    result = run_greybox(config)
    saved = plot_greybox_summary(result, output_dir=tmp_path, show=False)
    assert saved.get("ysys") is not None
    assert Path(saved["ysys"]).is_file()
    assert saved.get("zsys") is not None
    assert any(key.startswith("apparatus_layer12_mode") for key in saved)
