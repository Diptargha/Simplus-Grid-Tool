from pathlib import Path
import json

import pandas as pd

from simplusgt.export import export_greybox_excel, result_to_dashboard_data
from simplusgt.greybox import FrequencyGrid, GreyboxConfig, GreyboxLayerSelection, run_greybox
from simplusgt.pipeline import run_case


ROOT = Path(__file__).resolve().parents[1]
SG_CASE = ROOT / "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json"


def test_dashboard_export_contains_modes_and_participation():
    result = run_case(SG_CASE)
    data = result_to_dashboard_data(result, case_path="SgInfiniteBus.json", top_states=5)
    assert data["schema_version"] == 1
    assert data["eigenvalues"]
    assert data["modes"]
    assert data["modes"][0]["participation"]
    assert len(data["modes"][0]["participation"]) <= 5
    assert "component" in data["modes"][0]["participation"][0]
    assert data["states"][0]["component"]
    assert data["apparatus"]
    assert any("type" in item for item in data["apparatus"])
    assert any("SynchronousMachine" in state["component"] for state in data["states"])
    assert "power_flow" in data
    assert data["network_graph"]["nodes"]
    assert data["network_graph"]["edges"]
    assert "shunts" in data["network_graph"]
    assert data["network_graph"]["nodes"][0]["apparatus"]
    assert "shunts" in data["network_graph"]["nodes"][0]
    assert "matrices" in data
    assert data["matrices"]["A"]["values"]
    assert data["matrices"]["A"]["row_components"]
    json.dumps(data, allow_nan=False)


def test_greybox_excel_export_contains_zsys(tmp_path):
    greybox = run_greybox(
        GreyboxConfig(
            case_path=SG_CASE,
            layers=GreyboxLayerSelection.from_names(["app-l1", "app-l2"]),
            modes=(0,),
            frequency_grid=FrequencyGrid(values_hz=(1.0, 10.0, 50.0)),
        )
    )
    out = tmp_path / "greybox.xlsx"
    export_greybox_excel(greybox, out)
    assert out.is_file()
    zsys = pd.read_excel(out, sheet_name="Zsys")
    assert {"Frequency_Hz", "Mag", "Phase_deg", "Real", "Imag"} <= set(zsys.columns)
    assert len(zsys) == 3 * len(greybox.impedance.output_labels) * len(greybox.impedance.input_labels)
    sheets = set(pd.ExcelFile(out).sheet_names)
    assert {"Summary", "Zsys", "Ysys", "Zsys_MagPhase", "Zsys_RealImag", "Eigenvalues", "StatePF"} <= sheets
    eigs = pd.read_excel(out, sheet_name="Eigenvalues")
    assert {"mode_index", "real_rad_s", "imag_rad_s", "real_hz", "imag_hz", "frequency_hz", "damping_ratio"} <= set(eigs.columns)
    assert len(eigs) >= 1
    state_pf = pd.read_excel(out, sheet_name="StatePF")
    assert {"mode_index", "state", "description", "apparatus", "pf_abs"} <= set(state_pf.columns)
    assert len(state_pf) >= 1
    assert state_pf["apparatus"].notna().all()
    assert state_pf["description"].notna().all()
    assert any(
        "SynchronousMachine" in str(name) or "InfiniteBusAc" in str(name) or "Network" in str(name)
        for name in state_pf["apparatus"]
    )
    assert any(
        "current" in str(text).lower()
        or "angle" in str(text).lower()
        or "frequency" in str(text).lower()
        or "interconnection" in str(text).lower()
        for text in state_pf["description"]
    )
