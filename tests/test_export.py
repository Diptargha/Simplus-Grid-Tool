from pathlib import Path
import json

from simplusgt.export import result_to_dashboard_data
from simplusgt.pipeline import run_case


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_export_contains_modes_and_participation():
    result = run_case(ROOT / "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json")
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
