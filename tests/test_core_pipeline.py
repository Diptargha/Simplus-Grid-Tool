from pathlib import Path

import numpy as np

from simplusgt.dss import DescriptorStateSpace, append
from simplusgt.io import load_case
from simplusgt.netlists import normalize_case
from simplusgt.pipeline import run_case
from simplusgt.powerflow import run_power_flow


ROOT = Path(__file__).resolve().parents[1]


def test_load_and_normalize_user_data():
    case = load_case(ROOT / "UserData.json")
    netlists = normalize_case(case)
    assert netlists.buses.shape[1] == 12
    assert netlists.lines.shape[1] == 8
    assert len(netlists.apparatus_types) == len(case.Apparatus)


def test_gauss_seidel_power_flow_user_data():
    case = load_case(ROOT / "UserData.json")
    netlists = normalize_case(case)
    wbase = case.Basic.Fbase * 2 * np.pi
    result = run_power_flow(netlists.buses, netlists.lines, wbase, 1)
    assert len(result.power_flow) == netlists.buses.shape[0]
    assert np.all(np.isfinite(result.vm))
    assert result.ybus.shape[0] == netlists.buses.shape[0]


def test_dss_append_dimensions_and_labels():
    g1 = DescriptorStateSpace.from_state_space(
        np.array([[-1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[0.0]]),
        states=["x1"], inputs=["u1"], outputs=["y1"]
    )
    g2 = DescriptorStateSpace.static(np.array([[2.0]]), inputs=["u2"], outputs=["y2"])
    gout = append(g1, g2)
    assert gout.A.shape == (1, 1)
    assert gout.D.shape == (2, 2)
    assert gout.inputs == ["u1", "u2"]
    assert gout.outputs == ["y1", "y2"]


def test_run_single_sg_case_smoke():
    result = run_case(ROOT / "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json")
    assert result.apparatus_block.nx >= 0
    assert result.network.ybus.nu == result.network.ybus.ny
    assert result.whole_system_dss.nx >= result.apparatus_block.nx
    assert result.eigenvalues is not None


def test_run_dc_buck_case_smoke():
    result = run_case(ROOT / "Examples/DcPowerSystem/2Bus/TwoBusGfdBuck.json")
    assert result.netlists.buses.shape[0] == 2
    assert result.whole_system_dss.nx >= 0
    assert result.eigenvalues is not None
