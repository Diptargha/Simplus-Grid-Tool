import json
from pathlib import Path

import numpy as np

from simplusgt.dss import DescriptorStateSpace, inverse
from simplusgt.export import export_greybox_json, greybox_result_to_dashboard_data
from simplusgt.greybox import (
    FrequencyGrid,
    GreyboxConfig,
    GreyboxLayerSelection,
    evalfr,
    run_greybox,
    run_greybox_from_config,
    whole_system_impedance_bundle,
    whole_system_impedance_model,
)


ROOT = Path(__file__).resolve().parents[1]
SG_CASE = ROOT / "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json"


def test_evalfr_matches_first_order_transfer():
    model = DescriptorStateSpace.from_state_space(
        np.array([[-2.0]]),
        np.array([[3.0]]),
        np.array([[4.0]]),
        np.array([[5.0]]),
    )
    value = evalfr(model, 1j)
    assert np.allclose(value, 4.0 * (3.0 / (1j + 2.0)) + 5.0)


def test_descriptor_inverse_matches_static_inverse():
    model = DescriptorStateSpace.static(np.array([[2.0, 0.5], [0.25, 4.0]]))
    inverted = inverse(model)
    assert np.allclose(evalfr(inverted, 1j), np.linalg.inv(model.D))


def test_greybox_admittance_only_skips_optional_layers():
    config = GreyboxConfig(
        case_path=SG_CASE,
        layers=GreyboxLayerSelection.from_names("admittance-only"),
        frequency_grid=FrequencyGrid(values_hz=(1.0, 10.0)),
    )
    result = run_greybox(config)
    assert result.admittance.values.shape[0] == 2
    assert result.impedance.values.shape[0] == 2
    assert result.modes == []
    assert result.sensitivity == []


def test_greybox_layer_selection_exports_layer12():
    config = GreyboxConfig(
        case_path=SG_CASE,
        layers=GreyboxLayerSelection.from_names(["app-l1", "app-l2"]),
        modes=(0,),
        frequency_grid=FrequencyGrid(values_hz=(1.0,)),
    )
    data = greybox_result_to_dashboard_data(run_greybox(config))
    assert data["analysis_type"] == "greybox"
    assert data["whole_system_admittance"]["channels"]
    assert data["whole_system_impedance"]["channels"]
    assert data["whole_system_admittance"]["channels"][0]["output_component"]
    assert data["modes"]
    assert data["modes"][0]["layer1"]
    assert data["modes"][0]["layer2"]
    assert data["sensitivity"] == []
    json.dumps(data, allow_nan=False)


def test_greybox_config_entry_and_export(tmp_path):
    output = tmp_path / "greybox.json"
    config_path = tmp_path / "greybox_config.json"
    config_path.write_text(
        json.dumps(
            {
                "case_path": str(SG_CASE),
                "output_path": str(output),
                "layers": ["admittance-only"],
                "frequency_grid": {"values_hz": [1.0, 5.0]},
            }
        ),
        encoding="utf-8",
    )
    result = run_greybox_from_config(config_path)
    assert result.config.output_path == output
    exported = export_greybox_json(None, config_path=config_path)
    assert exported == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["whole_system_admittance"]["frequencies_hz"] == [1.0, 5.0]
    assert payload["whole_system_impedance"]["frequencies_hz"] == [1.0, 5.0]


def test_true_zsys_uses_current_inputs_and_voltage_outputs():
    result = run_greybox(GreyboxConfig(case_path=SG_CASE, frequency_grid=FrequencyGrid(values_hz=(1.0,))))
    zsys = whole_system_impedance_model(result.run_result)
    assert zsys.inputs[:2] == ["i_d1", "i_q1"]
    assert zsys.outputs[:2] == ["v_d1", "v_q1"]
    assert result.impedance.input_labels[:2] == ["i_d1", "i_q1"]
    assert result.impedance.output_labels[:2] == ["v_d1", "v_q1"]


def test_exported_zsys_matches_matlab_feedback_transfer_formula():
    result = run_greybox(GreyboxConfig(case_path=SG_CASE, frequency_grid=FrequencyGrid(values_hz=(1.0,))))
    bundle = whole_system_impedance_bundle(result.run_result)
    zm = evalfr(bundle.zm, 2j * np.pi)
    ybus = evalfr(bundle.ybus, 2j * np.pi)
    expected = np.linalg.solve(np.eye(zm.shape[0]) + zm @ ybus, zm)
    assert np.allclose(result.impedance.values[0], expected)
