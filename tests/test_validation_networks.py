from pathlib import Path

import pytest

from simplusgt.pipeline import run_case


ROOT = Path(__file__).resolve().parents[1]


MINIMUM_ACCEPTANCE_CASES = [
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/SgInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/GflInverterInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/GfmInverterInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/BessInfiniteBus.json",
    "Examples/DcPowerSystem/2Bus/TwoBusGfdBuck.json",
    "Examples/HybridPowerSystem/HVDC_4Bus/HVDC_Infbus_4Bus.json",
    "Examples/AcPowerSystem/IEEE_14Bus/IEEE_14Bus.json",
]

APPARATUS_FAMILY_CASES = [
    "UserData.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/GflInverterInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/GfmInverterInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/BessInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/PV_GfmInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/PV_GflInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/WtGfmInfiniteBus.json",
    "Examples/AcPowerSystem/SingleApparatusInfiniteBus/WtGflInfiniteBus.json",
    "Examples/HybridPowerSystem/HVDC_4Bus/HVDC_Infbus_4Bus.json",
    "Examples/HybridPowerSystem/HVDC_4Bus/MTDC_Infbus_4Bus.json",
]

ALL_EXAMPLE_CASES = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "Examples").glob("**/*.json"))


@pytest.mark.parametrize("relative_path", MINIMUM_ACCEPTANCE_CASES)
def test_minimum_acceptance_networks_run(relative_path):
    result = run_case(ROOT / relative_path)
    assert result.netlists.buses.shape[0] > 0
    assert result.whole_system_dss.D.shape == (result.whole_system_dss.ny, result.whole_system_dss.nu)
    assert result.eigenvalues is not None


@pytest.mark.parametrize("relative_path", APPARATUS_FAMILY_CASES)
def test_apparatus_families_do_not_use_placeholders(relative_path):
    result = run_case(ROOT / relative_path)
    assert not any("placeholder" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize("relative_path", ALL_EXAMPLE_CASES)
def test_all_json_examples_run(relative_path):
    result = run_case(ROOT / relative_path)
    assert result.netlists.buses.shape[0] > 0
    assert result.eigenvalues is not None
