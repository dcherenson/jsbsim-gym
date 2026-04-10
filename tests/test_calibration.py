import pytest
from jsbsim_gym.calibration import generate_nominal_calibration_package

def test_generate_nominal_calibration_package_exists():
    assert callable(generate_nominal_calibration_package)
