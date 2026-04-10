import pytest
from jsbsim_gym.uncertainty import RuntimeUncertaintySampler

def test_sampler_exists():
    assert RuntimeUncertaintySampler is not None
