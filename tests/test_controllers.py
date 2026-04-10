import pytest
from jsbsim_gym.controllers import PersistentExcitationController

def test_persistent_excitation_controller():
    controller = PersistentExcitationController(target_h=1000.0)
    state_dict = {
        'h': 1000.0,
        'u': 300.0,
        'theta': 0.0,
        'q': 0.0,
        'phi': 0.0,
        'p': 0.0,
        'p_E': 0.0
    }
    action = controller.get_action(state_dict, time_sec=0.0)
    assert len(action) == 4
    assert -1.0 <= action[0] <= 1.0  # throttle
