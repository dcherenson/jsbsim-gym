import pytest
import gymnasium as gym
import jsbsim_gym.env
import jsbsim_gym.canyon_env
from jsbsim_gym.data_collection_env import DataCollectionEnv

def test_make_jsbsim_env():
    env = gym.make("JSBSim-v0")
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    assert len(obs) > 0
    env.close()

def test_data_collection_env():
    env = DataCollectionEnv()
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    env.close()

def test_make_canyon_env():
    env = gym.make("JSBSimCanyon-v0")
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    assert len(obs) == 18
    env.close()
