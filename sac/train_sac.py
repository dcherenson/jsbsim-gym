import sys
from pathlib import Path

import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.env # This line makes sure the environment is registered
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0")

log_path = str(REPO_ROOT / 'logs')

try:
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_path, gradient_steps=-1, device='cuda')
    model.learn(3000000)
finally:
    model.save(str(REPO_ROOT / "models" / "jsbsim_sac"))
    model.save_replay_buffer(str(REPO_ROOT / "models" / "jsbsim_sac_buffer"))