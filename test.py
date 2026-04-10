import sys
import gymnasium as gym
import gymnasium.spaces
import gymnasium.envs

sys.modules["gym"] = gym
sys.modules["gym.spaces"] = gymnasium.spaces
sys.modules["gym.envs"] = gymnasium.envs

import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
import imageio as iio
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0", render_mode="rgb_array")

# Load model with custom_objects to bypass space mismatch from legacy models
custom_objects = {
    "observation_space": env.observation_space,
    "action_space": env.action_space
}
model = SAC.load("models/jsbsim_sac", env, custom_objects=custom_objects)

mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
gif_writer = iio.get_writer("video.gif", format="gif", fps=5)
obs, info = env.reset()
terminated = False
truncated = False
step = 0
while not (terminated or truncated):
    render_data = env.render()
    mp4_writer.append_data(render_data)
    if step % 6 == 0:
        gif_writer.append_data(render_data[::2,::2,:])

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
mp4_writer.close()
gif_writer.close()
env.close()