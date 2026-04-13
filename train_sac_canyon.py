import argparse
from os import makedirs, path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_env import OBS_LATERAL_NORM, OBS_U_FPS, OBS_V_FPS, OBS_W_FPS

DEM_PATH = "data/dem/black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)


class CanyonSACRewardWrapper(gym.Wrapper):
    """Reward shaping for low, fast canyon flight with wall/terrain safety."""

    def __init__(
        self,
        env,
        progress_gain=0.08,
        speed_gain=0.55,
        low_altitude_gain=0.60,
        wall_penalty_gain=0.90,
        target_clearance_ft=180.0,
    ):
        super().__init__(env)
        self.progress_gain = float(progress_gain)
        self.speed_gain = float(speed_gain)
        self.low_altitude_gain = float(low_altitude_gain)
        self.wall_penalty_gain = float(wall_penalty_gain)
        self.target_clearance_ft = float(target_clearance_ft)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        speed_fps = float(
            np.sqrt(
                float(obs[OBS_U_FPS]) ** 2
                + float(obs[OBS_V_FPS]) ** 2
                + float(obs[OBS_W_FPS]) ** 2
            )
        )
        speed_term = self.speed_gain * np.clip(speed_fps / 1000.0, 0.0, 2.0)

        progress_from_start_ft = float(info.get("progress_from_start_ft", 0.0))
        progress_term = self.progress_gain * np.clip(progress_from_start_ft / 40.0, -3.0, 3.0)

        clearance_ft = float(info.get("terrain_clearance_ft", 0.0))
        clearance_error = abs(clearance_ft - self.target_clearance_ft)
        low_altitude_term = self.low_altitude_gain * (1.0 - np.clip(clearance_error / self.target_clearance_ft, 0.0, 2.0))

        lateral_norm = abs(float(obs[OBS_LATERAL_NORM]))
        wall_term = -self.wall_penalty_gain * np.clip(lateral_norm, 0.0, 2.0)

        reward = progress_term + speed_term + low_altitude_term + wall_term + 0.05

        termination_reason = info.get("termination_reason", "running")
        if terminated:
            if termination_reason in {"terrain_collision", "ground_collision"}:
                reward -= 250.0
            elif termination_reason == "hit_canyon_wall":
                reward -= 180.0
            elif termination_reason == "altitude_out_of_bounds":
                reward -= 80.0

        info["reward_progress"] = float(progress_term)
        info["reward_speed"] = float(speed_term)
        info["reward_low_altitude"] = float(low_altitude_term)
        info["reward_wall"] = float(wall_term)
        info["reward_total"] = float(reward)

        return obs, float(reward), terminated, truncated, info


def make_canyon_env():
    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode=None,
        canyon_mode="dem",
        dem_path=DEM_PATH,
        dem_bbox=DEM_BBOX,
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=DEM_START_PIXEL,
        dem_start_heading_mode="follow_canyon",
        # Disable proxy wall bounds for DEM runs: the static proxy frame can
        # force identical early wall collisions and flatten rollout metrics.
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=250.0,
        entry_altitude_ft=250.0,
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=300,
        terrain_collision_buffer_ft=10.0,
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )
    return CanyonSACRewardWrapper(env)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC for canyon flight in JSBSimCanyon-v0")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda)")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=500,
        help="Warmup steps before gradient updates begin",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Replay batch size")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How many episodes between rollout log prints",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    root = path.abspath(path.dirname(__file__))
    makedirs(path.join(root, "logs"), exist_ok=True)
    makedirs(path.join(root, "models"), exist_ok=True)

    log_path = path.join(root, "logs", "canyon_sac")
    model_path = path.join(root, "models", "jsbsim_canyon_sac")
    replay_path = path.join(root, "models", "jsbsim_canyon_sac_buffer")
    monitor_path = path.join(root, "logs", "canyon_sac_monitor.csv")

    env = Monitor(make_canyon_env(), filename=monitor_path)
    print(
        f"Training config: timesteps={args.timesteps}, learning_starts={args.learning_starts}, "
        f"batch_size={args.batch_size}, log_interval={args.log_interval}"
    )

    model = None
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
            gradient_steps=-1,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps, log_interval=args.log_interval)
    finally:
        env.close()
        if model is not None:
            model.save(model_path)
            model.save_replay_buffer(replay_path)


if __name__ == "__main__":
    train()
