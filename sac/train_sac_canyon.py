import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_env import (
    OBS_LATERAL_NORM,
    OBS_P,
    OBS_Q,
    OBS_R,
    OBS_U_FPS,
    OBS_V_FPS,
    OBS_W_FPS,
)

DEM_PATH = REPO_ROOT / "data/dem/black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)


class CanyonSACRewardWrapper(gym.Wrapper):
    """Reward shaping for low, fast canyon flight with wall/terrain safety."""

    def __init__(
        self,
        env,
        progress_gain=0.70,
        speed_gain=0.35,
        low_altitude_gain=0.45,
        centerline_gain=0.60,
        offcenter_penalty_gain=0.30,
        alive_bonus=0.15,
        target_clearance_ft=180.0,
        terrain_crash_penalty=25.0,
        wall_crash_penalty=18.0,
        altitude_violation_penalty=8.0,
        early_termination_penalty_gain=80.0,
        time_limit_bonus=25.0,
        max_step_reward_abs=5.0,
        angular_rate_penalty_gain=0.8,
        angular_rate_threshold_deg_s=35.0,
        start_pixel_jitter_px=40,
        heading_jitter_deg=12.0,
    ):
        super().__init__(env)
        self.progress_gain = float(progress_gain)
        self.speed_gain = float(speed_gain)
        self.low_altitude_gain = float(low_altitude_gain)
        self.centerline_gain = float(centerline_gain)
        self.offcenter_penalty_gain = float(offcenter_penalty_gain)
        self.alive_bonus = float(alive_bonus)
        self.target_clearance_ft = float(target_clearance_ft)
        self.terrain_crash_penalty = float(terrain_crash_penalty)
        self.wall_crash_penalty = float(wall_crash_penalty)
        self.altitude_violation_penalty = float(altitude_violation_penalty)
        self.early_termination_penalty_gain = float(early_termination_penalty_gain)
        self.time_limit_bonus = float(time_limit_bonus)
        self.max_step_reward_abs = float(max_step_reward_abs)
        self.angular_rate_penalty_gain = max(float(angular_rate_penalty_gain), 0.0)
        self.angular_rate_threshold_deg_s = max(float(angular_rate_threshold_deg_s), 1.0)
        self.start_pixel_jitter_px = max(int(start_pixel_jitter_px), 0)
        self.heading_jitter_deg = max(float(heading_jitter_deg), 0.0)
        self.max_episode_steps = int(getattr(env.unwrapped, "max_episode_steps", 300))
        self.episode_step = 0
        self.rng = np.random.default_rng()

        base_env = env.unwrapped
        dem_start_pixel = getattr(base_env, "dem_start_pixel", None)
        dem_start_info = getattr(base_env, "dem_start_info", None)

        if dem_start_pixel is not None:
            self.base_start_pixel = (int(dem_start_pixel[0]), int(dem_start_pixel[1]))
        elif dem_start_info is not None:
            self.base_start_pixel = (
                int(dem_start_info.get("pixel_x", 0)),
                int(dem_start_info.get("pixel_y", 0)),
            )
        else:
            self.base_start_pixel = None

        dem_start_heading_deg = getattr(base_env, "dem_start_heading_deg", None)
        self.base_start_heading_deg = (
            float(dem_start_heading_deg) if dem_start_heading_deg is not None else None
        )

        dem_start_elev_ft = float(getattr(base_env, "dem_start_elev_ft", 0.0))
        self.entry_altitude_offset_ft = float(getattr(base_env, "entry_altitude_ft", 0.0) - dem_start_elev_ft)
        self.target_altitude_offset_ft = float(getattr(base_env, "target_altitude_ft", 0.0) - dem_start_elev_ft)
        self.min_altitude_offset_ft = float(getattr(base_env, "min_altitude_ft", 0.0) - dem_start_elev_ft)
        self.max_altitude_offset_ft = float(getattr(base_env, "max_altitude_ft", 0.0) - dem_start_elev_ft)

        self.current_start_pixel = self.base_start_pixel
        self.current_start_heading_deg = self.base_start_heading_deg

    def _randomize_dem_start(self):
        base_env = self.env.unwrapped
        if getattr(base_env, "canyon_mode", "") != "dem":
            return

        canyon = getattr(base_env, "canyon", None)
        if canyon is None or not hasattr(canyon, "get_pixel_info"):
            return

        if self.base_start_pixel is None:
            self.base_start_pixel = (int(canyon.cols // 2), int(canyon.rows // 2))

        base_px, base_py = self.base_start_pixel
        if self.start_pixel_jitter_px > 0:
            px = int(
                self.rng.integers(
                    base_px - self.start_pixel_jitter_px,
                    base_px + self.start_pixel_jitter_px + 1,
                )
            )
            py = int(
                self.rng.integers(
                    base_py - self.start_pixel_jitter_px,
                    base_py + self.start_pixel_jitter_px + 1,
                )
            )
        else:
            px, py = int(base_px), int(base_py)

        px = int(np.clip(px, 0, int(canyon.cols) - 1))
        py = int(np.clip(py, 0, int(canyon.rows) - 1))

        dem_start_info = canyon.get_pixel_info(px, py)
        dem_start_elev_ft = float(dem_start_info["elevation_msl_ft"])

        base_env.dem_start_pixel = (px, py)
        base_env.dem_start_info = dem_start_info
        base_env.dem_start_elev_ft = dem_start_elev_ft
        base_env.entry_altitude_ft = dem_start_elev_ft + self.entry_altitude_offset_ft
        base_env.target_altitude_ft = dem_start_elev_ft + self.target_altitude_offset_ft
        base_env.min_altitude_ft = dem_start_elev_ft + self.min_altitude_offset_ft
        base_env.max_altitude_ft = dem_start_elev_ft + self.max_altitude_offset_ft

        heading_mode = str(getattr(base_env, "dem_start_heading_mode", "keep_initial")).lower()
        heading_deg = None
        if heading_mode == "follow_canyon" and hasattr(canyon, "get_heading_for_pixel"):
            heading_deg = float(canyon.get_heading_for_pixel(px, py))
        elif heading_mode == "toward_center" and hasattr(canyon, "get_total_length_ft"):
            target_local_north_ft = 0.5 * float(canyon.get_total_length_ft())
            target_local_east_ft = 0.0
            dn = target_local_north_ft - float(dem_start_info["local_north_ft"])
            de = target_local_east_ft - float(dem_start_info["local_east_ft"])
            if abs(dn) + abs(de) > 1e-6:
                heading_deg = float(np.degrees(np.arctan2(de, dn)))
            else:
                heading_deg = 0.0
        elif self.base_start_heading_deg is not None:
            heading_deg = float(self.base_start_heading_deg)

        if heading_deg is not None and self.heading_jitter_deg > 0.0:
            heading_deg += float(self.rng.uniform(-self.heading_jitter_deg, self.heading_jitter_deg))
        if heading_deg is not None:
            heading_deg = float((heading_deg + 360.0) % 360.0)
            base_env.dem_start_heading_deg = heading_deg

        self.current_start_pixel = (px, py)
        self.current_start_heading_deg = heading_deg

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_dem_start()
        self.episode_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.episode_step += 1
        obs, _, terminated, truncated, info = self.env.step(action)

        speed_fps = float(
            np.sqrt(
                float(obs[OBS_U_FPS]) ** 2
                + float(obs[OBS_V_FPS]) ** 2
                + float(obs[OBS_W_FPS]) ** 2
            )
        )
        speed_term = self.speed_gain * np.clip((speed_fps - 350.0) / 450.0, -0.5, 1.5)

        progress_ft = float(info.get("progress_ft", 0.0))
        progress_from_start_ft = float(info.get("progress_from_start_ft", 0.0))
        progress_local = np.clip(progress_ft / 25.0, -2.0, 3.0)
        progress_global = np.clip(progress_from_start_ft / 25.0, -2.0, 2.0)
        progress_term = self.progress_gain * (0.8 * progress_local + 0.2 * progress_global)

        clearance_ft = float(info.get("terrain_clearance_ft", 0.0))
        clearance_error = abs(clearance_ft - self.target_clearance_ft)
        low_altitude_term = self.low_altitude_gain * (1.0 - np.clip(clearance_error / self.target_clearance_ft, 0.0, 2.0))

        lateral_norm = abs(float(obs[OBS_LATERAL_NORM]))
        centerline_term = self.centerline_gain * (1.0 - np.clip(lateral_norm / 1.5, 0.0, 1.0))
        offcenter_term = -self.offcenter_penalty_gain * np.clip(lateral_norm - 1.5, 0.0, 2.0)

        rate_mag_rad_s = float(
            np.sqrt(
                float(obs[OBS_P]) ** 2
                + float(obs[OBS_Q]) ** 2
                + float(obs[OBS_R]) ** 2
            )
        )
        rate_mag_deg_s = float(np.degrees(rate_mag_rad_s))
        angular_rate_term = -self.angular_rate_penalty_gain * np.clip(
            (rate_mag_deg_s - self.angular_rate_threshold_deg_s)
            / self.angular_rate_threshold_deg_s,
            0.0,
            3.0,
        )

        reward = (
            self.alive_bonus
            + progress_term
            + speed_term
            + low_altitude_term
            + centerline_term
            + offcenter_term
            + angular_rate_term
        )
        reward = float(np.clip(reward, -self.max_step_reward_abs, self.max_step_reward_abs))

        termination_reason = info.get("termination_reason", "running")
        if terminated:
            remaining_frac = np.clip(
                (self.max_episode_steps - self.episode_step) / max(float(self.max_episode_steps), 1.0),
                0.0,
                1.0,
            )
            early_termination_penalty = self.early_termination_penalty_gain * remaining_frac
            if termination_reason in {"terrain_collision", "ground_collision"}:
                reward -= self.terrain_crash_penalty + early_termination_penalty
            elif termination_reason == "hit_canyon_wall":
                reward -= self.wall_crash_penalty + 0.75 * early_termination_penalty
            elif termination_reason == "altitude_out_of_bounds":
                reward -= self.altitude_violation_penalty + 0.5 * early_termination_penalty
        elif truncated:
            reward += self.time_limit_bonus

        info["reward_progress"] = float(progress_term)
        info["reward_speed"] = float(speed_term)
        info["reward_low_altitude"] = float(low_altitude_term)
        info["reward_centerline"] = float(centerline_term)
        info["reward_offcenter"] = float(offcenter_term)
        info["reward_angular_rate"] = float(angular_rate_term)
        info["angular_rate_mag_deg_s"] = float(rate_mag_deg_s)
        info["reward_wall"] = float(centerline_term + offcenter_term)
        info["reward_alive"] = float(self.alive_bonus)
        info["reward_time_limit_bonus"] = float(self.time_limit_bonus if truncated else 0.0)
        info["reward_total"] = float(reward)
        if self.current_start_pixel is not None:
            info["start_pixel_x"] = float(self.current_start_pixel[0])
            info["start_pixel_y"] = float(self.current_start_pixel[1])
        if self.current_start_heading_deg is not None:
            info["start_heading_deg"] = float(self.current_start_heading_deg)

        return obs, float(reward), terminated, truncated, info


def make_canyon_env(
    start_pixel_jitter_px=40,
    start_heading_jitter_deg=12.0,
    angular_rate_penalty_gain=0.45,
    angular_rate_threshold_deg_s=45.0,
):
    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode=None,
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
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
        # Slight stochasticity helps avoid collapsing into one deterministic
        # bad trajectory with constant episode returns.
        wind_sigma=1.5,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )
    return CanyonSACRewardWrapper(
        env,
        angular_rate_penalty_gain=angular_rate_penalty_gain,
        angular_rate_threshold_deg_s=angular_rate_threshold_deg_s,
        start_pixel_jitter_px=start_pixel_jitter_px,
        heading_jitter_deg=start_heading_jitter_deg,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC for canyon flight in JSBSimCanyon-v0")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda)")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=10_000,
        help="Warmup steps before gradient updates begin",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Replay batch size")
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient updates per environment step batch",
    )
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto_0.2",
        help="Entropy coefficient (e.g. auto, auto_0.1, or fixed numeric like 0.05)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How many episodes between rollout log prints",
    )
    parser.add_argument(
        "--start-pixel-jitter-px",
        type=int,
        default=40,
        help="Uniform random start-pixel jitter radius in DEM pixels (0 disables)",
    )
    parser.add_argument(
        "--start-heading-jitter-deg",
        type=float,
        default=12.0,
        help="Uniform random start-heading jitter around canyon heading (0 disables)",
    )
    parser.add_argument(
        "--angular-rate-penalty-gain",
        type=float,
        default=0.45,
        help="Penalty gain for high total angular rate (0 disables)",
    )
    parser.add_argument(
        "--angular-rate-threshold-deg-s",
        type=float,
        default=45.0,
        help="Angular-rate magnitude threshold in deg/s before penalty begins",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    log_dir = REPO_ROOT / "logs"
    model_dir = REPO_ROOT / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "canyon_sac"
    model_path = model_dir / "jsbsim_canyon_sac"
    replay_path = model_dir / "jsbsim_canyon_sac_buffer"
    monitor_path = log_dir / "canyon_sac_monitor.csv"

    env = Monitor(
        make_canyon_env(
            start_pixel_jitter_px=args.start_pixel_jitter_px,
            start_heading_jitter_deg=args.start_heading_jitter_deg,
            angular_rate_penalty_gain=args.angular_rate_penalty_gain,
            angular_rate_threshold_deg_s=args.angular_rate_threshold_deg_s,
        ),
        filename=str(monitor_path),
        info_keywords=(
            "termination_reason",
            "start_pixel_x",
            "start_pixel_y",
            "start_heading_deg",
            "terrain_clearance_ft",
            "distance_from_start_ft",
            "progress_from_start_ft",
            "reward_progress",
            "reward_speed",
            "reward_low_altitude",
            "reward_centerline",
            "reward_offcenter",
            "reward_angular_rate",
            "angular_rate_mag_deg_s",
            "reward_alive",
            "reward_time_limit_bonus",
            "reward_total",
        ),
    )
    try:
        ent_coef_value = float(args.ent_coef)
    except ValueError:
        ent_coef_value = args.ent_coef

    print(
        f"Training config: timesteps={args.timesteps}, learning_starts={args.learning_starts}, "
        f"learning_rate={args.learning_rate}, batch_size={args.batch_size}, "
        f"gradient_steps={args.gradient_steps}, "
        f"ent_coef={args.ent_coef}, log_interval={args.log_interval}, "
        f"start_pixel_jitter_px={args.start_pixel_jitter_px}, "
        f"start_heading_jitter_deg={args.start_heading_jitter_deg}, "
        f"angular_rate_penalty_gain={args.angular_rate_penalty_gain}, "
        f"angular_rate_threshold_deg_s={args.angular_rate_threshold_deg_s}"
    )

    model = None
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_path),
            learning_rate=args.learning_rate,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            ent_coef=ent_coef_value,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps, log_interval=args.log_interval)
    finally:
        env.close()
        if model is not None:
            model.save(str(model_path))
            model.save_replay_buffer(str(replay_path))


if __name__ == "__main__":
    train()
