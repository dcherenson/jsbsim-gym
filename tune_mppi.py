import argparse
import os
import time
from pathlib import Path
import functools

import gymnasium as gym
import numpy as np
import optuna

# Ensure jax uses CPU for parallel tuning to avoid GPU memory OOM and fast JIT overhead
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController
from jsbsim_gym.simple_controller import SimpleCanyonController

# Re-use helpers from run_mppi
from run_mppi import to_mppi_state

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)

# Max steps for tuning episodes — used for both the loop limit and crash penalty.
# This is separate from the env's internal max_episode_steps to avoid the Gymnasium
# TimeLimit wrapper bug where env.unwrapped.max_episode_steps returns a different
# value than what was passed to gym.make.
TUNE_MAX_STEPS = 600

# Seeds to evaluate each trial on. Using multiple seeds prevents overfitting
# to a single canyon configuration. Seed 3 matches run_mppi.py.
EVAL_SEEDS = [3, 7, 42]


def objective(trial: optuna.Trial, target_speed_fps=800.0):
    # ----------------------------------------------------
    # 1. Parameter Sampling
    # ----------------------------------------------------
    lambda_ = trial.suggest_float("lambda_", 0.01, 20.0, log=True)
    gamma_ = trial.suggest_float("gamma_", 0.001, 0.2, log=True)
    horizon = trial.suggest_int("horizon", 15, 80, step=5)
    action_diff_weight = trial.suggest_float("action_diff_weight", 0.0, 5.0)
    action_l2_weight = trial.suggest_float("action_l2_weight", 0.0, 2.0)
    smoothness_penalty_weight = trial.suggest_float("smoothness_penalty_weight", 0.0, 3.0)

    # Tuning specific weights for contour following and altitude
    low_altitude_gain = trial.suggest_float("low_altitude_gain", 0.0, 15.0)
    centerline_gain = trial.suggest_float("centerline_gain", 0.0, 10.0)
    offcenter_penalty_gain = trial.suggest_float("offcenter_penalty_gain", 0.0, 10.0)
    heading_alignment_gain = trial.suggest_float("heading_alignment_gain", 0.0, 10.0)
    target_alt_tune_ft = trial.suggest_float("target_alt_tune_ft", 100.0, 1000.0)

    # Added parameters for additional tuning power
    progress_gain = trial.suggest_float("progress_gain", 0.0, 5.0)
    speed_gain = trial.suggest_float("speed_gain", 0.0, 2.0)
    heading_alignment_scale_rad = trial.suggest_float("heading_alignment_scale_rad", 0.1, 3.14)
    alive_bonus = trial.suggest_float("alive_bonus", 0.0, 5.0)
    terrain_crash_penalty = trial.suggest_float("terrain_crash_penalty", 10.0, 200.0)
    wall_crash_penalty = trial.suggest_float("wall_crash_penalty", 10.0, 200.0)
    angular_rate_penalty_gain = trial.suggest_float("angular_rate_penalty_gain", 0.0, 5.0)

    # Noise / exploration bounds — controls how far the planner can deviate
    delta_roll_bound = trial.suggest_float("delta_roll_bound", 0.05, 0.60)
    delta_pitch_bound = trial.suggest_float("delta_pitch_bound", 0.05, 0.60)

    # Evaluate across multiple seeds and average the reward
    seed_rewards = []
    for seed in EVAL_SEEDS:
        reward = _run_episode(
            seed=seed,
            lambda_=lambda_, gamma_=gamma_, horizon=horizon,
            action_diff_weight=action_diff_weight, action_l2_weight=action_l2_weight,
            smoothness_penalty_weight=smoothness_penalty_weight,
            low_altitude_gain=low_altitude_gain, centerline_gain=centerline_gain,
            offcenter_penalty_gain=offcenter_penalty_gain,
            heading_alignment_gain=heading_alignment_gain,
            target_alt_tune_ft=target_alt_tune_ft,
            progress_gain=progress_gain, speed_gain=speed_gain,
            heading_alignment_scale_rad=heading_alignment_scale_rad,
            alive_bonus=alive_bonus,
            terrain_crash_penalty=terrain_crash_penalty,
            wall_crash_penalty=wall_crash_penalty,
            angular_rate_penalty_gain=angular_rate_penalty_gain,
            delta_roll_bound=delta_roll_bound,
            delta_pitch_bound=delta_pitch_bound,
            target_speed_fps=target_speed_fps,
        )
        seed_rewards.append(reward)

    return float(np.mean(seed_rewards))


def _run_episode(
    seed, lambda_, gamma_, horizon,
    action_diff_weight, action_l2_weight, smoothness_penalty_weight,
    low_altitude_gain, centerline_gain, offcenter_penalty_gain,
    heading_alignment_gain, target_alt_tune_ft,
    progress_gain, speed_gain, heading_alignment_scale_rad,
    alive_bonus, terrain_crash_penalty, wall_crash_penalty,
    angular_rate_penalty_gain,
    delta_roll_bound, delta_pitch_bound,
    target_speed_fps=800.0,
):
    """Run a single episode with the given parameters and return the reward."""
    # ----------------------------------------------------
    # Environment Setup
    # ----------------------------------------------------
    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode=None,  # Headless for tuning
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
        dem_bbox=DEM_BBOX,
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=DEM_START_PIXEL,
        dem_start_heading_mode="follow_canyon",
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
        max_episode_steps=TUNE_MAX_STEPS,
        terrain_collision_buffer_ft=10.0,
        wind_sigma=1.0,
        canyon_segment_spacing_ft=12.0,
        entry_speed_kts=target_speed_fps / 1.68781,
    )

    obs, info = env.reset(seed=seed)
    state = env.unwrapped.get_full_state_dict()

    altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
    initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)

    target_altitude_ft = float(env.unwrapped.target_altitude_ft - altitude_ref_ft)
    min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
    max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)
    mppi_target_altitude_ft = target_alt_tune_ft

    canyon = env.unwrapped.canyon
    if hasattr(canyon, "north_samples_ft") and hasattr(canyon, "width_samples_ft"):
        north_samples_ft = canyon.north_samples_ft
        width_samples_ft = canyon.width_samples_ft
        center_east_samples_ft = canyon.center_east_samples_ft
        centerline_heading_samples_rad = getattr(canyon, "centerline_heading_samples_rad", np.zeros_like(center_east_samples_ft))
    else:
        # Fallback for procedural or other types
        north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
        width_samples_ft = np.ones(256, dtype=np.float32) * 1000.0
        center_east_samples_ft = np.zeros(256, dtype=np.float32)
        centerline_heading_samples_rad = np.zeros(256, dtype=np.float32)

    # Configure Smooth MPPI Controller
    config = JaxSmoothMPPIConfig(
        horizon=horizon,
        num_samples=768,
        optimization_steps=3,
        lambda_=lambda_,
        gamma_=gamma_,
        progress_gain=progress_gain,
        speed_gain=speed_gain,
        low_altitude_gain=low_altitude_gain,
        centerline_gain=centerline_gain,
        offcenter_penalty_gain=offcenter_penalty_gain,
        heading_alignment_gain=heading_alignment_gain,
        heading_alignment_scale_rad=heading_alignment_scale_rad,
        alive_bonus=alive_bonus,
        target_altitude_ft=mppi_target_altitude_ft,
        min_altitude_ft=min_altitude_ft,
        max_altitude_ft=max_altitude_ft,
        terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
        wall_margin_ft=float(env.unwrapped.wall_margin_ft),
        terrain_crash_penalty=terrain_crash_penalty,
        wall_crash_penalty=wall_crash_penalty,
        altitude_violation_penalty=8.0,
        early_termination_penalty_gain=120.0,
        time_limit_bonus=35.0,
        max_step_reward_abs=15.0,
        angular_rate_penalty_gain=angular_rate_penalty_gain,
        angular_rate_threshold_deg_s=0.0,
        target_speed_fps=target_speed_fps,
        action_noise_std=(delta_roll_bound, delta_pitch_bound, 0.12, 0.10),
        delta_noise_std=(delta_roll_bound * 0.6, delta_pitch_bound * 0.6, 0.08, 0.06),
        delta_action_bounds=(delta_roll_bound, delta_pitch_bound, 0.14, 0.10),
        noise_smoothing_kernel=(0.10, 0.20, 0.40, 0.20, 0.10),
        smoothness_penalty_weight=smoothness_penalty_weight,
        action_diff_weight=action_diff_weight,
        action_l2_weight=action_l2_weight,
        debug_render_plans=False,
    )

    controller = JaxSmoothMPPIController(
        config=config,
        canyon_north_samples_ft=north_samples_ft,
        canyon_width_samples_ft=width_samples_ft,
        canyon_center_east_samples_ft=center_east_samples_ft,
        canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
    )

    # JIT warm-up: compile the planner before we start timing/scoring
    _ = controller.get_action(initial_controller_state)

    total_reward = 0.0
    steps = 0
    sum_alt = 0.0
    sum_lat_err = 0.0

    try:
        while steps < TUNE_MAX_STEPS:
            controller_state = to_mppi_state(env, state, altitude_ref_ft)
            action = controller.get_action(controller_state)

            width_ft = float(controller.get_canyon_width_ft(controller_state["p_N"]))
            lateral_ft = float(controller.get_lateral_error_ft(
                controller_state["p_N"], controller_state["p_E"],
            ))

            if np.isnan(action).any() or np.isinf(action).any():
                total_reward -= 5000
                break

            _, step_reward, terminated, truncated, info_ = env.step(action)
            steps += 1
            state = env.unwrapped.get_full_state_dict()

            h_ft = float(controller_state["h"])
            sum_alt += h_ft
            sum_lat_err += abs(lateral_ft)

            # Reward: survival is king, but we also want low altitude + centerline
            alt_penalty = max(0.0, h_ft - 100.0) / 50.0
            lat_penalty = abs(lateral_ft) / 50.0
            step_score = 10.0 - alt_penalty - lat_penalty
            total_reward += step_score

            if terminated or truncated:
                termination_reason = info_.get("termination_reason", "running")
                if termination_reason not in ["success", "time_limit", "running"]:
                    # Penalty proportional to how early the crash is.
                    # Use TUNE_MAX_STEPS (not env.unwrapped) so it matches the loop limit.
                    missed_steps = TUNE_MAX_STEPS - steps
                    total_reward -= missed_steps * 10.0
                break
    finally:
        env.close()

    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run.")
    parser.add_argument("--study-name", type=str, default="mppi_canyon_tuning", help="Name of the Optuna study.")
    parser.add_argument("--target-speed-kts", type=float, default=450.0, help="Target speed in kts.")
    args = parser.parse_args()

    # Create Optuna study to maximize the objective (total reward)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage="sqlite:///mppi_tuning.db", # Store results across runs
        load_if_exists=True,
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"Running {args.trials} trials...")

    study.optimize(
        functools.partial(objective, target_speed_fps=args.target_speed_kts * 1.68781),
        n_trials=args.trials,
        n_jobs=-1
    )

    print("\n==============================")
    print("Optimization finished!")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (reward): {study.best_value:.2f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==============================")
