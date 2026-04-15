import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna

# Ensure jax uses CPU for parallel tuning to avoid GPU memory OOM and fast JIT overhead.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from jsbsim_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController
from run_scenario import to_mppi_state

DEM_PATH = REPO_ROOT / "data/dem/black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)

KTS_TO_FPS = 1.68781
DEFAULT_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'mppi_tuning.db').as_posix()}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune MPPI gains with Optuna.",
    )
    parser.add_argument("--trials", type=int, default=80, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="mppi_canyon_tuning",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE,
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 42],
        help="Episode seeds used to evaluate each trial.",
    )
    parser.add_argument("--max-steps", type=int, default=1200, help="Max steps per episode.")
    parser.add_argument(
        "--mppi-controller",
        type=str,
        choices=["smooth_mppi", "mppi"],
        default="mppi",
        help="Which MPPI variant to tune.",
    )
    parser.add_argument(
        "--target-speed-kts",
        type=float,
        default=450.0,
        help="Controller target speed in knots.",
    )
    parser.add_argument(
        "--target-clearance-ft",
        type=float,
        default=100.0,
        help="Reference clearance used by scoring and altitude search bounds.",
    )
    parser.add_argument(
        "--wind-sigma",
        type=float,
        default=0.6,
        help="Wind sigma in the environment for robustness.",
    )
    parser.add_argument(
        "--robustness-weight",
        type=float,
        default=0.20,
        help="Penalty weight on per-seed score stddev.",
    )
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel Optuna jobs.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional optimization timeout in seconds.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=123,
        help="Random seed for Optuna sampler.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "output/canyon_mppi/mppi_optuna_best.json",
        help="Path to write the best-parameter JSON summary.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--terrain-following",
        dest="terrain_following",
        action="store_true",
        help="Tune in terrain-following mode.",
    )
    mode_group.add_argument(
        "--altitude-hold",
        dest="terrain_following",
        action="store_false",
        help="Tune in altitude-hold mode.",
    )
    parser.set_defaults(terrain_following=True)

    return parser.parse_args()


def _base_mppi_config(args):
    return {
        "mppi_controller": str(args.mppi_controller),
        "target_speed_fps": float(args.target_speed_kts) * KTS_TO_FPS,
        "target_clearance_ft": float(args.target_clearance_ft),
        "terrain_following": bool(args.terrain_following),
    }


def _make_env(args, target_speed_kts):
    if args.terrain_following:
        target_altitude_ft = 250.0
        entry_altitude_ft = 250.0
        max_altitude_ft = 3000.0
    else:
        target_altitude_ft = 900.0
        entry_altitude_ft = 900.0
        max_altitude_ft = 5000.0

    return gym.make(
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
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=target_altitude_ft,
        entry_altitude_ft=entry_altitude_ft,
        min_altitude_ft=-500.0,
        max_altitude_ft=max_altitude_ft,
        max_episode_steps=int(args.max_steps),
        terrain_collision_buffer_ft=10.0,
        wind_sigma=float(args.wind_sigma),
        canyon_segment_spacing_ft=12.0,
        entry_speed_kts=float(target_speed_kts),
    )


def _sample_mppi_params(trial, args):
    base_clearance_ft = max(float(args.target_clearance_ft), 60.0)
    max_target_altitude_ft = max(base_clearance_ft + 900.0, 1000.0)

    params = {
        "lambda_": trial.suggest_float("lambda_", 0.01, 20.0, log=True),
        "gamma_": trial.suggest_float("gamma_", 0.001, 0.2, log=True),
        "action_diff_weight": trial.suggest_float("action_diff_weight", 0.0, 5.0),
        "action_l2_weight": trial.suggest_float("action_l2_weight", 0.0, 2.0),
        "low_altitude_gain": trial.suggest_float("low_altitude_gain", 0.0, 15.0),
        "centerline_gain": trial.suggest_float("centerline_gain", 0.0, 10.0),
        "offcenter_penalty_gain": trial.suggest_float("offcenter_penalty_gain", 0.0, 10.0),
        "heading_alignment_gain": trial.suggest_float("heading_alignment_gain", 0.0, 10.0),
        "target_alt_tune_ft": trial.suggest_float("target_alt_tune_ft", base_clearance_ft, max_target_altitude_ft),
        "progress_gain": trial.suggest_float("progress_gain", 0.0, 5.0),
        "speed_gain": trial.suggest_float("speed_gain", 0.0, 2.0),
        "heading_alignment_scale_rad": trial.suggest_float("heading_alignment_scale_rad", 0.1, 3.14),
        "alive_bonus": trial.suggest_float("alive_bonus", 0.0, 5.0),
        "terrain_crash_penalty": trial.suggest_float("terrain_crash_penalty", 10.0, 200.0),
        "wall_crash_penalty": trial.suggest_float("wall_crash_penalty", 10.0, 200.0),
        "angular_rate_penalty_gain": trial.suggest_float("angular_rate_penalty_gain", 0.0, 5.0),
        "delta_roll_bound": trial.suggest_float("delta_roll_bound", 0.05, 0.60),
        "delta_pitch_bound": trial.suggest_float("delta_pitch_bound", 0.05, 0.60),
    }

    if args.mppi_controller == "smooth_mppi":
        params["smoothness_penalty_weight"] = trial.suggest_float("smoothness_penalty_weight", 0.0, 3.0)

    return params


def _run_episode(args, mppi_params, seed):
    env = _make_env(args=args, target_speed_kts=args.target_speed_kts)

    total_score = 0.0
    total_abs_lateral_norm = 0.0
    total_abs_altitude_error_norm = 0.0
    total_speed_error_norm = 0.0
    steps = 0
    termination_reason = "running"

    try:
        _, _ = env.reset(seed=int(seed))
        state = env.unwrapped.get_full_state_dict()

        altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
        initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)

        min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
        max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)

        canyon = env.unwrapped.canyon
        if hasattr(canyon, "north_samples_ft") and hasattr(canyon, "width_samples_ft"):
            north_samples_ft = canyon.north_samples_ft
            width_samples_ft = canyon.width_samples_ft
            center_east_samples_ft = canyon.center_east_samples_ft
            centerline_heading_samples_rad = getattr(
                canyon,
                "centerline_heading_samples_rad",
                np.zeros_like(center_east_samples_ft),
            )
        else:
            north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
            width_samples_ft = np.ones(256, dtype=np.float32) * 1000.0
            center_east_samples_ft = np.zeros(256, dtype=np.float32)
            centerline_heading_samples_rad = np.zeros(256, dtype=np.float32)

        common_config_kwargs = dict(
            horizon=30,
            num_samples=1000,
            optimization_steps=3,
            lambda_=mppi_params["lambda_"],
            gamma_=mppi_params["gamma_"],
            progress_gain=mppi_params["progress_gain"],
            speed_gain=mppi_params["speed_gain"],
            low_altitude_gain=mppi_params["low_altitude_gain"],
            centerline_gain=mppi_params["centerline_gain"],
            offcenter_penalty_gain=mppi_params["offcenter_penalty_gain"],
            heading_alignment_gain=mppi_params["heading_alignment_gain"],
            heading_alignment_scale_rad=mppi_params["heading_alignment_scale_rad"],
            alive_bonus=mppi_params["alive_bonus"],
            target_altitude_ft=mppi_params["target_alt_tune_ft"],
            min_altitude_ft=min_altitude_ft,
            max_altitude_ft=max_altitude_ft,
            terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
            wall_margin_ft=float(env.unwrapped.wall_margin_ft),
            terrain_crash_penalty=mppi_params["terrain_crash_penalty"],
            wall_crash_penalty=mppi_params["wall_crash_penalty"],
            altitude_violation_penalty=8.0,
            early_termination_penalty_gain=120.0,
            time_limit_bonus=35.0,
            max_step_reward_abs=15.0,
            angular_rate_penalty_gain=mppi_params["angular_rate_penalty_gain"],
            angular_rate_threshold_deg_s=0.0,
            target_speed_fps=float(args.target_speed_kts) * KTS_TO_FPS,
            action_noise_std=(mppi_params["delta_roll_bound"], mppi_params["delta_pitch_bound"], 0.12, 0.10),
            action_diff_weight=mppi_params["action_diff_weight"],
            action_l2_weight=mppi_params["action_l2_weight"],
            debug_render_plans=False,
        )

        if args.mppi_controller == "mppi":
            config = JaxMPPIConfig(**common_config_kwargs)
            controller = JaxMPPIController(
                config=config,
                canyon_north_samples_ft=north_samples_ft,
                canyon_width_samples_ft=width_samples_ft,
                canyon_center_east_samples_ft=center_east_samples_ft,
                canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
            )
        else:
            config = JaxSmoothMPPIConfig(
                **common_config_kwargs,
                delta_noise_std=(mppi_params["delta_roll_bound"] * 0.6, mppi_params["delta_pitch_bound"] * 0.6, 0.08, 0.06),
                delta_action_bounds=(mppi_params["delta_roll_bound"], mppi_params["delta_pitch_bound"], 0.14, 0.10),
                noise_smoothing_kernel=(0.10, 0.20, 0.40, 0.20, 0.10),
                smoothness_penalty_weight=mppi_params["smoothness_penalty_weight"],
            )
            controller = JaxSmoothMPPIController(
                config=config,
                canyon_north_samples_ft=north_samples_ft,
                canyon_width_samples_ft=width_samples_ft,
                canyon_center_east_samples_ft=center_east_samples_ft,
                canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
            )

        _ = controller.get_action(initial_controller_state)

        for _ in range(int(args.max_steps)):
            controller_state = to_mppi_state(env, state, altitude_ref_ft)
            action = controller.get_action(controller_state)

            if not np.isfinite(action).all():
                termination_reason = "invalid_action"
                total_score -= 3000.0
                break

            width_ft = float(controller.get_canyon_width_ft(controller_state["p_N"]))
            lateral_ft = float(controller.get_lateral_error_ft(controller_state["p_N"], controller_state["p_E"]))

            _, _, terminated, truncated, info = env.step(action)
            state = env.unwrapped.get_full_state_dict()
            steps += 1

            h_ft = float(controller_state["h"])
            speed_fps = float(np.sqrt(float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2))

            lateral_norm = abs(lateral_ft) / max(width_ft * 0.5, 1.0)
            altitude_error_norm = abs(h_ft - mppi_params["target_alt_tune_ft"]) / max(float(args.target_clearance_ft), 50.0)
            speed_error_norm = abs(speed_fps - float(args.target_speed_kts) * KTS_TO_FPS) / max(float(args.target_speed_kts) * KTS_TO_FPS, 1.0)

            step_score = (
                2.0
                - 1.30 * np.clip(lateral_norm, 0.0, 2.5)
                - 0.90 * np.clip(altitude_error_norm, 0.0, 2.5)
                - 0.45 * np.clip(speed_error_norm, 0.0, 3.0)
            )
            total_score += float(step_score)

            total_abs_lateral_norm += float(lateral_norm)
            total_abs_altitude_error_norm += float(altitude_error_norm)
            total_speed_error_norm += float(speed_error_norm)

            if terminated or truncated:
                termination_reason = info.get("termination_reason", "time_limit" if truncated else "terminated")
                break

        survival_frac = float(steps) / max(float(args.max_steps), 1.0)
        total_score += 130.0 * survival_frac

        if termination_reason in {"running", "time_limit"}:
            total_score += 80.0
        elif termination_reason in {"terrain_collision", "hit_canyon_wall", "ground_collision", "invalid_action"}:
            total_score -= 220.0 + 320.0 * (1.0 - survival_frac)
        elif termination_reason == "altitude_out_of_bounds":
            total_score -= 160.0 + 200.0 * (1.0 - survival_frac)
        else:
            total_score -= 180.0
    finally:
        env.close()

    denom = max(float(steps), 1.0)
    episode_summary = {
        "score": float(total_score),
        "steps": int(steps),
        "termination_reason": str(termination_reason),
        "mean_abs_lateral_norm": float(total_abs_lateral_norm / denom),
        "mean_abs_altitude_error_norm": float(total_abs_altitude_error_norm / denom),
        "mean_speed_error_norm": float(total_speed_error_norm / denom),
    }
    return float(total_score), episode_summary


def _objective(trial, args, base_config):
    del base_config
    mppi_params = _sample_mppi_params(trial, args)
    trial.set_user_attr("mppi_controller", args.mppi_controller)

    seed_scores = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _run_episode(args=args, mppi_params=mppi_params, seed=seed)
        seed_scores.append(float(score))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lateral_norm", summary["mean_abs_lateral_norm"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_altitude_error_norm", summary["mean_abs_altitude_error_norm"])
        trial.set_user_attr(f"seed_{seed}_mean_speed_error_norm", summary["mean_speed_error_norm"])

        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(seed_scores))
    std_score = float(np.std(seed_scores))
    robust_score = mean_score - float(args.robustness_weight) * std_score

    trial.set_user_attr("mean_seed_score", mean_score)
    trial.set_user_attr("std_seed_score", std_score)
    return robust_score


def _save_best_summary(args, study, base_config):
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_name": str(study.study_name),
        "storage": str(args.storage),
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "base_config": dict(base_config),
        "evaluation": {
            "seeds": [int(s) for s in args.seeds],
            "max_steps": int(args.max_steps),
            "mppi_controller": str(args.mppi_controller),
            "terrain_following": bool(args.terrain_following),
            "target_speed_kts": float(args.target_speed_kts),
            "target_clearance_ft": float(args.target_clearance_ft),
            "wind_sigma": float(args.wind_sigma),
            "robustness_weight": float(args.robustness_weight),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    base_config = _base_mppi_config(args)

    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, len(args.seeds)),
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(
        "Evaluation setup: "
        f"seeds={args.seeds}, "
        f"max_steps={args.max_steps}, "
        f"mppi_controller={args.mppi_controller}, "
        f"terrain_following={args.terrain_following}, "
        f"target_speed_kts={args.target_speed_kts:.1f}, "
        f"target_clearance_ft={args.target_clearance_ft:.1f}, "
        f"wind_sigma={args.wind_sigma:.2f}"
    )

    study.optimize(
        lambda trial: _objective(trial, args, base_config),
        n_trials=int(args.trials),
        n_jobs=int(args.jobs),
        timeout=args.timeout,
    )

    summary_path = _save_best_summary(args, study, base_config)

    print("\n==============================")
    print("Optimization finished")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best robust score: {study.best_value:.3f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Saved summary JSON: {summary_path}")
    print("==============================")


if __name__ == "__main__":
    main()
