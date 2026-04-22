import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna

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
LOW_FLIGHT_REFERENCE_FT = 100.0


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
        default=700.0,
        help="Entry speed in knots for the evaluation scenario.",
    )
    parser.add_argument(
        "--centerline-error-scale-ft",
        type=float,
        default=200.0,
        help="Normalization scale in feet for centerline tracking penalties.",
    )
    parser.add_argument(
        "--altitude-error-scale-ft",
        type=float,
        default=150.0,
        help="Normalization scale in feet for altitude tracking penalties.",
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

    return parser.parse_args()


def _base_mppi_config(args):
    return {
        "mppi_controller": str(args.mppi_controller),
        "entry_speed_fps": float(args.target_speed_kts) * KTS_TO_FPS,
        "low_flight_reference_ft": float(LOW_FLIGHT_REFERENCE_FT),
    }


def _make_env(args, target_speed_kts):
    target_altitude_ft = 500.0
    entry_altitude_ft = 500.0
    max_altitude_ft = 1500.0

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
    params = {
        "lambda_": trial.suggest_float("lambda_", 0.01, 20.0, log=True),
        "gamma_": trial.suggest_float("gamma_", 0.001, 2.0, log=True),
        "action_diff_weight": trial.suggest_float("action_diff_weight", 0.0, 5.0),
        "action_l2_weight": trial.suggest_float("action_l2_weight", 0.0, 2.0),
        "low_altitude_gain": trial.suggest_float("low_altitude_gain", 0.0, 15.0),
        "centerline_gain": trial.suggest_float("centerline_gain", 0.0, 5.0),
        "offcenter_penalty_gain": trial.suggest_float("offcenter_penalty_gain", 0.0, 5.0),
        "progress_gain": trial.suggest_float("progress_gain", 0.0, 0.0),
        "speed_gain": trial.suggest_float("speed_gain", 0.0, 20.0),
        "terrain_crash_penalty": trial.suggest_float("terrain_crash_penalty", 100.0, 600.0),
        "action_noise_std_roll": trial.suggest_float("action_noise_std_roll", 0.05, 1.00),
        "action_noise_std_pitch": trial.suggest_float("action_noise_std_pitch", 0.05, 1.00),
        "action_noise_std_yaw": trial.suggest_float("action_noise_std_yaw", 0.02, 1.00),
        "action_noise_std_throttle": trial.suggest_float("action_noise_std_throttle", 0.02, 1.00),
    }

    if args.mppi_controller == "smooth_mppi":
        params["smoothness_penalty_weight"] = trial.suggest_float("smoothness_penalty_weight", 0.05, 3.0)

    return params


def _run_episode(args, mppi_params, seed):
    env = _make_env(args=args, target_speed_kts=args.target_speed_kts)

    total_abs_lateral_ft = 0.0
    total_altitude_above_min_ft = 0.0
    total_abs_lateral_norm = 0.0
    total_altitude_above_min_norm = 0.0
    total_speed_fps = 0.0
    max_abs_lateral_ft = 0.0
    max_altitude_above_min_ft = 0.0
    lateral_abs_samples_ft = []
    altitude_above_min_samples_ft = []
    steps = 0
    termination_reason = "running"

    try:
        _, _ = env.reset(seed=int(seed))
        state = env.unwrapped.get_full_state_dict()

        altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
        initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)

        min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
        max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)
        low_flight_altitude_ft = float(
            np.clip(
                float(LOW_FLIGHT_REFERENCE_FT),
                min_altitude_ft + 40.0,
                max_altitude_ft,
            )
        )

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
            horizon=40,
            num_samples=4000,
            optimization_steps=3,
            lambda_=mppi_params["lambda_"],
            gamma_=mppi_params["gamma_"],
            progress_gain=mppi_params["progress_gain"],
            speed_gain=mppi_params["speed_gain"],
            low_altitude_gain=mppi_params["low_altitude_gain"],
            centerline_gain=mppi_params["centerline_gain"],
            offcenter_penalty_gain=mppi_params["offcenter_penalty_gain"],
            target_altitude_ft=low_flight_altitude_ft,
            min_altitude_ft=min_altitude_ft,
            max_altitude_ft=max_altitude_ft,
            terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
            wall_margin_ft=float(env.unwrapped.wall_margin_ft),
            terrain_crash_penalty=mppi_params["terrain_crash_penalty"],
            early_termination_penalty_gain=120.0,
            max_step_reward_abs=15.0,
            target_speed_fps=float(args.target_speed_kts) * KTS_TO_FPS,
            action_noise_std=(
                mppi_params["action_noise_std_roll"],
                mppi_params["action_noise_std_pitch"],
                mppi_params["action_noise_std_yaw"],
                mppi_params["action_noise_std_throttle"],
            ),
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
                delta_noise_std=(
                    mppi_params["action_noise_std_roll"] * 0.6,
                    mppi_params["action_noise_std_pitch"] * 0.6,
                    mppi_params["action_noise_std_yaw"] * 0.6,
                    mppi_params["action_noise_std_throttle"] * 0.6,
                ),
                delta_action_bounds=(
                    mppi_params["action_noise_std_roll"],
                    mppi_params["action_noise_std_pitch"],
                    mppi_params["action_noise_std_yaw"],
                    mppi_params["action_noise_std_throttle"],
                ),
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
                break

            p_n_ft = float(controller_state["p_N"])
            p_e_ft = float(controller_state["p_E"])
            width_ft = float(np.interp(p_n_ft, north_samples_ft, width_samples_ft))
            center_east_ft = float(np.interp(p_n_ft, north_samples_ft, center_east_samples_ft))
            lateral_ft = p_e_ft - center_east_ft

            _, _, terminated, truncated, info = env.step(action)
            state = env.unwrapped.get_full_state_dict()
            steps += 1

            h_ft = float(controller_state["h"])
            speed_fps = float(np.sqrt(float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2))
            altitude_above_min_ft = max(h_ft - min_altitude_ft, 0.0)
            abs_lateral_ft = abs(lateral_ft)

            lateral_norm = abs_lateral_ft / max(width_ft * 0.5, 1.0)
            altitude_above_min_norm = altitude_above_min_ft / max(float(args.altitude_error_scale_ft), 1.0)
            total_abs_lateral_norm += float(lateral_norm)
            total_altitude_above_min_norm += float(altitude_above_min_norm)
            total_speed_fps += float(speed_fps)
            total_abs_lateral_ft += float(abs_lateral_ft)
            total_altitude_above_min_ft += float(altitude_above_min_ft)
            lateral_abs_samples_ft.append(float(abs_lateral_ft))
            altitude_above_min_samples_ft.append(float(altitude_above_min_ft))
            max_abs_lateral_ft = max(max_abs_lateral_ft, float(abs_lateral_ft))
            max_altitude_above_min_ft = max(max_altitude_above_min_ft, float(altitude_above_min_ft))

            if terminated or truncated:
                termination_reason = info.get("termination_reason", "time_limit" if truncated else "terminated")
                break

        survival_frac = float(steps) / max(float(args.max_steps), 1.0)
    finally:
        env.close()

    denom = max(float(steps), 1.0)
    lateral_arr = np.asarray(lateral_abs_samples_ft, dtype=np.float64)
    altitude_arr = np.asarray(altitude_above_min_samples_ft, dtype=np.float64)

    if lateral_arr.size > 0:
        rms_lateral_ft = float(np.sqrt(np.mean(np.square(lateral_arr))))
        p95_abs_lateral_ft = float(np.percentile(lateral_arr, 95.0))
    else:
        rms_lateral_ft = 0.0
        p95_abs_lateral_ft = 0.0

    if altitude_arr.size > 0:
        rms_altitude_above_min_ft = float(np.sqrt(np.mean(np.square(altitude_arr))))
        p95_altitude_above_min_ft = float(np.percentile(altitude_arr, 95.0))
    else:
        rms_altitude_above_min_ft = 0.0
        p95_altitude_above_min_ft = 0.0

    mean_abs_lateral_ft = float(total_abs_lateral_ft / denom)
    mean_altitude_above_min_ft = float(total_altitude_above_min_ft / denom)
    centerline_scale_ft = max(float(args.centerline_error_scale_ft), 1.0)
    altitude_scale_ft = max(float(args.altitude_error_scale_ft), 1.0)

    centerline_cost = (
        1.00 * (mean_abs_lateral_ft / centerline_scale_ft)
        + 1.50 * (rms_lateral_ft / centerline_scale_ft)
        + 0.50 * (p95_abs_lateral_ft / centerline_scale_ft)
    )
    altitude_cost = (
        1.00 * (mean_altitude_above_min_ft / altitude_scale_ft)
        + 1.50 * (rms_altitude_above_min_ft / altitude_scale_ft)
        + 0.50 * (p95_altitude_above_min_ft / altitude_scale_ft)
    )
    mean_speed_fps = float(total_speed_fps / denom)
    speed_reward = mean_speed_fps / 200.0

    termination_penalty = 0.0
    if termination_reason in {"running", "time_limit"}:
        termination_penalty = 0.0
    elif termination_reason in {"terrain_collision", "hit_canyon_wall", "ground_collision", "invalid_action"}:
        termination_penalty = 400.0 + 600.0 * (1.0 - survival_frac)
    elif termination_reason in {"altitude_out_of_bounds", "above_canyon_top"}:
        termination_penalty = 220.0 + 320.0 * (1.0 - survival_frac)
    else:
        termination_penalty = 260.0

    total_score = speed_reward - (centerline_cost + altitude_cost + termination_penalty)
    episode_summary = {
        "score": float(total_score),
        "steps": int(steps),
        "termination_reason": str(termination_reason),
        "mean_abs_lateral_ft": float(mean_abs_lateral_ft),
        "rms_lateral_ft": float(rms_lateral_ft),
        "p95_abs_lateral_ft": float(p95_abs_lateral_ft),
        "max_abs_lateral_ft": float(max_abs_lateral_ft),
        "mean_altitude_above_min_ft": float(mean_altitude_above_min_ft),
        "rms_altitude_above_min_ft": float(rms_altitude_above_min_ft),
        "p95_altitude_above_min_ft": float(p95_altitude_above_min_ft),
        "max_altitude_above_min_ft": float(max_altitude_above_min_ft),
        "mean_abs_lateral_norm": float(total_abs_lateral_norm / denom),
        "mean_altitude_above_min_norm": float(total_altitude_above_min_norm / denom),
        "mean_speed_fps": float(mean_speed_fps),
    }
    return float(total_score), episode_summary


def _objective(trial, args, base_config):
    del base_config
    mppi_params = _sample_mppi_params(trial, args)
    trial.set_user_attr("mppi_controller", args.mppi_controller)

    seed_scores = []
    seed_summaries = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _run_episode(args=args, mppi_params=mppi_params, seed=seed)
        seed_scores.append(float(score))
        seed_summaries.append((int(seed), summary))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lateral_ft", summary["mean_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_lateral_ft", summary["rms_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_abs_lateral_ft", summary["p95_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_max_abs_lateral_ft", summary["max_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_altitude_above_min_ft", summary["mean_altitude_above_min_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_altitude_above_min_ft", summary["rms_altitude_above_min_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_altitude_above_min_ft", summary["p95_altitude_above_min_ft"])
        trial.set_user_attr(f"seed_{seed}_max_altitude_above_min_ft", summary["max_altitude_above_min_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lateral_norm", summary["mean_abs_lateral_norm"])
        trial.set_user_attr(f"seed_{seed}_mean_altitude_above_min_norm", summary["mean_altitude_above_min_norm"])
        trial.set_user_attr(f"seed_{seed}_mean_speed_fps", summary["mean_speed_fps"])

        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(seed_scores))
    std_score = float(np.std(seed_scores))
    robust_score = mean_score - float(args.robustness_weight) * std_score

    trial.set_user_attr("mean_seed_score", mean_score)
    trial.set_user_attr("std_seed_score", std_score)

    for seed, summary in seed_summaries:
        print(
            f"Trial {trial.number:4d} | seed {seed:>3d} | "
            f"score {summary['score']:8.2f} | "
            f"term {summary['termination_reason']:<20} | "
            f"steps {summary['steps']:4d} | "
            f"speed {summary['mean_speed_fps']:7.1f} | "
            f"lat {summary['mean_abs_lateral_ft']:7.1f} | "
            f"alt {summary['mean_altitude_above_min_ft']:7.1f}"
        )

    print(
        f"Trial {trial.number:4d} | aggregate | "
        f"mean {mean_score:8.2f} | std {std_score:7.2f} | robust {robust_score:8.2f}"
    )
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
            "entry_speed_kts": float(args.target_speed_kts),
            "low_flight_reference_ft": float(LOW_FLIGHT_REFERENCE_FT),
            "centerline_error_scale_ft": float(args.centerline_error_scale_ft),
            "altitude_error_scale_ft": float(args.altitude_error_scale_ft),
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
        f"entry_speed_kts={args.target_speed_kts:.1f}, "
        f"low_flight_reference_ft={LOW_FLIGHT_REFERENCE_FT:.1f}, "
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
