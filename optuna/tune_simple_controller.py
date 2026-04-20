import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.simple_controller import SimpleCanyonController, SimpleCanyonControllerConfig

DEM_PATH = REPO_ROOT / "data/dem/black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)

KTS_TO_FPS = 1.68781
M_TO_FT = 3.28084
MAX_ABOVE_CANYON_TOP_FT = 500.0
DEFAULT_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'simple_controller_tuning.db').as_posix()}"
FIXED_LIMIT_KEYS = frozenset(
    {
        "roll_max_rad",
        "nz_min_cmd",
        "nz_max_cmd",
        "yaw_max_cmd",
        "throttle_max",
        "track_accel_max_fps2",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune SimpleCanyonController gains with Optuna.",
    )
    parser.add_argument("--trials", type=int, default=80, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="simple_controller_gain_tuning",
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
        "--target-speed-kts",
        type=float,
        default=450.0,
        help="Controller target speed in knots.",
    )
    parser.add_argument(
        "--centerline-error-scale-ft",
        type=float,
        default=250.0,
        help="Centerline-error normalization scale (ft) used by the tuner score.",
    )
    parser.add_argument(
        "--altitude-error-scale-ft",
        type=float,
        default=200.0,
        help="Altitude-error normalization scale (ft) used by the tuner score.",
    )
    parser.add_argument(
        "--lookahead-rows",
        type=int,
        default=10,
        help="DEM lookahead rows for centerline guidance.",
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
        default=REPO_ROOT / "output/simple_controller/simple_controller_optuna_best.json",
        help="Path to write the best-parameter JSON summary.",
    )

    return parser.parse_args()


def _base_controller_config(args):
    return SimpleCanyonControllerConfig(
        target_speed_fps=float(args.target_speed_kts) * KTS_TO_FPS,
        lookahead_rows=int(max(1, args.lookahead_rows)),
        use_dem_centerline=True,
    )


def _suggest_signed_magnitude(trial, name, baseline, min_scale, max_scale):
    magnitude = trial.suggest_float(
        name,
        abs(float(baseline)) * float(min_scale),
        abs(float(baseline)) * float(max_scale),
        log=True,
    )
    return -magnitude if baseline < 0.0 else magnitude


def _suggest_signed_range(trial, name, baseline, magnitude_low, magnitude_high, *, log=False):
    magnitude = trial.suggest_float(
        name,
        float(magnitude_low),
        float(magnitude_high),
        log=bool(log),
    )
    return -magnitude if float(baseline) < 0.0 else magnitude


def _sample_controller_config(trial, base_config):
    return replace(
        base_config,
        lateral_lookahead_time_s=trial.suggest_float("lateral_lookahead_time_s", 0.6, 3.0),
        lateral_lookahead_min_ft=trial.suggest_float("lateral_lookahead_min_ft", 60.0, 300.0),
        lateral_lookahead_max_ft=trial.suggest_float("lateral_lookahead_max_ft", 400.0, 2500.0),
        lateral_lookahead_width_gain=trial.suggest_float("lateral_lookahead_width_gain", 0.0, 1.5),
        lateral_nonlinear_cross_track_gain=trial.suggest_float("lateral_nonlinear_cross_track_gain", 0.2, 4.0, log=True),
        lateral_damping_gain=trial.suggest_float("lateral_damping_gain", 0.0, 1.5),
        lateral_curvature_ff_gain=trial.suggest_float("lateral_curvature_ff_gain", 0.0, 2.0),
    )


def _filtered_best_params(params):
    if not isinstance(params, dict):
        return {}
    return {
        key: value
        for key, value in params.items()
        if key not in FIXED_LIMIT_KEYS
    }


def _effective_trial_params(trial):
    effective = trial.user_attrs.get("effective_params")
    if isinstance(effective, dict) and effective:
        return _filtered_best_params(effective)
    return _filtered_best_params(trial.params)


def _make_env(max_steps, target_speed_kts, wind_sigma):
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
        dem_start_heading_deg=None,
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=900.0,
        entry_altitude_ft=900.0,
        min_altitude_ft=-500.0,
        max_altitude_ft=5000.0,
        max_episode_steps=int(max_steps),
        terrain_collision_buffer_ft=10.0,
        wind_sigma=float(wind_sigma),
        entry_speed_kts=float(target_speed_kts),
        canyon_segment_spacing_ft=12.0,
    )


def _run_episode(
    config,
    seed,
    max_steps,
    target_speed_kts,
    wind_sigma,
    centerline_error_scale_ft,
    altitude_error_scale_ft,
):
    env = _make_env(max_steps=max_steps, target_speed_kts=target_speed_kts, wind_sigma=wind_sigma)

    total_env_reward = 0.0
    total_abs_lateral_ft = 0.0
    total_abs_lateral_norm = 0.0
    total_abs_altitude_error_norm = 0.0
    total_speed_error_norm = 0.0
    max_abs_lateral_ft = 0.0
    max_abs_altitude_error_ft = 0.0
    lateral_abs_samples_ft = []
    altitude_abs_samples_ft = []
    steps = 0
    termination_reason = "running"

    try:
        _, _ = env.reset(seed=int(seed))
        state = env.unwrapped.get_full_state_dict()
        canyon = env.unwrapped.canyon
        canyon_top_profile_ft = None
        canyon_top_north_samples_ft = None
        if (
            hasattr(canyon, "ordered_dem_msl_m")
            and hasattr(canyon, "north_samples_ft")
        ):
            canyon_top_profile_ft = np.nanmax(np.asarray(canyon.ordered_dem_msl_m, dtype=np.float64), axis=1) * M_TO_FT
            canyon_top_north_samples_ft = np.asarray(canyon.north_samples_ft, dtype=np.float64)

        controller = SimpleCanyonController(env=env, config=config)
        controller.reset(state)

        for _ in range(int(max_steps)):
            action = controller.get_action(state)
            guidance = dict(controller.last_guidance)

            if not np.isfinite(action).all():
                termination_reason = "invalid_action"
                break

            _, env_reward, terminated, truncated, info = env.step(action)
            state = env.unwrapped.get_full_state_dict()

            steps += 1
            total_env_reward += float(env_reward)

            if canyon_top_profile_ft is not None and canyon_top_north_samples_ft is not None:
                local_north_ft = float(state["p_N"]) - float(getattr(canyon, "anchor_north_ft", 0.0))
                local_north_ft = float(np.clip(local_north_ft, canyon_top_north_samples_ft[0], canyon_top_north_samples_ft[-1]))
                canyon_top_msl_ft = float(np.interp(local_north_ft, canyon_top_north_samples_ft, canyon_top_profile_ft))
                if float(state["h"]) > canyon_top_msl_ft + MAX_ABOVE_CANYON_TOP_FT:
                    termination_reason = "above_canyon_top"
                    break

            lateral_error_ft = float(abs(info.get("lateral_error_ft", guidance.get("lateral_error_ft", 0.0))))
            lateral_norm = float(abs(info.get("lateral_error_norm", guidance.get("lateral_norm", 0.0))))
            altitude_error_ft = float(abs(info.get("altitude_error_ft", guidance.get("altitude_error_ft", 0.0))))
            speed_fps = float(guidance.get("speed_fps", 0.0))

            centerline_scale_ft = max(float(centerline_error_scale_ft), 1.0)
            altitude_scale_ft = max(float(altitude_error_scale_ft), 1.0)
            altitude_error_norm = altitude_error_ft / altitude_scale_ft
            speed_error_norm = abs(speed_fps - float(config.target_speed_fps)) / max(float(config.target_speed_fps), 1.0)

            total_abs_lateral_ft += lateral_error_ft
            total_abs_lateral_norm += lateral_norm
            total_abs_altitude_error_norm += altitude_error_norm
            total_speed_error_norm += speed_error_norm
            lateral_abs_samples_ft.append(lateral_error_ft)
            altitude_abs_samples_ft.append(altitude_error_ft)
            max_abs_lateral_ft = max(max_abs_lateral_ft, lateral_error_ft)
            max_abs_altitude_error_ft = max(max_abs_altitude_error_ft, altitude_error_ft)

            if terminated or truncated:
                termination_reason = info.get("termination_reason", "time_limit" if truncated else "terminated")
                break

        survival_frac = float(steps) / max(float(max_steps), 1.0)
    finally:
        env.close()

    denom = max(float(steps), 1.0)
    lateral_arr = np.asarray(lateral_abs_samples_ft, dtype=np.float64)
    altitude_arr = np.asarray(altitude_abs_samples_ft, dtype=np.float64)
    if lateral_arr.size > 0:
        rms_lateral_ft = float(np.sqrt(np.mean(np.square(lateral_arr))))
        p95_abs_lateral_ft = float(np.percentile(lateral_arr, 95.0))
    else:
        rms_lateral_ft = 0.0
        p95_abs_lateral_ft = 0.0
    if altitude_arr.size > 0:
        rms_altitude_error_ft = float(np.sqrt(np.mean(np.square(altitude_arr))))
        p95_abs_altitude_error_ft = float(np.percentile(altitude_arr, 95.0))
    else:
        rms_altitude_error_ft = 0.0
        p95_abs_altitude_error_ft = 0.0

    mean_abs_lateral_ft = float(total_abs_lateral_ft / denom)
    mean_abs_altitude_error_ft = float(np.sum(altitude_arr) / denom)
    centerline_scale_ft = max(float(centerline_error_scale_ft), 1.0)
    altitude_scale_ft = max(float(altitude_error_scale_ft), 1.0)
    centerline_cost = (
        1.00 * (mean_abs_lateral_ft / centerline_scale_ft)
        + 1.50 * (rms_lateral_ft / centerline_scale_ft)
        + 0.50 * (p95_abs_lateral_ft / centerline_scale_ft)
    )
    altitude_cost = (
        1.00 * (mean_abs_altitude_error_ft / altitude_scale_ft)
        + 1.50 * (rms_altitude_error_ft / altitude_scale_ft)
        + 0.50 * (p95_abs_altitude_error_ft / altitude_scale_ft)
    )

    termination_penalty = 0.0
    if termination_reason in {"running", "time_limit"}:
        termination_penalty = 0.0
    elif termination_reason in {"terrain_collision", "hit_canyon_wall", "ground_collision", "invalid_action"}:
        termination_penalty = 400.0 + 600.0 * (1.0 - survival_frac)
    elif termination_reason in {"altitude_out_of_bounds", "above_canyon_top"}:
        termination_penalty = 220.0 + 320.0 * (1.0 - survival_frac)
    else:
        termination_penalty = 260.0

    total_score = -(centerline_cost + altitude_cost + termination_penalty)

    episode_summary = {
        "score": float(total_score),
        "env_reward": float(total_env_reward),
        "steps": int(steps),
        "termination_reason": str(termination_reason),
        "mean_abs_lateral_ft": mean_abs_lateral_ft,
        "rms_lateral_ft": float(rms_lateral_ft),
        "p95_abs_lateral_ft": float(p95_abs_lateral_ft),
        "max_abs_lateral_ft": float(max_abs_lateral_ft),
        "mean_abs_altitude_error_ft": float(mean_abs_altitude_error_ft),
        "rms_altitude_error_ft": float(rms_altitude_error_ft),
        "p95_abs_altitude_error_ft": float(p95_abs_altitude_error_ft),
        "max_abs_altitude_error_ft": float(max_abs_altitude_error_ft),
        "mean_abs_lateral_norm": float(total_abs_lateral_norm / denom),
        "mean_abs_altitude_error_norm": float(total_abs_altitude_error_norm / denom),
        "mean_speed_error_norm": float(total_speed_error_norm / denom),
    }
    return float(total_score), episode_summary


def _objective(trial, args, base_config):
    config = _sample_controller_config(trial, base_config)
    trial.set_user_attr(
        "effective_params",
        {
            key: getattr(config, key)
            for key in trial.params.keys()
            if key not in FIXED_LIMIT_KEYS
        },
    )

    seed_scores = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _run_episode(
            config=config,
            seed=seed,
            max_steps=args.max_steps,
            target_speed_kts=args.target_speed_kts,
            wind_sigma=args.wind_sigma,
            centerline_error_scale_ft=args.centerline_error_scale_ft,
            altitude_error_scale_ft=args.altitude_error_scale_ft,
        )
        seed_scores.append(float(score))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lateral_ft", summary["mean_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_lateral_ft", summary["rms_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_abs_lateral_ft", summary["p95_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_max_abs_lateral_ft", summary["max_abs_lateral_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_altitude_error_ft", summary["mean_abs_altitude_error_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_altitude_error_ft", summary["rms_altitude_error_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_abs_altitude_error_ft", summary["p95_abs_altitude_error_ft"])
        trial.set_user_attr(f"seed_{seed}_max_abs_altitude_error_ft", summary["max_abs_altitude_error_ft"])
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
    best_trial = study.best_trial

    payload = {
        "study_name": str(study.study_name),
        "storage": str(args.storage),
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": _effective_trial_params(best_trial),
        "base_config": asdict(base_config),
        "evaluation": {
            "seeds": [int(s) for s in args.seeds],
            "max_steps": int(args.max_steps),
            "target_speed_kts": float(args.target_speed_kts),
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
    base_config = _base_controller_config(args)

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
        f"target_speed_kts={args.target_speed_kts:.1f}, "
        f"centerline_error_scale_ft={args.centerline_error_scale_ft:.1f}, "
        f"altitude_error_scale_ft={args.altitude_error_scale_ft:.1f}, "
        f"wind_sigma={args.wind_sigma:.2f}"
    )

    study.optimize(
        lambda trial: _objective(trial, args, base_config),
        n_trials=int(args.trials),
        n_jobs=int(args.jobs),
        timeout=args.timeout,
    )

    best_trial = study.best_trial
    summary_path = _save_best_summary(args, study, base_config)

    print("\n==============================")
    print("Optimization finished")
    print(f"Best trial: #{best_trial.number}")
    print(f"Best robust score: {float(best_trial.value):.3f}")
    print("Best parameters:")
    for key, value in _effective_trial_params(best_trial).items():
        print(f"  {key}: {value}")
    print(f"Saved summary JSON: {summary_path}")
    print("==============================")


if __name__ == "__main__":
    main()
