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
DEFAULT_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'simple_controller_tuning.db').as_posix()}"


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
        "--target-clearance-ft",
        type=float,
        default=100.0,
        help="Target terrain clearance used by the controller.",
    )
    parser.add_argument(
        "--lookahead-rows",
        type=int,
        default=40,
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


def _base_controller_config(args):
    return SimpleCanyonControllerConfig(
        target_speed_fps=float(args.target_speed_kts) * KTS_TO_FPS,
        target_clearance_ft=float(args.target_clearance_ft),
        lookahead_rows=int(max(1, args.lookahead_rows)),
        use_dem_centerline=True,
        use_terrain_following=bool(args.terrain_following),
    )


def _suggest_signed_magnitude(trial, name, baseline, min_scale, max_scale):
    magnitude = trial.suggest_float(
        name,
        abs(float(baseline)) * float(min_scale),
        abs(float(baseline)) * float(max_scale),
        log=True,
    )
    return -magnitude if baseline < 0.0 else magnitude


def _sample_controller_config(trial, base_config):
    return replace(
        base_config,
        roll_lateral_gain=_suggest_signed_magnitude(trial, "roll_lateral_gain", base_config.roll_lateral_gain, 0.35, 3.5),
        roll_rate_gain=_suggest_signed_magnitude(trial, "roll_rate_gain", base_config.roll_rate_gain, 0.2, 6.0),
        roll_heading_gain=_suggest_signed_magnitude(trial, "roll_heading_gain", base_config.roll_heading_gain, 0.3, 3.5),
        roll_p_gain=_suggest_signed_magnitude(trial, "roll_p_gain", base_config.roll_p_gain, 0.35, 4.0),
        roll_rate_damping=_suggest_signed_magnitude(trial, "roll_rate_damping", base_config.roll_rate_damping, 0.2, 5.0),
        pitch_clearance_gain=_suggest_signed_magnitude(trial, "pitch_clearance_gain", base_config.pitch_clearance_gain, 0.25, 5.0),
        pitch_ahead_gain=_suggest_signed_magnitude(trial, "pitch_ahead_gain", base_config.pitch_ahead_gain, 0.25, 5.0),
        pitch_integral_gain=_suggest_signed_magnitude(trial, "pitch_integral_gain", base_config.pitch_integral_gain, 0.2, 8.0),
        pitch_rate_gain=_suggest_signed_magnitude(trial, "pitch_rate_gain", base_config.pitch_rate_gain, 0.25, 5.0),
        pitch_q_gain=_suggest_signed_magnitude(trial, "pitch_q_gain", base_config.pitch_q_gain, 0.35, 3.0),
        pitch_rate_damping=_suggest_signed_magnitude(trial, "pitch_rate_damping", base_config.pitch_rate_damping, 0.2, 5.0),
        yaw_rate_damping=_suggest_signed_magnitude(trial, "yaw_rate_damping", base_config.yaw_rate_damping, 0.2, 5.0),
        yaw_beta_gain=_suggest_signed_magnitude(trial, "yaw_beta_gain", base_config.yaw_beta_gain, 0.2, 8.0),
        yaw_roll_trim=trial.suggest_float("yaw_roll_trim", -0.25, 0.25),
        throttle_base=trial.suggest_float("throttle_base", 0.10, 0.75),
        throttle_speed_gain=trial.suggest_float("throttle_speed_gain", 0.0002, 0.0080, log=True),
        throttle_climb_penalty=-trial.suggest_float("throttle_climb_penalty_mag", 0.0, 0.0040),
    )


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


def _run_episode(config, seed, max_steps, target_speed_kts, wind_sigma):
    env = _make_env(max_steps=max_steps, target_speed_kts=target_speed_kts, wind_sigma=wind_sigma)

    total_score = 0.0
    total_env_reward = 0.0
    total_abs_lateral_norm = 0.0
    total_abs_clearance_error_norm = 0.0
    total_speed_error_norm = 0.0
    steps = 0
    termination_reason = "running"
    prev_action = None

    try:
        _, _ = env.reset(seed=int(seed))
        state = env.unwrapped.get_full_state_dict()

        controller = SimpleCanyonController(env=env, config=config)
        controller.reset(state)

        for _ in range(int(max_steps)):
            action = controller.get_action(state)
            guidance = dict(controller.last_guidance)

            if not np.isfinite(action).all():
                termination_reason = "invalid_action"
                total_score -= 3000.0
                break

            _, env_reward, terminated, truncated, info = env.step(action)
            state = env.unwrapped.get_full_state_dict()

            steps += 1
            total_env_reward += float(env_reward)

            lateral_norm = float(abs(guidance.get("lateral_norm", 0.0)))
            clearance_error_ft = float(abs(guidance.get("clearance_error_ft", 0.0)))
            speed_fps = float(guidance.get("speed_fps", 0.0))

            clearance_scale_ft = max(float(config.target_clearance_ft), 200.0)
            clearance_error_norm = clearance_error_ft / clearance_scale_ft
            speed_error_norm = abs(speed_fps - float(config.target_speed_fps)) / max(float(config.target_speed_fps), 1.0)

            tracking_score = (
                2.0
                - 1.10 * np.clip(lateral_norm, 0.0, 2.5)
                - 0.85 * np.clip(clearance_error_norm, 0.0, 2.5)
                - 0.45 * np.clip(speed_error_norm, 0.0, 3.0)
            )

            smoothness_penalty = 0.0
            if prev_action is not None:
                smoothness_penalty = 0.12 * float(np.linalg.norm(action - prev_action))
            prev_action = action

            total_score += float(env_reward) + float(tracking_score) - smoothness_penalty

            total_abs_lateral_norm += lateral_norm
            total_abs_clearance_error_norm += clearance_error_norm
            total_speed_error_norm += speed_error_norm

            if terminated or truncated:
                termination_reason = info.get("termination_reason", "time_limit" if truncated else "terminated")
                break

        survival_frac = float(steps) / max(float(max_steps), 1.0)
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
        "env_reward": float(total_env_reward),
        "steps": int(steps),
        "termination_reason": str(termination_reason),
        "mean_abs_lateral_norm": float(total_abs_lateral_norm / denom),
        "mean_abs_clearance_error_norm": float(total_abs_clearance_error_norm / denom),
        "mean_speed_error_norm": float(total_speed_error_norm / denom),
    }
    return float(total_score), episode_summary


def _objective(trial, args, base_config):
    config = _sample_controller_config(trial, base_config)

    seed_scores = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _run_episode(
            config=config,
            seed=seed,
            max_steps=args.max_steps,
            target_speed_kts=args.target_speed_kts,
            wind_sigma=args.wind_sigma,
        )
        seed_scores.append(float(score))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lateral_norm", summary["mean_abs_lateral_norm"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_clearance_error_norm", summary["mean_abs_clearance_error_norm"])
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
        "base_config": asdict(base_config),
        "evaluation": {
            "seeds": [int(s) for s in args.seeds],
            "max_steps": int(args.max_steps),
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
