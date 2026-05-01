from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsbsim_gym.env import JSBSimEnv
from jsbsim_gym.mppi_run_config import (
    KTS_TO_FPS,
    apply_mppi_optuna_params,
    build_mppi_base_config_kwargs,
    build_mppi_controller,
)


FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
G_FTPS2 = 32.174
TURN_SPEED_SAFETY_FACTOR = 0.85
CONTROL_HZ = 30.0
CIRCLE_MPPI_TUNING_STUDY_NAME = "circle_mppi_contouring_tuning"
CIRCLE_MPPI_TUNING_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'circle_mppi_tuning.db').as_posix()}"
CIRCLE_MPPI_TUNING_JSON_PATH = REPO_ROOT / "output" / "circle_mppi" / "circle_mppi_optuna_best.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune contouring MPPI parameters for the simple-circle tracking scenario.",
    )
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi"],
        default="mppi",
        help="Controller variant to tune.",
    )
    parser.add_argument("--trials", type=int, default=1000, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=CIRCLE_MPPI_TUNING_STUDY_NAME,
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=CIRCLE_MPPI_TUNING_STORAGE,
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 42],
        help="Episode seeds used to evaluate each trial.",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode.")
    parser.add_argument("--radius-ft", type=float, default=2800.0, help="Circle radius in feet.")
    parser.add_argument(
        "--direction",
        choices=["cw", "ccw"],
        default="cw",
        help="Circle direction.",
    )
    parser.add_argument(
        "--target-speed-kts",
        type=float,
        default=450.0,
        help="Requested target speed for the circle reference.",
    )
    parser.add_argument(
        "--target-altitude-ft",
        type=float,
        default=None,
        help="Optional fixed target altitude in feet MSL.",
    )
    parser.add_argument(
        "--reference-points-per-lap",
        type=int,
        default=2048,
        help="Reference trajectory samples per lap.",
    )
    parser.add_argument(
        "--reference-laps",
        type=int,
        default=None,
        help="Optional explicit number of laps in the nominal reference.",
    )
    parser.add_argument(
        "--robustness-weight",
        type=float,
        default=0.15,
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
        help="Random seed for the Optuna sampler.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CIRCLE_MPPI_TUNING_JSON_PATH,
        help="Path to write the best-parameter JSON summary.",
    )
    return parser.parse_args()


def _max_turn_feasible_speed_kts(radius_ft, roll_max_rad, safety_factor=TURN_SPEED_SAFETY_FACTOR):
    radius_ft = max(float(radius_ft), 1.0)
    roll_max_rad = float(np.clip(roll_max_rad, np.deg2rad(5.0), np.deg2rad(85.0)))
    safety_factor = float(np.clip(safety_factor, 0.2, 1.0))
    max_speed_fps = np.sqrt(G_FTPS2 * radius_ft * np.tan(roll_max_rad))
    return float((safety_factor * max_speed_fps) / KTS_TO_FPS)


def _extract_state(env, start_n_m, start_e_m):
    sim = env.simulation
    state = env.state
    n_ft = float((state[0] - start_n_m) * M_TO_FT)
    e_ft = float((state[1] - start_e_m) * M_TO_FT)

    return {
        "n_ft": n_ft,
        "e_ft": e_ft,
        "h_ft": float(sim.get_property_value("position/h-sl-ft")),
        "u_fps": float(sim.get_property_value("velocities/u-fps")),
        "v_fps": float(sim.get_property_value("velocities/v-fps")),
        "w_fps": float(sim.get_property_value("velocities/w-fps")),
        "ny": float(sim.get_property_value("accelerations/Ny")),
        "nz": float(sim.get_property_value("accelerations/Nz")),
        "p_rad_s": float(state[6]),
        "q_rad_s": float(state[7]),
        "r_rad_s": float(state[8]),
        "phi_rad": float(state[9]),
        "theta_rad": float(state[10]),
        "psi_rad": float(state[11]),
        "beta_rad": float(state[5]),
    }


def _to_controller_state(state):
    return {
        "p_N": float(state["n_ft"]),
        "p_E": float(state["e_ft"]),
        "h": float(state["h_ft"]),
        "u": float(state["u_fps"]),
        "v": float(state["v_fps"]),
        "w": float(state["w_fps"]),
        "p": float(state["p_rad_s"]),
        "q": float(state["q_rad_s"]),
        "r": float(state["r_rad_s"]),
        "phi": float(state["phi_rad"]),
        "theta": float(state["theta_rad"]),
        "psi": float(state["psi_rad"]),
        "beta": float(state["beta_rad"]),
        "ny": float(state["ny"]),
        "nz": float(state["nz"]),
    }


def _build_circle_nominal_reference(
    *,
    start_n_ft,
    start_e_ft,
    center_n_ft,
    center_e_ft,
    radius_ft,
    direction,
    target_altitude_ft,
    target_speed_fps,
    max_steps,
    control_hz,
    points_per_lap,
    laps_override,
):
    radius_ft = float(max(radius_ft, 1.0))
    points_per_lap = int(max(points_per_lap, 64))

    circumference_ft = float(2.0 * np.pi * radius_ft)
    if laps_override is None:
        distance_budget_ft = float(max_steps) * (float(target_speed_fps) / float(max(control_hz, 1.0)))
        laps = int(max(1, np.ceil(distance_budget_ft / max(circumference_ft, 1.0)) + 1))
    else:
        laps = int(max(laps_override, 1))

    total_points = int(max(2, points_per_lap * laps))
    theta0 = float(np.arctan2(float(start_e_ft) - float(center_e_ft), float(start_n_ft) - float(center_n_ft)))
    angle_progress = np.linspace(0.0, 2.0 * np.pi * float(laps), total_points, endpoint=False, dtype=np.float64)
    if direction == "cw":
        theta = theta0 - angle_progress
    else:
        theta = theta0 + angle_progress

    north_ft = float(center_n_ft) + radius_ft * np.cos(theta)
    east_ft = float(center_e_ft) + radius_ft * np.sin(theta)
    altitude_ft = np.full_like(north_ft, float(target_altitude_ft), dtype=np.float64)

    d_north = np.gradient(north_ft)
    d_east = np.gradient(east_ft)
    heading_rad = np.unwrap(np.arctan2(d_east, d_north))

    reference_states_ft_rad = np.column_stack(
        [
            north_ft,
            east_ft,
            altitude_ft,
            np.zeros_like(north_ft),
            np.zeros_like(north_ft),
            heading_rad,
        ]
    ).astype(np.float32)

    speed_fps = np.full((total_points,), float(target_speed_fps), dtype=np.float32)

    return {
        "reference_states_ft_rad": reference_states_ft_rad,
        "speed_fps": speed_fps,
        "north_ft": north_ft.astype(np.float32),
        "east_ft": east_ft.astype(np.float32),
        "circle_laps": int(laps),
    }


def _build_flat_terrain_grid(reference_north_ft, reference_east_ft, margin_ft=6000.0, samples_per_axis=64):
    margin_ft = float(max(margin_ft, 1000.0))
    samples_per_axis = int(max(samples_per_axis, 8))

    min_n = float(np.min(reference_north_ft)) - margin_ft
    max_n = float(np.max(reference_north_ft)) + margin_ft
    min_e = float(np.min(reference_east_ft)) - margin_ft
    max_e = float(np.max(reference_east_ft)) + margin_ft

    north_axis = np.linspace(min_n, max_n, samples_per_axis, dtype=np.float32)
    east_axis = np.linspace(min_e, max_e, samples_per_axis, dtype=np.float32)
    terrain = np.zeros((samples_per_axis, samples_per_axis), dtype=np.float32)
    return north_axis, east_axis, terrain


def _sample_controller_overrides(trial):
    contour_weight = trial.suggest_float("contour_weight", 0.05, 20.0, log=True)
    lag_ratio = trial.suggest_float("lag_ratio", 1.0e-3, 0.25, log=True)
    return {
        "lambda_": trial.suggest_float("lambda_", 0.10, 10.0, log=True),
        "gamma_": trial.suggest_float("gamma_", 0.002, 0.50, log=True),
        "action_noise_std": (
            trial.suggest_float("action_noise_std_aileron", 0.02, 1.0, log=True),
            trial.suggest_float("action_noise_std_elevator", 0.02, 1.0, log=True),
            trial.suggest_float("action_noise_std_rudder", 0.01, 1.0, log=True),
            trial.suggest_float("action_noise_std_throttle", 0.005, 1.0, log=True),
        ),
        "contour_weight": contour_weight,
        "lag_weight": contour_weight * lag_ratio,
        "progress_reward_weight": trial.suggest_float("progress_reward_weight", 1.0, 200.0, log=True),
        "virtual_speed_weight": trial.suggest_float("virtual_speed_weight", 1.0e-4, 0.5, log=True),
        "control_rate_weights": (
            trial.suggest_float("control_rate_weight_aileron", 1.0, 500.0, log=True),
            trial.suggest_float("control_rate_weight_elevator", 1.0, 700.0, log=True),
            trial.suggest_float("control_rate_weight_rudder", 0.01, 100.0, log=True),
            trial.suggest_float("control_rate_weight_throttle", 0.01, 50.0, log=True),
        ),
    }


def _episode_cost_summary(args, config_kwargs, *, seed: int):
    env = JSBSimEnv(render_mode=None)

    contour_errors_ft = []
    lag_errors_ft = []
    position_errors_ft = []
    altitude_errors_ft = []
    virtual_speeds_fps = []
    progress_samples_ft = []
    action_rates = []
    alpha_excess_deg = []
    nz_excess_g = []

    steps = 0
    termination_reason = "running"

    try:
        target_speed_fps = float(args.effective_target_speed_kts) * KTS_TO_FPS
        env.simulation.set_property_value("ic/u-fps", target_speed_fps)
        if args.target_altitude_ft is not None:
            env.simulation.set_property_value("ic/h-sl-ft", float(args.target_altitude_ft))
        env.reset(seed=int(seed))

        # Disable accidental task completion from JSBSimEnv's random goal logic.
        env.goal[:] = 1.0e12

        start_n_m = float(env.state[0])
        start_e_m = float(env.state[1])
        init_state = _extract_state(env, start_n_m, start_e_m)

        target_altitude_ft = (
            float(args.target_altitude_ft)
            if args.target_altitude_ft is not None
            else float(init_state["h_ft"])
        )

        psi0 = float(init_state["psi_rad"])
        right_n = -np.sin(psi0)
        right_e = np.cos(psi0)
        if args.direction == "cw":
            center_n_ft = init_state["n_ft"] - float(args.radius_ft) * right_n
            center_e_ft = init_state["e_ft"] - float(args.radius_ft) * right_e
        else:
            center_n_ft = init_state["n_ft"] + float(args.radius_ft) * right_n
            center_e_ft = init_state["e_ft"] + float(args.radius_ft) * right_e

        nominal_reference = _build_circle_nominal_reference(
            start_n_ft=float(init_state["n_ft"]),
            start_e_ft=float(init_state["e_ft"]),
            center_n_ft=float(center_n_ft),
            center_e_ft=float(center_e_ft),
            radius_ft=float(args.radius_ft),
            direction=str(args.direction),
            target_altitude_ft=float(target_altitude_ft),
            target_speed_fps=float(target_speed_fps),
            max_steps=int(args.max_steps),
            control_hz=CONTROL_HZ,
            points_per_lap=int(args.reference_points_per_lap),
            laps_override=args.reference_laps,
        )
        terrain_north_ft, terrain_east_ft, terrain_elevation_ft = _build_flat_terrain_grid(
            reference_north_ft=np.asarray(nominal_reference["north_ft"], dtype=np.float32),
            reference_east_ft=np.asarray(nominal_reference["east_ft"], dtype=np.float32),
            margin_ft=max(2.0 * float(args.radius_ft), 5000.0),
            samples_per_axis=64,
        )

        controller, _ = build_mppi_controller(
            args.controller,
            config_base_kwargs=config_kwargs,
            reference_trajectory=nominal_reference,
            terrain_north_samples_ft=terrain_north_ft,
            terrain_east_samples_ft=terrain_east_ft,
            terrain_elevation_ft=terrain_elevation_ft,
        )
        controller.reset(seed=int(seed))

        prev_action = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float32)

        for _ in range(int(args.max_steps)):
            state = _extract_state(env, start_n_m, start_e_m)
            controller_state = _to_controller_state(state)
            action = np.asarray(controller.get_action(controller_state), dtype=np.float32)
            if not np.all(np.isfinite(action)):
                termination_reason = "invalid_action"
                break

            _, _, done, truncated, _ = env.step(action)
            steps += 1

            post_state = _extract_state(env, start_n_m, start_e_m)
            post_controller_state = _to_controller_state(post_state)
            tracking = dict(controller.get_tracking_metrics(post_controller_state))

            contour_errors_ft.append(float(tracking.get("contour_error_ft", np.nan)))
            lag_errors_ft.append(float(tracking.get("lag_error_ft", np.nan)))
            position_errors_ft.append(float(tracking.get("position_error_ft", np.nan)))
            altitude_errors_ft.append(abs(float(tracking.get("altitude_error_ft", np.nan))))
            virtual_speeds_fps.append(float(tracking.get("virtual_speed_fps", np.nan)))
            progress_samples_ft.append(float(tracking.get("progress_s_ft", np.nan)))

            action_rate = np.abs(action - prev_action).astype(np.float64, copy=False)
            action_rates.append(action_rate)
            prev_action = action.copy()

            alpha_rad = float(np.arctan2(float(post_controller_state["w"]), max(float(post_controller_state["u"]), 1.0)))
            alpha_excess_rad = max(alpha_rad - float(controller.config.alpha_limit_rad), 0.0)
            alpha_excess_deg.append(float(np.degrees(alpha_excess_rad)))
            nz_excess_g.append(max(abs(float(post_controller_state.get("nz", 1.0))) - float(controller.config.nz_limit_g), 0.0))

            if done:
                termination_reason = "terminated"
                break
            if truncated:
                termination_reason = "time_limit"
                break

        survival_frac = float(steps) / max(float(args.max_steps), 1.0)
    finally:
        env.close()

    contour_arr = np.asarray(contour_errors_ft, dtype=np.float64)
    lag_arr = np.asarray(lag_errors_ft, dtype=np.float64)
    position_arr = np.asarray(position_errors_ft, dtype=np.float64)
    altitude_arr = np.asarray(altitude_errors_ft, dtype=np.float64)
    virtual_speed_arr = np.asarray(virtual_speeds_fps, dtype=np.float64)
    progress_arr = np.asarray(progress_samples_ft, dtype=np.float64)
    action_rate_arr = np.asarray(action_rates, dtype=np.float64).reshape(-1, 4) if action_rates else np.zeros((0, 4))
    alpha_excess_arr = np.asarray(alpha_excess_deg, dtype=np.float64)
    nz_excess_arr = np.asarray(nz_excess_g, dtype=np.float64)

    mean_contour_ft = float(np.nanmean(contour_arr)) if contour_arr.size else 1.0e4
    rms_contour_ft = float(np.sqrt(np.nanmean(np.square(contour_arr)))) if contour_arr.size else 1.0e4
    p95_contour_ft = float(np.nanpercentile(contour_arr, 95.0)) if contour_arr.size else 1.0e4
    mean_abs_lag_ft = float(np.nanmean(np.abs(lag_arr))) if lag_arr.size else 1.0e4
    mean_pos_ft = float(np.nanmean(position_arr)) if position_arr.size else 1.0e4
    mean_altitude_ft = float(np.nanmean(altitude_arr)) if altitude_arr.size else 1.0e4

    mean_action_rate = (
        np.nanmean(np.abs(action_rate_arr), axis=0)
        if action_rate_arr.size
        else np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    mean_alpha_excess_deg = float(np.nanmean(alpha_excess_arr)) if alpha_excess_arr.size else 30.0
    mean_nz_excess_g = float(np.nanmean(nz_excess_arr)) if nz_excess_arr.size else 5.0

    progress_gain_ft = 0.0
    if progress_arr.size >= 2:
        progress_gain_ft = float(progress_arr[-1] - progress_arr[0])

    target_speed_fps = float(args.effective_target_speed_kts) * KTS_TO_FPS
    expected_progress_ft = float(target_speed_fps) * (float(steps) / CONTROL_HZ)
    progress_ratio = float(progress_gain_ft / max(expected_progress_ft, 1.0))
    mean_virtual_speed_ratio = float(np.nanmean(virtual_speed_arr) / max(target_speed_fps, 1.0)) if virtual_speed_arr.size else 0.0

    track_cost = (
        1.00 * mean_contour_ft
        + 0.75 * rms_contour_ft
        + 0.50 * p95_contour_ft
        + 0.20 * mean_abs_lag_ft
        + 0.10 * mean_pos_ft
        + 0.15 * mean_altitude_ft
    )
    rate_cost = float(40.0 * np.sum(mean_action_rate))
    limit_cost = float(120.0 * mean_alpha_excess_deg + 250.0 * mean_nz_excess_g)
    progress_cost = float(
        120.0 * max(1.0 - mean_virtual_speed_ratio, 0.0)
        + 180.0 * max(1.0 - progress_ratio, 0.0)
    )

    if termination_reason in {"running", "time_limit"}:
        termination_penalty = 0.0
    elif termination_reason == "invalid_action":
        termination_penalty = 4.0e4
    else:
        termination_penalty = 2.0e4 + 3.0e4 * (1.0 - survival_frac)

    total_cost = float(track_cost + rate_cost + limit_cost + progress_cost + termination_penalty)
    score = -total_cost

    return float(score), {
        "score": float(score),
        "steps": int(steps),
        "survival_frac": float(survival_frac),
        "termination_reason": str(termination_reason),
        "mean_contour_error_ft": float(mean_contour_ft),
        "rms_contour_error_ft": float(rms_contour_ft),
        "p95_contour_error_ft": float(p95_contour_ft),
        "mean_abs_lag_error_ft": float(mean_abs_lag_ft),
        "mean_position_error_ft": float(mean_pos_ft),
        "mean_altitude_error_ft": float(mean_altitude_ft),
        "mean_aileron_rate": float(mean_action_rate[0]),
        "mean_elevator_rate": float(mean_action_rate[1]),
        "mean_rudder_rate": float(mean_action_rate[2]),
        "mean_throttle_rate": float(mean_action_rate[3]),
        "mean_alpha_excess_deg": float(mean_alpha_excess_deg),
        "mean_nz_excess_g": float(mean_nz_excess_g),
        "progress_ratio": float(progress_ratio),
        "mean_virtual_speed_ratio": float(mean_virtual_speed_ratio),
        "track_cost": float(track_cost),
        "rate_cost": float(rate_cost),
        "limit_cost": float(limit_cost),
        "progress_cost": float(progress_cost),
        "termination_penalty": float(termination_penalty),
    }


def _objective(trial, args, base_config_kwargs):
    effective_params = _sample_controller_overrides(trial)
    config_kwargs, applied_keys = apply_mppi_optuna_params(base_config_kwargs, effective_params)
    trial.set_user_attr("effective_params", {key: config_kwargs[key] for key in applied_keys})

    seed_scores = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _episode_cost_summary(args, config_kwargs, seed=int(seed))
        seed_scores.append(float(score))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_survival_frac", summary["survival_frac"])
        trial.set_user_attr(f"seed_{seed}_mean_contour_error_ft", summary["mean_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_contour_error_ft", summary["rms_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_contour_error_ft", summary["p95_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lag_error_ft", summary["mean_abs_lag_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_position_error_ft", summary["mean_position_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_altitude_error_ft", summary["mean_altitude_error_ft"])
        trial.set_user_attr(f"seed_{seed}_progress_ratio", summary["progress_ratio"])
        trial.set_user_attr(f"seed_{seed}_mean_virtual_speed_ratio", summary["mean_virtual_speed_ratio"])
        trial.set_user_attr(f"seed_{seed}_mean_aileron_rate", summary["mean_aileron_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_elevator_rate", summary["mean_elevator_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_rudder_rate", summary["mean_rudder_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_throttle_rate", summary["mean_throttle_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_alpha_excess_deg", summary["mean_alpha_excess_deg"])
        trial.set_user_attr(f"seed_{seed}_mean_nz_excess_g", summary["mean_nz_excess_g"])

        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(seed_scores))
    std_score = float(np.std(seed_scores))
    robust_score = float(mean_score - float(args.robustness_weight) * std_score)

    trial.set_user_attr("mean_seed_score", mean_score)
    trial.set_user_attr("std_seed_score", std_score)
    return robust_score


def _save_best_summary(args, study, base_config_kwargs):
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_trial = study.best_trial

    payload = {
        "study_name": str(study.study_name),
        "storage": str(args.storage),
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.user_attrs.get("effective_params", {})),
        "base_config": dict(base_config_kwargs),
        "evaluation": {
            "controller": str(args.controller),
            "seeds": [int(s) for s in args.seeds],
            "max_steps": int(args.max_steps),
            "radius_ft": float(args.radius_ft),
            "direction": str(args.direction),
            "requested_target_speed_kts": float(args.target_speed_kts),
            "effective_target_speed_kts": float(args.effective_target_speed_kts),
            "target_altitude_ft": None if args.target_altitude_ft is None else float(args.target_altitude_ft),
            "reference_points_per_lap": int(args.reference_points_per_lap),
            "reference_laps": None if args.reference_laps is None else int(args.reference_laps),
            "robustness_weight": float(args.robustness_weight),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main():
    args = parse_args()

    max_turn_speed_kts = _max_turn_feasible_speed_kts(
        radius_ft=float(args.radius_ft),
        roll_max_rad=np.deg2rad(70.0),
    )
    args.effective_target_speed_kts = float(min(float(args.target_speed_kts), max_turn_speed_kts))

    base_config_kwargs = build_mppi_base_config_kwargs()
    # Circle tuning is terrain-free; keep the contouring objective and dynamics identical otherwise.
    base_config_kwargs["terrain_collision_penalty"] = 0.0
    base_config_kwargs["terrain_repulsion_scale"] = 0.0

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
        f"controller={args.controller}, "
        f"seeds={args.seeds}, "
        f"max_steps={args.max_steps}, "
        f"radius_ft={args.radius_ft:.1f}, "
        f"direction={args.direction}, "
        f"requested_speed_kts={args.target_speed_kts:.1f}, "
        f"effective_speed_kts={args.effective_target_speed_kts:.1f}, "
        f"target_altitude_ft={'auto' if args.target_altitude_ft is None else f'{args.target_altitude_ft:.1f}'}"
    )

    if args.effective_target_speed_kts + 1e-9 < float(args.target_speed_kts):
        print(
            "Speed capped for turn feasibility: "
            f"requested={float(args.target_speed_kts):.1f} kts, "
            f"effective={args.effective_target_speed_kts:.1f} kts"
        )

    study.optimize(
        lambda trial: _objective(trial, args, base_config_kwargs),
        n_trials=int(args.trials),
        n_jobs=int(args.jobs),
        timeout=args.timeout,
    )

    best_trial = study.best_trial
    summary_path = _save_best_summary(args, study, base_config_kwargs)

    print("\n==============================")
    print("Optimization finished")
    print(f"Best trial: #{best_trial.number}")
    print(f"Best robust score: {float(best_trial.value):.3f}")
    print("Best parameters:")
    for key, value in dict(best_trial.user_attrs.get("effective_params", {})).items():
        print(f"  {key}: {value}")
    print(f"Saved summary JSON: {summary_path}")
    print("==============================")


if __name__ == "__main__":
    main()
