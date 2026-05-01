from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0.
from jsbsim_gym.canyon import DEMCanyon
from jsbsim_gym.mppi_run_config import (
    MPPI_TUNING_JSON_PATH,
    MPPI_TUNING_STORAGE,
    MPPI_TUNING_STUDY_NAME,
    apply_mppi_optuna_params,
    build_mppi_base_config_kwargs,
    build_mppi_controller,
)
from jsbsim_gym.nominal_trajectory import (
    build_nominal_reference_from_dyn,
    load_nominal_initial_conditions_from_dyn,
)

DEM_PATH = REPO_ROOT / "data" / "dem" / "black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
M_TO_FT = 3.28084
SAFE_CLEARANCE_FT = 40.0 * M_TO_FT
ALPHA_LIMIT_DEG = 25.0
NZ_LIMIT_G = 9.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune the contouring MPPI controller against an offline nominal trajectory.",
    )
    parser.add_argument(
        "--nominal-dyn-path",
        type=Path,
        required=True,
        help="Offline dyn.asb trajectory used as the MPPI nominal reference.",
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=MPPI_TUNING_STUDY_NAME,
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=MPPI_TUNING_STORAGE,
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
        "--wind-sigma",
        type=float,
        default=0.0,
        help="Wind sigma used in the evaluation environment.",
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
        default=MPPI_TUNING_JSON_PATH,
        help="Path to write the best-parameter JSON summary.",
    )
    return parser.parse_args()


def _to_mppi_state(env, state_dict, altitude_ref_ft):
    p_n_ft = float(state_dict["p_N"])
    p_e_ft = float(state_dict["p_E"])

    canyon = getattr(env.unwrapped, "canyon", None)
    if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
        lat_deg = float(env.unwrapped.simulation.get_property_value("position/lat-gc-deg"))
        lon_deg = float(env.unwrapped.simulation.get_property_value("position/long-gc-deg"))
        p_n_ft, p_e_ft = canyon.get_local_from_latlon(lat_deg, lon_deg)

    return {
        "p_N": float(p_n_ft),
        "p_E": float(p_e_ft),
        "h": float(state_dict["h"] - altitude_ref_ft),
        "u": float(state_dict["u"]),
        "v": float(state_dict["v"]),
        "w": float(state_dict["w"]),
        "p": float(state_dict["p"]),
        "q": float(state_dict["q"]),
        "r": float(state_dict["r"]),
        "phi": float(state_dict["phi"]),
        "theta": float(state_dict["theta"]),
        "psi": float(state_dict["psi"]),
        "beta": float(state_dict.get("beta", 0.0)),
        "ny": float(state_dict.get("ny", 0.0)),
        "nz": float(state_dict.get("nz", 1.0)),
    }


def _load_nominal_bundle(nominal_dyn_path: Path):
    nominal_canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=DEM_BBOX[0],
        north=DEM_BBOX[1],
        west=DEM_BBOX[2],
        east=DEM_BBOX[3],
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=DEM_START_PIXEL,
    )
    initial_conditions = load_nominal_initial_conditions_from_dyn(
        nominal_dyn_path,
        canyon=nominal_canyon,
    )
    runtime_canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=DEM_BBOX[0],
        north=DEM_BBOX[1],
        west=DEM_BBOX[2],
        east=DEM_BBOX[3],
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=tuple(initial_conditions["start_pixel"]),
    )
    start_info = runtime_canyon.get_pixel_info(*tuple(initial_conditions["start_pixel"]))
    altitude_ref_ft = float(start_info["elevation_msl_ft"])
    reference_trajectory = build_nominal_reference_from_dyn(
        nominal_dyn_path,
        canyon=runtime_canyon,
        altitude_ref_ft=altitude_ref_ft,
        resample_spacing_ft=12.0,
    )
    return {
        "initial_conditions": initial_conditions,
        "reference_trajectory": reference_trajectory,
        "terrain_north_samples_ft": np.asarray(runtime_canyon.north_samples_ft, dtype=np.float32),
        "terrain_east_samples_ft": np.asarray(runtime_canyon.east_samples_ft, dtype=np.float32),
        "terrain_elevation_ft": np.asarray(runtime_canyon.ordered_dem_msl_m, dtype=np.float32) * M_TO_FT
        - altitude_ref_ft,
        "altitude_ref_ft": altitude_ref_ft,
    }


def _make_env(nominal_bundle, *, max_steps: int, wind_sigma: float):
    ic = nominal_bundle["initial_conditions"]
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
        dem_start_pixel=tuple(ic["start_pixel"]),
        dem_start_heading_mode="follow_canyon",
        dem_start_heading_deg=float(ic["heading_deg"]),
        dem_render_mesh=False,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=500.0,
        entry_altitude_ft=float(ic["entry_altitude_ft"]),
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=int(max_steps),
        terrain_collision_buffer_ft=10.0,
        entry_speed_kts=float(ic["speed_kts"]),
        entry_roll_deg=float(ic["roll_deg"]),
        entry_pitch_deg=float(ic["pitch_deg"]),
        entry_alpha_deg=float(ic["alpha_deg"]),
        entry_beta_deg=float(ic["beta_deg"]),
        wind_sigma=float(wind_sigma),
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )


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
        "terrain_collision_penalty": trial.suggest_float("terrain_collision_penalty", 1.0e5, 1.0e7, log=True),
        "terrain_repulsion_scale": trial.suggest_float("terrain_repulsion_scale", 1.0e3, 1.0e6, log=True),
        "control_rate_weights": (
            trial.suggest_float("control_rate_weight_aileron", 1.0, 500.0, log=True),
            trial.suggest_float("control_rate_weight_elevator", 1.0, 700.0, log=True),
            trial.suggest_float("control_rate_weight_rudder", 0.01, 100.0, log=True),
            trial.suggest_float("control_rate_weight_throttle", 0.01, 50.0, log=True),
        ),
    }


def _episode_cost_summary(
    config_base_kwargs,
    nominal_bundle,
    *,
    seed: int,
    max_steps: int,
    wind_sigma: float,
):
    env = _make_env(nominal_bundle, max_steps=max_steps, wind_sigma=wind_sigma)
    prev_action = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
    termination_reason = "running"

    contour_errors_ft = []
    lag_errors_ft = []
    position_errors_ft = []
    altitude_errors_ft = []
    clearance_shortfalls = []
    action_rates = []
    alpha_excess_deg = []
    nz_excess_g = []
    contouring_costs = []
    terrain_costs = []
    rate_costs = []
    limit_costs = []
    total_stage_costs = []
    min_clearance_ft = np.inf
    steps = 0

    try:
        np.random.seed(int(seed))
        _, _ = env.reset(seed=int(seed))
        altitude_ref_ft = float(nominal_bundle["altitude_ref_ft"])
        state = env.unwrapped.get_full_state_dict()
        controller_state = _to_mppi_state(env, state, altitude_ref_ft)

        controller, _ = build_mppi_controller(
            "mppi",
            config_base_kwargs=config_base_kwargs,
            reference_trajectory=nominal_bundle["reference_trajectory"],
            terrain_north_samples_ft=nominal_bundle["terrain_north_samples_ft"],
            terrain_east_samples_ft=nominal_bundle["terrain_east_samples_ft"],
            terrain_elevation_ft=nominal_bundle["terrain_elevation_ft"],
        )
        controller.reset(seed=int(seed))

        for step in range(int(max_steps)):
            action = np.asarray(controller.get_action(controller_state), dtype=np.float32)
            if not np.all(np.isfinite(action)):
                termination_reason = "invalid_action"
                break

            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            state = env.unwrapped.get_full_state_dict()
            post_state = _to_mppi_state(env, state, altitude_ref_ft)
            tracking = dict(controller.get_tracking_metrics(post_state))

            clearance_ft = float(info.get("terrain_clearance_ft", np.nan))
            min_clearance_ft = min(min_clearance_ft, clearance_ft)
            clearance_shortfall = 0.0
            if np.isfinite(clearance_ft):
                clearance_shortfall = max(SAFE_CLEARANCE_FT - clearance_ft, 0.0) / max(SAFE_CLEARANCE_FT, 1.0)

            action_rate = action - prev_action

            terrain_cost_est = float(controller.config.terrain_collision_penalty)
            if np.isfinite(clearance_ft):
                if clearance_ft <= 0.0:
                    terrain_cost_est = float(controller.config.terrain_collision_penalty)
                else:
                    terrain_cost_est = float(
                        min(
                            float(controller.config.terrain_collision_penalty),
                            float(controller.config.terrain_repulsion_scale)
                            * np.exp(
                                -float(controller.config.terrain_decay_rate_ft_inv)
                                * (clearance_ft - float(controller.config.terrain_safe_clearance_ft))
                            ),
                        )
                    )

            rate_cost_est = float(
                np.sum(
                    np.asarray(controller.config.control_rate_weights, dtype=np.float64)
                    * np.square(action_rate)
                )
            )
            prev_action = action.copy()
            alpha_deg = float(np.degrees(np.arctan2(float(post_state["w"]), max(float(post_state["u"]), 1.0))))
            alpha_excess = max(alpha_deg - ALPHA_LIMIT_DEG, 0.0)
            nz_excess = max(abs(float(post_state.get("nz", 1.0))) - NZ_LIMIT_G, 0.0)
            alpha_excess_rad = max(
                np.deg2rad(alpha_deg) - float(controller.config.alpha_limit_rad),
                0.0,
            )
            limit_cost_est = float(
                float(controller.config.nz_penalty_weight) * (nz_excess ** 2)
                + float(controller.config.alpha_penalty_weight) * (alpha_excess_rad ** 2)
            )
            contouring_cost_est = float(tracking["contouring_cost_est"])
            total_stage_cost_est = float(contouring_cost_est + terrain_cost_est + rate_cost_est + limit_cost_est)

            contour_errors_ft.append(float(tracking["contour_error_ft"]))
            lag_errors_ft.append(float(tracking["lag_error_ft"]))
            position_errors_ft.append(float(tracking["position_error_ft"]))
            altitude_errors_ft.append(abs(float(tracking["altitude_error_ft"])))
            clearance_shortfalls.append(float(clearance_shortfall))
            action_rates.append(np.abs(action_rate).astype(np.float64, copy=False))
            alpha_excess_deg.append(float(alpha_excess))
            nz_excess_g.append(float(nz_excess))
            contouring_costs.append(float(contouring_cost_est))
            terrain_costs.append(float(terrain_cost_est))
            rate_costs.append(float(rate_cost_est))
            limit_costs.append(float(limit_cost_est))
            total_stage_costs.append(float(total_stage_cost_est))

            controller_state = post_state

            if terminated or truncated:
                termination_reason = info.get("termination_reason", "time_limit" if truncated else "terminated")
                break

        survival_frac = float(steps) / max(float(max_steps), 1.0)
    finally:
        env.close()

    contour_arr = np.asarray(contour_errors_ft, dtype=np.float64)
    lag_arr = np.asarray(lag_errors_ft, dtype=np.float64)
    position_arr = np.asarray(position_errors_ft, dtype=np.float64)
    altitude_arr = np.asarray(altitude_errors_ft, dtype=np.float64)
    clearance_shortfall_arr = np.asarray(clearance_shortfalls, dtype=np.float64)
    action_rate_arr = np.asarray(action_rates, dtype=np.float64).reshape(-1, 4) if action_rates else np.zeros((0, 4))
    alpha_excess_arr = np.asarray(alpha_excess_deg, dtype=np.float64)
    nz_excess_arr = np.asarray(nz_excess_g, dtype=np.float64)
    contouring_cost_arr = np.asarray(contouring_costs, dtype=np.float64)
    terrain_cost_arr = np.asarray(terrain_costs, dtype=np.float64)
    rate_cost_arr = np.asarray(rate_costs, dtype=np.float64)
    limit_cost_arr = np.asarray(limit_costs, dtype=np.float64)
    total_stage_cost_arr = np.asarray(total_stage_costs, dtype=np.float64)

    mean_contour_ft = float(np.mean(contour_arr)) if contour_arr.size else 1.0e4
    rms_contour_ft = float(np.sqrt(np.mean(np.square(contour_arr)))) if contour_arr.size else 1.0e4
    p95_contour_ft = float(np.percentile(contour_arr, 95.0)) if contour_arr.size else 1.0e4
    mean_abs_lag_ft = float(np.mean(np.abs(lag_arr))) if lag_arr.size else 1.0e4
    p95_abs_lag_ft = float(np.percentile(np.abs(lag_arr), 95.0)) if lag_arr.size else 1.0e4
    mean_pos_ft = float(np.mean(position_arr)) if position_arr.size else 1.0e4
    mean_altitude_ft = float(np.mean(altitude_arr)) if altitude_arr.size else 1.0e4
    mean_clearance_shortfall = float(np.mean(clearance_shortfall_arr)) if clearance_shortfall_arr.size else 1.0
    max_clearance_shortfall = float(np.max(clearance_shortfall_arr)) if clearance_shortfall_arr.size else 1.0
    mean_action_rate = (
        np.mean(np.abs(action_rate_arr), axis=0) if action_rate_arr.size else np.asarray([1.0, 1.0, 1.0, 1.0])
    )
    mean_alpha_excess_deg = float(np.mean(alpha_excess_arr)) if alpha_excess_arr.size else 25.0
    mean_nz_excess_g = float(np.mean(nz_excess_arr)) if nz_excess_arr.size else 5.0
    mean_contouring_cost = float(np.mean(contouring_cost_arr)) if contouring_cost_arr.size else 1.0e6
    mean_terrain_cost = float(np.mean(terrain_cost_arr)) if terrain_cost_arr.size else 1.0e6
    mean_rate_cost = float(np.mean(rate_cost_arr)) if rate_cost_arr.size else 1.0e4
    mean_limit_cost = float(np.mean(limit_cost_arr)) if limit_cost_arr.size else 1.0e6
    mean_total_stage_cost = float(np.mean(total_stage_cost_arr)) if total_stage_cost_arr.size else 1.0e6
    p95_total_stage_cost = (
        float(np.percentile(total_stage_cost_arr, 95.0)) if total_stage_cost_arr.size else 1.0e6
    )
    if not np.isfinite(min_clearance_ft):
        min_clearance_ft = np.nan

    contour_track_cost = (
        1.00 * mean_contour_ft
        + 0.75 * rms_contour_ft
        + 0.50 * p95_contour_ft
        + 0.20 * mean_abs_lag_ft
        + 0.10 * p95_abs_lag_ft
        + 0.20 * mean_altitude_ft
    )
    stage_cost_term = mean_total_stage_cost + 0.25 * p95_total_stage_cost
    terrain_margin_cost = 100.0 * mean_clearance_shortfall + 150.0 * max_clearance_shortfall

    if termination_reason in {"running", "time_limit"}:
        termination_penalty = 0.0
    elif termination_reason in {"terrain_collision", "ground_collision"}:
        termination_penalty = 2.0e4 + 3.0e4 * (1.0 - survival_frac)
    elif termination_reason in {"hit_canyon_wall", "altitude_out_of_bounds"}:
        termination_penalty = 1.0e4 + 1.5e4 * (1.0 - survival_frac)
    elif termination_reason == "invalid_action":
        termination_penalty = 4.0e4
    else:
        termination_penalty = 1.2e4

    total_cost = float(contour_track_cost + stage_cost_term + terrain_margin_cost + termination_penalty)
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
        "p95_abs_lag_error_ft": float(p95_abs_lag_ft),
        "mean_position_error_ft": float(mean_pos_ft),
        "mean_altitude_error_ft": float(mean_altitude_ft),
        "min_terrain_clearance_ft": float(min_clearance_ft),
        "mean_clearance_shortfall": float(mean_clearance_shortfall),
        "max_clearance_shortfall": float(max_clearance_shortfall),
        "mean_aileron_rate": float(mean_action_rate[0]),
        "mean_elevator_rate": float(mean_action_rate[1]),
        "mean_rudder_rate": float(mean_action_rate[2]),
        "mean_throttle_rate": float(mean_action_rate[3]),
        "mean_alpha_excess_deg": float(mean_alpha_excess_deg),
        "mean_nz_excess_g": float(mean_nz_excess_g),
        "mean_contouring_cost": float(mean_contouring_cost),
        "mean_terrain_cost": float(mean_terrain_cost),
        "mean_rate_cost": float(mean_rate_cost),
        "mean_limit_cost": float(mean_limit_cost),
        "mean_total_stage_cost": float(mean_total_stage_cost),
        "p95_total_stage_cost": float(p95_total_stage_cost),
        "contour_track_cost": float(contour_track_cost),
        "stage_cost_term": float(stage_cost_term),
        "terrain_margin_cost": float(terrain_margin_cost),
        "termination_penalty": float(termination_penalty),
    }


def _objective(trial, args, nominal_bundle, base_config_kwargs):
    effective_params = _sample_controller_overrides(trial)
    config_kwargs, applied_keys = apply_mppi_optuna_params(base_config_kwargs, effective_params)
    trial.set_user_attr("effective_params", {key: config_kwargs[key] for key in applied_keys})

    seed_scores = []
    for idx, seed in enumerate(args.seeds):
        score, summary = _episode_cost_summary(
            config_kwargs,
            nominal_bundle,
            seed=int(seed),
            max_steps=int(args.max_steps),
            wind_sigma=float(args.wind_sigma),
        )
        seed_scores.append(float(score))

        trial.set_user_attr(f"seed_{seed}_termination", summary["termination_reason"])
        trial.set_user_attr(f"seed_{seed}_steps", summary["steps"])
        trial.set_user_attr(f"seed_{seed}_survival_frac", summary["survival_frac"])
        trial.set_user_attr(f"seed_{seed}_mean_contour_error_ft", summary["mean_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_rms_contour_error_ft", summary["rms_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_contour_error_ft", summary["p95_contour_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_abs_lag_error_ft", summary["mean_abs_lag_error_ft"])
        trial.set_user_attr(f"seed_{seed}_p95_abs_lag_error_ft", summary["p95_abs_lag_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_position_error_ft", summary["mean_position_error_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_altitude_error_ft", summary["mean_altitude_error_ft"])
        trial.set_user_attr(f"seed_{seed}_min_terrain_clearance_ft", summary["min_terrain_clearance_ft"])
        trial.set_user_attr(f"seed_{seed}_mean_aileron_rate", summary["mean_aileron_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_elevator_rate", summary["mean_elevator_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_rudder_rate", summary["mean_rudder_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_throttle_rate", summary["mean_throttle_rate"])
        trial.set_user_attr(f"seed_{seed}_mean_alpha_excess_deg", summary["mean_alpha_excess_deg"])
        trial.set_user_attr(f"seed_{seed}_mean_nz_excess_g", summary["mean_nz_excess_g"])
        trial.set_user_attr(f"seed_{seed}_mean_contouring_cost", summary["mean_contouring_cost"])
        trial.set_user_attr(f"seed_{seed}_mean_terrain_cost", summary["mean_terrain_cost"])
        trial.set_user_attr(f"seed_{seed}_mean_rate_cost", summary["mean_rate_cost"])
        trial.set_user_attr(f"seed_{seed}_mean_limit_cost", summary["mean_limit_cost"])
        trial.set_user_attr(f"seed_{seed}_mean_total_stage_cost", summary["mean_total_stage_cost"])
        trial.set_user_attr(f"seed_{seed}_p95_total_stage_cost", summary["p95_total_stage_cost"])

        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(seed_scores))
    std_score = float(np.std(seed_scores))
    robust_score = mean_score - 0.15 * std_score

    trial.set_user_attr("mean_seed_score", mean_score)
    trial.set_user_attr("std_seed_score", std_score)
    return float(robust_score)


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
            "nominal_dyn_path": str(Path(args.nominal_dyn_path).expanduser()),
            "seeds": [int(s) for s in args.seeds],
            "max_steps": int(args.max_steps),
            "wind_sigma": float(args.wind_sigma),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    nominal_bundle = _load_nominal_bundle(Path(args.nominal_dyn_path).expanduser())
    base_config_kwargs = build_mppi_base_config_kwargs()

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
        f"nominal_dyn_path={Path(args.nominal_dyn_path).expanduser()}, "
        f"seeds={args.seeds}, "
        f"max_steps={args.max_steps}, "
        f"wind_sigma={args.wind_sigma:.2f}"
    )

    study.optimize(
        lambda trial: _objective(trial, args, nominal_bundle, base_config_kwargs),
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
