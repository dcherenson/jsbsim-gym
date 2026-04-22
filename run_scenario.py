import argparse
import csv
from dataclasses import fields, is_dataclass
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from drs_gatekeeper import DRSGatekeeper, GatekeeperParams, TrackBoundsEstimate
import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_env import OBS_ALTITUDE_ERROR_FT, OBS_PHI, OBS_P, OBS_Q, OBS_R, OBS_THETA
from jsbsim_gym.canyon_artifacts import CanyonRunRecorder
from jsbsim_gym.mppi_run_config import (
    KTS_TO_FPS,
    MPPI_TUNING_JSON_PATH,
    build_mppi_base_config_kwargs,
    build_mppi_controller,
    load_mppi_optuna_params,
)
from jsbsim_gym.mppi_support import f16_kinematics_step_with_load_factors, load_nominal_weights
from jsbsim_gym.simple_controller import (
    SimpleCanyonController,
    SimpleCanyonControllerConfig,
    SimpleTrajectoryController,
    build_simple_trajectory_policy_jax,
    build_reference_trajectory,
    with_default_simple_controller_optuna_gains,
)
from jsbsim_gym.uncertainty import RuntimeUncertaintySampler, sample_empirical_jax as sample_empirical_coeff_jax

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
REPO_ROOT = Path(__file__).resolve().parent
DEM_PATH = REPO_ROOT / DEM_PATH
OUTPUT_ROOT = REPO_ROOT / "output"
UNCERTAINTY_ARTIFACT_PATH = REPO_ROOT / "f16_uncertainty_model.pkl"
M_TO_FT = 3.28084
G_FTPS2 = 32.174

def render_state(lateral_error_ft, width_ft):
    y_norm = 2.0 * lateral_error_ft / max(width_ft, 1.0)

    viz = "Wall |"
    pos = int(np.clip((y_norm + 1.0) * 15, 0, 30))
    for i in range(31):
        viz += "X" if i == pos else " "
    viz += "| Wall"
    return viz


def _wrap_heading_deg(angle_rad):
    return float(np.mod(np.degrees(float(angle_rad)), 360.0))


def _wrap_angle_rad(angle_rad):
    return float(np.arctan2(np.sin(float(angle_rad)), np.cos(float(angle_rad))))


def _print_mppi_config(controller_tag, config):
    if not is_dataclass(config):
        print(f"{controller_tag.upper()} config: {config}")
        return

    print(f"\n{controller_tag.upper()} effective configuration:")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {field.name}: {value}")


def save_simple_controller_diagnostics(output_dir, file_stem, rows, termination_reason):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    csv_path = output_dir / f"{file_stem}_diagnostics.csv"
    plot_path = output_dir / f"{file_stem}_diagnostics.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    xtrack_ft = np.asarray([row["lateral_error_ft"] for row in rows], dtype=np.float64)
    xtrack_norm = np.asarray([row["lateral_error_norm"] for row in rows], dtype=np.float64)
    heading_error_deg = np.asarray([row["heading_error_deg"] for row in rows], dtype=np.float64)
    roll_cmd = np.asarray([row["roll_cmd"] for row in rows], dtype=np.float64)
    roll_des_deg = np.asarray([row["roll_des_deg"] for row in rows], dtype=np.float64)
    phi_deg = np.asarray([row["phi_deg"] for row in rows], dtype=np.float64)
    track_accel_cmd_fps2 = np.asarray([row["track_accel_cmd_fps2"] for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    axs[0].plot(time_s, xtrack_ft, color="tab:red", linewidth=2.0)
    axs[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[0].set_ylabel("Xtrack (ft)")
    axs[0].set_title(f"Simple Controller Diagnostics | end={termination_reason} | steps={len(rows)}")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(time_s, xtrack_norm, color="tab:orange", linewidth=2.0, label="xtrack norm")
    axs[1].axhline(1.0, color="black", linewidth=1.0, alpha=0.4, linestyle="--")
    axs[1].axhline(-1.0, color="black", linewidth=1.0, alpha=0.4, linestyle="--")
    axs[1].plot(time_s, heading_error_deg, color="tab:blue", linewidth=1.5, label="heading error (deg)")
    axs[1].set_ylabel("Norm / Deg")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(time_s, roll_des_deg, color="tab:purple", linewidth=2.0, label="roll des (deg)")
    axs[2].plot(time_s, phi_deg, color="tab:green", linewidth=1.5, label="phi (deg)")
    axs[2].set_ylabel("Bank (deg)")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.25)

    axs[3].plot(time_s, roll_cmd, color="tab:brown", linewidth=2.0, label="roll cmd")
    axs[3].plot(time_s, track_accel_cmd_fps2, color="tab:gray", linewidth=1.5, label="track accel (fps^2)")
    axs[3].set_ylabel("Cmd / Accel")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.25)

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return csv_path, plot_path


def save_mppi_tracking_diagnostics(output_dir, file_stem, rows, termination_reason, controller_label):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{file_stem}_tracking_diagnostics.csv"
    plot_path = output_dir / f"{file_stem}_tracking_diagnostics.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    along_track_progress_ft = np.asarray([row["along_track_progress_ft"] for row in rows], dtype=np.float64)
    cross_track_error_ft = np.asarray([row["cross_track_error_ft"] for row in rows], dtype=np.float64)
    altitude_error_ft = np.asarray([row["altitude_error_ft"] for row in rows], dtype=np.float64)
    speed_error_kts = np.asarray([row["speed_error_kts"] for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    axs = axs.reshape(-1)

    axs[0].plot(time_s, along_track_progress_ft, color="tab:blue", linewidth=2.0)
    axs[0].set_ylabel("Feet")
    axs[0].set_title("Along-Track Progress")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(time_s, cross_track_error_ft, color="tab:red", linewidth=2.0)
    axs[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[1].set_ylabel("Feet")
    axs[1].set_title("Cross-Track Error")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(time_s, altitude_error_ft, color="tab:purple", linewidth=2.0)
    axs[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Feet")
    axs[2].set_title("Altitude Error")
    axs[2].grid(True, alpha=0.25)

    axs[3].plot(time_s, speed_error_kts, color="tab:orange", linewidth=2.0)
    axs[3].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Knots")
    axs[3].set_title("Speed Error")
    axs[3].grid(True, alpha=0.25)

    fig.suptitle(
        f"{controller_label.upper()} Tracking Diagnostics | end={termination_reason} | steps={len(rows)}",
        fontsize=13,
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return csv_path, plot_path


def to_mppi_state(env, state, altitude_ref_ft):
    p_n_ft = float(state["p_N"])
    p_e_ft = float(state["p_E"])

    canyon = getattr(env.unwrapped, "canyon", None)
    if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
        try:
            lat_deg = float(env.unwrapped.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(env.unwrapped.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = canyon.get_local_from_latlon(lat_deg, lon_deg)
            p_n_ft = float(local_north_ft)
            p_e_ft = float(local_east_ft)
        except Exception:
            pass

    return {
        "p_N": p_n_ft,
        "p_E": p_e_ft,
        "h": float(state["h"] - altitude_ref_ft),
        "u": float(state["u"]),
        "v": float(state["v"]),
        "w": float(state["w"]),
        "p": float(state["p"]),
        "q": float(state["q"]),
        "r": float(state["r"]),
        "phi": float(state["phi"]),
        "theta": float(state["theta"]),
        "psi": float(state["psi"]),
        "beta": float(state.get("beta", 0.0)),
        "ny": float(state.get("ny", 0.0)),
        "nz": float(state.get("nz", 1.0)),
    }


def get_active_canyon_reference(env, pad_back_ft=500.0, pad_front_ft=1500.0):
    canyon = env.unwrapped.canyon
    if not (
        hasattr(canyon, "north_samples_ft")
        and hasattr(canyon, "width_samples_ft")
        and hasattr(canyon, "center_east_samples_ft")
    ):
        north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
        width_samples_ft = np.ones(256, dtype=np.float32) * 1000.0
        center_east_samples_ft = np.zeros(256, dtype=np.float32)
        centerline_heading_samples_rad = np.zeros(256, dtype=np.float32)
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    north_samples_ft = np.asarray(canyon.north_samples_ft, dtype=np.float32)
    width_samples_ft = np.asarray(canyon.width_samples_ft, dtype=np.float32)
    center_east_samples_ft = np.asarray(canyon.center_east_samples_ft, dtype=np.float32)
    centerline_heading_samples_rad = np.asarray(
        getattr(canyon, "centerline_heading_samples_rad", np.zeros_like(center_east_samples_ft)),
        dtype=np.float32,
    )

    start_info = getattr(env.unwrapped, "dem_start_info", None)
    canyon_span_ft = float(getattr(env.unwrapped, "canyon_span_ft", 0.0))
    if start_info is None or canyon_span_ft <= 0.0 or north_samples_ft.size < 2:
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    start_north_ft = float(start_info["local_north_ft"])
    min_north_ft = max(float(north_samples_ft[0]), start_north_ft - float(pad_back_ft))
    max_north_ft = min(float(north_samples_ft[-1]), start_north_ft + canyon_span_ft + float(pad_front_ft))
    mask = (north_samples_ft >= min_north_ft) & (north_samples_ft <= max_north_ft)
    if int(np.count_nonzero(mask)) < 2:
        return north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad

    return (
        north_samples_ft[mask],
        width_samples_ft[mask],
        center_east_samples_ft[mask],
        centerline_heading_samples_rad[mask],
    )


def controller_state_to_gatekeeper_flat(state_dict):
    return jnp.asarray(
        [
            float(state_dict["p_N"]),
            float(state_dict["p_E"]),
            float(state_dict["h"]),
            float(state_dict["u"]),
            float(state_dict["v"]),
            float(state_dict["w"]),
            float(state_dict["p"]),
            float(state_dict["q"]),
            float(state_dict["r"]),
            float(state_dict["phi"]),
            float(state_dict["theta"]),
            float(state_dict["psi"]),
            float(state_dict.get("ny", 0.0)),
            float(state_dict.get("nz", 1.0)),
        ],
        dtype=jnp.float32,
    )


def _pad_action_plan(action_plan, horizon):
    horizon = int(max(horizon, 1))
    if action_plan is None:
        return np.zeros((horizon, 4), dtype=np.float32)

    plan = np.asarray(action_plan, dtype=np.float32).reshape(-1, 4)
    if plan.shape[0] == 0:
        return np.zeros((horizon, 4), dtype=np.float32)
    if plan.shape[0] >= horizon:
        return plan[:horizon].copy()

    pad = np.repeat(plan[-1:, :], horizon - plan.shape[0], axis=0)
    return np.concatenate([plan, pad], axis=0).astype(np.float32, copy=False)


def build_jsbsim_gatekeeper(
    env,
    initial_controller_state,
    nominal_horizon,
    debug_timing=False,
):
    canyon = env.unwrapped.canyon

    north_samples_ft, width_samples_ft, center_east_samples_ft, heading_samples_rad = get_active_canyon_reference(env)
    width_grad_samples_ft = np.gradient(width_samples_ft, north_samples_ft).astype(np.float32)

    canyon_north_full_ft = np.asarray(
        getattr(canyon, "north_samples_ft", north_samples_ft),
        dtype=np.float32,
    )
    if hasattr(canyon, "ordered_dem_msl_m"):
        terrain_floor_full_msl_ft = (
            np.nanmin(np.asarray(canyon.ordered_dem_msl_m, dtype=np.float32), axis=1) * M_TO_FT
        ).astype(np.float32)
        terrain_floor_msl_ft = np.interp(
            north_samples_ft,
            canyon_north_full_ft,
            terrain_floor_full_msl_ft,
        ).astype(np.float32)
    else:
        terrain_floor_msl_ft = np.zeros_like(north_samples_ft, dtype=np.float32)
    if hasattr(canyon, "wall_height_samples_ft"):
        wall_height_full_ft = np.asarray(canyon.wall_height_samples_ft, dtype=np.float32)
        wall_height_samples_ft = np.interp(
            north_samples_ft,
            canyon_north_full_ft,
            wall_height_full_ft,
        ).astype(np.float32)
    else:
        wall_height_samples_ft = np.zeros_like(north_samples_ft, dtype=np.float32)

    altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
    terrain_floor_rel_ft = terrain_floor_msl_ft - altitude_ref_ft
    canyon_top_rel_ft = terrain_floor_rel_ft + wall_height_samples_ft
    backup_peek_ft =00.0
    pcis_centerline_tol_ft = 500.0
    pcis_altitude_tol_ft = 500.0
    backup_target_speed_fps = 350.0 * KTS_TO_FPS
    pcis_speed_limit_fps = 400.0 * KTS_TO_FPS
    backup_target_altitude_ft = float(np.nanmax(canyon_top_rel_ft) + backup_peek_ft)

    backup_reference = build_reference_trajectory(
        north_ft=north_samples_ft,
        east_ft=center_east_samples_ft,
        heading_rad=heading_samples_rad,
        width_ft=width_samples_ft,
        closed_loop=False,
    )

    backup_config = SimpleCanyonControllerConfig(
        target_speed_fps=backup_target_speed_fps,
        use_dem_centerline=False,
    )
    backup_config, _, _ = with_default_simple_controller_optuna_gains(backup_config)
    backup_controller = SimpleTrajectoryController(
        config=backup_config,
        target_altitude_ft=backup_target_altitude_ft,
        wall_margin_ft=float(getattr(env.unwrapped, "wall_margin_ft", 0.0)),
        altitude_reference_offset_ft=0.0,
        reference_trajectory=backup_reference,
    )
    backup_controller.reset(
        state_dict=initial_controller_state,
        target_altitude_ft=backup_target_altitude_ft,
        reference_trajectory=backup_reference,
    )

    W, B, poly_powers = load_nominal_weights()
    W_jax = jnp.asarray(W, dtype=jnp.float32)
    B_jax = jnp.asarray(B, dtype=jnp.float32)
    poly_powers_jax = jnp.asarray(poly_powers, dtype=jnp.int32)
    north_samples_jax = jnp.asarray(north_samples_ft, dtype=jnp.float32)
    center_east_jax = jnp.asarray(center_east_samples_ft, dtype=jnp.float32)
    heading_samples_jax = jnp.asarray(heading_samples_rad, dtype=jnp.float32)
    width_samples_jax = jnp.asarray(width_samples_ft, dtype=jnp.float32)
    width_grad_jax = jnp.asarray(width_grad_samples_ft, dtype=jnp.float32)
    terrain_floor_jax = jnp.asarray(terrain_floor_rel_ft, dtype=jnp.float32)

    latest_nominal = {
        "action": jnp.zeros((4,), dtype=jnp.float32),
    }
    uncertainty_sampler = RuntimeUncertaintySampler(str(UNCERTAINTY_ARTIFACT_PATH))
    active_empirical_features = uncertainty_sampler.configure_active_features()
    active_empirical_feature_set = set(active_empirical_features)

    def _interp(profile, p_n_ft):
        return jnp.interp(p_n_ft, north_samples_jax, profile)

    def _speed_fps(state_flat):
        return jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(state_flat[3:6])), 1.0))

    def nominal_policy_fn(_state_flat):
        return latest_nominal["action"]

    backup_policy_fn = build_simple_trajectory_policy_jax(
        config=backup_config,
        reference_trajectory=backup_reference,
        target_altitude_ft=backup_target_altitude_ft,
        wall_margin_ft=float(getattr(env.unwrapped, "wall_margin_ft", 0.0)),
        altitude_reference_offset_ft=0.0,
    )

    def dynamics_fn(state_flat, action, noise):
        noise = jnp.asarray(noise, dtype=jnp.float32)
        return f16_kinematics_step_with_load_factors(state_flat, action, W_jax, B_jax + noise, poly_powers_jax)

    def safety_fn(state_flat, _env_param):
        terrain_floor_ft = _interp(terrain_floor_jax, state_flat[0])
        return state_flat[2] - terrain_floor_ft

    def pcis_fn(state_flat):
        speed_margin = pcis_speed_limit_fps - _speed_fps(state_flat)
        centerline_margin = pcis_centerline_tol_ft - jnp.abs(state_flat[1] - _interp(center_east_jax, state_flat[0]))
        altitude_margin = pcis_altitude_tol_ft - jnp.abs(state_flat[2] - backup_target_altitude_ft)
        return jnp.minimum(speed_margin, jnp.minimum(centerline_margin, altitude_margin))

    def empirical_feature_fn(state_flat, action, prev_action, step_idx):
        del step_idx
        u = state_flat[3]
        v = state_flat[4]
        w = state_flat[5]
        p_n_ft = state_flat[0]

        need_speed_terms = bool(active_empirical_feature_set & {"beta", "mach", "qbar"})
        need_alpha = "alpha" in active_empirical_feature_set
        need_width = "canyon_width" in active_empirical_feature_set
        need_width_grad = "canyon_width_grad" in active_empirical_feature_set

        v_sq = u * u + v * v + w * w if need_speed_terms else None
        v_total = jnp.sqrt(jnp.maximum(v_sq, 1.0)) if need_speed_terms else None

        feature_map = {
            "p": state_flat[6],
            "q": state_flat[7],
            "r": state_flat[8],
            "delta_t": action[3],
            "delta_e": action[1],
            "delta_a": action[0],
            "delta_r": action[2],
            "prev_delta_t": prev_action[3],
            "prev_delta_e": prev_action[1],
            "prev_delta_a": prev_action[0],
            "prev_delta_r": prev_action[2],
        }
        if need_alpha:
            feature_map["alpha"] = jnp.arctan2(w, jnp.maximum(u, 1.0))
        if "beta" in active_empirical_feature_set:
            feature_map["beta"] = jnp.arcsin(jnp.clip(v / v_total, -1.0, 1.0))
        if "mach" in active_empirical_feature_set:
            feature_map["mach"] = v_total / 1116.45
        if "qbar" in active_empirical_feature_set:
            feature_map["qbar"] = jnp.maximum(0.5 * 0.0023769 * v_sq, 1.0)
        if need_width:
            feature_map["canyon_width"] = _interp(width_samples_jax, p_n_ft)
        if need_width_grad:
            feature_map["canyon_width_grad"] = _interp(width_grad_jax, p_n_ft)

        return jnp.asarray([feature_map[name] for name in active_empirical_features], dtype=jnp.float32)

    params = GatekeeperParams(
        M=20,
        T=100,
        N=250,
        delta=0.1,
        epsilon=0.10,
        beta=0.00,
        alpha=0.0,
        p=1,
        lipschitz_mode="fixed",
        lipschitz_constant=0.0,
        debug_timing=bool(debug_timing),
    )
    initial_bounds = TrackBoundsEstimate.from_track_width(
        half_width=max(float(np.nanmean(width_samples_ft) * 0.5), 1.0),
        relative_uncertainty=0.0,
    )
    theta_dim = 0  # Disable environment-parameter sampling for now.
    gatekeeper = DRSGatekeeper(
        params=params,
        dynamics_fn=dynamics_fn,
        nominal_policy_fn=nominal_policy_fn,
        backup_policy_fn=backup_policy_fn,
        safety_fn=safety_fn,
        pcis_fn=pcis_fn,
        noise_dim=6,
        theta_dim=theta_dim,
        uncertainty_model=uncertainty_sampler,
        empirical_feature_fn=empirical_feature_fn,
        empirical_sampler_fn=sample_empirical_coeff_jax,
        initial_track_bounds=initial_bounds,
        seed=3,
    )
    gatekeeper.reset(controller_state_to_gatekeeper_flat(initial_controller_state), t=0)

    return {
        "gatekeeper": gatekeeper,
        "backup_controller": backup_controller,
        "backup_reference": backup_reference,
        "backup_target_speed_fps": backup_target_speed_fps,
        "backup_target_altitude_ft": backup_target_altitude_ft,
        "latest_nominal": latest_nominal,
        "pcis_centerline_tol_ft": pcis_centerline_tol_ft,
    }


class AltitudeHoldController:
    """Straight-flight + altitude-hold controller for visualization/debug runs."""

    def __init__(self):
        self.altitude_error_integral = 0.0

    def get_action(self, obs):
        altitude_error_ft = float(obs[OBS_ALTITUDE_ERROR_FT])
        roll_angle = float(obs[OBS_PHI])
        roll_rate = float(obs[OBS_P])
        pitch_rate = float(obs[OBS_Q])
        yaw_rate = float(obs[OBS_R])
        pitch_angle = float(obs[OBS_THETA])

        self.altitude_error_integral += altitude_error_ft * (1.0 / 30.0)
        self.altitude_error_integral = np.clip(self.altitude_error_integral, -2000.0, 2000.0)

        roll_cmd = np.clip(-1.6 * roll_angle - 0.45 * roll_rate, -0.35, 0.35)
        yaw_cmd = np.clip(-0.25 * yaw_rate, -0.15, 0.15)

        pitch_cmd = np.clip(
            0.0022 * altitude_error_ft
            + 0.70 * pitch_angle
            - 0.30 * pitch_rate
            + 0.00006 * self.altitude_error_integral,
            -0.50,
            0.50,
        )

        throttle_cmd = np.clip(0.64 - 0.00008 * altitude_error_ft, 0.50, 0.95)
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)




def parse_args():
    parser = argparse.ArgumentParser(description="Run canyon scenarios across available controllers.")
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi", "simple", "altitude_hold"],
        default="mppi",
        help="Controller variant to run.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable interactive rendering window (default is headless).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for run artifacts. The final subdirectory is controller-specific.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum control steps to run.",
    )
    parser.add_argument(
        "--target-speed-kts",
        type=float,
        default=450.0,
        help="Target flight speed in knots (kts).",
    )
    parser.add_argument(
        "--initial-speed-kts",
        type=float,
        default=450,
        help="Initial entry speed in knots (kts). Defaults to --target-speed-kts.",
    )
    parser.add_argument(
        "--initial-altitude-ft",
        type=float,
        default=500.0,
        help="Initial entry altitude in feet, relative to the DEM start elevation in canyon DEM mode.",
    )
    parser.add_argument(
        "--initial-roll-deg",
        type=float,
        default=None,
        help="Optional initial roll attitude in degrees.",
    )
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=None,
        help="Optional initial pitch attitude in degrees.",
    )
    parser.add_argument(
        "--initial-heading-deg",
        type=float,
        default=None,
        help="Optional initial true heading in degrees. Defaults to the DEM follow-canyon heading.",
    )
    parser.add_argument(
        "--initial-alpha-deg",
        type=float,
        default=None,
        help="Optional initial angle of attack in degrees.",
    )
    parser.add_argument(
        "--initial-beta-deg",
        type=float,
        default=None,
        help="Optional initial sideslip angle in degrees.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mppi_canyon_tuning",
        help="Name of the Optuna study to load tuned parameters from.",
    )
    parser.add_argument(
        "--gatekeeper",
        action="store_true",
        help="Wrap the nominal MPPI controller with the DRS gatekeeper and a conservative simple-controller backup.",
    )
    parser.add_argument(
        "--gatekeeper-debug-timing",
        action="store_true",
        help="Print gatekeeper timing diagnostics each step.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.gatekeeper and args.controller not in {"mppi", "smooth_mppi"}:
        raise ValueError("--gatekeeper currently requires --controller mppi or --controller smooth_mppi.")
    initial_speed_kts = float(args.target_speed_kts if args.initial_speed_kts is None else args.initial_speed_kts)
    initial_altitude_ft = float(args.initial_altitude_ft)

    output_subdirs = {
        "mppi": "canyon_mppi",
        "smooth_mppi": "canyon_smooth_mppi",
        "simple": "canyon_simple",
        "altitude_hold": "canyon_altitude_hold",
    }
    output_dir = args.output_dir
    if output_dir is None:
        subdir = output_subdirs[args.controller]
        if args.gatekeeper:
            subdir = f"{subdir}_gatekeeper"
        output_dir = Path(args.output_root) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "render" if args.render else "headless"
    controller_tag = str(args.controller)

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode="human" if args.render else "rgb_array",
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
        dem_bbox=DEM_BBOX,
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=DEM_START_PIXEL,
        dem_start_heading_mode="follow_canyon",
        dem_start_heading_deg=args.initial_heading_deg,
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=500.0,
        entry_altitude_ft=initial_altitude_ft,
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=10.0,
        entry_speed_kts=initial_speed_kts,
        entry_roll_deg=args.initial_roll_deg,
        entry_pitch_deg=args.initial_pitch_deg,
        entry_alpha_deg=args.initial_alpha_deg,
        entry_beta_deg=args.initial_beta_deg,
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    obs, _ = env.reset(seed=3)

    state = env.unwrapped.get_full_state_dict()
    actual_dem_start_pixel = tuple(getattr(env.unwrapped, "dem_start_pixel", DEM_START_PIXEL))

    altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
    initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)
    start_path_north_ft = float(initial_controller_state["p_N"])

    target_altitude_ft = float(env.unwrapped.target_altitude_ft - altitude_ref_ft)
    min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
    max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)

    canyon = env.unwrapped.canyon
    north_samples_ft, width_samples_ft, center_east_samples_ft, centerline_heading_samples_rad = get_active_canyon_reference(env)

    mppi_target_altitude_ft = target_altitude_ft

    optuna_params, optuna_source = load_mppi_optuna_params(study_name=args.study_name)
    if optuna_params:
        print(
            f"Auto-loaded MPPI tuned parameters from {optuna_source} "
            f"({len(optuna_params)} parameters)."
        )
        if "target_alt_tune_ft" in optuna_params:
            mppi_target_altitude_ft = float(optuna_params["target_alt_tune_ft"])
    else:
        print("Note: No tuned MPPI parameters found; using built-in defaults.")

    config_base_kwargs = build_mppi_base_config_kwargs(
        optuna_params=optuna_params,
        target_speed_fps=args.target_speed_kts * KTS_TO_FPS,
        target_altitude_ft=mppi_target_altitude_ft,
        min_altitude_ft=min_altitude_ft,
        max_altitude_ft=max_altitude_ft,
        wall_margin_ft=float(env.unwrapped.wall_margin_ft),
        horizon=40,
        num_samples=10000,
        optimization_steps=3,
        terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
    )

    if controller_tag == "simple":
        simple_config = SimpleCanyonControllerConfig(
            target_speed_fps=args.target_speed_kts * 1.68781,
        )
        simple_config, simple_tuning_source, simple_tuned_keys = with_default_simple_controller_optuna_gains(simple_config)
        if simple_tuned_keys:
            print(
                f"Auto-loaded simple-controller tuned gains from {simple_tuning_source} "
                f"({len(simple_tuned_keys)} parameters)."
            )
        else:
            print("Note: No tuned simple-controller gains found; using built-in defaults.")
        controller = SimpleCanyonController(
            env=env,
            config=simple_config,
        )
        controller.reset(initial_controller_state)
    elif controller_tag == "altitude_hold":
        controller = AltitudeHoldController()
    else:
        controller, config = build_mppi_controller(
            controller_tag,
            optuna_params=optuna_params,
            config_base_kwargs=config_base_kwargs,
            canyon_north_samples_ft=north_samples_ft,
            canyon_width_samples_ft=width_samples_ft,
            canyon_center_east_samples_ft=center_east_samples_ft,
            canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
        )
        _print_mppi_config(controller_tag, config)

    recorder = CanyonRunRecorder(
        env=env,
        dem_path=DEM_PATH,
        dem_bbox=DEM_BBOX,
        dem_start_pixel=actual_dem_start_pixel,
        output_dir=output_dir,
        file_stem=f"canyon_{controller_tag}_{mode_tag}",
        title_prefix=f"{controller_tag.upper()} Trajectory Overlay",
        fps=30,
    )
    recorder.initialize()
    recorder.set_centerline_profile(north_samples_ft, center_east_samples_ft)
    termination_reason = "running"
    simple_diagnostic_rows = []
    mppi_tracking_rows = []

    gatekeeper_bundle = None
    gatekeeper_prev_using_backup = False

    print(f"Initializing {controller_tag} canyon controller...")
    if controller_tag in {"mppi", "smooth_mppi"}:
        print("Compiling JAX JIT... (this takes a moment)", flush=True)
        _ = controller.get_action(initial_controller_state)
        print("JIT compilation finished.", flush=True)
        if args.gatekeeper:
            if not UNCERTAINTY_ARTIFACT_PATH.exists():
                raise FileNotFoundError(f"Missing uncertainty artifact: {UNCERTAINTY_ARTIFACT_PATH}")
            gatekeeper_bundle = build_jsbsim_gatekeeper(
                env=env,
                initial_controller_state=initial_controller_state,
                nominal_horizon=int(controller.config.horizon),
                debug_timing=bool(args.gatekeeper_debug_timing),
            )
            print(
                f"Initialized gatekeeper with backup simple controller at {gatekeeper_bundle['backup_target_speed_fps'] / KTS_TO_FPS:.0f} kts and "
                f"target altitude {gatekeeper_bundle['backup_target_altitude_ft']:.0f} ft above DEM origin."
            )
            print("Compiling Gatekeeper JAX JIT... (this takes a moment)", flush=True)
            gatekeeper = gatekeeper_bundle["gatekeeper"]
            latest_nominal = gatekeeper_bundle["latest_nominal"]
            nominal_action = controller.get_action(initial_controller_state)
            latest_nominal["action"] = jnp.asarray(np.asarray(nominal_action, dtype=np.float32), dtype=jnp.float32)
            warmup_nominal_trajectory = _pad_action_plan(getattr(controller, "base_plan", None), gatekeeper.params.T)
            _ = gatekeeper.update(
                controller_state_to_gatekeeper_flat(initial_controller_state),
                track_bounds=None,
                nominal_trajectory=jnp.asarray(warmup_nominal_trajectory, dtype=jnp.float32),
                max_steps=int(args.max_steps),
            )
            gatekeeper.reset(controller_state_to_gatekeeper_flat(initial_controller_state), t=0)
            gatekeeper_prev_using_backup = False
            print("Gatekeeper JIT compilation finished.", flush=True)

    print("\nStarting Canyon Flight...")
    print(
        f"Initial conditions: {initial_speed_kts:.1f} kts entry speed, "
        f"{initial_altitude_ft:.1f} ft entry altitude, "
        f"start_px={actual_dem_start_pixel}, "
        f"phi={np.degrees(float(initial_controller_state['phi'])):.1f} deg, "
        f"theta={np.degrees(float(initial_controller_state['theta'])):.1f} deg, "
        f"psi={_wrap_heading_deg(initial_controller_state['psi']):.1f} deg, "
        f"alpha={np.degrees(float(state['alpha'])):.1f} deg, "
        f"beta={np.degrees(float(state.get('beta', 0.0))):.1f} deg."
    )
    print(
        f"{'Step':<5} | {'p_N_rel':<8} | {'LatErr':<8} | {'h_rel':<8} | "
        f"{'V':<6} | {'W_c':<6} | {'Plan(ms)':<8} | {'gk(ms)':<8} | {'gk_ws':<7} | {'gk_upd':<7} | {'gk_bak':<7}"
    )
    print("-" * 110)

    try:
        for step in range(int(args.max_steps)):
            controller_state = to_mppi_state(env, state, altitude_ref_ft)

            t0 = time.time()
            if controller_tag == "altitude_hold":
                action = controller.get_action(obs)
            else:
                nominal_action = controller.get_action(controller_state)
                action = nominal_action
            plan_ms = (time.time() - t0) * 1000.0

            t0 = time.time()
            gatekeeper_state = None
            gk_nom_prep = 0.0
            gk_update_ms = 0.0
            gk_backup_ms = 0.0
            if gatekeeper_bundle is not None:
                gatekeeper = gatekeeper_bundle["gatekeeper"]
                latest_nominal = gatekeeper_bundle["latest_nominal"]
                latest_nominal["action"] = jnp.asarray(np.asarray(nominal_action, dtype=np.float32), dtype=jnp.float32)

                t_ws = time.time()
                nominal_trajectory = _pad_action_plan(getattr(controller, "base_plan", None), gatekeeper.params.T)
                nominal_trajectory_jax = jnp.asarray(nominal_trajectory, dtype=jnp.float32)
                gk_nom_prep = (time.time() - t_ws) * 1000.0

                track_bounds = None
                if gatekeeper.theta_dim > 0:
                    current_width_ft = float(state.get("canyon_width", np.nanmean(width_samples_ft)))
                    track_bounds = TrackBoundsEstimate.from_track_width(
                        half_width=max(0.5 * current_width_ft, 1.0),
                        relative_uncertainty=0.0,
                    )

                t_update = time.time()
                gatekeeper_state = gatekeeper.update(
                    controller_state_to_gatekeeper_flat(controller_state),
                    track_bounds=track_bounds,
                    nominal_trajectory=nominal_trajectory_jax,
                    max_steps=int(args.max_steps),
                )
                gk_update_ms = (time.time() - t_update) * 1000.0

                if gatekeeper_state.using_backup:
                    t_backup = time.time()
                    backup_controller = gatekeeper_bundle["backup_controller"]
                    if not gatekeeper_prev_using_backup:
                        backup_controller.reset(
                            state_dict=controller_state,
                            target_altitude_ft=gatekeeper_bundle["backup_target_altitude_ft"],
                            reference_trajectory=gatekeeper_bundle["backup_reference"],
                        )
                    action = backup_controller.get_action(controller_state)
                    gk_backup_ms = (time.time() - t_backup) * 1000.0
                gatekeeper.tick()
                gatekeeper_prev_using_backup = bool(gatekeeper_state.using_backup)
            gk_ms = (time.time() - t0) * 1000.0

            planner_debug = None
            raw_debug = None
            debug_getter = getattr(controller, "get_render_debug", None)
            if callable(debug_getter):
                raw_debug = debug_getter()
            if gatekeeper_state is None:
                if raw_debug is not None:
                    planner_debug = {
                        "candidate_xy": np.asarray(
                            raw_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "candidate_h_ft": np.asarray(
                            raw_debug.get("candidate_h_ft", np.zeros((0, 0), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "final_xy": np.asarray(
                            raw_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "final_h_ft": np.asarray(
                            raw_debug.get("final_h_ft", np.zeros((0,), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "warm_start_xy": np.asarray(
                            raw_debug.get("warm_start_xy", np.zeros((0, 2), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "warm_start_h_ft": np.asarray(
                            raw_debug.get("warm_start_h_ft", np.zeros((0,), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                    }
            else:
                predicted_xy = np.asarray(
                    gatekeeper_state.predicted_trajectories
                    if gatekeeper_state.predicted_trajectories is not None
                    else np.zeros((0, 0, 2), dtype=np.float32),
                    dtype=np.float32,
                )
                predicted_h_ft = np.zeros((0, 0), dtype=np.float32)
                if predicted_xy.ndim == 3 and predicted_xy.shape[-1] == 2:
                    predicted_h_ft = np.full(
                        (predicted_xy.shape[0], predicted_xy.shape[1]),
                        float(controller_state["h"]),
                        dtype=np.float32,
                    )
                planner_debug = {
                    "gk_trajectories": predicted_xy,
                    "gk_h_ft": predicted_h_ft,
                    "failure_mask": np.asarray(
                        gatekeeper_state.failure_mask
                        if gatekeeper_state.failure_mask is not None
                        else np.zeros((0,), dtype=bool),
                        dtype=bool,
                    ),
                    "m_star": int(gatekeeper_state.m_star),
                    "s_t": int(gatekeeper_state.s_prev),
                    "plan_start_t": int(gatekeeper_state.plan_start_t),
                    "is_reverting": bool(gatekeeper_state.is_reverting),
                    "using_backup": bool(gatekeeper_state.using_backup),
                    "q_bar_star": float(gatekeeper_state.q_bar_star),
                    "epsilon": float(gatekeeper_bundle["gatekeeper"].params.epsilon),
                }
                if raw_debug is not None:
                    planner_debug["warm_start_xy"] = np.asarray(
                        raw_debug.get("warm_start_xy", np.zeros((0, 2), dtype=np.float32)),
                        dtype=np.float32,
                    ).copy()
                    planner_debug["warm_start_h_ft"] = np.asarray(
                        raw_debug.get("warm_start_h_ft", np.zeros((0,), dtype=np.float32)),
                        dtype=np.float32,
                    ).copy()

            if controller_tag == "simple" and gatekeeper_state is None:
                guidance = dict(getattr(controller, "last_guidance", {}) or {})
                lookahead_north_ft = float(guidance.get("lookahead_north_ft", controller_state["p_N"]))
                lookahead_center_east_ft = float(guidance.get("lookahead_center_east_ft", controller_state["p_E"]))

                if planner_debug is None:
                    planner_debug = {
                        "candidate_xy": np.zeros((0, 0, 2), dtype=np.float32),
                        "candidate_h_ft": np.zeros((0, 0), dtype=np.float32),
                        "final_xy": np.zeros((0, 2), dtype=np.float32),
                        "final_h_ft": np.zeros((0,), dtype=np.float32),
                    }

                planner_debug["lookahead_xy"] = np.asarray(
                    [[lookahead_north_ft, lookahead_center_east_ft]],
                    dtype=np.float32,
                )
                planner_debug["lookahead_h_ft"] = np.asarray(
                    # Match the centerline overlay reference height so the marker
                    # visually sits on the blue centerline ribbon.
                    [0.0],
                    dtype=np.float32,
                )

            heading_deg = _wrap_heading_deg(controller_state["psi"])
            heading_cmd_deg = heading_deg
            if controller_tag == "simple":
                simple_guidance = dict(getattr(controller, "last_guidance", {}) or {})
                heading_cmd_deg = float(simple_guidance.get("centerline_heading_deg", heading_deg))
            else:
                get_hdg_cmd = getattr(controller, "get_centerline_heading_rad", None)
                if callable(get_hdg_cmd):
                    try:
                        heading_cmd_deg = _wrap_heading_deg(get_hdg_cmd(controller_state["p_N"]))
                    except Exception:
                        heading_cmd_deg = heading_deg

            hud_debug = {
                "heading_deg": float(heading_deg),
                "heading_cmd_deg": float(heading_cmd_deg),
                "action_cmd": np.asarray(action, dtype=np.float32).copy(),
            }
            if gatekeeper_state is not None:
                hud_debug.update(
                    {
                        "gatekeeper_active": True,
                        "using_backup": bool(gatekeeper_state.using_backup),
                        "is_reverting": bool(gatekeeper_state.is_reverting),
                        "s_t": int(gatekeeper_state.s_prev),
                        "m_star": int(gatekeeper_state.m_star),
                        "q_bar_star": float(gatekeeper_state.q_bar_star),
                        "epsilon": float(gatekeeper_bundle["gatekeeper"].params.epsilon),
                    }
                )

            if np.isnan(action).any() or np.isinf(action).any():
                print(f"Invalid action at step {step}: {action}")
                break

            set_hud_commands = getattr(env.unwrapped, "set_hud_commands", None)
            if callable(set_hud_commands):
                controller_mode_label = {
                    "mppi": "MPPI",
                    "smooth_mppi": "SMPPI",
                    "simple": "SIMPLE",
                    "altitude_hold": "HOLD",
                }.get(controller_tag, str(controller_tag).upper())
                gate_label = "TRACK"
                if gatekeeper_state is not None:
                    gate_label = "BACKUP" if gatekeeper_state.using_backup else "NOMINAL"
                guidance_label = f"HDG {int(round(heading_cmd_deg)) % 360:03d}"
                set_hud_commands(
                    heading_cmd_deg=heading_cmd_deg,
                    mode_labels=(controller_mode_label, gate_label, guidance_label),
                )

            obs, _, terminated, truncated, info = env.step(action)
            termination_reason = info.get("termination_reason", "running")
            state = env.unwrapped.get_full_state_dict()
            post_controller_state = to_mppi_state(env, state, altitude_ref_ft)
            recorder.record_step(planner_debug=planner_debug, hud_debug=hud_debug)

            width_ft = float(info.get("canyon_width_ft", np.nan))
            lateral_ft = float(info.get("lateral_error_ft", np.nan))
            speed_fps = float(
                np.sqrt(
                    float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2
                )
            )
            speed_error_kts = float((speed_fps - float(config_base_kwargs["target_speed_fps"])) / KTS_TO_FPS)
            if controller_tag in {"mppi", "smooth_mppi"}:
                mppi_tracking_rows.append(
                    {
                        "step": int(step),
                        "time_s": float(step / 30.0),
                        "along_track_progress_ft": float(post_controller_state["p_N"] - start_path_north_ft),
                        "cross_track_error_ft": float(info.get("lateral_error_ft", np.nan)),
                        "altitude_error_ft": float(info.get("altitude_error_ft", np.nan)),
                        "speed_error_kts": float(speed_error_kts),
                        "speed_fps": float(speed_fps),
                        "target_speed_kts": float(config_base_kwargs["target_speed_fps"] / KTS_TO_FPS),
                        "terrain_clearance_ft": float(info.get("terrain_clearance_ft", np.nan)),
                        "termination_reason": str(termination_reason),
                        "using_backup": bool(gatekeeper_state.using_backup) if gatekeeper_state is not None else False,
                        "plan_ms": float(plan_ms),
                        "gatekeeper_ms": float(gk_ms),
                        "gatekeeper_nominal_prep_ms": float(gk_nom_prep),
                        "gatekeeper_update_ms": float(gk_update_ms),
                        "gatekeeper_backup_ms": float(gk_backup_ms),
                    }
                )
            if controller_tag == "simple":
                simple_guidance = dict(getattr(controller, "last_guidance", {}) or {})
                simple_diagnostic_rows.append(
                    {
                        "step": int(step),
                        "time_s": float(step / 30.0),
                        "lateral_error_ft": float(info.get("lateral_error_ft", np.nan)),
                        "lateral_error_norm": float(info.get("lateral_error_norm", np.nan)),
                        "heading_error_deg": float(simple_guidance.get("heading_error_deg", np.nan)),
                        "roll_cmd": float(simple_guidance.get("roll_cmd", np.nan)),
                        "roll_des_deg": float(simple_guidance.get("roll_des_deg", np.nan)),
                        "phi_deg": float(np.degrees(float(controller_state["phi"]))),
                        "track_accel_cmd_fps2": float(simple_guidance.get("track_accel_cmd_fps2", np.nan)),
                        "nz_des": float(simple_guidance.get("nz_des", np.nan)),
                        "pitch_cmd": float(simple_guidance.get("pitch_cmd", np.nan)),
                        "speed_fps": float(speed_fps),
                        "altitude_error_ft": float(info.get("altitude_error_ft", np.nan)),
                        "terrain_clearance_ft": float(info.get("terrain_clearance_ft", np.nan)),
                    }
                )

            if step % 5 == 0:
                rel_north_ft = float(controller_state["p_N"] - start_path_north_ft)
                rel_alt_ft = float(controller_state["h"])
                print(
                    f"{step:<5} | {rel_north_ft:<8.0f} | {lateral_ft:<8.0f} | {rel_alt_ft:<8.0f} | "
                    f"{speed_fps:<6.0f} | {width_ft:<6.0f} | {plan_ms:<8.1f} | {gk_ms:<8.1f} | "
                    f"{gk_nom_prep:<7.1f} | {gk_update_ms:<7.1f} | {gk_backup_ms:<7.1f}"
                )

            if terminated or truncated:
                print(f"Episode ended at step {step}: {termination_reason}")
                break

            if args.render:
                time.sleep(1.0 / 30.0)
    finally:
        env.close()

    artifacts = recorder.finalize(termination_reason)
    print(f"Saved video: {artifacts['video_path']}")
    print(f"Saved trajectory overlay: {artifacts['overlay_path']}")
    print(f"Saved trajectory CSV: {artifacts['trajectory_csv_path']}")
    if controller_tag == "simple":
        diag_csv_path, diag_plot_path = save_simple_controller_diagnostics(
            output_dir=output_dir,
            file_stem=f"canyon_{controller_tag}_{mode_tag}",
            rows=simple_diagnostic_rows,
            termination_reason=termination_reason,
        )
        if diag_csv_path is not None and diag_plot_path is not None:
            print(f"Saved diagnostics CSV: {diag_csv_path}")
            print(f"Saved diagnostics plot: {diag_plot_path}")
    if controller_tag in {"mppi", "smooth_mppi"}:
        diag_csv_path, diag_plot_path = save_mppi_tracking_diagnostics(
            output_dir=output_dir,
            file_stem=f"canyon_{controller_tag}_{mode_tag}",
            rows=mppi_tracking_rows,
            termination_reason=termination_reason,
            controller_label=controller_tag,
        )
        if diag_csv_path is not None and diag_plot_path is not None:
            print(f"Saved tracking diagnostics CSV: {diag_csv_path}")
            print(f"Saved tracking diagnostics plot: {diag_plot_path}")


if __name__ == "__main__":
    main()
