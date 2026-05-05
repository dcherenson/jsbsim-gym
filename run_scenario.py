import argparse
from dataclasses import fields, is_dataclass
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp

from drs_gatekeeper import DRSGatekeeper, GatekeeperParams, TrackBoundsEstimate
import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon import DEMCanyon
from jsbsim_gym.canyon_env import OBS_ALTITUDE_ERROR_FT, OBS_PHI, OBS_P, OBS_Q, OBS_R, OBS_THETA, OBS_U_FPS, OBS_V_FPS, OBS_W_FPS
from jsbsim_gym.canyon_artifacts import CanyonRunRecorder
from jsbsim_gym.mppi_run_config import (
    KTS_TO_FPS,
    build_mppi_base_config_kwargs,
    build_mppi_controller,
    with_default_mppi_optuna_params,
)
from jsbsim_gym.mppi_support import f16_kinematics_step_with_load_factors, load_nominal_weights
from jsbsim_gym.nominal_trajectory import (
    build_nominal_reference_from_dyn,
    load_nominal_initial_conditions_from_dyn,
)
from jsbsim_gym.simple_controller import (
    SimpleCanyonController,
    SimpleCanyonControllerConfig,
    SimpleTrajectoryController,
    build_simple_trajectory_policy_jax,
    build_reference_trajectory,
    with_default_simple_controller_optuna_gains,
)
from jsbsim_gym.cascaded_pid_controller import PIDTrajectoryController, build_pid_trajectory_policy_jax
from jsbsim_gym.run_diagnostics import (
    _append_state_fields,
    save_mppi_plan_diagnostics,
    save_mppi_tracking_diagnostics,
    save_pid_traj_diagnostics,
    save_simple_controller_diagnostics,
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
DEFAULT_INITIAL_SPEED_KTS = 450.0
DEFAULT_INITIAL_ALTITUDE_FT = 500.0
DEFAULT_INITIAL_HEADING_DEG = None
DEFAULT_INITIAL_ROLL_DEG = None
DEFAULT_INITIAL_PITCH_DEG = None
DEFAULT_INITIAL_ALPHA_DEG = None
DEFAULT_INITIAL_BETA_DEG = None
MPPI_STATE_KEYS = (
    "p_N",
    "p_E",
    "h",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "phi",
    "theta",
    "psi",
    "ny",
    "nz",
)


def _fraction_0_to_1(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a floating-point value in [0, 1], got {value!r}.") from exc
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise argparse.ArgumentTypeError(f"Expected a floating-point value in [0, 1], got {value!r}.")
    return fraction


def _set_mppi_nominal_start_progress(controller, progress_fraction: float) -> float | None:
    params = getattr(controller, "params", None)
    if params is None or not hasattr(params, "path_s_np"):
        return None
    if not hasattr(controller, "_progress_s_ft"):
        return None

    path_s = np.asarray(params.path_s_np, dtype=np.float64).reshape(-1)
    if path_s.size < 1:
        return None

    s0 = float(path_s[0])
    s1 = float(path_s[-1])
    progress_s_ft = float(s0 + float(np.clip(progress_fraction, 0.0, 1.0)) * (s1 - s0))
    controller._progress_s_ft = progress_s_ft
    warm_start_fn = getattr(controller, "_initialize_plans_from_nominal_progress", None)
    if callable(warm_start_fn):
        try:
            warm_start_fn(progress_s_ft)
        except Exception:
            pass
    return progress_s_ft


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


def _print_mppi_config(controller_tag, config):
    if not is_dataclass(config):
        print(f"{controller_tag.upper()} config: {config}")
        return

    print(f"\n{controller_tag.upper()} effective configuration:")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {field.name}: {value}")





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
        "alpha": float(state.get("alpha", 0.0)),
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


_GATEKEEPER_FLAT_BUF = np.zeros(14, dtype=np.float32)

def controller_state_to_gatekeeper_flat(state_dict):
    """Convert a controller state dict to a flat JAX float32 vector [14].

    Uses a module-level numpy buffer to avoid per-call Python list allocation.
    """
    buf = _GATEKEEPER_FLAT_BUF
    buf[0]  = state_dict["p_N"]
    buf[1]  = state_dict["p_E"]
    buf[2]  = state_dict["h"]
    buf[3]  = state_dict["u"]
    buf[4]  = state_dict["v"]
    buf[5]  = state_dict["w"]
    buf[6]  = state_dict["p"]
    buf[7]  = state_dict["q"]
    buf[8]  = state_dict["r"]
    buf[9]  = state_dict["phi"]
    buf[10] = state_dict["theta"]
    buf[11] = state_dict["psi"]
    buf[12] = state_dict.get("ny", 0.0)
    buf[13] = state_dict.get("nz", 1.0)
    return jnp.asarray(buf, dtype=jnp.float32)


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


def _build_gatekeeper_nominal_trajectory(controller, nominal_action, horizon):
    """
    Build a [T, 4] nominal action sequence for gatekeeper rollout.

    MPPI-style controllers provide `base_plan`; controllers without an open-loop
    plan (e.g. PID trajectory tracking) fall back to repeating the current
    nominal action.
    """
    horizon = int(max(horizon, 1))

    base_plan = getattr(controller, "base_plan", None)
    if base_plan is not None:
        return _pad_action_plan(base_plan, horizon)

    action = np.asarray(nominal_action, dtype=np.float32).reshape(-1)
    if action.size != 4:
        raise ValueError(f"Expected nominal action shape (4,), got {action.shape}.")
    return np.repeat(action[np.newaxis, :], horizon, axis=0).astype(np.float32, copy=False)


def build_jsbsim_gatekeeper(
    env,
    initial_controller_state,
    nominal_horizon,
    debug_timing=False,
    nominal_policy_fn_override=None,
    backup_controller_type="altitude_hold",
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
    backup_peek_ft =0.0
    pcis_centerline_tol_ft = 1000.0
    pcis_altitude_tol_ft = 1000.0
    backup_target_speed_fps = 300.0 * KTS_TO_FPS
    pcis_speed_limit_fps = 600.0 * KTS_TO_FPS
    backup_target_altitude_ft = float(np.nanmax(canyon_top_rel_ft) + backup_peek_ft)

    backup_reference = build_reference_trajectory(
        north_ft=north_samples_ft,
        east_ft=center_east_samples_ft,
        heading_rad=heading_samples_rad,
        width_ft=width_samples_ft,
        closed_loop=False,
    )

    if backup_controller_type == "altitude_hold":
        backup_controller = AltitudeHoldController(
            target_speed_fps=backup_target_speed_fps, 
            target_altitude_ft=backup_target_altitude_ft
            )
    else:
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

    W, B, poly_powers, throttle_force_coeffs = load_nominal_weights()
    W_jax = jnp.asarray(W, dtype=jnp.float32)
    B_jax = jnp.asarray(B, dtype=jnp.float32)
    poly_powers_jax = jnp.asarray(poly_powers, dtype=jnp.int32)
    throttle_force_coeffs_jax = jnp.asarray(throttle_force_coeffs, dtype=jnp.float32)
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

    if nominal_policy_fn_override is None:
        def nominal_policy_fn(_state_flat):
            return latest_nominal["action"]
    else:
        nominal_policy_fn = nominal_policy_fn_override

    if backup_controller_type == "altitude_hold":
        backup_policy_fn = build_altitude_hold_policy_jax(
            target_altitude_ft=backup_target_altitude_ft,
            target_speed_fps=backup_target_speed_fps
        )
    else:
        backup_policy_fn = build_simple_trajectory_policy_jax(
            config=backup_config,
            reference_trajectory=backup_reference,
            target_altitude_ft=backup_target_altitude_ft,
            wall_margin_ft=float(getattr(env.unwrapped, "wall_margin_ft", 0.0)),
            altitude_reference_offset_ft=0.0,
        )

    def dynamics_fn(state_flat, action, noise):
        noise = jnp.asarray(noise, dtype=jnp.float32)
        return f16_kinematics_step_with_load_factors(
            state_flat,
            action,
            W_jax,
            B_jax + noise,
            poly_powers_jax,
            throttle_force_coeffs_jax,
        )

    def safety_fn(state_flat, _env_param):
        terrain_floor_ft = _interp(terrain_floor_jax, state_flat[0])
        val = state_flat[2] - terrain_floor_ft
        # jax.debug.print("Safety function value: {v}", v=val)
        return val

    def pcis_fn(state_flat):
        speed_margin = pcis_speed_limit_fps - _speed_fps(state_flat)
        centerline_margin = 100 #pcis_centerline_tol_ft - jnp.abs(state_flat[1] - _interp(center_east_jax, state_flat[0]))
        altitude_margin = pcis_altitude_tol_ft - jnp.abs(state_flat[2] - backup_target_altitude_ft)
        # jax.debug.print("PCIS function values - Speed: {s}, Centerline: {c}, Altitude: {a}", s=speed_margin, c=centerline_margin, a=altitude_margin)
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
        M=30,
        T=100,
        N=100,
        delta=0.1,
        epsilon=0.90,
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

    def __init__(self, target_altitude_ft = 1000.0, target_speed_fps = 400.0):
        self.target_altitude_ft = target_altitude_ft
        self.target_speed_fps = target_speed_fps

    def reset(self, state_dict=None, target_altitude_ft=None, reference_trajectory=None, target_speed_fps=None, **kwargs):
        self.altitude_error_integral = 0.0
        if target_altitude_ft is not None:
            self.target_altitude_ft = float(target_altitude_ft)
        if target_speed_fps is not None:
            self.target_speed_fps = float(target_speed_fps)

    def get_action(self, obs_or_state):
        h = float(obs_or_state["h"])
        altitude_error_ft = float(self.target_altitude_ft - h)
        roll_angle = float(obs_or_state["phi"])
        pitch_angle = float(obs_or_state["theta"])
        roll_rate = float(obs_or_state["p"])
        pitch_rate = float(obs_or_state["q"])
        yaw_rate = float(obs_or_state["r"])
        u = float(obs_or_state["u"])
        v = float(obs_or_state["v"])
        w = float(obs_or_state["w"])
        speed_fps = float(np.sqrt(u**2 + v**2 + w**2))
        v_rate = u * np.sin(pitch_angle) - v * np.sin(roll_angle) * np.cos(pitch_angle) - w * np.cos(roll_angle) * np.cos(pitch_angle)
        # self.altitude_error_integral += altitude_error_ft * (1.0 / 30.0)
        # self.altitude_error_integral = np.clip(self.altitude_error_integral, -2000.0, 2000.0)

        roll_cmd = np.clip(-1.6 * roll_angle - 0.45 * roll_rate, -1.0, 1.0)
        yaw_cmd = np.clip(-0.25 * yaw_rate, -0.15, 0.15)

        pitch_cmd = np.clip(
            -0.0022 * altitude_error_ft
            + 0.004 * v_rate
            + 0.70 * pitch_angle
            + 0.30 * pitch_rate,
            # + 0.00006 * self.altit÷=ude_error_integral,
            -1.0,
            1.0,
        )

        speed_error_fps = self.target_speed_fps - speed_fps
        throttle_cmd = np.clip(0.64 + 0.01 * speed_error_fps, 0.0, 1.0)
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


def build_altitude_hold_policy_jax(target_altitude_ft, target_speed_fps):
    """Build a JAX policy closure that mirrors the altitude hold controller."""
    target_altitude_ft = float(target_altitude_ft)
    target_speed_fps = float(target_speed_fps)
    
    def policy_fn(state_flat):
        h = state_flat[2]
        u = state_flat[3]
        v = state_flat[4]
        w = state_flat[5]
        roll_rate = state_flat[6]
        pitch_rate = state_flat[7]
        yaw_rate = state_flat[8]
        roll_angle = state_flat[9]
        pitch_angle = state_flat[10]

        speed_fps = jnp.sqrt(u**2 + v**2 + w**2)
        altitude_error_ft = target_altitude_ft - h
        
        roll_cmd = jnp.clip(-1.6 * roll_angle - 0.45 * roll_rate, -0.35, 0.35)
        yaw_cmd = jnp.clip(-0.25 * yaw_rate, -0.15, 0.15)
        v_rate = u * jnp.sin(pitch_angle) - v * jnp.sin(roll_angle) * jnp.cos(pitch_angle) - w * jnp.cos(roll_angle) * jnp.cos(pitch_angle)


        pitch_cmd = jnp.clip(
            -0.0022 * altitude_error_ft
            + 0.004 * v_rate
            + 0.70 * pitch_angle
            - 0.30 * pitch_rate,
            -1.0,
            1.0,
        )

        speed_error_fps = target_speed_fps - speed_fps
        throttle_cmd = jnp.clip(0.64 + 0.01 * speed_error_fps, 0.0, 1.0)
        return jnp.stack([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd])

    return policy_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Run canyon scenarios across available controllers.")
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi", "simple", "altitude_hold", "pid_traj"],
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
        "--gatekeeper",
        action="store_true",
        help=(
            "Wrap the nominal controller with the DRS gatekeeper and a conservative "
            "simple-controller backup (supported: mppi, smooth_mppi, pid_traj)."
        ),
    )
    parser.add_argument(
        "--gatekeeper-debug-timing",
        action="store_true",
        help="Print gatekeeper timing diagnostics each step.",
    )
    parser.add_argument(
        "--backup-controller",
        type=str,
        default="altitude_hold",
        choices=["simple", "altitude_hold"],
        help="Which controller to use as the gatekeeper backup.",
    )

    parser.add_argument(
        "--nominal-dyn-path",
        type=Path,
        default=None,
        help="Optional Aerosandbox dyn.asb path to use as the nominal reference trajectory.",
    )
    parser.add_argument(
        "--nominal-start-fraction",
        type=_fraction_0_to_1,
        default=0.0,
        help="Start at this fraction of nominal dyn trajectory progress in [0,1] (0=start, 1=end).",
    )
    parser.add_argument(
        "--nominal-end-fraction",
        type=_fraction_0_to_1,
        default=1.0,
        help="Truncate the nominal dyn trajectory at this fraction in [0,1] (1=full traj, 0.5=first half).",
    )
    return parser.parse_args()

def collect_pre_step_diagnostics(
    step,
    controller_tag,
    controller,
    gatekeeper_state,
    gatekeeper_bundle,
    controller_state,
    action,
    nominal_action,
    mppi_plan_call_index,
    env,
    mppi_plan_rows,
    mppi_plan_action_sequences,
    mppi_plan_virtual_speed_sequences,
):
    if controller_tag in {"mppi", "smooth_mppi"}:
        plan_debug = None
        plan_debug_getter = getattr(controller, "get_plan_debug", None)
        if callable(plan_debug_getter):
            try:
                plan_debug = dict(plan_debug_getter())
            except Exception:
                plan_debug = None
        if isinstance(plan_debug, dict):
            action_plan = np.asarray(plan_debug.get("action_plan", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
            virtual_speed_plan = np.asarray(
                plan_debug.get("virtual_speed_plan", np.zeros((0,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if (
                action_plan.ndim == 2
                and action_plan.shape[1] == 4
                and virtual_speed_plan.ndim == 1
                and action_plan.shape[0] == virtual_speed_plan.shape[0]
                and action_plan.shape[0] >= 1
            ):
                nominal_action_vec = np.asarray(nominal_action, dtype=np.float32)
                applied_action_vec = np.asarray(action, dtype=np.float32)
                mppi_plan_rows.append(
                    {
                        "call_index": int(mppi_plan_call_index),
                        "step": int(step),
                        "time_s": float(step / 30.0),
                        "progress_s_ft": float(plan_debug.get("progress_s_ft", np.nan)),
                        "controller_step_index": int(plan_debug.get("step_index", -1)),
                        "using_backup": bool(gatekeeper_state.using_backup) if gatekeeper_state is not None else False,
                        "nominal_aileron_cmd": float(nominal_action_vec[0]),
                        "nominal_elevator_cmd": float(nominal_action_vec[1]),
                        "nominal_rudder_cmd": float(nominal_action_vec[2]),
                        "nominal_throttle_cmd": float(nominal_action_vec[3]),
                        "applied_aileron_cmd": float(applied_action_vec[0]),
                        "applied_elevator_cmd": float(applied_action_vec[1]),
                        "applied_rudder_cmd": float(applied_action_vec[2]),
                        "applied_throttle_cmd": float(applied_action_vec[3]),
                    }
                )
                mppi_plan_action_sequences.append(action_plan.copy())
                mppi_plan_virtual_speed_sequences.append(virtual_speed_plan.copy())
                mppi_plan_call_index += 1

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
                "pid_error_xy": np.asarray(
                    raw_debug.get("pid_error_xy", np.zeros((0, 2), dtype=np.float32)),
                    dtype=np.float32,
                ).copy(),
                "pid_error_h_ft": np.asarray(
                    raw_debug.get("pid_error_h_ft", np.zeros((0,), dtype=np.float32)),
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
        lookahead_h_ft = float(controller_state["h"])
        planner_debug["lookahead_h_ft"] = np.asarray(
            [lookahead_h_ft],
            dtype=np.float32,
        )

    heading_deg = _wrap_heading_deg(controller_state["psi"])
    heading_cmd_deg = heading_deg
    if controller_tag == "simple":
        simple_guidance = dict(getattr(controller, "last_guidance", {}) or {})
        heading_cmd_deg = float(simple_guidance.get("centerline_heading_deg", heading_deg))
    else:
        get_hdg_cmd = getattr(controller, "get_reference_heading_rad", None)
        if callable(get_hdg_cmd):
            try:
                heading_cmd_deg = _wrap_heading_deg(get_hdg_cmd())
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

    invalid_action = False
    if np.isnan(action).any() or np.isinf(action).any():
        print(f"Invalid action at step {step}: {action}")
        invalid_action = True

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
    return mppi_plan_call_index, planner_debug, hud_debug, invalid_action

def collect_post_step_diagnostics(
    step,
    controller_tag,
    controller,
    gatekeeper_state,
    controller_state,
    post_controller_state,
    action,
    prev_applied_action,
    info,
    state,
    env,
    termination_reason,
    plan_ms,
    gk_ms,
    gk_nom_prep,
    gk_update_ms,
    gk_backup_ms,
    start_path_north_ft,
    mppi_tracking_rows,
    simple_diagnostic_rows,
    pid_traj_diagnostic_rows,
):
    width_ft = float(info.get("canyon_width_ft", np.nan))
    lateral_ft = float(info.get("lateral_error_ft", np.nan))
    speed_fps = float(
        np.sqrt(
            float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2
        )
    )
    if controller_tag in {"mppi", "smooth_mppi"}:
        mppi_cfg = getattr(controller, "config", None)
        if mppi_cfg is None:
            raise RuntimeError("MPPI controller missing config while collecting debug diagnostics.")
        tracking_metrics = dict(controller.get_tracking_metrics(post_controller_state))
        action_vec = np.asarray(action, dtype=np.float32)
        action_rate = action_vec - prev_applied_action

        progress_s_ft = float(tracking_metrics["progress_s_ft"])
        virtual_speed_fps = float(tracking_metrics["virtual_speed_fps"])
        contour_error_ft = float(tracking_metrics["contour_error_ft"])
        lag_error_ft = float(tracking_metrics["lag_error_ft"])
        position_error_ft = float(tracking_metrics["position_error_ft"])
        altitude_error_ft = float(tracking_metrics["altitude_error_ft"])
        contour_cost_est = float(tracking_metrics["contour_cost_est"])
        lag_cost_est = float(tracking_metrics["lag_cost_est"])
        progress_reward_est = float(tracking_metrics["progress_reward_est"])
        virtual_speed_cost_est = float(tracking_metrics["virtual_speed_cost_est"])
        contouring_cost_est = float(tracking_metrics["contouring_cost_est"])

        terrain_clearance_ft = float(info.get("terrain_clearance_ft", np.nan))
        terrain_safe_clearance_ft = float(mppi_cfg.terrain_safe_clearance_ft)
        if np.isfinite(terrain_clearance_ft):
            if terrain_clearance_ft <= 0.0:
                terrain_cost_est = float(mppi_cfg.terrain_collision_penalty)
            else:
                terrain_cost_est = float(
                    min(
                        float(mppi_cfg.terrain_collision_penalty),
                        float(mppi_cfg.terrain_repulsion_scale)
                        * np.exp(
                            -float(mppi_cfg.terrain_decay_rate_ft_inv)
                            * (terrain_clearance_ft - terrain_safe_clearance_ft)
                        ),
                    )
                )
        else:
            terrain_cost_est = np.nan

        rate_weights = np.asarray(mppi_cfg.control_rate_weights, dtype=np.float64)
        rate_cost_est = float(np.sum(rate_weights * np.square(action_rate)))

        alpha_rad = float(np.arctan2(float(post_controller_state["w"]), max(float(post_controller_state["u"]), 1.0)))
        alpha_deg = float(np.degrees(alpha_rad))
        alpha_limit_deg = float(np.degrees(float(mppi_cfg.alpha_limit_rad)))
        alpha_excess_rad = max(alpha_rad - float(mppi_cfg.alpha_limit_rad), 0.0)
        nz_g = float(post_controller_state.get("nz", 1.0))
        nz_excess_g = max(
            float(mppi_cfg.nz_min_g) - nz_g,
            nz_g - float(mppi_cfg.nz_max_g),
            0.0,
        )
        limit_cost_est = float(
            float(mppi_cfg.nz_penalty_weight) * (nz_excess_g ** 2)
            + float(mppi_cfg.alpha_penalty_weight) * (alpha_excess_rad ** 2)
        )
        total_stage_cost_est = float(
            contouring_cost_est + terrain_cost_est + rate_cost_est + limit_cost_est
        )
        rudder_pos_norm = np.nan
        rudder_pos_rad = np.nan
        sim = getattr(env.unwrapped, "simulation", None)
        if sim is not None:
            try:
                rudder_pos_norm = float(sim.get_property_value("fcs/rudder-pos-norm"))
            except Exception:
                pass
            try:
                rudder_pos_rad = float(sim.get_property_value("fcs/rudder-pos-rad"))
            except Exception:
                pass

        tracking_row = {
            "step": int(step),
            "time_s": float(step / 30.0),
            "progress_s_ft": float(progress_s_ft),
            "virtual_speed_fps": float(virtual_speed_fps),
            "reference_north_ft": float(tracking_metrics["reference_north_ft"]),
            "reference_east_ft": float(tracking_metrics["reference_east_ft"]),
            "reference_altitude_ft": float(tracking_metrics["reference_altitude_ft"]),
            "reference_heading_deg": float(np.degrees(float(tracking_metrics["reference_heading_rad"]))),
            "contour_error_ft": float(contour_error_ft),
            "lag_error_ft": float(lag_error_ft),
            "position_error_ft": float(position_error_ft),
            "altitude_error_ft": float(altitude_error_ft),
            "terrain_clearance_ft": float(terrain_clearance_ft),
            "terrain_safe_clearance_ft": float(terrain_safe_clearance_ft),
            "nz_g": float(nz_g),
            "nz_min_g": float(mppi_cfg.nz_min_g),
            "nz_max_g": float(mppi_cfg.nz_max_g),
            "alpha_deg": float(alpha_deg),
            "alpha_limit_deg": float(alpha_limit_deg),
            "contour_cost_est": float(contour_cost_est),
            "lag_cost_est": float(lag_cost_est),
            "progress_reward_est": float(progress_reward_est),
            "virtual_speed_cost_est": float(virtual_speed_cost_est),
            "contouring_cost_est": float(contouring_cost_est),
            "terrain_cost_est": float(terrain_cost_est),
            "rate_cost_est": float(rate_cost_est),
            "limit_cost_est": float(limit_cost_est),
            "total_stage_cost_est": float(total_stage_cost_est),
            "aileron_cmd": float(action_vec[0]),
            "elevator_cmd": float(action_vec[1]),
            "rudder_cmd": float(action_vec[2]),
            "throttle_cmd": float(action_vec[3]),
            "rudder_pos_norm": float(rudder_pos_norm),
            "rudder_pos_rad": float(rudder_pos_rad),
            "aileron_rate": float(action_rate[0]),
            "elevator_rate": float(action_rate[1]),
            "rudder_rate": float(action_rate[2]),
            "throttle_rate": float(action_rate[3]),
            "termination_reason": str(termination_reason),
            "using_backup": bool(gatekeeper_state.using_backup) if gatekeeper_state is not None else False,
            "plan_ms": float(plan_ms),
            "gatekeeper_ms": float(gk_ms),
            "gatekeeper_nominal_prep_ms": float(gk_nom_prep),
            "gatekeeper_update_ms": float(gk_update_ms),
            "gatekeeper_backup_ms": float(gk_backup_ms),
        }
        _append_state_fields(tracking_row, prefix="pre", state_dict=controller_state)
        _append_state_fields(tracking_row, prefix="post", state_dict=post_controller_state)
        mppi_tracking_rows.append(tracking_row)
        prev_applied_action = action_vec.copy()
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

    if controller_tag == "pid_traj":
        pid_diag = dict(getattr(controller, "last_diagnostics", {}) or {})
        pid_traj_diagnostic_rows.append(
            {
                "step": int(step),
                "time_s": float(step / 30.0),
                "e_xtrk": float(pid_diag.get("e_xtrk", np.nan)),
                "e_z": float(pid_diag.get("e_z", np.nan)),
                "phi_ref": float(pid_diag.get("phi_ref", np.nan)),
                "phi_cmd": float(pid_diag.get("phi_cmd", np.nan)),
                "phi": float(controller_state["phi"]),
                "alpha_ref": float(pid_diag.get("alpha_ref", np.nan)),
                "alpha_cmd": float(pid_diag.get("alpha_cmd", np.nan)),
                "alpha": float(controller_state["alpha"]),
                "p_cmd": float(pid_diag.get("p_cmd", np.nan)),
                "p": float(controller_state["p"]),
                "q_cmd": float(pid_diag.get("q_cmd", np.nan)),
                "q": float(controller_state["q"]),
                "elevator_cmd": float(pid_diag.get("elevator_cmd", np.nan)),
                "aileron_cmd": float(pid_diag.get("aileron_cmd", np.nan)),
                "rudder_cmd": float(pid_diag.get("rudder_cmd", np.nan)),
                "throttle_cmd": float(pid_diag.get("throttle_cmd", np.nan)),
                "v_opt_val": float(pid_diag.get("v_opt_val", np.nan)),
                "V": float(speed_fps),
            }
        )

    if step % 5 == 0:
        if controller_tag in {"mppi", "smooth_mppi"}:
            mode_label = "NOM"
            if gatekeeper_state is not None and bool(gatekeeper_state.using_backup):
                mode_label = "BACKUP"
            print(
                f"{step:<5} | {progress_s_ft:<8.0f} | {contour_error_ft:<8.1f} | {lag_error_ft:<8.1f} | "
                f"{position_error_ft:<8.1f} | {altitude_error_ft:<8.1f} | {terrain_clearance_ft:<7.1f} | "
                f"{total_stage_cost_est:<10.1f} | {mode_label:<6} | "
                f"{plan_ms:<8.1f} | {gk_ms:<8.1f}"
            )
        else:
            rel_north_ft = float(controller_state["p_N"] - start_path_north_ft)
            rel_alt_ft = float(controller_state["h"])
            print(
                f"{step:<5} | {rel_north_ft:<8.0f} | {lateral_ft:<8.0f} | {rel_alt_ft:<8.0f} | "
                f"{speed_fps:<6.0f} | {width_ft:<6.0f} | {plan_ms:<8.1f} | {gk_ms:<8.1f} | "
                f"{gk_nom_prep:<7.1f} | {gk_update_ms:<7.1f} | {gk_backup_ms:<7.1f}"
            )
    return prev_applied_action

def main():
    args = parse_args()
    if args.gatekeeper and args.controller not in {"mppi", "smooth_mppi", "pid_traj"}:
        raise ValueError("--gatekeeper currently requires --controller mppi, --controller smooth_mppi, or --controller pid_traj.")
    nominal_start_fraction = float(args.nominal_start_fraction)
    nominal_end_fraction = float(args.nominal_end_fraction)
    if args.nominal_dyn_path is None and nominal_start_fraction > 0.0:
        raise ValueError("--nominal-start-fraction requires --nominal-dyn-path.")
    if args.nominal_dyn_path is None and nominal_end_fraction < 1.0:
        raise ValueError("--nominal-end-fraction requires --nominal-dyn-path.")
    if nominal_end_fraction <= nominal_start_fraction:
        raise ValueError(
            f"--nominal-end-fraction ({nominal_end_fraction}) must be greater than "
            f"--nominal-start-fraction ({nominal_start_fraction})."
        )
    initial_speed_kts = float(DEFAULT_INITIAL_SPEED_KTS)
    initial_altitude_ft = float(DEFAULT_INITIAL_ALTITUDE_FT)
    dem_start_pixel = DEM_START_PIXEL
    initial_heading_deg = DEFAULT_INITIAL_HEADING_DEG
    initial_roll_deg = DEFAULT_INITIAL_ROLL_DEG
    initial_pitch_deg = DEFAULT_INITIAL_PITCH_DEG
    initial_alpha_deg = DEFAULT_INITIAL_ALPHA_DEG
    initial_beta_deg = DEFAULT_INITIAL_BETA_DEG

    if args.nominal_dyn_path is not None:
        nominal_canyon = DEMCanyon(
            dem_path=str(DEM_PATH),
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
        nominal_initial_conditions = load_nominal_initial_conditions_from_dyn(
            args.nominal_dyn_path,
            canyon=nominal_canyon,
            progress_fraction=nominal_start_fraction,
        )
        dem_start_pixel = tuple(nominal_initial_conditions["start_pixel"])
        initial_speed_kts = float(nominal_initial_conditions["speed_kts"])
        initial_altitude_ft = float(nominal_initial_conditions["entry_altitude_ft"])
        initial_heading_deg = float(nominal_initial_conditions["heading_deg"])
        initial_roll_deg = float(nominal_initial_conditions["roll_deg"])
        initial_pitch_deg = float(nominal_initial_conditions["pitch_deg"])
        initial_alpha_deg = float(nominal_initial_conditions["alpha_deg"])
        initial_beta_deg = float(nominal_initial_conditions["beta_deg"])
        print(
            "Using nominal offline initial conditions: "
            f"start_pixel={dem_start_pixel}, "
            f"speed={initial_speed_kts:.1f} kts, "
            f"altitude={initial_altitude_ft:.1f} ft AGL, "
            f"heading={initial_heading_deg:.1f} deg, "
            f"roll={initial_roll_deg:.1f} deg, "
            f"pitch={initial_pitch_deg:.1f} deg, "
            f"alpha={initial_alpha_deg:.1f} deg, "
            f"beta={initial_beta_deg:.1f} deg, "
            f"progress={float(nominal_initial_conditions['progress_fraction']):.3f} "
            f"(sample {int(nominal_initial_conditions['sample_index']) + 1}/"
            f"{int(nominal_initial_conditions['sample_count'])})."
        )

    output_subdirs = {
        "mppi": "canyon_mppi",
        "smooth_mppi": "canyon_smooth_mppi",
        "simple": "canyon_simple",
        "altitude_hold": "canyon_altitude_hold",
        "pid_traj": "canyon_pid_traj",
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
        dem_start_pixel=dem_start_pixel,
        dem_start_heading_mode="follow_canyon",
        dem_start_heading_deg=initial_heading_deg,
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
        entry_roll_deg=initial_roll_deg,
        entry_pitch_deg=initial_pitch_deg,
        entry_alpha_deg=initial_alpha_deg,
        entry_beta_deg=initial_beta_deg,
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

    canyon = env.unwrapped.canyon
    north_samples_ft, width_samples_ft, center_east_samples_ft, _ = get_active_canyon_reference(env)
    nominal_reference = None
    if args.nominal_dyn_path is not None:
        nominal_reference = build_nominal_reference_from_dyn(
            args.nominal_dyn_path,
            canyon=canyon,
            altitude_ref_ft=altitude_ref_ft,
            resample_spacing_ft=float(getattr(env.unwrapped, "canyon_segment_spacing_ft", 12.0)),
            end_fraction=nominal_end_fraction,
        )
        n_samples = len(np.asarray(nominal_reference['reference_states_ft_rad']))
        end_frac_str = f" (end={nominal_end_fraction:.2f})" if nominal_end_fraction < 1.0 else ""
        print(
            f"Loaded nominal dyn reference from {args.nominal_dyn_path} "
            f"({n_samples} samples{end_frac_str})."
        )

    config_base_kwargs = build_mppi_base_config_kwargs()
    if controller_tag == "mppi":
        config_base_kwargs, mppi_tuning_source, mppi_tuned_keys = with_default_mppi_optuna_params(config_base_kwargs)
        if mppi_tuned_keys:
            print(
                f"Auto-loaded MPPI tuned parameters from {mppi_tuning_source} "
                f"({len(mppi_tuned_keys)} parameters)."
            )
        else:
            print("Note: No tuned MPPI parameters found; using built-in defaults.")

    if controller_tag == "simple":
        simple_config = SimpleCanyonControllerConfig(
            target_speed_fps=float(initial_speed_kts) * KTS_TO_FPS,
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
        _simple_target_alt_rel = float(getattr(controller._core, "target_altitude_ft", float("nan")))
        _simple_target_alt_msl = _simple_target_alt_rel + altitude_ref_ft
        print(
            f"Simple controller target altitude: {_simple_target_alt_rel:.1f} ft (rel DEM origin) "
            f"/ {_simple_target_alt_msl:.1f} ft MSL"
        )
    elif controller_tag == "pid_traj":
        if nominal_reference is None:
            raise ValueError("pid_traj requires --nominal-dyn-path.")
        controller = PIDTrajectoryController(env=env, reference_trajectory=nominal_reference)
        controller.reset(initial_controller_state)
    elif controller_tag == "altitude_hold":
        controller = AltitudeHoldController()
    else:
        if nominal_reference is None:
            raise ValueError("MPPI controllers now require --nominal-dyn-path.")
        if not all(
            hasattr(canyon, attr)
            for attr in ("north_samples_ft", "east_samples_ft", "ordered_dem_msl_m")
        ):
            raise TypeError("MPPI controllers require a DEM canyon with a queryable terrain grid.")
        controller, config = build_mppi_controller(
            controller_tag,
            config_base_kwargs=config_base_kwargs,
            reference_trajectory=nominal_reference,
            terrain_north_samples_ft=np.asarray(canyon.north_samples_ft, dtype=np.float32),
            terrain_east_samples_ft=np.asarray(canyon.east_samples_ft, dtype=np.float32),
            terrain_elevation_ft=np.asarray(canyon.ordered_dem_msl_m, dtype=np.float32) * M_TO_FT - altitude_ref_ft,
        )
        _print_mppi_config(controller_tag, config)
        start_progress_s_ft = _set_mppi_nominal_start_progress(controller, nominal_start_fraction)
        if start_progress_s_ft is not None:
            print(
                "Initialized MPPI nominal progress at "
                f"{start_progress_s_ft:.1f} ft (fraction={nominal_start_fraction:.3f})."
            )

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
    if nominal_reference is not None:
        recorder.set_reference_profile(
            north_samples_ft=np.asarray(nominal_reference["display_north_ft"], dtype=np.float32),
            east_samples_ft=np.asarray(nominal_reference["display_east_ft"], dtype=np.float32),
            altitude_samples_ft=np.asarray(nominal_reference["display_altitude_ft"], dtype=np.float32),
            label="Nominal offline trajectory",
        )
    else:
        recorder.set_centerline_profile(north_samples_ft, center_east_samples_ft)
    termination_reason = "running"
    simple_diagnostic_rows = []
    pid_traj_diagnostic_rows = []
    mppi_tracking_rows = []
    mppi_plan_rows = []
    mppi_plan_action_sequences = []
    mppi_plan_virtual_speed_sequences = []
    mppi_plan_call_index = 0

    gatekeeper_bundle = None
    gatekeeper_prev_using_backup = False

    print(f"Initializing {controller_tag} canyon controller...")
    if controller_tag in {"mppi", "smooth_mppi"}:
        print("Compiling JAX JIT... (this takes a moment)", flush=True)
        _set_mppi_nominal_start_progress(controller, nominal_start_fraction)
        _ = controller.get_action(initial_controller_state)
        controller.reset(seed=3)
        _set_mppi_nominal_start_progress(controller, nominal_start_fraction)
        print("JIT compilation finished.", flush=True)
        if args.gatekeeper:
            if not UNCERTAINTY_ARTIFACT_PATH.exists():
                raise FileNotFoundError(f"Missing uncertainty artifact: {UNCERTAINTY_ARTIFACT_PATH}")
            gatekeeper_bundle = build_jsbsim_gatekeeper(
                env=env,
                initial_controller_state=initial_controller_state,
                nominal_horizon=int(controller.config.horizon),
                debug_timing=bool(args.gatekeeper_debug_timing),
                backup_controller_type=args.backup_controller,
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
            warmup_nominal_trajectory = _build_gatekeeper_nominal_trajectory(
                controller=controller,
                nominal_action=nominal_action,
                horizon=gatekeeper.params.T,
            )
            _ = gatekeeper.update(
                controller_state_to_gatekeeper_flat(initial_controller_state),
                track_bounds=None,
                nominal_trajectory=jnp.asarray(warmup_nominal_trajectory, dtype=jnp.float32),
                max_steps=int(args.max_steps),
            )
            gatekeeper.reset(controller_state_to_gatekeeper_flat(initial_controller_state), t=0)
            gatekeeper_prev_using_backup = False
            controller.reset(seed=3)
            _set_mppi_nominal_start_progress(controller, nominal_start_fraction)
            print("Gatekeeper JIT compilation finished.", flush=True)
    elif controller_tag == "pid_traj":
        if args.gatekeeper:
            if not UNCERTAINTY_ARTIFACT_PATH.exists():
                raise FileNotFoundError(f"Missing uncertainty artifact: {UNCERTAINTY_ARTIFACT_PATH}")
            pid_nominal_policy_fn = build_pid_trajectory_policy_jax(
                reference_trajectory=nominal_reference,
                config=getattr(controller, "config", None),
            )
            gatekeeper_bundle = build_jsbsim_gatekeeper(
                env=env,
                initial_controller_state=initial_controller_state,
                nominal_horizon=100,
                debug_timing=bool(args.gatekeeper_debug_timing),
                nominal_policy_fn_override=pid_nominal_policy_fn,
                backup_controller_type=args.backup_controller,
            )
            print(
                f"Initialized gatekeeper with backup simple controller at {gatekeeper_bundle['backup_target_speed_fps'] / KTS_TO_FPS:.0f} kts and "
                f"target altitude {gatekeeper_bundle['backup_target_altitude_ft']:.0f} ft above DEM origin."
            )
            print("Compiling PID policy JAX JIT... (this takes a moment)", flush=True)
            # Warm up the JIT-compiled nominal policy in isolation first.
            _warmup_state = controller_state_to_gatekeeper_flat(initial_controller_state)
            _ = pid_nominal_policy_fn(_warmup_state)
            _.block_until_ready()
            print("Compiling Gatekeeper JAX JIT... (this takes a moment)", flush=True)
            gatekeeper = gatekeeper_bundle["gatekeeper"]
            _ = gatekeeper.update(
                controller_state_to_gatekeeper_flat(initial_controller_state),
                track_bounds=None,
                nominal_trajectory=None,
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
    if controller_tag in {"mppi", "smooth_mppi"}:
        print(
            f"{'Step':<5} | {'Prog':<8} | {'CErr':<8} | {'Lag':<8} | {'PosErr':<8} | {'dH':<8} | "
            f"{'Clr':<7} | {'Cost':<10} | {'Mode':<6} | {'Plan(ms)':<8} | {'gk(ms)':<8}"
        )
        print("-" * 117)
    else:
        print(
            f"{'Step':<5} | {'p_N_rel':<8} | {'LatErr':<8} | {'h_rel':<8} | "
            f"{'V':<6} | {'W_c':<6} | {'Plan(ms)':<8} | {'gk(ms)':<8} | {'gk_ws':<7} | {'gk_upd':<7} | {'gk_bak':<7}"
        )
        print("-" * 110)

    # MAIN LOOP
    try:
        prev_applied_action = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
        for step in range(int(args.max_steps)):
            controller_state = to_mppi_state(env, state, altitude_ref_ft)

            t0 = time.time()
            # if controller_tag == "altitude_hold":
            #     nominal_action = controller.get_action(obs)
            #     action = nominal_action
            # else:
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
                if controller_tag in {"mppi", "smooth_mppi"}:
                    latest_nominal["action"] = jnp.asarray(np.asarray(nominal_action, dtype=np.float32), dtype=jnp.float32)

                t_ws = time.time()
                if controller_tag in {"mppi", "smooth_mppi"}:
                    nominal_trajectory = _build_gatekeeper_nominal_trajectory(
                        controller=controller,
                        nominal_action=nominal_action,
                        horizon=gatekeeper.params.T,
                    )
                    nominal_trajectory_jax = jnp.asarray(nominal_trajectory, dtype=jnp.float32)
                else:
                    nominal_trajectory_jax = None
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

            mppi_plan_call_index, planner_debug, hud_debug, invalid_action = collect_pre_step_diagnostics(
                step=step,
                controller_tag=controller_tag,
                controller=controller,
                gatekeeper_state=gatekeeper_state,
                gatekeeper_bundle=gatekeeper_bundle,
                controller_state=controller_state,
                action=action,
                nominal_action=nominal_action,
                mppi_plan_call_index=mppi_plan_call_index,
                env=env,
                mppi_plan_rows=mppi_plan_rows,
                mppi_plan_action_sequences=mppi_plan_action_sequences,
                mppi_plan_virtual_speed_sequences=mppi_plan_virtual_speed_sequences,
            )
            if invalid_action:
                break

            obs, _, terminated, truncated, info = env.step(action)
            termination_reason = info.get("termination_reason", "running")
            state = env.unwrapped.get_full_state_dict()
            post_controller_state = to_mppi_state(env, state, altitude_ref_ft)
            recorder.record_step(planner_debug=planner_debug, hud_debug=hud_debug)

            prev_applied_action = collect_post_step_diagnostics(
                step=step,
                controller_tag=controller_tag,
                controller=controller,
                gatekeeper_state=gatekeeper_state,
                controller_state=controller_state,
                post_controller_state=post_controller_state,
                action=action,
                prev_applied_action=prev_applied_action,
                info=info,
                state=state,
                env=env,
                termination_reason=termination_reason,
                plan_ms=plan_ms,
                gk_ms=gk_ms,
                gk_nom_prep=gk_nom_prep,
                gk_update_ms=gk_update_ms,
                gk_backup_ms=gk_backup_ms,
                start_path_north_ft=start_path_north_ft,
                mppi_tracking_rows=mppi_tracking_rows,
                simple_diagnostic_rows=simple_diagnostic_rows,
                pid_traj_diagnostic_rows=pid_traj_diagnostic_rows,
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
    if controller_tag == "pid_traj":
        diag_csv_path, diag_plot_path = save_pid_traj_diagnostics(
            output_dir=output_dir,
            file_stem=f"canyon_{controller_tag}_{mode_tag}",
            rows=pid_traj_diagnostic_rows,
            termination_reason=termination_reason,
        )
        if diag_csv_path is not None and diag_plot_path is not None:
            print(f"Saved pid_traj diagnostics CSV: {diag_csv_path}")
            print(f"Saved pid_traj diagnostics plot: {diag_plot_path}")
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
        plan_diag_csv_path, plan_diag_npz_path = save_mppi_plan_diagnostics(
            output_dir=output_dir,
            file_stem=f"canyon_{controller_tag}_{mode_tag}",
            rows=mppi_plan_rows,
            action_plans=mppi_plan_action_sequences,
            virtual_speed_plans=mppi_plan_virtual_speed_sequences,
        )
        if plan_diag_csv_path is not None and plan_diag_npz_path is not None:
            print(f"Saved MPPI plan diagnostics CSV: {plan_diag_csv_path}")
            print(f"Saved MPPI plan diagnostics NPZ: {plan_diag_npz_path}")
        if mppi_tracking_rows:
            contour_error_ft = np.asarray(
                [float(row.get("contour_error_ft", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            lag_error_ft = np.asarray(
                [float(row.get("lag_error_ft", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            position_error_ft = np.asarray(
                [float(row.get("position_error_ft", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            terrain_clearance_ft = np.asarray(
                [float(row.get("terrain_clearance_ft", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            total_stage_cost = np.asarray(
                [float(row.get("total_stage_cost_est", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            limit_cost = np.asarray(
                [float(row.get("limit_cost_est", np.nan)) for row in mppi_tracking_rows],
                dtype=np.float64,
            )
            print(
                "MPPI debug summary: "
                f"mean(contour_error_ft)={np.nanmean(contour_error_ft):.1f}, "
                f"mean(|lag_error_ft|)={np.nanmean(np.abs(lag_error_ft)):.1f}, "
                f"mean(position_error_ft)={np.nanmean(position_error_ft):.1f}, "
                f"max(position_error_ft)={np.nanmax(position_error_ft):.1f}, "
                f"min(terrain_clearance_ft)={np.nanmin(terrain_clearance_ft):.1f}, "
                f"mean(limit_cost)={np.nanmean(limit_cost):.1f}, "
                f"mean(total_stage_cost)={np.nanmean(total_stage_cost):.1f}"
            )


if __name__ == "__main__":
    main()
