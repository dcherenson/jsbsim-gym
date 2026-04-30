from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym import _mppi_backend as backend


Array = jax.Array

DEFAULT_ACTION_LOW = (-1.0, -1.0, -1.0, 0.0)
DEFAULT_ACTION_HIGH = (1.0, 1.0, 1.0, 1.0)
DEFAULT_CONTOUR_WEIGHT = 1.0
DEFAULT_LAG_WEIGHT = 0.05
DEFAULT_PROGRESS_REWARD_WEIGHT = 25.0
DEFAULT_VIRTUAL_SPEED_WEIGHT = 0.015
DEFAULT_TERRAIN_SAFE_CLEARANCE_FT = 40.0 * 3.28084
DEFAULT_CONTROL_RATE_WEIGHTS = (15.0, 20.0, 5.0, 2.0)
DEFAULT_ALPHA_LIMIT_RAD = float(np.deg2rad(25.0))
DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS = 120.0
DEFAULT_VIRTUAL_SPEED_MIN_FPS = 0.0
DEFAULT_VIRTUAL_SPEED_MAX_FPS = 1200.0
DEFAULT_VIRTUAL_SPEED_TRIM_FPS = 800.0


@dataclass(slots=True)
class MPPICostConfig:
    horizon: int = 24
    lambda_: float = 1.0
    gamma_: float = 0.05
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH
    contour_weight: float = DEFAULT_CONTOUR_WEIGHT
    lag_weight: float = DEFAULT_LAG_WEIGHT
    progress_reward_weight: float = DEFAULT_PROGRESS_REWARD_WEIGHT
    virtual_speed_weight: float = DEFAULT_VIRTUAL_SPEED_WEIGHT
    terrain_collision_penalty: float = 1.0e6
    terrain_repulsion_scale: float = 1.0e5
    terrain_decay_rate_ft_inv: float = 0.03
    terrain_safe_clearance_ft: float = DEFAULT_TERRAIN_SAFE_CLEARANCE_FT
    control_rate_weights: tuple[float, float, float, float] = DEFAULT_CONTROL_RATE_WEIGHTS
    nz_limit_g: float = 9.0
    nz_penalty_weight: float = 1.0e4
    alpha_limit_rad: float = DEFAULT_ALPHA_LIMIT_RAD
    alpha_penalty_weight: float = 1.0e4


@dataclass(slots=True)
class NominalJSBSimParams:
    W: Array
    B: Array
    poly_powers: Array
    reference_states_ft_rad: Array
    path_s_ft: Array
    path_positions_ft: Array
    path_tangents_ft: Array
    terrain_north_samples_ft: Array
    terrain_east_samples_ft: Array
    terrain_elevation_ft: Array
    reference_states_np: np.ndarray
    reference_speed_fps_np: np.ndarray
    path_s_np: np.ndarray
    path_positions_np: np.ndarray
    path_tangents_np: np.ndarray
    terrain_north_np: np.ndarray
    terrain_east_np: np.ndarray
    terrain_elevation_ft_np: np.ndarray


load_nominal_weights = backend.load_nominal_weights
f16_kinematics_step = backend.f16_kinematics_step
f16_kinematics_step_with_load_factors = backend.f16_kinematics_step_with_load_factors
smooth_noise_batch = backend.smooth_noise_batch
softmax_weights = backend.softmax_weights


def clip_action(
    action: Array,
    low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW,
    high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH,
) -> Array:
    return backend.clip_action(
        action,
        jnp.asarray(low, dtype=jnp.float32),
        jnp.asarray(high, dtype=jnp.float32),
    )


def make_trim_action_plan(
    horizon: int,
    *,
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW,
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH,
    throttle_cmd: float = 0.55,
) -> np.ndarray:
    plan = np.zeros((int(max(horizon, 1)), 4), dtype=np.float32)
    throttle = float(np.clip(throttle_cmd, action_low[3], action_high[3]))
    plan[:, 3] = throttle
    return plan


def make_trim_virtual_speed_plan(
    horizon: int,
    *,
    trim_speed_fps: float = DEFAULT_VIRTUAL_SPEED_TRIM_FPS,
) -> np.ndarray:
    return np.full((int(max(horizon, 1)),), float(trim_speed_fps), dtype=np.float32)


def jsbsim_state_to_jax_with_load_factors(state_dict: dict[str, float]) -> Array:
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


def _build_spatial_path(reference_states_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(reference_states_np[:, :3], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Reference positions must have shape (N, 3).")

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    keep_mask = np.concatenate([[True], segment_lengths > 1e-3])
    positions = positions[keep_mask]
    if positions.shape[0] < 2:
        raise ValueError("Reference path collapsed after removing duplicate position samples.")

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_s_ft = np.concatenate([[0.0], np.cumsum(segment_lengths, dtype=np.float64)]).astype(np.float32)
    if float(path_s_ft[-1]) <= 1e-3:
        raise ValueError("Reference path has zero spatial arc length.")

    tangents = np.zeros_like(positions, dtype=np.float32)
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]
    if positions.shape[0] > 2:
        tangents[1:-1] = positions[2:] - positions[:-2]
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tangent_norms, 1e-6)
    return path_s_ft, positions.astype(np.float32), tangents.astype(np.float32)


def build_nominal_params(
    *,
    reference_trajectory: dict[str, np.ndarray],
    terrain_north_samples_ft: np.ndarray,
    terrain_east_samples_ft: np.ndarray,
    terrain_elevation_ft: np.ndarray,
) -> NominalJSBSimParams:
    W, B, poly_powers = load_nominal_weights()

    reference_states_np = np.asarray(reference_trajectory["reference_states_ft_rad"], dtype=np.float32)
    if reference_states_np.ndim != 2 or reference_states_np.shape[1] != 6:
        raise ValueError("reference_trajectory['reference_states_ft_rad'] must have shape (N, 6).")
    if reference_states_np.shape[0] < 2:
        raise ValueError("Reference trajectory must contain at least two samples.")
    if not np.all(np.isfinite(reference_states_np)):
        raise ValueError("Reference trajectory contains non-finite values.")

    reference_speed_fps_np = np.asarray(reference_trajectory.get("speed_fps", []), dtype=np.float32).reshape(-1)
    if reference_speed_fps_np.size == 0:
        reference_speed_fps_np = np.ones((reference_states_np.shape[0],), dtype=np.float32) * DEFAULT_VIRTUAL_SPEED_TRIM_FPS
    if reference_speed_fps_np.shape[0] != reference_states_np.shape[0]:
        raise ValueError("reference_trajectory['speed_fps'] must match the reference state sample count.")

    path_s_np, path_positions_np, path_tangents_np = _build_spatial_path(reference_states_np)

    terrain_north_np = np.asarray(terrain_north_samples_ft, dtype=np.float32).reshape(-1)
    terrain_east_np = np.asarray(terrain_east_samples_ft, dtype=np.float32).reshape(-1)
    terrain_elevation_np = np.asarray(terrain_elevation_ft, dtype=np.float32)
    if terrain_north_np.ndim != 1 or terrain_east_np.ndim != 1:
        raise ValueError("Terrain north/east samples must be 1D arrays.")
    if terrain_north_np.size < 2 or terrain_east_np.size < 2:
        raise ValueError("Terrain grid axes must each contain at least two samples.")
    if terrain_elevation_np.shape != (terrain_north_np.size, terrain_east_np.size):
        raise ValueError("Terrain elevation grid shape must match the north/east sample axes.")
    if not np.all(np.isfinite(terrain_elevation_np)):
        raise ValueError("Terrain elevation grid contains non-finite values.")

    return NominalJSBSimParams(
        W=jnp.asarray(W, dtype=jnp.float32),
        B=jnp.asarray(B, dtype=jnp.float32),
        poly_powers=jnp.asarray(poly_powers, dtype=jnp.int32),
        reference_states_ft_rad=jnp.asarray(reference_states_np, dtype=jnp.float32),
        path_s_ft=jnp.asarray(path_s_np, dtype=jnp.float32),
        path_positions_ft=jnp.asarray(path_positions_np, dtype=jnp.float32),
        path_tangents_ft=jnp.asarray(path_tangents_np, dtype=jnp.float32),
        terrain_north_samples_ft=jnp.asarray(terrain_north_np, dtype=jnp.float32),
        terrain_east_samples_ft=jnp.asarray(terrain_east_np, dtype=jnp.float32),
        terrain_elevation_ft=jnp.asarray(terrain_elevation_np, dtype=jnp.float32),
        reference_states_np=reference_states_np,
        reference_speed_fps_np=reference_speed_fps_np,
        path_s_np=path_s_np,
        path_positions_np=path_positions_np,
        path_tangents_np=path_tangents_np,
        terrain_north_np=terrain_north_np,
        terrain_east_np=terrain_east_np,
        terrain_elevation_ft_np=terrain_elevation_np,
    )


def reference_state_for_index(params: NominalJSBSimParams, index: int) -> np.ndarray:
    clamped_index = int(np.clip(int(index), 0, params.reference_states_np.shape[0] - 1))
    return np.asarray(params.reference_states_np[clamped_index], dtype=np.float32)


def reference_heading_for_index(params: NominalJSBSimParams, index: int) -> float:
    return float(reference_state_for_index(params, index)[5])


def _interp_path_position_and_tangent_np(
    params: NominalJSBSimParams,
    progress_s_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(progress_s_ft, float(params.path_s_np[0]), float(params.path_s_np[-1])))
    idx_hi = int(np.clip(np.searchsorted(params.path_s_np, s, side="right"), 1, params.path_s_np.shape[0] - 1))
    idx_lo = idx_hi - 1
    s0 = float(params.path_s_np[idx_lo])
    s1 = float(params.path_s_np[idx_hi])
    tau = 0.0 if abs(s1 - s0) <= 1e-6 else float(np.clip((s - s0) / (s1 - s0), 0.0, 1.0))

    pos = (1.0 - tau) * params.path_positions_np[idx_lo] + tau * params.path_positions_np[idx_hi]
    tangent = (1.0 - tau) * params.path_tangents_np[idx_lo] + tau * params.path_tangents_np[idx_hi]
    tangent_norm = float(np.linalg.norm(tangent))
    tangent = tangent / max(tangent_norm, 1e-6)
    return np.asarray(pos, dtype=np.float32), np.asarray(tangent, dtype=np.float32)


def contouring_reference_for_progress(
    params: NominalJSBSimParams,
    progress_s_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _interp_path_position_and_tangent_np(params, progress_s_ft)


def reference_state_for_progress(params: NominalJSBSimParams, progress_s_ft: float) -> np.ndarray:
    position_ft, tangent = contouring_reference_for_progress(params, progress_s_ft)
    heading_rad = float(np.arctan2(float(tangent[1]), float(tangent[0])))
    pitch_rad = float(np.arctan2(float(tangent[2]), max(float(np.hypot(tangent[0], tangent[1])), 1e-6)))
    return np.asarray(
        [
            float(position_ft[0]),
            float(position_ft[1]),
            float(position_ft[2]),
            0.0,
            float(pitch_rad),
            float(heading_rad),
        ],
        dtype=np.float32,
    )


def reference_heading_for_progress(params: NominalJSBSimParams, progress_s_ft: float) -> float:
    _, tangent = contouring_reference_for_progress(params, progress_s_ft)
    return float(np.arctan2(float(tangent[1]), float(tangent[0])))


def _backend_cost_config(cost_config: MPPICostConfig) -> backend.JaxMPPIConfig:
    return backend.JaxMPPIConfig(
        horizon=int(cost_config.horizon),
        num_samples=1,
        optimization_steps=1,
        replan_interval=1,
        lambda_=float(cost_config.lambda_),
        gamma_=float(cost_config.gamma_),
        action_noise_std=(0.0, 0.0, 0.0, 0.0),
        action_low=tuple(cost_config.action_low),
        action_high=tuple(cost_config.action_high),
        contour_weight=float(cost_config.contour_weight),
        lag_weight=float(cost_config.lag_weight),
        progress_reward_weight=float(cost_config.progress_reward_weight),
        virtual_speed_weight=float(cost_config.virtual_speed_weight),
        terrain_collision_penalty=float(cost_config.terrain_collision_penalty),
        terrain_repulsion_scale=float(cost_config.terrain_repulsion_scale),
        terrain_decay_rate_ft_inv=float(cost_config.terrain_decay_rate_ft_inv),
        terrain_safe_clearance_ft=float(cost_config.terrain_safe_clearance_ft),
        control_rate_weights=tuple(float(x) for x in cost_config.control_rate_weights),
        nz_limit_g=float(cost_config.nz_limit_g),
        nz_penalty_weight=float(cost_config.nz_penalty_weight),
        alpha_limit_rad=float(cost_config.alpha_limit_rad),
        alpha_penalty_weight=float(cost_config.alpha_penalty_weight),
        debug_render_plans=False,
        debug_num_trajectories=0,
        seed=0,
    )


def build_rollout_cost_fn(params: NominalJSBSimParams, cost_config: MPPICostConfig) -> Any:
    backend_config = _backend_cost_config(cost_config)
    rollout_states = backend.build_rollout_state_batch_fn(params.W, params.B, params.poly_powers)
    rollout_costs_from_states = backend.build_rollout_cost_from_states_fn(
        params.path_s_ft,
        params.path_positions_ft,
        params.path_tangents_ft,
        params.terrain_north_samples_ft,
        params.terrain_east_samples_ft,
        params.terrain_elevation_ft,
        backend_config,
    )

    def rollout_costs(
        initial_state: Array,
        action_batch: Array,
        virtual_speed_batch: Array,
        initial_prev_action: Array,
        initial_progress_s_ft: float,
    ) -> Array:
        state_batch = rollout_states(initial_state, action_batch)
        return rollout_costs_from_states(
            initial_state,
            state_batch,
            action_batch,
            virtual_speed_batch,
            initial_prev_action,
            jnp.asarray(initial_progress_s_ft, dtype=jnp.float32),
        )

    return rollout_costs


def build_rollout_positions_fn(params: NominalJSBSimParams) -> Any:
    return backend.build_rollout_positions_fn(params.W, params.B, params.poly_powers)
