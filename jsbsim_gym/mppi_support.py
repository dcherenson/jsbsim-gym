from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym import _mppi_backend as backend
from jsbsim_gym.simple_controller import (
    SimpleCanyonControllerConfig,
    SimpleTrajectoryController,
    build_reference_trajectory,
    with_default_simple_controller_optuna_gains,
)


Array = jax.Array

DEFAULT_ACTION_LOW = (-1.0, -1.0, -1.0, 0.0)
DEFAULT_ACTION_HIGH = (1.0, 1.0, 1.0, 1.0)


@dataclass(slots=True)
class MPPICostConfig:
    horizon: int = 24
    lambda_: float = 2.0
    gamma_: float = 0.015
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH
    progress_gain: float = 0.70
    speed_gain: float = 0.35
    low_altitude_gain: float = 0.45
    altitude_target_gain: float = 0.0
    altitude_target_scale_ft: float = 250.0
    centerline_gain: float = 0.60
    offcenter_penalty_gain: float = 0.30
    heading_alignment_gain: float = 0.45
    heading_alignment_scale_rad: float = 0.70
    alive_bonus: float = 0.15
    target_speed_fps: float = 800.0
    speed_target_gain: float = 0.0
    speed_target_scale_fps: float = 120.0
    target_altitude_ft: float = 250.0
    min_altitude_ft: float = -500.0
    max_altitude_ft: float = 3000.0
    terrain_collision_height_ft: float = 60.0
    wall_margin_ft: float = 30.0
    terrain_crash_penalty: float = 250.0
    wall_crash_penalty: float = 18.0
    altitude_violation_penalty: float = 8.0
    early_termination_penalty_gain: float = 80.0
    time_limit_bonus: float = 25.0
    max_step_reward_abs: float = 15.0
    angular_rate_penalty_gain: float = 0.45
    angular_rate_threshold_deg_s: float = 45.0
    bank_angle_penalty_gain: float = 0.0
    bank_angle_threshold_deg: float = 85.0
    action_diff_weight: float = 2.5
    action_l2_weight: float = 0.4


@dataclass(slots=True)
class NominalJSBSimParams:
    W: Array
    B: Array
    poly_powers: Array
    canyon_north_samples_ft: Array
    canyon_width_samples_ft: Array
    canyon_center_east_samples_ft: Array
    canyon_heading_samples_rad: Array
    reference_altitude_samples_ft: Array | None
    reference_speed_samples_fps: Array | None
    canyon_north_np: np.ndarray
    canyon_width_np: np.ndarray
    canyon_center_east_np: np.ndarray
    canyon_heading_rad_np: np.ndarray
    reference_altitude_ft_np: np.ndarray | None
    reference_speed_fps_np: np.ndarray | None


@dataclass(slots=True)
class F16CenterlineDriver:
    target_speed_fps: float = 450.0
    target_altitude_ft: float = 250.0
    wall_margin_ft: float = 0.0
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH
    controller_config: SimpleCanyonControllerConfig | None = None
    _reference_trajectory: dict[str, np.ndarray] | None = None
    _reference_params_id: int | None = None
    _tail_controller: SimpleTrajectoryController | None = None

    def act(self, state: dict[str, float], params: NominalJSBSimParams) -> np.ndarray:
        controller = self._ensure_tail_controller(state, params)
        controller.target_altitude_ft = self._target_altitude_ft_for_p_n(float(state["p_N"]), params)
        controller.target_speed_fps = self._target_speed_fps_for_p_n(float(state["p_N"]), params)
        action = controller.get_action(self._state_for_controller(state))
        return np.asarray(
            clip_action(action, low=self.action_low, high=self.action_high),
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._tail_controller = None

    def rollout_plan(
        self,
        *,
        state: dict[str, float],
        params: NominalJSBSimParams,
        horizon: int,
    ) -> np.ndarray:
        controller = self._make_controller(params)
        simulated = np.asarray(
            jsbsim_state_to_jax_with_load_factors(self._state_for_controller(state)),
            dtype=np.float32,
        )
        initial_p_n_ft = float(simulated[0])
        controller.reset(
            state_dict=_flat_state_to_dict(simulated),
            target_altitude_ft=self._target_altitude_ft_for_p_n(initial_p_n_ft, params),
        )
        controller.target_speed_fps = self._target_speed_fps_for_p_n(initial_p_n_ft, params)

        actions: list[np.ndarray] = []
        for _ in range(int(horizon)):
            sim_dict = _flat_state_to_dict(simulated)
            controller.target_altitude_ft = self._target_altitude_ft_for_p_n(float(simulated[0]), params)
            controller.target_speed_fps = self._target_speed_fps_for_p_n(float(simulated[0]), params)
            action = np.asarray(controller.get_action(sim_dict), dtype=np.float32)
            action = np.asarray(
                clip_action(action, low=self.action_low, high=self.action_high),
                dtype=np.float32,
            )
            actions.append(action)
            simulated = np.asarray(
                f16_kinematics_step_with_load_factors(
                    jnp.asarray(simulated, dtype=jnp.float32),
                    jnp.asarray(action, dtype=jnp.float32),
                    params.W,
                    params.B,
                    params.poly_powers,
                ),
                dtype=np.float32,
            )
        return np.asarray(actions, dtype=np.float32)

    def _target_altitude_ft_for_p_n(self, p_n_ft: float, params: NominalJSBSimParams) -> float:
        if params.reference_altitude_ft_np is None:
            return float(self.target_altitude_ft)
        return float(np.interp(float(p_n_ft), params.canyon_north_np, params.reference_altitude_ft_np))

    def _target_speed_fps_for_p_n(self, p_n_ft: float, params: NominalJSBSimParams) -> float:
        if params.reference_speed_fps_np is None:
            return float(self.target_speed_fps)
        return float(np.interp(float(p_n_ft), params.canyon_north_np, params.reference_speed_fps_np))

    def _state_for_controller(self, state: dict[str, float]) -> dict[str, float]:
        state_dict = dict(state)
        state_dict.setdefault("ny", 0.0)
        state_dict.setdefault("nz", 1.0)
        state_dict.setdefault("beta", 0.0)
        return state_dict

    def _ensure_reference_trajectory(self, params: NominalJSBSimParams) -> dict[str, np.ndarray]:
        params_id = id(params)
        if self._reference_trajectory is None or self._reference_params_id != params_id:
            self._reference_trajectory = build_reference_trajectory(
                north_ft=params.canyon_north_np,
                east_ft=params.canyon_center_east_np,
                heading_rad=params.canyon_heading_rad_np,
                width_ft=params.canyon_width_np,
                closed_loop=False,
            )
            self._reference_params_id = params_id
            self._tail_controller = None
        return self._reference_trajectory

    def _make_controller(self, params: NominalJSBSimParams) -> SimpleTrajectoryController:
        return SimpleTrajectoryController(
            config=self.controller_config if self.controller_config is not None else SimpleCanyonControllerConfig(),
            target_altitude_ft=self.target_altitude_ft,
            wall_margin_ft=self.wall_margin_ft,
            reference_trajectory=self._ensure_reference_trajectory(params),
        )

    def _ensure_tail_controller(
        self,
        state: dict[str, float],
        params: NominalJSBSimParams,
    ) -> SimpleTrajectoryController:
        state_dict = self._state_for_controller(state)
        if self._tail_controller is None:
            self._tail_controller = self._make_controller(params)
            self._tail_controller.reset(
                state_dict=state_dict,
                target_altitude_ft=self.target_altitude_ft,
            )
        return self._tail_controller


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


def clip_delta_action(delta: Array, bounds: Array) -> Array:
    return jnp.clip(delta, -bounds, bounds)


def build_heuristic_driver(
    *,
    target_speed_fps: float = 450.0,
    target_altitude_ft: float = 250.0,
    wall_margin_ft: float = 0.0,
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW,
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH,
) -> F16CenterlineDriver:
    config, _, _ = with_default_simple_controller_optuna_gains(
        SimpleCanyonControllerConfig(target_speed_fps=float(target_speed_fps))
    )
    config = replace(config, target_speed_fps=float(target_speed_fps))
    return F16CenterlineDriver(
        target_speed_fps=target_speed_fps,
        target_altitude_ft=target_altitude_ft,
        wall_margin_ft=wall_margin_ft,
        action_low=action_low,
        action_high=action_high,
        controller_config=config,
    )
def jsbsim_state_to_jax(state_dict: dict[str, float]) -> Array:
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
        ],
        dtype=jnp.float32,
    )


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


def _flat_state_to_dict(state_flat: np.ndarray) -> dict[str, float]:
    state_dict = {
        "p_N": float(state_flat[0]),
        "p_E": float(state_flat[1]),
        "h": float(state_flat[2]),
        "u": float(state_flat[3]),
        "v": float(state_flat[4]),
        "w": float(state_flat[5]),
        "p": float(state_flat[6]),
        "q": float(state_flat[7]),
        "r": float(state_flat[8]),
        "phi": float(state_flat[9]),
        "theta": float(state_flat[10]),
        "psi": float(state_flat[11]),
    }
    if len(state_flat) >= 13:
        state_dict["ny"] = float(state_flat[12])
    if len(state_flat) >= 14:
        state_dict["nz"] = float(state_flat[13])
    return state_dict


def build_nominal_params(
    canyon_north_samples_ft: np.ndarray | None = None,
    canyon_width_samples_ft: np.ndarray | None = None,
    canyon_center_east_samples_ft: np.ndarray | None = None,
    canyon_centerline_heading_rad_samples: np.ndarray | None = None,
    reference_altitude_samples_ft: np.ndarray | None = None,
    reference_speed_samples_fps: np.ndarray | None = None,
) -> NominalJSBSimParams:
    W, B, poly_powers = load_nominal_weights()

    if canyon_north_samples_ft is None or canyon_width_samples_ft is None:
        north_np = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
        width_np = np.asarray(backend.canyon_width(jnp.asarray(north_np)), dtype=np.float32)
        center_east_np = np.zeros_like(north_np, dtype=np.float32)
        heading_np = np.zeros_like(north_np, dtype=np.float32)
    else:
        north_np = np.asarray(canyon_north_samples_ft, dtype=np.float32).reshape(-1)
        width_np = np.asarray(canyon_width_samples_ft, dtype=np.float32).reshape(-1)
        if north_np.size != width_np.size:
            raise ValueError("canyon_north_samples_ft and canyon_width_samples_ft must have same length")
        if north_np.size < 2:
            raise ValueError("canyon profile arrays must contain at least two samples")

        if canyon_center_east_samples_ft is None:
            center_east_np = np.zeros_like(north_np, dtype=np.float32)
        else:
            center_east_np = np.asarray(canyon_center_east_samples_ft, dtype=np.float32).reshape(-1)
            if center_east_np.size != north_np.size:
                raise ValueError("canyon_center_east_samples_ft must have same length as canyon_north_samples_ft")

        if canyon_centerline_heading_rad_samples is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.gradient(center_east_np, north_np, edge_order=1)
            slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
            heading_np = np.arctan(slope).astype(np.float32)
        else:
            heading_np = np.asarray(canyon_centerline_heading_rad_samples, dtype=np.float32).reshape(-1)
            if heading_np.size != north_np.size:
                raise ValueError(
                    "canyon_centerline_heading_rad_samples must have same length as canyon_north_samples_ft"
                )

        order = np.argsort(north_np)
        north_np = north_np[order]
        width_np = width_np[order]
        center_east_np = center_east_np[order]
        heading_np = heading_np[order]

    reference_altitude_np = None
    if reference_altitude_samples_ft is not None:
        reference_altitude_np = np.asarray(reference_altitude_samples_ft, dtype=np.float32).reshape(-1)
        if reference_altitude_np.size != north_np.size:
            raise ValueError("reference_altitude_samples_ft must have same length as canyon_north_samples_ft")
        if canyon_north_samples_ft is not None and canyon_width_samples_ft is not None:
            reference_altitude_np = reference_altitude_np[order]

    reference_speed_np = None
    if reference_speed_samples_fps is not None:
        reference_speed_np = np.asarray(reference_speed_samples_fps, dtype=np.float32).reshape(-1)
        if reference_speed_np.size != north_np.size:
            raise ValueError("reference_speed_samples_fps must have same length as canyon_north_samples_ft")
        if canyon_north_samples_ft is not None and canyon_width_samples_ft is not None:
            reference_speed_np = reference_speed_np[order]

    return NominalJSBSimParams(
        W=jnp.asarray(W, dtype=jnp.float32),
        B=jnp.asarray(B, dtype=jnp.float32),
        poly_powers=jnp.asarray(poly_powers, dtype=jnp.int32),
        canyon_north_samples_ft=jnp.asarray(north_np, dtype=jnp.float32),
        canyon_width_samples_ft=jnp.asarray(width_np, dtype=jnp.float32),
        canyon_center_east_samples_ft=jnp.asarray(center_east_np, dtype=jnp.float32),
        canyon_heading_samples_rad=jnp.asarray(heading_np, dtype=jnp.float32),
        reference_altitude_samples_ft=(
            None if reference_altitude_np is None else jnp.asarray(reference_altitude_np, dtype=jnp.float32)
        ),
        reference_speed_samples_fps=(
            None if reference_speed_np is None else jnp.asarray(reference_speed_np, dtype=jnp.float32)
        ),
        canyon_north_np=north_np,
        canyon_width_np=width_np,
        canyon_center_east_np=center_east_np,
        canyon_heading_rad_np=heading_np,
        reference_altitude_ft_np=reference_altitude_np,
        reference_speed_fps_np=reference_speed_np,
    )


def action_plan_tail(
    *,
    state: dict[str, float],
    params: NominalJSBSimParams,
    driver: F16CenterlineDriver,
) -> np.ndarray:
    return driver.act(state=state, params=params)


def rollout_heuristic_plan(
    *,
    state: dict[str, float],
    params: NominalJSBSimParams,
    horizon: int,
    driver: F16CenterlineDriver,
) -> np.ndarray:
    return driver.rollout_plan(
        state=state,
        params=params,
        horizon=horizon,
    )


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
        progress_gain=float(cost_config.progress_gain),
        speed_gain=float(cost_config.speed_gain),
        low_altitude_gain=float(cost_config.low_altitude_gain),
        altitude_target_gain=float(cost_config.altitude_target_gain),
        altitude_target_scale_ft=float(cost_config.altitude_target_scale_ft),
        centerline_gain=float(cost_config.centerline_gain),
        offcenter_penalty_gain=float(cost_config.offcenter_penalty_gain),
        heading_alignment_gain=float(cost_config.heading_alignment_gain),
        heading_alignment_scale_rad=float(cost_config.heading_alignment_scale_rad),
        alive_bonus=float(cost_config.alive_bonus),
        target_speed_fps=float(cost_config.target_speed_fps),
        speed_target_gain=float(cost_config.speed_target_gain),
        speed_target_scale_fps=float(cost_config.speed_target_scale_fps),
        target_altitude_ft=float(cost_config.target_altitude_ft),
        min_altitude_ft=float(cost_config.min_altitude_ft),
        max_altitude_ft=float(cost_config.max_altitude_ft),
        terrain_collision_height_ft=float(cost_config.terrain_collision_height_ft),
        wall_margin_ft=float(cost_config.wall_margin_ft),
        terrain_crash_penalty=float(cost_config.terrain_crash_penalty),
        wall_crash_penalty=float(cost_config.wall_crash_penalty),
        altitude_violation_penalty=float(cost_config.altitude_violation_penalty),
        early_termination_penalty_gain=float(cost_config.early_termination_penalty_gain),
        time_limit_bonus=float(cost_config.time_limit_bonus),
        max_step_reward_abs=float(cost_config.max_step_reward_abs),
        angular_rate_penalty_gain=float(cost_config.angular_rate_penalty_gain),
        angular_rate_threshold_deg_s=float(cost_config.angular_rate_threshold_deg_s),
        bank_angle_penalty_gain=float(cost_config.bank_angle_penalty_gain),
        bank_angle_threshold_deg=float(cost_config.bank_angle_threshold_deg),
        action_diff_weight=float(cost_config.action_diff_weight),
        action_l2_weight=float(cost_config.action_l2_weight),
        debug_render_plans=False,
        debug_num_trajectories=0,
        seed=0,
    )


def build_rollout_cost_fn(params: NominalJSBSimParams, cost_config: MPPICostConfig) -> Any:
    backend_config = _backend_cost_config(cost_config)
    rollout_states = backend.build_rollout_state_batch_fn(params.W, params.B, params.poly_powers)
    reference_altitude_samples_ft = params.reference_altitude_samples_ft
    if reference_altitude_samples_ft is None:
        reference_altitude_samples_ft = jnp.full_like(
            params.canyon_north_samples_ft,
            float(cost_config.target_altitude_ft),
        )
    reference_speed_samples_fps = params.reference_speed_samples_fps
    if reference_speed_samples_fps is None:
        reference_speed_samples_fps = jnp.full_like(
            params.canyon_north_samples_ft,
            float(cost_config.target_speed_fps),
        )
    rollout_costs_from_states = backend.build_rollout_cost_from_states_fn(
        params.canyon_north_samples_ft,
        params.canyon_width_samples_ft,
        params.canyon_center_east_samples_ft,
        params.canyon_heading_samples_rad,
        reference_altitude_samples_ft,
        reference_speed_samples_fps,
        backend_config,
    )

    def rollout_costs(initial_state: Array, action_batch: Array, initial_prev_action: Array) -> Array:
        state_batch = rollout_states(initial_state, action_batch)
        return rollout_costs_from_states(initial_state, state_batch, action_batch, initial_prev_action)

    return rollout_costs


def build_rollout_positions_fn(params: NominalJSBSimParams) -> Any:
    return backend.build_rollout_positions_fn(params.W, params.B, params.poly_powers)
