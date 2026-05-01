from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym.mppi_defaults import (
    MPPI_DEFAULT_ACTION_HIGH,
    MPPI_DEFAULT_ACTION_LOW,
    MPPI_DEFAULT_ACTION_NOISE_STD,
    MPPI_DEFAULT_ALPHA_LIMIT_RAD,
    MPPI_DEFAULT_ALPHA_PENALTY_WEIGHT,
    MPPI_DEFAULT_CONTOUR_WEIGHT,
    MPPI_DEFAULT_CONTROL_RATE_WEIGHTS,
    MPPI_DEFAULT_DEBUG_NUM_TRAJECTORIES,
    MPPI_DEFAULT_DEBUG_RENDER_PLANS,
    MPPI_DEFAULT_GAMMA,
    MPPI_DEFAULT_HORIZON,
    MPPI_DEFAULT_LAG_WEIGHT,
    MPPI_DEFAULT_LAMBDA,
    MPPI_DEFAULT_NZ_LIMIT_G,
    MPPI_DEFAULT_NZ_PENALTY_WEIGHT,
    MPPI_DEFAULT_NUM_SAMPLES,
    MPPI_DEFAULT_OPTIMIZATION_STEPS,
    MPPI_DEFAULT_PROGRESS_REWARD_WEIGHT,
    MPPI_DEFAULT_REPLAN_INTERVAL,
    MPPI_DEFAULT_SEED,
    MPPI_DEFAULT_TERRAIN_COLLISION_PENALTY,
    MPPI_DEFAULT_TERRAIN_DECAY_RATE_FT_INV,
    MPPI_DEFAULT_TERRAIN_REPULSION_SCALE,
    MPPI_DEFAULT_TERRAIN_SAFE_CLEARANCE_FT,
    MPPI_DEFAULT_VIRTUAL_SPEED_MAX_FPS,
    MPPI_DEFAULT_VIRTUAL_SPEED_MIN_FPS,
    MPPI_DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS,
    MPPI_DEFAULT_VIRTUAL_SPEED_TRIM_FPS,
    MPPI_DEFAULT_VIRTUAL_SPEED_WEIGHT,
)
from jsbsim_gym.mppi_support import (
    MPPICostConfig,
    build_nominal_params,
    build_rollout_cost_fn,
    build_rollout_positions_fn,
    clip_action,
    contouring_reference_for_progress,
    jsbsim_state_to_jax_with_load_factors,
    make_trim_action_plan,
    make_trim_virtual_speed_plan,
    reference_heading_for_index,
    reference_heading_for_progress,
    reference_state_for_progress,
    reference_state_for_index,
    softmax_weights,
)


@dataclass(slots=True)
class JaxMPPIConfig:
    horizon: int = MPPI_DEFAULT_HORIZON
    num_samples: int = MPPI_DEFAULT_NUM_SAMPLES
    optimization_steps: int = MPPI_DEFAULT_OPTIMIZATION_STEPS
    replan_interval: int = MPPI_DEFAULT_REPLAN_INTERVAL
    lambda_: float = MPPI_DEFAULT_LAMBDA
    gamma_: float = MPPI_DEFAULT_GAMMA
    action_noise_std: tuple[float, float, float, float] = MPPI_DEFAULT_ACTION_NOISE_STD
    action_low: tuple[float, float, float, float] = MPPI_DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float] = MPPI_DEFAULT_ACTION_HIGH
    contour_weight: float = MPPI_DEFAULT_CONTOUR_WEIGHT
    lag_weight: float = MPPI_DEFAULT_LAG_WEIGHT
    progress_reward_weight: float = MPPI_DEFAULT_PROGRESS_REWARD_WEIGHT
    virtual_speed_weight: float = MPPI_DEFAULT_VIRTUAL_SPEED_WEIGHT
    terrain_collision_penalty: float = MPPI_DEFAULT_TERRAIN_COLLISION_PENALTY
    terrain_repulsion_scale: float = MPPI_DEFAULT_TERRAIN_REPULSION_SCALE
    terrain_decay_rate_ft_inv: float = MPPI_DEFAULT_TERRAIN_DECAY_RATE_FT_INV
    terrain_safe_clearance_ft: float = MPPI_DEFAULT_TERRAIN_SAFE_CLEARANCE_FT
    control_rate_weights: tuple[float, float, float, float] = MPPI_DEFAULT_CONTROL_RATE_WEIGHTS
    nz_limit_g: float = MPPI_DEFAULT_NZ_LIMIT_G
    nz_penalty_weight: float = MPPI_DEFAULT_NZ_PENALTY_WEIGHT
    alpha_limit_rad: float = MPPI_DEFAULT_ALPHA_LIMIT_RAD
    alpha_penalty_weight: float = MPPI_DEFAULT_ALPHA_PENALTY_WEIGHT
    virtual_speed_noise_std_fps: float = MPPI_DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS
    virtual_speed_min_fps: float = MPPI_DEFAULT_VIRTUAL_SPEED_MIN_FPS
    virtual_speed_max_fps: float = MPPI_DEFAULT_VIRTUAL_SPEED_MAX_FPS
    virtual_speed_trim_fps: float = MPPI_DEFAULT_VIRTUAL_SPEED_TRIM_FPS
    debug_render_plans: bool = MPPI_DEFAULT_DEBUG_RENDER_PLANS
    debug_num_trajectories: int = MPPI_DEFAULT_DEBUG_NUM_TRAJECTORIES
    seed: int = MPPI_DEFAULT_SEED


class JaxMPPIController:
    def __init__(
        self,
        config: JaxMPPIConfig | None = None,
        *,
        reference_trajectory: dict[str, np.ndarray],
        terrain_north_samples_ft: np.ndarray,
        terrain_east_samples_ft: np.ndarray,
        terrain_elevation_ft: np.ndarray,
    ):
        self.config = config or JaxMPPIConfig()
        self.name = "mppi_jax"
        self.params = build_nominal_params(
            reference_trajectory=reference_trajectory,
            terrain_north_samples_ft=terrain_north_samples_ft,
            terrain_east_samples_ft=terrain_east_samples_ft,
            terrain_elevation_ft=terrain_elevation_ft,
        )
        self.cost_config = MPPICostConfig(
            horizon=self.config.horizon,
            lambda_=self.config.lambda_,
            gamma_=self.config.gamma_,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
            contour_weight=self.config.contour_weight,
            lag_weight=self.config.lag_weight,
            progress_reward_weight=self.config.progress_reward_weight,
            virtual_speed_weight=self.config.virtual_speed_weight,
            terrain_collision_penalty=self.config.terrain_collision_penalty,
            terrain_repulsion_scale=self.config.terrain_repulsion_scale,
            terrain_decay_rate_ft_inv=self.config.terrain_decay_rate_ft_inv,
            terrain_safe_clearance_ft=self.config.terrain_safe_clearance_ft,
            control_rate_weights=self.config.control_rate_weights,
            nz_limit_g=self.config.nz_limit_g,
            nz_penalty_weight=self.config.nz_penalty_weight,
            alpha_limit_rad=self.config.alpha_limit_rad,
            alpha_penalty_weight=self.config.alpha_penalty_weight,
        )
        self._rollout_costs = build_rollout_cost_fn(self.params, self.cost_config)
        self._rollout_positions = build_rollout_positions_fn(self.params)
        self._key = jax.random.PRNGKey(self.config.seed)
        nominal_trim_speed_fps = float(self.params.reference_speed_fps_np[0])
        if not np.isfinite(nominal_trim_speed_fps):
            nominal_trim_speed_fps = float(self.config.virtual_speed_trim_fps)
        self._trim_virtual_speed_fps = float(
            np.clip(
                nominal_trim_speed_fps,
                float(self.config.virtual_speed_min_fps),
                float(self.config.virtual_speed_max_fps),
            )
        )
        self._action_plan = jnp.asarray(make_trim_action_plan(self.config.horizon), dtype=jnp.float32)
        self._virtual_speed_plan = jnp.asarray(
            make_trim_virtual_speed_plan(self.config.horizon, trim_speed_fps=self._trim_virtual_speed_fps),
            dtype=jnp.float32,
        )
        self.base_plan = self._action_plan
        self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_virtual_speed_fps = float(self._trim_virtual_speed_fps)
        self._cached_virtual_speed_fps = float(self._trim_virtual_speed_fps)
        self._progress_s_ft = float(self.params.path_s_np[0])
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug: dict[str, np.ndarray] | None = None

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None:
        del case_id
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed) + self.config.seed)
        else:
            self._key = jax.random.PRNGKey(self.config.seed)
        self._action_plan = jnp.asarray(make_trim_action_plan(self.config.horizon), dtype=jnp.float32)
        self._virtual_speed_plan = jnp.asarray(
            make_trim_virtual_speed_plan(self.config.horizon, trim_speed_fps=self._trim_virtual_speed_fps),
            dtype=jnp.float32,
        )
        self.base_plan = self._action_plan
        self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_virtual_speed_fps = float(self._trim_virtual_speed_fps)
        self._cached_virtual_speed_fps = float(self._trim_virtual_speed_fps)
        self._progress_s_ft = float(self.params.path_s_np[0])
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

    def _current_state(self, state_dict: dict[str, float]) -> jax.Array:
        return jsbsim_state_to_jax_with_load_factors(state_dict)

    def _shift_plan(self) -> None:
        shifted = np.roll(np.asarray(self._action_plan, dtype=np.float32), shift=-1, axis=0)
        shifted[-1] = self._last_action
        self._action_plan = jnp.asarray(shifted, dtype=jnp.float32)

        shifted_virtual_speed = np.roll(np.asarray(self._virtual_speed_plan, dtype=np.float32), shift=-1, axis=0)
        shifted_virtual_speed[-1] = self._last_virtual_speed_fps
        self._virtual_speed_plan = jnp.asarray(shifted_virtual_speed, dtype=jnp.float32)
        self.base_plan = self._action_plan

    def _optimize(self, state: jax.Array) -> np.ndarray:
        sigma = jnp.asarray(self.config.action_noise_std, dtype=jnp.float32)
        virtual_speed_sigma = jnp.asarray(self.config.virtual_speed_noise_std_fps, dtype=jnp.float32)
        base_plan = self._action_plan
        base_virtual_speed_plan = self._virtual_speed_plan
        last_candidates = None
        prev_action = jnp.asarray(self._last_action, dtype=jnp.float32)
        progress_lower = jnp.asarray(float(self.params.path_s_np[0]), dtype=jnp.float32)
        progress_upper = jnp.asarray(float(self.params.path_s_np[-1]), dtype=jnp.float32)

        for _ in range(self.config.optimization_steps):
            self._key, noise_key = jax.random.split(self._key)
            noise = jax.random.normal(
                noise_key,
                shape=(self.config.num_samples, self.config.horizon, 4),
                dtype=jnp.float32,
            ) * sigma
            self._key, virtual_speed_noise_key = jax.random.split(self._key)
            virtual_speed_noise = jax.random.normal(
                virtual_speed_noise_key,
                shape=(self.config.num_samples, self.config.horizon),
                dtype=jnp.float32,
            ) * virtual_speed_sigma
            candidate_actions = clip_action(
                base_plan[None, :, :] + noise,
                low=self.config.action_low,
                high=self.config.action_high,
            )
            candidate_virtual_speed = jnp.clip(
                base_virtual_speed_plan[None, :] + virtual_speed_noise,
                float(self.config.virtual_speed_min_fps),
                float(self.config.virtual_speed_max_fps),
            )
            costs = self._rollout_costs(
                state,
                candidate_actions,
                candidate_virtual_speed,
                prev_action,
                self._progress_s_ft,
            )
            perturbation_cost = self.cost_config.gamma_ * (
                jnp.sum(jnp.square(noise / jnp.maximum(sigma, 1e-6)), axis=(1, 2))
                + jnp.sum(jnp.square(virtual_speed_noise / jnp.maximum(virtual_speed_sigma, 1e-6)), axis=1)
            )
            total_costs = costs + perturbation_cost
            weights = softmax_weights(total_costs, self.config.lambda_)
            weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
            weighted_virtual_speed_noise = jnp.tensordot(weights, virtual_speed_noise, axes=(0, 0))
            base_plan = clip_action(
                base_plan + weighted_noise,
                low=self.config.action_low,
                high=self.config.action_high,
            )
            base_virtual_speed_plan = jnp.clip(
                base_virtual_speed_plan + weighted_virtual_speed_noise,
                float(self.config.virtual_speed_min_fps),
                float(self.config.virtual_speed_max_fps),
            )
            last_candidates = candidate_actions

        self._action_plan = base_plan
        self._virtual_speed_plan = base_virtual_speed_plan
        self.base_plan = self._action_plan

        if self.config.debug_render_plans and last_candidates is not None:
            limit = min(int(self.config.debug_num_trajectories), int(last_candidates.shape[0]))
            sampled = np.asarray(self._rollout_positions(state, last_candidates[:limit]), dtype=np.float32)
            final_plan = np.asarray(self._rollout_positions(state, base_plan[None, :, :])[0], dtype=np.float32)
            self._latest_render_debug = {
                "candidate_xy": sampled[:, :, :2].copy(),
                "candidate_h_ft": sampled[:, :, 2].copy(),
                "final_xy": final_plan[:, :2].copy(),
                "final_h_ft": final_plan[:, 2].copy(),
            }

        action = np.asarray(base_plan[0], dtype=np.float32)
        virtual_speed_fps = float(base_virtual_speed_plan[0])
        self._last_action = action.copy()
        self._cached_action = action.copy()
        self._last_virtual_speed_fps = virtual_speed_fps
        self._cached_virtual_speed_fps = virtual_speed_fps
        self._progress_s_ft = float(
            np.clip(
                self._progress_s_ft + (1.0 / 30.0) * virtual_speed_fps,
                float(progress_lower),
                float(progress_upper),
            )
        )
        return action

    def get_reference_state(self, index: int | None = None) -> np.ndarray:
        if index is None:
            return reference_state_for_progress(self.params, self._progress_s_ft)
        return reference_state_for_index(self.params, int(index))

    def get_reference_heading_rad(self, index: int | None = None) -> float:
        if index is None:
            return reference_heading_for_progress(self.params, self._progress_s_ft)
        return reference_heading_for_index(self.params, int(index))

    def get_tracking_metrics(self, state_dict: dict[str, float]) -> dict[str, float]:
        reference_position_ft, reference_tangent = contouring_reference_for_progress(self.params, self._progress_s_ft)
        position_error_ft = np.asarray(
            [
                float(state_dict["p_N"]) - float(reference_position_ft[0]),
                float(state_dict["p_E"]) - float(reference_position_ft[1]),
                float(state_dict["h"]) - float(reference_position_ft[2]),
            ],
            dtype=np.float64,
        )
        lag_error_ft = float(np.dot(position_error_ft, np.asarray(reference_tangent, dtype=np.float64)))
        contour_error_vec_ft = position_error_ft - lag_error_ft * np.asarray(reference_tangent, dtype=np.float64)
        contour_error_ft = float(np.linalg.norm(contour_error_vec_ft))
        position_error_norm_ft = float(np.linalg.norm(position_error_ft))
        contour_cost_est = float(self.config.contour_weight) * (contour_error_ft ** 2)
        lag_cost_est = float(self.config.lag_weight) * (lag_error_ft ** 2)
        progress_reward_est = -float(self.config.progress_reward_weight) * float(self._last_virtual_speed_fps)
        virtual_speed_cost_est = float(self.config.virtual_speed_weight) * (float(self._last_virtual_speed_fps) ** 2)
        return {
            "progress_s_ft": float(self._progress_s_ft),
            "virtual_speed_fps": float(self._last_virtual_speed_fps),
            "reference_north_ft": float(reference_position_ft[0]),
            "reference_east_ft": float(reference_position_ft[1]),
            "reference_altitude_ft": float(reference_position_ft[2]),
            "reference_heading_rad": float(np.arctan2(float(reference_tangent[1]), float(reference_tangent[0]))),
            "contour_error_ft": contour_error_ft,
            "lag_error_ft": lag_error_ft,
            "position_error_ft": position_error_norm_ft,
            "altitude_error_ft": float(position_error_ft[2]),
            "contour_cost_est": contour_cost_est,
            "lag_cost_est": lag_cost_est,
            "progress_reward_est": progress_reward_est,
            "virtual_speed_cost_est": virtual_speed_cost_est,
            "contouring_cost_est": contour_cost_est + lag_cost_est + progress_reward_est + virtual_speed_cost_est,
        }

    def get_render_debug(self) -> dict[str, np.ndarray] | None:
        return self._latest_render_debug

    def get_action(self, state_dict: dict[str, float]) -> np.ndarray:
        if self._step_index - self._last_replan_step < self.config.replan_interval:
            self._progress_s_ft = float(
                np.clip(
                    self._progress_s_ft + (1.0 / 30.0) * self._cached_virtual_speed_fps,
                    float(self.params.path_s_np[0]),
                    float(self.params.path_s_np[-1]),
                )
            )
            self._last_virtual_speed_fps = float(self._cached_virtual_speed_fps)
            self._step_index += 1
            return self._cached_action.copy()

        self._shift_plan()
        state = self._current_state(state_dict)
        action = self._optimize(state)
        self._last_replan_step = self._step_index
        self._step_index += 1
        return action
