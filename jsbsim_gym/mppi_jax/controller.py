from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym.mppi_support import (
    DEFAULT_ACTION_HIGH,
    DEFAULT_ACTION_LOW,
    DEFAULT_CONTROL_RATE_WEIGHTS,
    DEFAULT_STATE_TRACKING_WEIGHTS,
    DEFAULT_TERRAIN_SAFE_CLEARANCE_FT,
    MPPICostConfig,
    build_nominal_params,
    build_rollout_cost_fn,
    build_rollout_positions_fn,
    clip_action,
    jsbsim_state_to_jax_with_load_factors,
    make_trim_action_plan,
    reference_heading_for_index,
    reference_state_for_index,
    softmax_weights,
)


@dataclass(slots=True)
class JaxMPPIConfig:
    horizon: int = 40
    num_samples: int = 4000
    optimization_steps: int = 2
    replan_interval: int = 1
    lambda_: float = 1.0
    gamma_: float = 0.05
    action_noise_std: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    action_low: tuple[float, float, float, float] = DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float] = DEFAULT_ACTION_HIGH
    state_tracking_weights: tuple[float, float, float, float, float, float] = DEFAULT_STATE_TRACKING_WEIGHTS
    terrain_collision_penalty: float = 1.0e6
    terrain_repulsion_scale: float = 1.0e5
    terrain_decay_rate_ft_inv: float = 0.03
    terrain_safe_clearance_ft: float = DEFAULT_TERRAIN_SAFE_CLEARANCE_FT
    control_rate_weights: tuple[float, float, float, float] = DEFAULT_CONTROL_RATE_WEIGHTS
    nz_limit_g: float = 9.0
    nz_penalty_weight: float = 1.0e4
    alpha_limit_rad: float = float(np.deg2rad(25.0))
    alpha_penalty_weight: float = 1.0e6
    debug_render_plans: bool = True
    debug_num_trajectories: int = 96
    seed: int = 42


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
            state_tracking_weights=self.config.state_tracking_weights,
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
        self._action_plan = jnp.asarray(make_trim_action_plan(self.config.horizon), dtype=jnp.float32)
        self.base_plan = self._action_plan
        self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
        self._cached_action = self._last_action.copy()
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
        self.base_plan = self._action_plan
        self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

    def _current_state(self, state_dict: dict[str, float]) -> jax.Array:
        return jsbsim_state_to_jax_with_load_factors(state_dict)

    def _shift_plan(self) -> None:
        shifted = np.roll(np.asarray(self._action_plan, dtype=np.float32), shift=-1, axis=0)
        shifted[-1] = self._last_action
        self._action_plan = jnp.asarray(shifted, dtype=jnp.float32)
        self.base_plan = self._action_plan

    def _optimize(self, state: jax.Array) -> np.ndarray:
        sigma = jnp.asarray(self.config.action_noise_std, dtype=jnp.float32)
        base_plan = self._action_plan
        last_candidates = None
        prev_action = jnp.asarray(self._last_action, dtype=jnp.float32)
        reference_start_index = int(self._step_index)

        for _ in range(self.config.optimization_steps):
            self._key, noise_key = jax.random.split(self._key)
            noise = jax.random.normal(
                noise_key,
                shape=(self.config.num_samples, self.config.horizon, 4),
                dtype=jnp.float32,
            ) * sigma
            candidate_actions = clip_action(
                base_plan[None, :, :] + noise,
                low=self.config.action_low,
                high=self.config.action_high,
            )
            costs = self._rollout_costs(state, candidate_actions, prev_action, reference_start_index)
            perturbation_cost = self.cost_config.gamma_ * jnp.sum(
                jnp.square(noise / jnp.maximum(sigma, 1e-6)),
                axis=(1, 2),
            )
            total_costs = costs + perturbation_cost
            weights = softmax_weights(total_costs, self.config.lambda_)
            weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
            base_plan = clip_action(
                base_plan + weighted_noise,
                low=self.config.action_low,
                high=self.config.action_high,
            )
            last_candidates = candidate_actions

        self._action_plan = base_plan
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
        self._last_action = action.copy()
        self._cached_action = action.copy()
        return action

    def get_reference_state(self, index: int | None = None) -> np.ndarray:
        target_index = self._step_index if index is None else int(index)
        return reference_state_for_index(self.params, target_index)

    def get_reference_heading_rad(self, index: int | None = None) -> float:
        target_index = self._step_index if index is None else int(index)
        return reference_heading_for_index(self.params, target_index)

    def get_render_debug(self) -> dict[str, np.ndarray] | None:
        return self._latest_render_debug

    def get_action(self, state_dict: dict[str, float]) -> np.ndarray:
        if self._step_index - self._last_replan_step < self.config.replan_interval:
            self._step_index += 1
            return self._cached_action.copy()

        self._shift_plan()
        state = self._current_state(state_dict)
        action = self._optimize(state)
        self._last_replan_step = self._step_index
        self._step_index += 1
        return action
