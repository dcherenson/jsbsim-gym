from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym.mppi_jax.controller import JaxMPPIConfig
from jsbsim_gym.mppi_support import (
    MPPICostConfig,
    action_plan_tail,
    build_heuristic_driver,
    build_nominal_params,
    build_rollout_cost_fn,
    build_rollout_positions_fn,
    clip_action,
    jsbsim_state_to_jax,
    rollout_heuristic_plan,
    smooth_noise_batch,
    softmax_weights,
)


@dataclass(slots=True)
class JaxSmoothMPPIConfig(JaxMPPIConfig):
    delta_noise_std: tuple[float, float, float, float] = (0.08, 0.12, 0.08, 0.06)
    delta_action_bounds: tuple[float, float, float, float] = (0.18, 0.26, 0.14, 0.10)
    noise_smoothing_kernel: tuple[float, float, float, float, float] = (0.10, 0.20, 0.40, 0.20, 0.10)
    smoothness_penalty_weight: float = 0.35
    seed: int = 101


class JaxSmoothMPPIController:
    def __init__(
        self,
        config: JaxSmoothMPPIConfig | None = None,
        canyon_north_samples_ft=None,
        canyon_width_samples_ft=None,
        canyon_center_east_samples_ft=None,
        canyon_centerline_heading_rad_samples=None,
    ):
        self.config = config or JaxSmoothMPPIConfig()
        self.name = "smooth_mppi_jax"
        self.params = build_nominal_params(
            canyon_north_samples_ft=canyon_north_samples_ft,
            canyon_width_samples_ft=canyon_width_samples_ft,
            canyon_center_east_samples_ft=canyon_center_east_samples_ft,
            canyon_centerline_heading_rad_samples=canyon_centerline_heading_rad_samples,
        )
        self.driver = build_heuristic_driver(
            target_speed_fps=self.config.target_speed_fps,
            target_altitude_ft=self.config.target_altitude_ft,
            wall_margin_ft=self.config.wall_margin_ft,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
        )
        self.cost_config = MPPICostConfig(
            horizon=self.config.horizon,
            lambda_=self.config.lambda_,
            gamma_=self.config.gamma_,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
            progress_gain=self.config.progress_gain,
            speed_gain=self.config.speed_gain,
            low_altitude_gain=self.config.low_altitude_gain,
            altitude_target_gain=self.config.altitude_target_gain,
            altitude_target_scale_ft=self.config.altitude_target_scale_ft,
            centerline_gain=self.config.centerline_gain,
            offcenter_penalty_gain=self.config.offcenter_penalty_gain,
            heading_alignment_gain=self.config.heading_alignment_gain,
            heading_alignment_scale_rad=self.config.heading_alignment_scale_rad,
            alive_bonus=self.config.alive_bonus,
            target_speed_fps=self.config.target_speed_fps,
            target_altitude_ft=self.config.target_altitude_ft,
            min_altitude_ft=self.config.min_altitude_ft,
            max_altitude_ft=self.config.max_altitude_ft,
            terrain_collision_height_ft=self.config.terrain_collision_height_ft,
            wall_margin_ft=self.config.wall_margin_ft,
            terrain_crash_penalty=self.config.terrain_crash_penalty,
            wall_crash_penalty=self.config.wall_crash_penalty,
            altitude_violation_penalty=self.config.altitude_violation_penalty,
            early_termination_penalty_gain=self.config.early_termination_penalty_gain,
            time_limit_bonus=self.config.time_limit_bonus,
            max_step_reward_abs=self.config.max_step_reward_abs,
            angular_rate_penalty_gain=self.config.angular_rate_penalty_gain,
            angular_rate_threshold_deg_s=self.config.angular_rate_threshold_deg_s,
            bank_angle_penalty_gain=self.config.bank_angle_penalty_gain,
            bank_angle_threshold_deg=self.config.bank_angle_threshold_deg,
            action_diff_weight=self.config.action_diff_weight,
            action_l2_weight=self.config.action_l2_weight,
        )
        self._rollout_costs = build_rollout_cost_fn(self.params, self.cost_config)
        self._rollout_positions = build_rollout_positions_fn(self.params)
        self._key = jax.random.PRNGKey(self.config.seed)
        self._action_plan = jnp.zeros((self.config.horizon, 4), dtype=jnp.float32)
        self._action_plan = self._action_plan.at[:, 3].set(0.55)
        self.base_plan = self._action_plan
        self._initialized = False
        self.driver.reset()
        self._cached_action = np.array([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug: dict[str, np.ndarray] | None = None

    def reset(self, *, seed: int | None = None, case_id: str | None = None) -> None:
        del case_id
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed) + self.config.seed)
        else:
            self._key = jax.random.PRNGKey(self.config.seed)
        self._action_plan = jnp.zeros((self.config.horizon, 4), dtype=jnp.float32)
        self._action_plan = self._action_plan.at[:, 3].set(0.55)
        self.base_plan = self._action_plan
        self._initialized = False
        self.driver.reset()
        self._cached_action = np.array([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

    def _ensure_initialized(self, state_dict: dict[str, float]) -> None:
        if self._initialized:
            return
        heuristic_plan = rollout_heuristic_plan(
            state=state_dict,
            params=self.params,
            horizon=self.config.horizon,
            driver=self.driver,
        )
        self._action_plan = jnp.asarray(heuristic_plan, dtype=jnp.float32)
        self.base_plan = self._action_plan
        self._cached_action = np.asarray(heuristic_plan[0], dtype=np.float32)
        self._initialized = True

    def get_warm_start_plan(self, state_dict: dict[str, float], horizon: int | None = None) -> np.ndarray:
        plan_horizon = int(self.config.horizon if horizon is None else max(1, int(horizon)))
        return np.asarray(
            rollout_heuristic_plan(
                state=state_dict,
                params=self.params,
                horizon=plan_horizon,
                driver=self.driver,
            ),
            dtype=np.float32,
        )

    def _current_state(self, state_dict: dict[str, float]):
        return jsbsim_state_to_jax(state_dict)

    def _shift_plan(self, state_dict: dict[str, float]) -> None:
        tail = action_plan_tail(state=state_dict, params=self.params, driver=self.driver)
        shifted = np.roll(np.asarray(self._action_plan, dtype=np.float32), shift=-1, axis=0)
        shifted[-1] = tail
        self._action_plan = jnp.asarray(shifted, dtype=jnp.float32)
        self.base_plan = self._action_plan

    def _optimize(self, state) -> np.ndarray:
        sigma = jnp.asarray(self.config.delta_noise_std, dtype=jnp.float32)
        noise_bounds = jnp.asarray(self.config.delta_action_bounds, dtype=jnp.float32)
        base_plan = self._action_plan
        last_candidates = None
        for _ in range(self.config.optimization_steps):
            self._key, noise_key = jax.random.split(self._key)
            raw_noise = jax.random.normal(
                noise_key,
                shape=(self.config.num_samples, self.config.horizon, 4),
                dtype=jnp.float32,
            ) * sigma
            noise = jnp.clip(
                smooth_noise_batch(raw_noise, self.config.noise_smoothing_kernel),
                -noise_bounds,
                noise_bounds,
            )
            candidate_actions = clip_action(base_plan[None, :, :] + noise, low=self.config.action_low, high=self.config.action_high)
            costs = self._rollout_costs(state, candidate_actions, self._cached_action)
            perturbation_cost = self.cost_config.gamma_ * jnp.sum(
                jnp.square(noise / jnp.maximum(sigma, 1e-6)),
                axis=(1, 2),
            )
            action_diff = candidate_actions[:, 1:, :] - candidate_actions[:, :-1, :]
            smoothness_cost = self.config.smoothness_penalty_weight * jnp.sum(
                jnp.square(action_diff),
                axis=(1, 2),
            )
            total_costs = costs + perturbation_cost + smoothness_cost
            weights = softmax_weights(total_costs, self.config.lambda_)
            weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
            base_plan = clip_action(base_plan + weighted_noise, low=self.config.action_low, high=self.config.action_high)
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
        self._cached_action = action.copy()
        return action

    def get_render_debug(self) -> dict[str, np.ndarray] | None:
        return self._latest_render_debug

    def get_action(self, state_dict: dict[str, float]) -> np.ndarray:
        self._ensure_initialized(state_dict)
        if self._step_index - self._last_replan_step < self.config.replan_interval:
            self._step_index += 1
            return self._cached_action.copy()
        self._shift_plan(state_dict)
        state = self._current_state(state_dict)
        action = self._optimize(state)
        self._last_replan_step = self._step_index
        self._step_index += 1
        return action
