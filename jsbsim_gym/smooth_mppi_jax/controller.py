from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsbsim_gym.mppi_jax.controller import JaxMPPIConfig, JaxMPPIController
from jsbsim_gym.mppi_support import clip_action, smooth_noise_batch, softmax_weights


@dataclass(slots=True)
class JaxSmoothMPPIConfig(JaxMPPIConfig):
    action_noise_std: tuple[float, float, float, float] = (0.14, 0.22, 0.12, 0.10)
    delta_noise_std: tuple[float, float, float, float] = (0.08, 0.12, 0.08, 0.06)
    delta_action_bounds: tuple[float, float, float, float] = (0.18, 0.26, 0.14, 0.10)
    noise_smoothing_kernel: tuple[float, float, float, float, float] = (0.10, 0.20, 0.40, 0.20, 0.10)
    seed: int = 101


class JaxSmoothMPPIController(JaxMPPIController):
    def __init__(
        self,
        config: JaxSmoothMPPIConfig | None = None,
        *,
        reference_trajectory: dict[str, np.ndarray],
        terrain_north_samples_ft: np.ndarray,
        terrain_east_samples_ft: np.ndarray,
        terrain_elevation_ft: np.ndarray,
    ):
        super().__init__(
            config=config or JaxSmoothMPPIConfig(),
            reference_trajectory=reference_trajectory,
            terrain_north_samples_ft=terrain_north_samples_ft,
            terrain_east_samples_ft=terrain_east_samples_ft,
            terrain_elevation_ft=terrain_elevation_ft,
        )

    def _optimize(self, state: jax.Array) -> np.ndarray:
        sigma = jnp.asarray(self.config.delta_noise_std, dtype=jnp.float32)
        delta_bounds = jnp.asarray(self.config.delta_action_bounds, dtype=jnp.float32)
        base_plan = self._action_plan
        last_candidates = None
        prev_action = jnp.asarray(self._last_action, dtype=jnp.float32)
        reference_start_index = int(self._step_index)

        for _ in range(self.config.optimization_steps):
            self._key, noise_key = jax.random.split(self._key)
            raw_noise = jax.random.normal(
                noise_key,
                shape=(self.config.num_samples, self.config.horizon, 4),
                dtype=jnp.float32,
            ) * sigma
            noise = jnp.clip(
                smooth_noise_batch(raw_noise, self.config.noise_smoothing_kernel),
                -delta_bounds,
                delta_bounds,
            )
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
