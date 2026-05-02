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
    MPPI_DEFAULT_NZ_MAX_G,
    MPPI_DEFAULT_NZ_MIN_G,
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
    f16_kinematics_step_with_load_factors,
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
    nz_min_g: float = MPPI_DEFAULT_NZ_MIN_G
    nz_max_g: float = MPPI_DEFAULT_NZ_MAX_G
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
            nz_min_g=self.config.nz_min_g,
            nz_max_g=self.config.nz_max_g,
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
        self._initialize_plans_from_nominal_progress(self._progress_s_ft)
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
        self._initialize_plans_from_nominal_progress(self._progress_s_ft)
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

    def _set_trim_plans(self) -> None:
        self._action_plan = jnp.asarray(make_trim_action_plan(self.config.horizon), dtype=jnp.float32)
        self._virtual_speed_plan = jnp.asarray(
            make_trim_virtual_speed_plan(self.config.horizon, trim_speed_fps=self._trim_virtual_speed_fps),
            dtype=jnp.float32,
        )
        self.base_plan = self._action_plan
        self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_virtual_speed_fps = float(self._virtual_speed_plan[0])
        self._cached_virtual_speed_fps = float(self._virtual_speed_plan[0])

    def _progress_to_reference_index(self, progress_s_ft: float) -> int:
        path_s = np.asarray(self.params.path_s_np, dtype=np.float64).reshape(-1)
        ref_len = int(np.asarray(self.params.reference_states_np).shape[0])
        if path_s.size < 1 or ref_len < 1:
            return 0

        s = float(np.clip(progress_s_ft, float(path_s[0]), float(path_s[-1])))
        path_idx = int(np.clip(np.searchsorted(path_s, s, side="left"), 0, path_s.size - 1))
        if ref_len == path_s.size or path_s.size <= 1:
            return int(np.clip(path_idx, 0, ref_len - 1))

        frac = float(path_idx) / float(path_s.size - 1)
        ref_idx = int(np.rint(frac * float(ref_len - 1)))
        return int(np.clip(ref_idx, 0, ref_len - 1))

    @staticmethod
    def _reference_body_rates(phi_rad: np.ndarray, theta_rad: np.ndarray, psi_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = 1.0 / 30.0
        phi = np.unwrap(np.asarray(phi_rad, dtype=np.float64))
        theta = np.asarray(theta_rad, dtype=np.float64)
        psi = np.unwrap(np.asarray(psi_rad, dtype=np.float64))

        phi_dot = np.gradient(phi, dt)
        theta_dot = np.gradient(theta, dt)
        psi_dot = np.gradient(psi, dt)

        theta_clip = np.clip(theta, -np.deg2rad(89.0), np.deg2rad(89.0))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta_clip)
        tan_theta = np.tan(theta_clip)

        q = theta_dot * cos_phi + psi_dot * cos_theta * sin_phi
        r = -theta_dot * sin_phi + psi_dot * cos_theta * cos_phi
        p = phi_dot - q * sin_phi * tan_theta - r * cos_phi * tan_theta
        return p.astype(np.float64), q.astype(np.float64), r.astype(np.float64)

    def _build_nominal_warm_start_plans(self, progress_s_ft: float) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(max(self.config.horizon, 1))
        action_low = np.asarray(self.config.action_low, dtype=np.float64)
        action_high = np.asarray(self.config.action_high, dtype=np.float64)
        trim_action = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float64)
        trim_action = np.clip(trim_action, action_low, action_high)

        ref_states = np.asarray(self.params.reference_states_np, dtype=np.float64)
        ref_speed = np.asarray(self.params.reference_speed_fps_np, dtype=np.float64).reshape(-1)
        if ref_states.shape[0] < 2 or ref_speed.size < 1:
            return (
                np.asarray(make_trim_action_plan(horizon), dtype=np.float32),
                np.asarray(make_trim_virtual_speed_plan(horizon, trim_speed_fps=self._trim_virtual_speed_fps), dtype=np.float32),
            )

        start_idx = self._progress_to_reference_index(progress_s_ft)
        idx = np.arange(start_idx, start_idx + horizon + 1, dtype=np.int64)
        idx = np.clip(idx, 0, ref_states.shape[0] - 1)

        phi_seq = ref_states[idx, 3]
        theta_seq = ref_states[idx, 4]
        psi_seq = ref_states[idx, 5]
        p_des, q_des, r_des = self._reference_body_rates(phi_seq, theta_seq, psi_seq)

        idx_h = idx[:-1]
        speed_des = ref_speed[np.clip(idx_h, 0, ref_speed.size - 1)]
        speed_des = np.where(np.isfinite(speed_des), speed_des, float(self._trim_virtual_speed_fps))
        speed_des = np.clip(
            speed_des,
            float(self.config.virtual_speed_min_fps),
            float(self.config.virtual_speed_max_fps),
        )

        rep_state = np.zeros((14,), dtype=np.float32)
        rep_state[:3] = ref_states[idx_h[0], :3].astype(np.float32)
        rep_state[3] = float(max(speed_des[0], 1.0))
        rep_state[6] = float(p_des[0])
        rep_state[7] = float(q_des[0])
        rep_state[8] = float(r_des[0])
        rep_state[9] = float(phi_seq[0])
        rep_state[10] = float(theta_seq[0])
        rep_state[11] = float(psi_seq[0])
        rep_state[12] = 0.0
        rep_state[13] = 1.0

        def _outputs_from_action(action_cmd: np.ndarray) -> np.ndarray:
            next_state = np.asarray(
                f16_kinematics_step_with_load_factors(
                    jnp.asarray(rep_state, dtype=jnp.float32),
                    jnp.asarray(action_cmd, dtype=jnp.float32),
                    self.params.W,
                    self.params.B,
                    self.params.poly_powers,
                    self.params.throttle_force_coeffs,
                ),
                dtype=np.float64,
            )
            speed_next = float(np.sqrt(max(float(next_state[3] ** 2 + next_state[4] ** 2 + next_state[5] ** 2), 1.0)))
            return np.asarray([next_state[6], next_state[7], next_state[8], speed_next], dtype=np.float64)

        y_trim = _outputs_from_action(trim_action)
        jac = np.zeros((4, 4), dtype=np.float64)
        eps = 0.08
        for ch in range(4):
            a_plus = trim_action.copy()
            a_minus = trim_action.copy()
            a_plus[ch] = min(float(action_high[ch]), float(a_plus[ch] + eps))
            a_minus[ch] = max(float(action_low[ch]), float(a_minus[ch] - eps))
            denom = float(a_plus[ch] - a_minus[ch])
            if denom <= 1e-6:
                jac[:, ch] = 0.0
                continue
            y_plus = _outputs_from_action(a_plus)
            y_minus = _outputs_from_action(a_minus)
            jac[:, ch] = (y_plus - y_minus) / denom

        jac_pinv = np.linalg.pinv(jac, rcond=1e-3)
        desired_outputs = np.column_stack([p_des[:-1], q_des[:-1], r_des[:-1], speed_des])
        action_plan = np.zeros((horizon, 4), dtype=np.float64)
        prev = trim_action.copy()
        for k in range(horizon):
            delta_y = desired_outputs[k] - y_trim
            cmd = trim_action + jac_pinv @ delta_y
            # Blend toward previous command to avoid large one-step discontinuities.
            cmd = 0.35 * cmd + 0.65 * prev
            cmd = np.clip(cmd, action_low, action_high)
            action_plan[k] = cmd
            prev = cmd

        return action_plan.astype(np.float32), speed_des.astype(np.float32)

    def _initialize_plans_from_nominal_progress(self, progress_s_ft: float) -> None:
        try:
            action_plan_np, speed_plan_np = self._build_nominal_warm_start_plans(progress_s_ft)
            if action_plan_np.shape != (int(max(self.config.horizon, 1)), 4):
                raise ValueError("invalid nominal action warm-start shape")
            if speed_plan_np.shape != (int(max(self.config.horizon, 1)),):
                raise ValueError("invalid nominal speed warm-start shape")
            if not np.all(np.isfinite(action_plan_np)) or not np.all(np.isfinite(speed_plan_np)):
                raise ValueError("non-finite nominal warm-start values")
            self._action_plan = jnp.asarray(action_plan_np, dtype=jnp.float32)
            self._virtual_speed_plan = jnp.asarray(speed_plan_np, dtype=jnp.float32)
            self.base_plan = self._action_plan
            self._last_action = np.asarray(self._action_plan[0], dtype=np.float32)
            self._cached_action = self._last_action.copy()
            self._last_virtual_speed_fps = float(self._virtual_speed_plan[0])
            self._cached_virtual_speed_fps = float(self._virtual_speed_plan[0])
        except Exception:
            self._set_trim_plans()

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

    def get_plan_debug(self) -> dict[str, np.ndarray | float | int]:
        return {
            "action_plan": np.asarray(self._action_plan, dtype=np.float32).copy(),
            "virtual_speed_plan": np.asarray(self._virtual_speed_plan, dtype=np.float32).copy(),
            "progress_s_ft": float(self._progress_s_ft),
            "step_index": int(self._step_index),
        }

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
