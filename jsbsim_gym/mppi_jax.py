import jax
import jax.numpy as jnp
import numpy as np
import functools
from dataclasses import dataclass


MPPI_FEATURE_NAMES = (
    "alpha",
    "beta",
    "p",
    "q",
    "r",
    "delta_e",
    "delta_a",
    "delta_r",
    "delta_t",
)

# F-16 inertia terms from aircraft/f16/f16.xml (slug*ft^2)
IXX = 9496.0
IYY = 55814.0
IZZ = 63100.0
IXZ = -982.0
INERTIA_DET = IXX * IZZ - IXZ * IXZ

@dataclass(frozen=True)
class JaxMPPIConfig:
    horizon: int = 40
    num_samples: int = 4000
    optimization_steps: int = 2
    replan_interval: int = 1
    lambda_: float = 2.0
    gamma_: float = 0.015
    action_noise_std: tuple = (0.16, 0.14, 0.12, 0.08)  # roll, pitch, yaw, throttle
    action_low: tuple = (-1.0, -1.0, -1.0, 0.0)
    action_high: tuple = (1.0, 1.0, 1.0, 1.0)
    # SAC-style shaping gains used by the MPPI surrogate objective.
    progress_gain: float = 0.70
    speed_gain: float = 0.35
    low_altitude_gain: float = 0.45
    centerline_gain: float = 0.60
    offcenter_penalty_gain: float = 0.30
    heading_alignment_gain: float = 0.45
    heading_alignment_scale_rad: float = 0.70
    alive_bonus: float = 0.15
    target_speed_fps: float = 800.0
    target_altitude_ft: float = 250.0
    min_altitude_ft: float = -500.0
    max_altitude_ft: float = 3000.0
    terrain_collision_height_ft: float = 60.0
    wall_margin_ft: float = 30.0
    terrain_crash_penalty: float = 25.0
    wall_crash_penalty: float = 18.0
    altitude_violation_penalty: float = 8.0
    early_termination_penalty_gain: float = 80.0
    time_limit_bonus: float = 25.0
    max_step_reward_abs: float = 15.0
    angular_rate_penalty_gain: float = 0.45
    angular_rate_threshold_deg_s: float = 45.0
    action_diff_weight: float = 2.5
    action_l2_weight: float = 0.4
    debug_render_plans: bool = True
    debug_num_trajectories: int = 96
    seed: int = 42

    def tree_flatten(self):
        # Numeric values that we want to be dynamic (leaves)
        # We include floats and tuples of floats (which JAX will flatten further)
        children = (
            self.lambda_, self.gamma_, self.progress_gain, self.speed_gain,
            self.low_altitude_gain, self.centerline_gain, self.offcenter_penalty_gain,
            self.heading_alignment_gain, self.heading_alignment_scale_rad,
            self.alive_bonus, self.target_speed_fps, self.target_altitude_ft, self.min_altitude_ft,
            self.max_altitude_ft, self.terrain_collision_height_ft, self.wall_margin_ft,
            self.terrain_crash_penalty, self.wall_crash_penalty, self.altitude_violation_penalty,
            self.early_termination_penalty_gain, self.time_limit_bonus, self.max_step_reward_abs,
            self.angular_rate_penalty_gain, self.angular_rate_threshold_deg_s,
            self.action_diff_weight, self.action_l2_weight,
            # Tuples are children too
            self.action_noise_std, self.action_low, self.action_high,
        )
        # Structural values that require JIT recompile (aux_data)
        aux_data = (
            self.horizon, self.num_samples, self.optimization_steps,
            self.replan_interval, self.debug_render_plans,
            self.debug_num_trajectories, self.seed
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the dataclass. Order must match tree_flatten.
        (
            lambda_, gamma_, progress_gain, speed_gain,
            low_altitude_gain, centerline_gain, offcenter_penalty_gain,
            heading_alignment_gain, heading_alignment_scale_rad,
            alive_bonus, target_speed_fps, target_altitude_ft, min_altitude_ft,
            max_altitude_ft, terrain_collision_height_ft, wall_margin_ft,
            terrain_crash_penalty, wall_crash_penalty, altitude_violation_penalty,
            early_termination_penalty_gain, time_limit_bonus, max_step_reward_abs,
            angular_rate_penalty_gain, angular_rate_threshold_deg_s,
            action_diff_weight, action_l2_weight,
            action_noise_std, action_low, action_high,
        ) = children
        (
            horizon, num_samples, optimization_steps,
            replan_interval, debug_render_plans,
            debug_num_trajectories, seed
        ) = aux_data
        
        return cls(
            horizon=horizon, num_samples=num_samples, optimization_steps=optimization_steps,
            replan_interval=replan_interval, lambda_=lambda_, gamma_=gamma_,
            action_noise_std=action_noise_std, action_low=action_low, action_high=action_high,
            progress_gain=progress_gain, speed_gain=speed_gain, low_altitude_gain=low_altitude_gain,
            centerline_gain=centerline_gain, offcenter_penalty_gain=offcenter_penalty_gain,
            heading_alignment_gain=heading_alignment_gain, heading_alignment_scale_rad=heading_alignment_scale_rad,
            alive_bonus=alive_bonus, target_speed_fps=target_speed_fps, target_altitude_ft=target_altitude_ft,
            min_altitude_ft=min_altitude_ft, max_altitude_ft=max_altitude_ft,
            terrain_collision_height_ft=terrain_collision_height_ft, wall_margin_ft=wall_margin_ft,
            terrain_crash_penalty=terrain_crash_penalty, wall_crash_penalty=wall_crash_penalty,
            altitude_violation_penalty=altitude_violation_penalty,
            early_termination_penalty_gain=early_termination_penalty_gain,
            time_limit_bonus=time_limit_bonus, max_step_reward_abs=max_step_reward_abs,
            angular_rate_penalty_gain=angular_rate_penalty_gain,
            angular_rate_threshold_deg_s=angular_rate_threshold_deg_s,
            action_diff_weight=action_diff_weight, action_l2_weight=action_l2_weight,
            debug_render_plans=debug_render_plans, debug_num_trajectories=debug_num_trajectories,
            seed=seed
        )

jax.tree_util.register_pytree_node_class(JaxMPPIConfig)

@dataclass(frozen=True)
class JaxSmoothMPPIConfig(JaxMPPIConfig):
    delta_noise_std: tuple = (0.08, 0.12, 0.08, 0.06)
    delta_action_bounds: tuple = (0.18, 0.26, 0.14, 0.10)
    noise_smoothing_kernel: tuple = (0.10, 0.20, 0.40, 0.20, 0.10)
    smoothness_penalty_weight: float = 0.35
    seed: int = 101

    def tree_flatten(self):
        # Flatten the base class first
        base_children, base_aux = super().tree_flatten()
        
        # Add smooth-specific children (dynamic)
        smooth_children = base_children + (
            self.delta_noise_std, self.delta_action_bounds,
            self.noise_smoothing_kernel, self.smoothness_penalty_weight
        )
        # Add smooth-specific aux data (static)
        # We already have seed in base_aux, but smooth has its own default seed.
        # For simplicity, we'll just keep the base_aux.
        return (smooth_children, base_aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # The last 4 children are smooth-specific
        smoothness_penalty_weight = children[-1]
        noise_smoothing_kernel = children[-2]
        delta_action_bounds = children[-3]
        delta_noise_std = children[-4]
        
        base_children = children[:-4]
        # Reconstruct base part using its logic via super() is tricky with dataclasses,
        # so we just initialize ourself directly.
        
        obj = JaxMPPIConfig.tree_unflatten(aux_data, base_children)
        # Now create the smooth version using the unflattened base fields
        import dataclasses
        return cls(
            **{f.name: getattr(obj, f.name) for f in dataclasses.fields(JaxMPPIConfig)},
            delta_noise_std=delta_noise_std,
            delta_action_bounds=delta_action_bounds,
            noise_smoothing_kernel=noise_smoothing_kernel,
            smoothness_penalty_weight=smoothness_penalty_weight
        )

jax.tree_util.register_pytree_node_class(JaxSmoothMPPIConfig)

def load_nominal_weights():
    # Load from the npz file
    import os
    path = os.path.join(os.path.dirname(__file__), "mppi_nominal_weights.npz")
    data = np.load(path, allow_pickle=True)
    W = np.asarray(data['W'])
    B = np.asarray(data['B'])

    feature_names = tuple(MPPI_FEATURE_NAMES)
    if "feature_names" in data:
        feature_names = tuple(str(x) for x in data["feature_names"])

    if feature_names != MPPI_FEATURE_NAMES:
        raise ValueError(
            "mppi_nominal_weights.npz feature set mismatch. "
            "Expected throttle-inclusive features "
            f"{MPPI_FEATURE_NAMES}, got {feature_names}. "
            "Regenerate weights via: uv run python extract_nominal_weights.py"
        )

    n = len(MPPI_FEATURE_NAMES)
    expected_rows = 1 + n + (n * (n + 1)) // 2
    if W.shape[0] != expected_rows:
        raise ValueError(
            "mppi_nominal_weights.npz has incompatible polynomial size. "
            f"Expected W rows={expected_rows}, got {W.shape[0]}. "
            "Regenerate weights via: uv run python extract_nominal_weights.py"
        )
    if B.shape[0] != 6:
        raise ValueError(f"Expected 6 output channels in B, got shape {B.shape}")

    return jnp.asarray(W), jnp.asarray(B)

def expand_poly(x):
    # x shape (n_features,)
    # The polynomials features order: 1, x, x^2 + cross terms
    # Using the same order as sklearn PolynomialFeatures(degree=2)
    # Features: alpha, beta, p, q, r, delta_e, delta_a, delta_r, delta_t
    ones = jnp.ones((1,))

    n_features = int(x.shape[0])
    quad = []
    for i in range(n_features):
        for j in range(i, n_features):
            quad.append(x[i] * x[j])

    quad_stack = jnp.stack(quad)
    return jnp.concatenate([ones, x, quad_stack])

def canyon_width(p_N):
    W_base = 300.0
    W_amp = 220.0
    W_freq = 15000.0
    return W_base + W_amp * jnp.sin(2.0 * jnp.pi * p_N / W_freq)


def canyon_width_from_profile(p_N, north_samples_ft, width_samples_ft):
    return jnp.interp(p_N, north_samples_ft, width_samples_ft)


def canyon_center_east_from_profile(p_N, north_samples_ft, center_east_samples_ft):
    return jnp.interp(p_N, north_samples_ft, center_east_samples_ft)


def canyon_centerline_heading_from_profile(p_N, north_samples_ft, heading_samples_rad):
    return jnp.interp(p_N, north_samples_ft, heading_samples_rad)


def wrap_angle_rad(angle_rad):
    return jnp.arctan2(jnp.sin(angle_rad), jnp.cos(angle_rad))


def clip_action(action, low, high):
    return jnp.clip(action, low, high)


def softmax_weights(costs, temperature):
    minimum = jnp.min(costs)
    logits = -(costs - minimum) / jnp.maximum(temperature, 1e-6)
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    return weights / (jnp.sum(weights) + 1e-8)


def moments_to_angular_rate_derivatives(p, q, r, L, M, N):
    h_x = IXX * p + IXZ * r
    h_y = IYY * q
    h_z = IXZ * p + IZZ * r

    cross_x = q * h_z - r * h_y
    cross_y = r * h_x - p * h_z
    cross_z = p * h_y - q * h_x

    rhs_x = L - cross_x
    rhs_y = M - cross_y
    rhs_z = N - cross_z

    p_dot = (IZZ * rhs_x - IXZ * rhs_z) / INERTIA_DET
    q_dot = rhs_y / IYY
    r_dot = (IXX * rhs_z - IXZ * rhs_x) / INERTIA_DET
    return p_dot, q_dot, r_dot

def f16_kinematics_step(state, action, W, B):
    # State: p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi = state
    
    # FCS
    delta_a = action[0]
    delta_e = action[1]
    delta_r = action[2]
    delta_t = action[3]
    
    alpha = jnp.arctan2(w, jnp.maximum(u, 1.0))
    V_sq = u*u + v*v + w*w
    V = jnp.sqrt(jnp.maximum(V_sq, 1.0))
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    
    features = jnp.array([alpha, beta, p, q, r, delta_e, delta_a, delta_r, delta_t])
    phi_vec = expand_poly(features)
    
    preds = jnp.dot(phi_vec, W) + B
    X, Y, Z, L, M, N = preds
    
    G = 32.174
    u_dot = X + r*v - q*w - G*jnp.sin(theta)
    
    v_dot = Y + p*w - r*u + G*jnp.sin(phi)*jnp.cos(theta)
    w_dot = Z + q*u - p*v + G*jnp.cos(phi)*jnp.cos(theta)

    p_dot, q_dot, r_dot = moments_to_angular_rate_derivatives(p, q, r, L, M, N)
    
    u_dot = jnp.clip(u_dot, -1000.0, 1000.0)
    v_dot = jnp.clip(v_dot, -1000.0, 1000.0)
    w_dot = jnp.clip(w_dot, -1000.0, 1000.0)
    p_dot = jnp.clip(p_dot, -50.0, 50.0)
    q_dot = jnp.clip(q_dot, -50.0, 50.0)
    r_dot = jnp.clip(r_dot, -50.0, 50.0)
    
    t_theta = jnp.tan(theta)
    phi_dot = p + q*jnp.sin(phi)*t_theta + r*jnp.cos(phi)*t_theta
    theta_dot = q*jnp.cos(phi) - r*jnp.sin(phi)
    psi_dot = (q*jnp.sin(phi) + r*jnp.cos(phi)) / jnp.cos(theta)
    
    phi_dot = jnp.clip(phi_dot, -50.0, 50.0)
    theta_dot = jnp.clip(theta_dot, -50.0, 50.0)
    psi_dot = jnp.clip(psi_dot, -50.0, 50.0)
    
    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)
    
    p_N_dot = u*(c_theta*c_psi) + v*(s_phi*s_theta*c_psi - c_phi*s_psi) + w*(c_phi*s_theta*c_psi + s_phi*s_psi)
    p_E_dot = u*(c_theta*s_psi) + v*(s_phi*s_theta*s_psi + c_phi*c_psi) + w*(c_phi*s_theta*s_psi - s_phi*c_psi)
    h_dot = u*(s_theta) - v*(s_phi*c_theta) - w*(c_phi*c_theta)
    
    # Clip position velocities to prevent nan propagation
    p_N_dot = jnp.clip(p_N_dot, -3000.0, 3000.0)
    p_E_dot = jnp.clip(p_E_dot, -3000.0, 3000.0)
    h_dot = jnp.clip(h_dot, -3000.0, 3000.0)
    
    dt = 1.0/30.0
    state_next = jnp.array([
        p_N + dt*p_N_dot,
        p_E + dt*p_E_dot,
        h + dt*h_dot,
        u + dt*u_dot,
        v + dt*v_dot,
        w + dt*w_dot,
        p + dt*p_dot,
        q + dt*q_dot,
        r + dt*r_dot,
        phi + dt*phi_dot,
        theta + dt*theta_dot,
        psi + dt*psi_dot
    ])
    
    return state_next

def rollout_trajectory(initial_state, action_seq, W, B):
    def step_fn(state, action):
        next_state = f16_kinematics_step(state, action, W, B)
        return next_state, next_state
    
    _, state_seq = jax.lax.scan(step_fn, initial_state, action_seq)
    return state_seq


@jax.jit
def rollout_trajectory_batch(initial_state, action_batch, W, B):
    return jax.vmap(lambda action_seq: rollout_trajectory(initial_state, action_seq, W, B))(action_batch)


def smooth_noise_batch(noise, kernel_weights):
    kernel = jnp.asarray(kernel_weights, dtype=jnp.float32)
    kernel = kernel / jnp.maximum(jnp.sum(kernel), 1e-6)
    pad = int(kernel.shape[0] // 2)

    padded = jnp.pad(noise, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    kernel_conv = kernel[:, None, None]

    # Flatten sample and action channels so one 1D conv smooths each trajectory channel.
    flat = jnp.transpose(padded, (0, 2, 1)).reshape((-1, padded.shape[1], 1))
    smoothed = jax.lax.conv_general_dilated(
        flat,
        kernel_conv,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    smoothed = smoothed.reshape((noise.shape[0], noise.shape[2], noise.shape[1]))
    return jnp.transpose(smoothed, (0, 2, 1))

def single_rollout_cost(
    initial_state,
    action_seq,
    initial_prev_action,
    W,
    B,
    canyon_north_samples_ft,
    canyon_width_samples_ft,
    canyon_center_east_samples_ft,
    canyon_heading_samples_rad,
    config: JaxMPPIConfig,
):
    horizon_f = jnp.maximum(float(config.horizon), 1.0)

    def body(carry, inputs):
        state, active, prev_action, prev_dist_start_ft, cumulative_cost = carry
        action, step_idx = inputs

        low = jnp.asarray(config.action_low, dtype=jnp.float32)
        high = jnp.asarray(config.action_high, dtype=jnp.float32)
        bounded_action = clip_action(action, low, high)
        next_state_raw = f16_kinematics_step(state, bounded_action, W, B)

        p_N = next_state_raw[0]
        p_E = next_state_raw[1]
        h = next_state_raw[2]
        u = next_state_raw[3]
        v = next_state_raw[4]
        w = next_state_raw[5]
        p_rate = next_state_raw[6]
        q_rate = next_state_raw[7]
        r_rate = next_state_raw[8]
        psi = next_state_raw[11]

        width_ft = canyon_width_from_profile(p_N, canyon_north_samples_ft, canyon_width_samples_ft)
        center_east_ft = canyon_center_east_from_profile(
            p_N,
            canyon_north_samples_ft,
            canyon_center_east_samples_ft,
        )
        centerline_heading_rad = canyon_centerline_heading_from_profile(
            p_N,
            canyon_north_samples_ft,
            canyon_heading_samples_rad,
        )
        half_width_ft = 0.5 * width_ft
        usable_half_ft = jnp.maximum(half_width_ft - config.wall_margin_ft, 1.0)

        raw_lateral_error_ft = p_E - center_east_ft
        invalid_centerline = jnp.abs(raw_lateral_error_ft) > 4.0 * jnp.maximum(width_ft, 1.0)
        effective_center_east_ft = jnp.where(invalid_centerline, initial_state[1], center_east_ft)
        effective_heading_rad = jnp.where(invalid_centerline, psi, centerline_heading_rad)

        dn_ft = p_N - state[0]
        de_ft = p_E - state[1]
        progress_ft = dn_ft * jnp.cos(effective_heading_rad) + de_ft * jnp.sin(effective_heading_rad)
        dist_start_ft = jnp.sqrt(
            jnp.maximum(
                (p_N - initial_state[0]) * (p_N - initial_state[0])
                + (p_E - initial_state[1]) * (p_E - initial_state[1]),
                0.0,
            )
        )
        progress_from_start_ft = dist_start_ft - prev_dist_start_ft

        progress_local = jnp.clip(progress_ft / 25.0, -2.0, 3.0)
        progress_global = jnp.clip(progress_from_start_ft / 25.0, -2.0, 2.0)
        progress_term = config.progress_gain * (0.8 * progress_local + 0.2 * progress_global)

        speed_fps = jnp.sqrt(jnp.maximum(u * u + v * v + w * w, 1.0))
        # Penalty for deviation from target speed
        speed_error_norm = (speed_fps - config.target_speed_fps) / 100.0
        speed_term = -config.speed_gain * (speed_error_norm * speed_error_norm)

        clearance_error = jnp.abs(h - config.target_altitude_ft)
        low_altitude_term = config.low_altitude_gain * (
            1.0 - jnp.clip(clearance_error / jnp.maximum(config.target_altitude_ft, 1.0), 0.0, 2.0)
        )

        lateral_error_ft = p_E - effective_center_east_ft
        lateral_norm = jnp.abs(lateral_error_ft) / usable_half_ft
        centerline_term = config.centerline_gain * (1.0 - jnp.clip(lateral_norm, 0.0, 1.0))
        offcenter_term = -config.offcenter_penalty_gain * jnp.clip(lateral_norm - 0.5, 0.0, 2.0)

        heading_error_rad = wrap_angle_rad(psi - effective_heading_rad)
        heading_term = config.heading_alignment_gain * (
            1.0
            - jnp.clip(
                jnp.abs(heading_error_rad) / jnp.maximum(config.heading_alignment_scale_rad, 1e-3),
                0.0,
                2.0,
            )
        )

        rate_mag_rad_s = jnp.sqrt(jnp.maximum(p_rate * p_rate + q_rate * q_rate + r_rate * r_rate, 0.0))
        rate_mag_deg_s = jnp.degrees(rate_mag_rad_s)
        angular_rate_term = -config.angular_rate_penalty_gain * jnp.clip(
            (rate_mag_deg_s - config.angular_rate_threshold_deg_s)
            / jnp.maximum(config.angular_rate_threshold_deg_s, 1.0),
            0.0,
            3.0,
        )

        stage_reward = (
            config.alive_bonus
            + progress_term
            + speed_term
            + low_altitude_term
            + centerline_term
            + offcenter_term
            + heading_term
            + angular_rate_term
        )
        stage_reward = jnp.clip(stage_reward, -config.max_step_reward_abs, config.max_step_reward_abs)

        terrain_collision = h <= config.terrain_collision_height_ft
        out_of_canyon = jnp.abs(lateral_error_ft) > usable_half_ft
        out_of_altitude = (h < config.min_altitude_ft) | (h > config.max_altitude_ft)
        terminated_now = terrain_collision | out_of_canyon | out_of_altitude

        remaining_frac = (horizon_f - (step_idx.astype(jnp.float32) + 1.0)) / horizon_f
        early_penalty = config.early_termination_penalty_gain * jnp.clip(remaining_frac, 0.0, 1.0)
        termination_penalty = jnp.where(
            terrain_collision,
            config.terrain_crash_penalty + early_penalty,
            jnp.where(
                out_of_canyon,
                config.wall_crash_penalty + 0.75 * early_penalty,
                jnp.where(
                    out_of_altitude,
                    config.altitude_violation_penalty + 0.5 * early_penalty,
                    0.0,
                ),
            ),
        )

        survived_to_end = jnp.logical_and(step_idx == (config.horizon - 1), jnp.logical_not(terminated_now))
        stage_reward = stage_reward - termination_penalty + jnp.where(survived_to_end, config.time_limit_bonus, 0.0)

        action_cost = (
            config.action_l2_weight * jnp.sum(jnp.square(bounded_action))
            + config.action_diff_weight * jnp.sum(jnp.square(bounded_action - prev_action))
        )
        stage_cost = -stage_reward + action_cost
        stage_cost = jnp.where(active, stage_cost, 0.0)

        next_state = jnp.where(active, next_state_raw, state)
        next_prev_action = jnp.where(active, bounded_action, prev_action)
        next_prev_dist = jnp.where(active, dist_start_ft, prev_dist_start_ft)
        terminated_active = jnp.logical_and(active, terminated_now)
        next_active = jnp.logical_and(active, jnp.logical_not(terminated_active))
        next_cumulative_cost = cumulative_cost + stage_cost

        return (next_state, next_active, next_prev_action, next_prev_dist, next_cumulative_cost), stage_cost

    init_prev_action = clip_action(
        initial_prev_action,
        jnp.asarray(config.action_low, dtype=jnp.float32),
        jnp.asarray(config.action_high, dtype=jnp.float32),
    )
    init_dist_start_ft = jnp.asarray(0.0, dtype=jnp.float32)

    carry, _ = jax.lax.scan(
        body,
        (
            initial_state,
            jnp.asarray(True),
            init_prev_action,
            init_dist_start_ft,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        (action_seq, jnp.arange(config.horizon, dtype=jnp.int32)),
    )
    return carry[-1]

@jax.jit
def mppi_optimize_step(
    initial_state,
    base_action_plan,
    initial_prev_action,
    key,
    W,
    B,
    canyon_north_samples_ft,
    canyon_width_samples_ft,
    canyon_center_east_samples_ft,
    canyon_heading_samples_rad,
    config: JaxMPPIConfig,
):
    sigma = jnp.asarray(config.action_noise_std, dtype=jnp.float32)
    low = jnp.asarray(config.action_low, dtype=jnp.float32)
    high = jnp.asarray(config.action_high, dtype=jnp.float32)

    noise = jax.random.normal(
        key,
        shape=(config.num_samples, config.horizon, 4),
        dtype=jnp.float32,
    ) * sigma
    candidate_actions = clip_action(base_action_plan[None, :, :] + noise, low, high)

    costs = jax.vmap(
        lambda candidate_action_seq: single_rollout_cost(
            initial_state,
            candidate_action_seq,
            initial_prev_action,
            W,
            B,
            canyon_north_samples_ft,
            canyon_width_samples_ft,
            canyon_center_east_samples_ft,
            canyon_heading_samples_rad,
            config,
        )
    )(candidate_actions)

    perturbation_cost = config.gamma_ * jnp.sum(
        jnp.square(noise / jnp.maximum(sigma, 1e-6)),
        axis=(1, 2),
    )
    total_costs = costs + perturbation_cost

    weights = softmax_weights(total_costs, config.lambda_)
    weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
    optimized_plan = clip_action(base_action_plan + weighted_noise, low, high)

    return optimized_plan, total_costs, state_seq_best(initial_state, optimized_plan, W, B), candidate_actions


@jax.jit
def smooth_mppi_optimize_step(
    initial_state,
    base_action_plan,
    initial_prev_action,
    key,
    W,
    B,
    canyon_north_samples_ft,
    canyon_width_samples_ft,
    canyon_center_east_samples_ft,
    canyon_heading_samples_rad,
    config: JaxSmoothMPPIConfig,
):
    low = jnp.asarray(config.action_low, dtype=jnp.float32)
    high = jnp.asarray(config.action_high, dtype=jnp.float32)
    delta_sigma = jnp.asarray(config.delta_noise_std, dtype=jnp.float32)
    delta_bounds = jnp.asarray(config.delta_action_bounds, dtype=jnp.float32)

    raw_delta = jax.random.normal(
        key,
        shape=(config.num_samples, config.horizon, 4),
        dtype=jnp.float32,
    ) * delta_sigma
    smoothed_delta = smooth_noise_batch(raw_delta, config.noise_smoothing_kernel)
    bounded_delta = jnp.clip(smoothed_delta, -delta_bounds, delta_bounds)

    candidate_actions = clip_action(base_action_plan[None, :, :] + bounded_delta, low, high)

    costs = jax.vmap(
        lambda candidate_action_seq: single_rollout_cost(
            initial_state,
            candidate_action_seq,
            initial_prev_action,
            W,
            B,
            canyon_north_samples_ft,
            canyon_width_samples_ft,
            canyon_center_east_samples_ft,
            canyon_heading_samples_rad,
            config,
        )
    )(candidate_actions)

    perturbation_cost = config.gamma_ * jnp.sum(
        jnp.square(bounded_delta / jnp.maximum(delta_sigma, 1e-6)),
        axis=(1, 2),
    )
    action_diff = candidate_actions[:, 1:, :] - candidate_actions[:, :-1, :]
    smoothness_cost = config.smoothness_penalty_weight * jnp.sum(jnp.square(action_diff), axis=(1, 2))
    total_costs = costs + perturbation_cost + smoothness_cost

    weights = softmax_weights(total_costs, config.lambda_)
    weighted_delta = jnp.tensordot(weights, bounded_delta, axes=(0, 0))
    optimized_plan = clip_action(base_action_plan + weighted_delta, low, high)

    return optimized_plan, total_costs, state_seq_best(initial_state, optimized_plan, W, B), candidate_actions

def state_seq_best(initial_state, plan, W, B):
    return rollout_trajectory(initial_state, plan, W, B)

class JaxMPPIController:
    def __init__(
        self,
        config=None,
        canyon_north_samples_ft=None,
        canyon_width_samples_ft=None,
        canyon_center_east_samples_ft=None,
        canyon_centerline_heading_rad_samples=None,
    ):
        self.config = config or JaxMPPIConfig()
        self.W, self.B = load_nominal_weights()
        self.key = jax.random.PRNGKey(self.config.seed)
        self.base_plan = jnp.zeros((self.config.horizon, 4), dtype=jnp.float32)
        self.base_plan = self.base_plan.at[:, 3].set(0.55)
        self._last_action = np.array([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

        if canyon_north_samples_ft is None or canyon_width_samples_ft is None:
            north_default = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
            width_default = np.asarray(canyon_width(jnp.asarray(north_default)), dtype=np.float32)
            center_default = np.zeros_like(north_default, dtype=np.float32)
            heading_default = np.zeros_like(north_default, dtype=np.float32)

            self._canyon_north_np = north_default
            self._canyon_width_np = width_default
            self._canyon_center_east_np = center_default
            self._canyon_heading_rad_np = heading_default
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
            self._canyon_north_np = north_np[order]
            self._canyon_width_np = width_np[order]
            self._canyon_center_east_np = center_east_np[order]
            self._canyon_heading_rad_np = heading_np[order]

        self.canyon_north_samples_ft = jnp.asarray(self._canyon_north_np)
        self.canyon_width_samples_ft = jnp.asarray(self._canyon_width_np)
        self.canyon_center_east_samples_ft = jnp.asarray(self._canyon_center_east_np)
        self.canyon_heading_samples_rad = jnp.asarray(self._canyon_heading_rad_np)

    @staticmethod
    def _wrap_angle_rad(angle_rad):
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    def _centerline_east_ft(self, p_n_ft):
        return float(np.interp(float(p_n_ft), self._canyon_north_np, self._canyon_center_east_np))

    def _centerline_heading_rad(self, p_n_ft):
        return float(np.interp(float(p_n_ft), self._canyon_north_np, self._canyon_heading_rad_np))

    def get_lateral_error_ft(self, p_n_ft, p_e_ft):
        return float(p_e_ft - self._centerline_east_ft(p_n_ft))

    def get_canyon_width_ft(self, p_n_ft):
        return float(np.interp(float(p_n_ft), self._canyon_north_np, self._canyon_width_np))

    def get_centerline_heading_rad(self, p_n_ft):
        return self._centerline_heading_rad(p_n_ft)

    def reset(self, seed=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(int(seed) + int(self.config.seed))
        else:
            self.key = jax.random.PRNGKey(self.config.seed)

        self.base_plan = jnp.zeros((self.config.horizon, 4), dtype=jnp.float32)
        self.base_plan = self.base_plan.at[:, 3].set(0.55)
        self._last_action = np.array([0.0, 0.0, 0.0, 0.55], dtype=np.float32)
        self._cached_action = self._last_action.copy()
        self._last_replan_step = -10**9
        self._step_index = 0
        self._latest_render_debug = None

    def _update_render_debug(self, initial_state, candidate_actions, final_state_seq):
        if not bool(self.config.debug_render_plans):
            self._latest_render_debug = None
            return
        if candidate_actions is None or final_state_seq is None:
            self._latest_render_debug = None
            return

        limit = int(min(max(int(self.config.debug_num_trajectories), 1), int(candidate_actions.shape[0])))
        sampled_actions = candidate_actions[:limit]
        sampled_state_seq = rollout_trajectory_batch(initial_state, sampled_actions, self.W, self.B)

        initial_state_tiled = jnp.broadcast_to(
            initial_state[None, None, :],
            (limit, 1, int(initial_state.shape[0])),
        )
        sampled_with_start = jnp.concatenate([initial_state_tiled, sampled_state_seq], axis=1)
        final_with_start = jnp.concatenate([initial_state[None, :], final_state_seq], axis=0)

        sampled_np = np.asarray(sampled_with_start, dtype=np.float32)
        final_np = np.asarray(final_with_start, dtype=np.float32)

        self._latest_render_debug = {
            "candidate_xy": sampled_np[:, :, :2].copy(),
            "candidate_h_ft": sampled_np[:, :, 2].copy(),
            "final_xy": final_np[:, :2].copy(),
            "final_h_ft": final_np[:, 2].copy(),
        }

    def _tail_action(self, state_dict):
        alt_error_ft = float(state_dict["h"] - self.config.target_altitude_ft)
        p_rate = float(state_dict["p"])
        q_rate = float(state_dict["q"])
        r_rate = float(state_dict["r"])
        p_n_ft = float(state_dict["p_N"])
        p_e_ft = float(state_dict["p_E"])
        width_ft = self.get_canyon_width_ft(p_n_ft)
        raw_lateral_error_ft = self.get_lateral_error_ft(p_n_ft, p_e_ft)
        if abs(raw_lateral_error_ft) > 4.0 * max(width_ft, 1.0):
            lateral_error_ft = 0.0
            heading_error_rad = 0.0
        else:
            lateral_error_ft = raw_lateral_error_ft
            heading_ref_rad = self._centerline_heading_rad(p_n_ft)
            heading_error_rad = self._wrap_angle_rad(float(state_dict["psi"]) - heading_ref_rad)
        speed_fps = float(
            np.sqrt(
                float(state_dict["u"]) ** 2
                + float(state_dict["v"]) ** 2
                + float(state_dict["w"]) ** 2
            )
        )

        roll_cmd = np.clip(-0.0028 * lateral_error_ft - 0.60 * heading_error_rad - 0.20 * p_rate, -0.55, 0.55)
        pitch_cmd = np.clip(0.0018 * alt_error_ft + 0.18 * q_rate, -0.40, 0.40)
        yaw_cmd = np.clip(-0.20 * r_rate - 0.12 * heading_error_rad, -0.30, 0.30)
        throttle_cmd = np.clip(0.58 + 0.0012 * (450.0 - speed_fps), 0.35, 0.95)
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)

    def _shift_plan(self, state_dict):
        shifted = np.roll(np.asarray(self.base_plan, dtype=np.float32), shift=-1, axis=0)
        shifted[-1] = self._tail_action(state_dict)
        self.base_plan = jnp.asarray(shifted, dtype=jnp.float32)

    def _optimize_step(self, state, prev_action, key):
        return mppi_optimize_step(
            state,
            self.base_plan,
            prev_action,
            key,
            self.W,
            self.B,
            self.canyon_north_samples_ft,
            self.canyon_width_samples_ft,
            self.canyon_center_east_samples_ft,
            self.canyon_heading_samples_rad,
            self.config,
        )

    def get_action(self, state_dict):
        if self._step_index - self._last_replan_step < int(max(self.config.replan_interval, 1)):
            self._step_index += 1
            return self._cached_action.copy()

        # State: p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi
        state = jnp.array([
            state_dict['p_N'],
            state_dict['p_E'],
            state_dict['h'],
            state_dict['u'],
            state_dict['v'],
            state_dict['w'],
            state_dict['p'],
            state_dict['q'],
            state_dict['r'],
            state_dict['phi'],
            state_dict['theta'],
            state_dict['psi'],
        ], dtype=jnp.float32)

        self._shift_plan(state_dict)

        prev_action = jnp.asarray(self._last_action, dtype=jnp.float32)
        last_candidate_actions = None
        best_seq = None
        for idx in range(self.config.optimization_steps):
            self.key, subkey = jax.random.split(self.key)
            self.base_plan, costs, best_seq, candidate_actions = self._optimize_step(state, prev_action, subkey)
            last_candidate_actions = candidate_actions
            if jnp.isnan(costs).any() or jnp.isinf(costs).any():
                print(f"NAN/INF in costs at opt step {idx}!")
                print(f"Costs sample: {costs[:10]}")
                self._latest_render_debug = None
                self._step_index += 1
                return self._cached_action.copy()

        self._update_render_debug(state, last_candidate_actions, best_seq)

        action = np.asarray(self.base_plan[0], dtype=np.float32)
        self._last_action = action.copy()
        self._cached_action = action.copy()
        self._last_replan_step = self._step_index
        self._step_index += 1
        return action

    def get_render_debug(self):
        return self._latest_render_debug


class JaxSmoothMPPIController(JaxMPPIController):
    def __init__(
        self,
        config=None,
        canyon_north_samples_ft=None,
        canyon_width_samples_ft=None,
        canyon_center_east_samples_ft=None,
        canyon_centerline_heading_rad_samples=None,
    ):
        super().__init__(
            config=config or JaxSmoothMPPIConfig(),
            canyon_north_samples_ft=canyon_north_samples_ft,
            canyon_width_samples_ft=canyon_width_samples_ft,
            canyon_center_east_samples_ft=canyon_center_east_samples_ft,
            canyon_centerline_heading_rad_samples=canyon_centerline_heading_rad_samples,
        )

    def _optimize_step(self, state, prev_action, key):
        return smooth_mppi_optimize_step(
            state,
            self.base_plan,
            prev_action,
            key,
            self.W,
            self.B,
            self.canyon_north_samples_ft,
            self.canyon_width_samples_ft,
            self.canyon_center_east_samples_ft,
            self.canyon_heading_samples_rad,
            self.config,
        )
