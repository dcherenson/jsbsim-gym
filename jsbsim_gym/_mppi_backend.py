import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from itertools import combinations_with_replacement


MPPI_FEATURE_NAMES = (
    "alpha",
    "beta",
    "mach",
    "p",
    "q",
    "r",
    "delta_t",
    "delta_e",
    "delta_a",
    "delta_r",
)

MPPI_TARGET_NAMES = (
    "C_X",
    "C_Y",
    "C_Z",
    "C_L",
    "C_M",
    "C_N",
)

IXX = 9496.0
IYY = 55814.0
IZZ = 63100.0
IXZ = -982.0
INERTIA_DET = IXX * IZZ - IXZ * IXZ

WING_AREA_FT2 = 300.0
WING_SPAN_FT = 30.0
MEAN_AERODYNAMIC_CHORD_FT = 11.32
DEFAULT_MASS_LBS = 17400.0 + 230.0 + 2000.0
DEFAULT_MASS_SLUGS = DEFAULT_MASS_LBS / 32.174
STANDARD_SPEED_OF_SOUND_FPS = 1116.45
AIR_DENSITY_SLUG_FT3 = 0.0023769
MIN_QBAR_PSF = 1.0
DT = 1.0 / 30.0
G_FTPS2 = 32.174


@dataclass(frozen=True)
class JaxMPPIConfig:
    horizon: int = 40
    num_samples: int = 4000
    optimization_steps: int = 2
    replan_interval: int = 1
    lambda_: float = 1.0
    gamma_: float = 0.05
    action_noise_std: tuple = (0.16, 0.14, 0.12, 0.08)
    action_low: tuple = (-1.0, -1.0, -1.0, 0.0)
    action_high: tuple = (1.0, 1.0, 1.0, 1.0)
    state_tracking_weights: tuple = (0.05, 0.05, 0.05, 10.0, 10.0, 8.0)
    terrain_collision_penalty: float = 1.0e6
    terrain_repulsion_scale: float = 1.0e5
    terrain_decay_rate_ft_inv: float = 0.03
    terrain_safe_clearance_ft: float = 40.0 * 3.28084
    control_rate_weights: tuple = (15.0, 20.0, 5.0, 2.0)
    nz_limit_g: float = 9.0
    nz_penalty_weight: float = 1.0e4
    alpha_limit_rad: float = np.deg2rad(25.0)
    alpha_penalty_weight: float = 1.0e6
    debug_render_plans: bool = True
    debug_num_trajectories: int = 96
    seed: int = 42

    def tree_flatten(self):
        children = (
            self.lambda_,
            self.gamma_,
            self.action_noise_std,
            self.action_low,
            self.action_high,
            self.state_tracking_weights,
            self.terrain_collision_penalty,
            self.terrain_repulsion_scale,
            self.terrain_decay_rate_ft_inv,
            self.terrain_safe_clearance_ft,
            self.control_rate_weights,
            self.nz_limit_g,
            self.nz_penalty_weight,
            self.alpha_limit_rad,
            self.alpha_penalty_weight,
        )
        aux_data = (
            self.horizon,
            self.num_samples,
            self.optimization_steps,
            self.replan_interval,
            self.debug_render_plans,
            self.debug_num_trajectories,
            self.seed,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            lambda_,
            gamma_,
            action_noise_std,
            action_low,
            action_high,
            state_tracking_weights,
            terrain_collision_penalty,
            terrain_repulsion_scale,
            terrain_decay_rate_ft_inv,
            terrain_safe_clearance_ft,
            control_rate_weights,
            nz_limit_g,
            nz_penalty_weight,
            alpha_limit_rad,
            alpha_penalty_weight,
        ) = children
        (
            horizon,
            num_samples,
            optimization_steps,
            replan_interval,
            debug_render_plans,
            debug_num_trajectories,
            seed,
        ) = aux_data
        return cls(
            horizon=horizon,
            num_samples=num_samples,
            optimization_steps=optimization_steps,
            replan_interval=replan_interval,
            lambda_=lambda_,
            gamma_=gamma_,
            action_noise_std=action_noise_std,
            action_low=action_low,
            action_high=action_high,
            state_tracking_weights=state_tracking_weights,
            terrain_collision_penalty=terrain_collision_penalty,
            terrain_repulsion_scale=terrain_repulsion_scale,
            terrain_decay_rate_ft_inv=terrain_decay_rate_ft_inv,
            terrain_safe_clearance_ft=terrain_safe_clearance_ft,
            control_rate_weights=control_rate_weights,
            nz_limit_g=nz_limit_g,
            nz_penalty_weight=nz_penalty_weight,
            alpha_limit_rad=alpha_limit_rad,
            alpha_penalty_weight=alpha_penalty_weight,
            debug_render_plans=debug_render_plans,
            debug_num_trajectories=debug_num_trajectories,
            seed=seed,
        )


jax.tree_util.register_pytree_node_class(JaxMPPIConfig)


@dataclass(frozen=True)
class JaxSmoothMPPIConfig(JaxMPPIConfig):
    delta_noise_std: tuple = (0.08, 0.12, 0.08, 0.06)
    delta_action_bounds: tuple = (0.18, 0.26, 0.14, 0.10)
    noise_smoothing_kernel: tuple = (0.10, 0.20, 0.40, 0.20, 0.10)
    seed: int = 101

    def tree_flatten(self):
        base_children, base_aux = super().tree_flatten()
        return base_children + (
            self.delta_noise_std,
            self.delta_action_bounds,
            self.noise_smoothing_kernel,
        ), base_aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        delta_noise_std = children[-3]
        delta_action_bounds = children[-2]
        noise_smoothing_kernel = children[-1]
        base = JaxMPPIConfig.tree_unflatten(aux_data, children[:-3])
        import dataclasses

        return cls(
            **{field.name: getattr(base, field.name) for field in dataclasses.fields(JaxMPPIConfig)},
            delta_noise_std=delta_noise_std,
            delta_action_bounds=delta_action_bounds,
            noise_smoothing_kernel=noise_smoothing_kernel,
        )


jax.tree_util.register_pytree_node_class(JaxSmoothMPPIConfig)


def _build_polynomial_powers(n_features, degree, include_bias):
    powers = []
    if include_bias:
        powers.append(np.zeros(n_features, dtype=np.int32))
    for deg in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(n_features), deg):
            row = np.zeros(n_features, dtype=np.int32)
            for idx in combo:
                row[idx] += 1
            powers.append(row)
    return np.asarray(powers, dtype=np.int32)


def load_nominal_weights():
    import os

    path = os.path.join(os.path.dirname(__file__), "nominal_coeff_weights.npz")
    data = np.load(path, allow_pickle=True)
    W = np.asarray(data["W"])
    B = np.asarray(data["B"])

    if "feature_names" not in data:
        raise ValueError("nominal_coeff_weights.npz missing required metadata 'feature_names'.")
    feature_names = tuple(str(x) for x in data["feature_names"])
    if feature_names != MPPI_FEATURE_NAMES:
        raise ValueError(
            "nominal_coeff_weights.npz feature set mismatch. "
            f"Expected {MPPI_FEATURE_NAMES}, got {feature_names}. "
            "Regenerate weights via: uv run python -m jsbsim_gym.calibration"
        )

    if "target_names" not in data:
        raise ValueError("nominal_coeff_weights.npz missing required metadata 'target_names'.")
    target_names = tuple(str(x) for x in data["target_names"])
    if target_names != MPPI_TARGET_NAMES:
        raise ValueError(
            "nominal_coeff_weights.npz target set mismatch. "
            f"Expected {MPPI_TARGET_NAMES}, got {target_names}."
        )

    if "model_space" not in data:
        raise ValueError("nominal_coeff_weights.npz missing required metadata 'model_space'.")
    model_space = str(data["model_space"][0])
    if model_space != "aerodynamic_coefficients":
        raise ValueError(
            "nominal_coeff_weights.npz model space mismatch. "
            f"Expected 'aerodynamic_coefficients', got '{model_space}'."
        )

    if "poly_degree" not in data:
        raise ValueError("nominal_coeff_weights.npz missing required metadata 'poly_degree'.")
    poly_degree = int(data["poly_degree"][0])
    include_bias = bool(int(data["include_bias"][0])) if "include_bias" in data else True
    poly_powers = _build_polynomial_powers(len(MPPI_FEATURE_NAMES), poly_degree, include_bias)
    expected_rows = int(poly_powers.shape[0])

    if W.shape[0] != expected_rows:
        raise ValueError(
            "nominal_coeff_weights.npz has incompatible polynomial size. "
            f"Expected W rows={expected_rows}, got {W.shape[0]}. "
            "Regenerate weights via: uv run python -m jsbsim_gym.calibration"
        )
    if B.shape[0] != 6:
        raise ValueError(f"Expected 6 output channels in B, got shape {B.shape}")

    return jnp.asarray(W), jnp.asarray(B), jnp.asarray(poly_powers, dtype=jnp.int32)


def expand_poly(x, poly_powers):
    return jnp.prod(jnp.power(x[None, :], poly_powers), axis=1)


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


def f16_kinematics_step(state, action, W, B, poly_powers):
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi = state

    delta_a = action[0]
    delta_e = action[1]
    delta_r = action[2]
    delta_t = action[3]

    alpha = jnp.arctan2(w, jnp.maximum(u, 1.0))
    V_sq = u * u + v * v + w * w
    V = jnp.sqrt(jnp.maximum(V_sq, 1.0))
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    mach = V / STANDARD_SPEED_OF_SOUND_FPS

    features = jnp.array([alpha, beta, mach, p, q, r, delta_t, delta_e, delta_a, delta_r], dtype=jnp.float32)
    phi_vec = expand_poly(features, poly_powers)
    preds = jnp.dot(phi_vec, W) + B
    C_X, C_Y, C_Z, C_L, C_M, C_N = preds

    qbar_psf = jnp.maximum(0.5 * AIR_DENSITY_SLUG_FT3 * V_sq, MIN_QBAR_PSF)
    force_scale = qbar_psf * WING_AREA_FT2 / DEFAULT_MASS_SLUGS
    roll_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT
    pitch_moment_scale = qbar_psf * WING_AREA_FT2 * MEAN_AERODYNAMIC_CHORD_FT
    yaw_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT

    X = C_X * force_scale
    Y = C_Y * force_scale
    Z = C_Z * force_scale
    L = C_L * roll_moment_scale
    M = C_M * pitch_moment_scale
    N = C_N * yaw_moment_scale

    u_dot = X + r * v - q * w - G_FTPS2 * jnp.sin(theta)
    v_dot = Y + p * w - r * u + G_FTPS2 * jnp.sin(phi) * jnp.cos(theta)
    w_dot = Z + q * u - p * v + G_FTPS2 * jnp.cos(phi) * jnp.cos(theta)

    p_dot, q_dot, r_dot = moments_to_angular_rate_derivatives(p, q, r, L, M, N)

    u_dot = jnp.clip(u_dot, -1000.0, 1000.0)
    v_dot = jnp.clip(v_dot, -1000.0, 1000.0)
    w_dot = jnp.clip(w_dot, -1000.0, 1000.0)
    p_dot = jnp.clip(p_dot, -50.0, 50.0)
    q_dot = jnp.clip(q_dot, -50.0, 50.0)
    r_dot = jnp.clip(r_dot, -50.0, 50.0)

    t_theta = jnp.tan(theta)
    phi_dot = p + q * jnp.sin(phi) * t_theta + r * jnp.cos(phi) * t_theta
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = (q * jnp.sin(phi) + r * jnp.cos(phi)) / jnp.cos(theta)

    phi_dot = jnp.clip(phi_dot, -50.0, 50.0)
    theta_dot = jnp.clip(theta_dot, -50.0, 50.0)
    psi_dot = jnp.clip(psi_dot, -50.0, 50.0)

    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)

    p_N_dot = (
        u * (c_theta * c_psi)
        + v * (s_phi * s_theta * c_psi - c_phi * s_psi)
        + w * (c_phi * s_theta * c_psi + s_phi * s_psi)
    )
    p_E_dot = (
        u * (c_theta * s_psi)
        + v * (s_phi * s_theta * s_psi + c_phi * c_psi)
        + w * (c_phi * s_theta * s_psi - s_phi * c_psi)
    )
    h_dot = u * s_theta - v * (s_phi * c_theta) - w * (c_phi * c_theta)

    p_N_dot = jnp.clip(p_N_dot, -3000.0, 3000.0)
    p_E_dot = jnp.clip(p_E_dot, -3000.0, 3000.0)
    h_dot = jnp.clip(h_dot, -3000.0, 3000.0)

    return jnp.array(
        [
            p_N + DT * p_N_dot,
            p_E + DT * p_E_dot,
            h + DT * h_dot,
            u + DT * u_dot,
            v + DT * v_dot,
            w + DT * w_dot,
            p + DT * p_dot,
            q + DT * q_dot,
            r + DT * r_dot,
            phi + DT * phi_dot,
            theta + DT * theta_dot,
            psi + DT * psi_dot,
        ],
        dtype=jnp.float32,
    )


def f16_kinematics_step_with_load_factors(state, action, W, B, poly_powers):
    state_core = state[:12]
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi = state_core

    delta_a = action[0]
    delta_e = action[1]
    delta_r = action[2]
    delta_t = action[3]

    alpha = jnp.arctan2(w, jnp.maximum(u, 1.0))
    V_sq = u * u + v * v + w * w
    V = jnp.sqrt(jnp.maximum(V_sq, 1.0))
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    mach = V / STANDARD_SPEED_OF_SOUND_FPS

    features = jnp.array([alpha, beta, mach, p, q, r, delta_t, delta_e, delta_a, delta_r], dtype=jnp.float32)
    phi_vec = expand_poly(features, poly_powers)
    preds = jnp.dot(phi_vec, W) + B
    C_X, C_Y, C_Z, C_L, C_M, C_N = preds

    qbar_psf = jnp.maximum(0.5 * AIR_DENSITY_SLUG_FT3 * V_sq, MIN_QBAR_PSF)
    force_scale = qbar_psf * WING_AREA_FT2 / DEFAULT_MASS_SLUGS
    roll_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT
    pitch_moment_scale = qbar_psf * WING_AREA_FT2 * MEAN_AERODYNAMIC_CHORD_FT
    yaw_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT

    X = C_X * force_scale
    Y = C_Y * force_scale
    Z = C_Z * force_scale
    L = C_L * roll_moment_scale
    M = C_M * pitch_moment_scale
    N = C_N * yaw_moment_scale

    u_dot = X + r * v - q * w - G_FTPS2 * jnp.sin(theta)
    v_dot = Y + p * w - r * u + G_FTPS2 * jnp.sin(phi) * jnp.cos(theta)
    w_dot = Z + q * u - p * v + G_FTPS2 * jnp.cos(phi) * jnp.cos(theta)

    p_dot, q_dot, r_dot = moments_to_angular_rate_derivatives(p, q, r, L, M, N)

    u_dot = jnp.clip(u_dot, -1000.0, 1000.0)
    v_dot = jnp.clip(v_dot, -1000.0, 1000.0)
    w_dot = jnp.clip(w_dot, -1000.0, 1000.0)
    p_dot = jnp.clip(p_dot, -50.0, 50.0)
    q_dot = jnp.clip(q_dot, -50.0, 50.0)
    r_dot = jnp.clip(r_dot, -50.0, 50.0)

    t_theta = jnp.tan(theta)
    phi_dot = p + q * jnp.sin(phi) * t_theta + r * jnp.cos(phi) * t_theta
    theta_dot = q * jnp.cos(phi) - r * jnp.sin(phi)
    psi_dot = (q * jnp.sin(phi) + r * jnp.cos(phi)) / jnp.cos(theta)

    phi_dot = jnp.clip(phi_dot, -50.0, 50.0)
    theta_dot = jnp.clip(theta_dot, -50.0, 50.0)
    psi_dot = jnp.clip(psi_dot, -50.0, 50.0)

    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)

    p_N_dot = (
        u * (c_theta * c_psi)
        + v * (s_phi * s_theta * c_psi - c_phi * s_psi)
        + w * (c_phi * s_theta * c_psi + s_phi * s_psi)
    )
    p_E_dot = (
        u * (c_theta * s_psi)
        + v * (s_phi * s_theta * s_psi + c_phi * c_psi)
        + w * (c_phi * s_theta * s_psi - s_phi * c_psi)
    )
    h_dot = u * s_theta - v * (s_phi * c_theta) - w * (c_phi * c_theta)

    p_N_dot = jnp.clip(p_N_dot, -3000.0, 3000.0)
    p_E_dot = jnp.clip(p_E_dot, -3000.0, 3000.0)
    h_dot = jnp.clip(h_dot, -3000.0, 3000.0)

    state_next_core = jnp.array(
        [
            p_N + DT * p_N_dot,
            p_E + DT * p_E_dot,
            h + DT * h_dot,
            u + DT * u_dot,
            v + DT * v_dot,
            w + DT * w_dot,
            p + DT * p_dot,
            q + DT * q_dot,
            r + DT * r_dot,
            phi + DT * phi_dot,
            theta + DT * theta_dot,
            psi + DT * psi_dot,
        ],
        dtype=jnp.float32,
    )

    ny = jnp.clip(Y / G_FTPS2, -100.0, 100.0)
    nz = jnp.clip(-Z / G_FTPS2, -100.0, 100.0)
    return jnp.concatenate([state_next_core, jnp.asarray([ny, nz], dtype=jnp.float32)], axis=0)


def rollout_trajectory_with_load_factors(initial_state, action_seq, W, B, poly_powers):
    def step_fn(state, action):
        next_state = f16_kinematics_step_with_load_factors(state, action, W, B, poly_powers)
        return next_state, next_state

    _, state_seq = jax.lax.scan(step_fn, initial_state, action_seq)
    return state_seq


@jax.jit
def rollout_trajectory_batch_with_load_factors(initial_state, action_batch, W, B, poly_powers):
    return jax.vmap(
        lambda action_seq: rollout_trajectory_with_load_factors(initial_state, action_seq, W, B, poly_powers)
    )(action_batch)


def smooth_noise_batch(noise, kernel_weights):
    kernel = jnp.asarray(kernel_weights, dtype=jnp.float32)
    kernel = kernel / jnp.maximum(jnp.sum(kernel), 1e-6)
    pad = int(kernel.shape[0] // 2)

    padded = jnp.pad(noise, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    kernel_conv = kernel[:, None, None]

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


def _terrain_elevation_ft_at(p_n_ft, p_e_ft, north_axis_ft, east_axis_ft, terrain_elevation_ft):
    n_hi = jnp.clip(jnp.searchsorted(north_axis_ft, p_n_ft, side="right"), 1, north_axis_ft.shape[0] - 1)
    e_hi = jnp.clip(jnp.searchsorted(east_axis_ft, p_e_ft, side="right"), 1, east_axis_ft.shape[0] - 1)
    n_lo = n_hi - 1
    e_lo = e_hi - 1

    n0 = north_axis_ft[n_lo]
    n1 = north_axis_ft[n_hi]
    e0 = east_axis_ft[e_lo]
    e1 = east_axis_ft[e_hi]

    dn = jnp.where(jnp.abs(n1 - n0) > 1e-6, (p_n_ft - n0) / (n1 - n0), 0.0)
    de = jnp.where(jnp.abs(e1 - e0) > 1e-6, (p_e_ft - e0) / (e1 - e0), 0.0)
    dn = jnp.clip(dn, 0.0, 1.0)
    de = jnp.clip(de, 0.0, 1.0)

    z00 = terrain_elevation_ft[n_lo, e_lo]
    z01 = terrain_elevation_ft[n_lo, e_hi]
    z10 = terrain_elevation_ft[n_hi, e_lo]
    z11 = terrain_elevation_ft[n_hi, e_hi]

    z0 = (1.0 - de) * z00 + de * z01
    z1 = (1.0 - de) * z10 + de * z11
    return (1.0 - dn) * z0 + dn * z1


def _reference_window(reference_states_ft_rad, start_index, horizon):
    indices = jnp.minimum(
        jnp.asarray(start_index, dtype=jnp.int32) + jnp.arange(1, horizon + 1, dtype=jnp.int32),
        reference_states_ft_rad.shape[0] - 1,
    )
    return reference_states_ft_rad[indices]


def single_rollout_cost_from_states(
    initial_state,
    state_seq,
    action_seq,
    initial_prev_action,
    reference_start_index,
    reference_states_ft_rad,
    terrain_north_samples_ft,
    terrain_east_samples_ft,
    terrain_elevation_ft,
    config: JaxMPPIConfig,
):
    del initial_state

    horizon = state_seq.shape[0]
    low = jnp.asarray(config.action_low, dtype=jnp.float32)
    high = jnp.asarray(config.action_high, dtype=jnp.float32)
    bounded_actions = clip_action(action_seq, low, high)
    initial_prev_action = clip_action(initial_prev_action, low, high)
    prev_actions = jnp.concatenate([initial_prev_action[None, :], bounded_actions[:-1, :]], axis=0)

    ref_seq = _reference_window(reference_states_ft_rad, reference_start_index, horizon)

    state_errors = jnp.stack(
        [
            state_seq[:, 0] - ref_seq[:, 0],
            state_seq[:, 1] - ref_seq[:, 1],
            state_seq[:, 2] - ref_seq[:, 2],
            wrap_angle_rad(state_seq[:, 9] - ref_seq[:, 3]),
            wrap_angle_rad(state_seq[:, 10] - ref_seq[:, 4]),
            wrap_angle_rad(state_seq[:, 11] - ref_seq[:, 5]),
        ],
        axis=1,
    )
    state_weights = jnp.asarray(config.state_tracking_weights, dtype=jnp.float32)
    state_cost = jnp.sum(state_weights[None, :] * jnp.square(state_errors), axis=1)

    terrain_elevation_seq_ft = jax.vmap(
        lambda p_n_ft, p_e_ft: _terrain_elevation_ft_at(
            p_n_ft,
            p_e_ft,
            terrain_north_samples_ft,
            terrain_east_samples_ft,
            terrain_elevation_ft,
        )
    )(state_seq[:, 0], state_seq[:, 1])
    hagl_ft = state_seq[:, 2] - terrain_elevation_seq_ft
    terrain_soft_cost = config.terrain_repulsion_scale * jnp.exp(
        -config.terrain_decay_rate_ft_inv * (hagl_ft - config.terrain_safe_clearance_ft)
    )
    terrain_cost = jnp.where(
        hagl_ft <= 0.0,
        config.terrain_collision_penalty,
        jnp.minimum(terrain_soft_cost, config.terrain_collision_penalty),
    )

    action_rate = bounded_actions - prev_actions
    control_rate_weights = jnp.asarray(config.control_rate_weights, dtype=jnp.float32)
    rate_cost = jnp.sum(control_rate_weights[None, :] * jnp.square(action_rate), axis=1)

    alpha_rad = jnp.arctan2(state_seq[:, 5], jnp.maximum(state_seq[:, 3], 1.0))
    nz_excess = jnp.maximum(jnp.abs(state_seq[:, 13]) - config.nz_limit_g, 0.0)
    alpha_excess = jnp.maximum(alpha_rad - config.alpha_limit_rad, 0.0)
    limit_cost = (
        config.nz_penalty_weight * jnp.square(nz_excess)
        + config.alpha_penalty_weight * jnp.square(alpha_excess)
    )

    stage_cost = state_cost + terrain_cost + rate_cost + limit_cost
    terrain_collision = hagl_ft <= 0.0
    prior_collision = jnp.concatenate([jnp.asarray([False]), terrain_collision[:-1]], axis=0)
    active = jnp.cumsum(prior_collision.astype(jnp.int32)) == 0
    stage_cost = jnp.where(active, stage_cost, 0.0)
    return jnp.sum(stage_cost)


def build_rollout_state_batch_fn(W, B, poly_powers):
    def rollout_states(initial_state, action_batch):
        return rollout_trajectory_batch_with_load_factors(initial_state, action_batch, W, B, poly_powers)

    return jax.jit(rollout_states)


def build_rollout_cost_from_states_fn(
    reference_states_ft_rad,
    terrain_north_samples_ft,
    terrain_east_samples_ft,
    terrain_elevation_ft,
    config: JaxMPPIConfig,
):
    def rollout_costs_from_states(initial_state, state_batch, action_batch, initial_prev_action, reference_start_index):
        return jax.vmap(
            lambda state_seq, action_seq: single_rollout_cost_from_states(
                initial_state,
                state_seq,
                action_seq,
                initial_prev_action,
                reference_start_index,
                reference_states_ft_rad,
                terrain_north_samples_ft,
                terrain_east_samples_ft,
                terrain_elevation_ft,
                config,
            )
        )(state_batch, action_batch)

    return jax.jit(rollout_costs_from_states)


def build_rollout_positions_fn(W, B, poly_powers):
    def rollout_positions(initial_state, action_batch):
        state_seq = rollout_trajectory_batch_with_load_factors(initial_state, action_batch, W, B, poly_powers)
        initial_state_tiled = jnp.broadcast_to(
            initial_state[None, None, :],
            (action_batch.shape[0], 1, initial_state.shape[0]),
        )
        return jnp.concatenate([initial_state_tiled, state_seq], axis=1)

    return jax.jit(rollout_positions)
