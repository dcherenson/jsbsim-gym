import jax
import jax.numpy as jnp
import numpy as np
import functools
from dataclasses import dataclass

@dataclass(frozen=True)
class JaxMPPIConfig:
    horizon: int = 40
    num_samples: int = 1024
    optimization_steps: int = 2
    lambda_: float = 1.0
    action_noise_std: tuple = (0.2, 0.2, 0.2, 0.1) # roll, pitch, yaw, throttle
    action_low: tuple = (-1.0, -1.0, -1.0, 0.0)
    action_high: tuple = (1.0, 1.0, 1.0, 1.0)
    crash_penalty: float = 1e5
    progress_weight: float = 1.0
    center_weight: float = 0.5
    action_diff_weight: float = 5.0
    action_l2_weight: float = 1.0

def load_nominal_weights():
    # Load from the npz file
    import os
    path = os.path.join(os.path.dirname(__file__), "mppi_nominal_weights.npz")
    data = np.load(path)
    return jnp.asarray(data['W']), jnp.asarray(data['B'])

def expand_poly(x):
    # x shape (8,)
    # The polynomials features order: 1, x, x^2 + cross terms
    # Using the same order as sklearn PolynomialFeatures(degree=2)
    # Features: alpha, beta, p, q, r, delta_e, delta_a, delta_r
    ones = jnp.ones((1,))
    
    quad = []
    for i in range(8):
        for j in range(i, 8):
            quad.append(x[i] * x[j])
            
    quad_stack = jnp.stack(quad)
    return jnp.concatenate([ones, x, quad_stack])

def canyon_width(p_N):
    W_base = 300.0
    W_amp = 220.0
    W_freq = 15000.0
    return W_base + W_amp * jnp.sin(2.0 * jnp.pi * p_N / W_freq)

def f16_kinematics_step(state, action, W, B):
    # State: p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi = state
    
    # FCS
    delta_a = action[0]
    delta_e = action[1]
    delta_r = action[2]
    # action[3] is throttle, ignored by aerodynamic forces model directly, 
    # but we will just pass it, wait, nominal model features are only 8:
    # alpha, beta, p, q, r, delta_e, delta_a, delta_r
    
    alpha = jnp.arctan2(w, jnp.maximum(u, 1.0))
    V_sq = u*u + v*v + w*w
    V = jnp.sqrt(jnp.maximum(V_sq, 1.0))
    beta = jnp.arcsin(jnp.clip(v / V, -1.0, 1.0))
    
    features = jnp.array([alpha, beta, p, q, r, delta_e, delta_a, delta_r])
    phi_vec = expand_poly(features)
    
    preds = jnp.dot(phi_vec, W) + B
    X, Y, Z, L, M, N = preds
    
    G = 32.174
    u_dot = X + r*v - q*w - G*jnp.sin(theta)
    # Thrust approximation! Thrust pushes u forward. 
    # The nominal model was trained on data with throttle! 
    # Wait, the nominal features didn't include throttle! 
    # That means the Ridge model swallowed thrust into the bias or it wasn't varying.
    # In DataCollectionEnv persistent excitation, throttle was fixed or random? 
    # We will add an artificial thrust force loosely, or just rely on the baseline.
    # Let's add simple artificial thrust to u_dot proportional to throttle just in case.
    # Max thrust approx 30000 lbf, mass approx 20000 lb, so ~1.5 Gs -> ~ 50 fps2
    throttle = action[3]
    u_dot += throttle * 20.0  # basic push
    
    v_dot = Y + p*w - r*u + G*jnp.sin(phi)*jnp.cos(theta)
    w_dot = Z + q*u - p*v + G*jnp.cos(phi)*jnp.cos(theta)
    
    p_dot = L
    q_dot = M
    r_dot = N
    
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

def compute_cost(state_seq, action_seq, config: JaxMPPIConfig):
    p_N = state_seq[:, 0]
    p_E = state_seq[:, 1]
    h = state_seq[:, 2]
    
    W_c = jax.vmap(canyon_width)(p_N)
    
    oob = (jnp.abs(p_E) > (W_c / 2.0 - 25.0)) | (h < 50.0) | (h > 15000.0)
    collision_penalty = jnp.sum(oob * config.crash_penalty)
    
    progress_cost = -jnp.sum(p_N) * config.progress_weight
    center_cost = jnp.sum(jnp.square(p_E)) * config.center_weight
    
    action_diff = jnp.sum(jnp.square(action_seq[1:] - action_seq[:-1])) * config.action_diff_weight
    action_l2 = jnp.sum(jnp.square(action_seq)) * config.action_l2_weight
    
    return collision_penalty + progress_cost + center_cost + action_diff + action_l2

@functools.partial(jax.jit, static_argnames=['config'])
def mppi_optimize_step(initial_state, base_action_plan, key, W, B, config: JaxMPPIConfig):
    sigma = jnp.array(config.action_noise_std)
    low = jnp.array(config.action_low)
    high = jnp.array(config.action_high)
    
    def evaluate_candidate(noise):
        candidate_actions = jnp.clip(base_action_plan + noise, low, high)
        state_seq = rollout_trajectory(initial_state, candidate_actions, W, B)
        cost = compute_cost(state_seq, candidate_actions, config)
        return cost
    
    keys = jax.random.split(key, config.num_samples)
    noise = jax.random.normal(keys[0], shape=(config.num_samples, config.horizon, 4)) * sigma
    
    costs = jax.vmap(evaluate_candidate)(noise)
    
    beta = jnp.min(costs)
    weights = jnp.exp(- (costs - beta) / config.lambda_)
    weights = weights / (jnp.sum(weights) + 1e-8)
    
    weighted_noise = jnp.tensordot(weights, noise, axes=(0, 0))
    optimized_plan = jnp.clip(base_action_plan + weighted_noise, low, high)
    
    return optimized_plan, costs, state_seq_best(initial_state, optimized_plan, W, B)

def state_seq_best(initial_state, plan, W, B):
    return rollout_trajectory(initial_state, plan, W, B)

class JaxMPPIController:
    def __init__(self, config=None):
        self.config = config or JaxMPPIConfig()
        self.W, self.B = load_nominal_weights()
        self.key = jax.random.PRNGKey(42)
        self.base_plan = jnp.zeros((self.config.horizon, 4))
        # Warm start throttle at 0.5
        self.base_plan = self.base_plan.at[:, 3].set(0.5)

    def get_action(self, state_dict):
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
            state_dict['psi']
        ])
        
        # Shift plan
        shifted_plan = jnp.roll(self.base_plan, shift=-1, axis=0)
        shifted_plan = shifted_plan.at[-1].set(shifted_plan[-2])
        self.base_plan = shifted_plan
        
        for idx in range(self.config.optimization_steps):
            self.key, subkey = jax.random.split(self.key)
            self.base_plan, costs, best_seq = mppi_optimize_step(
                state, self.base_plan, subkey, self.W, self.B, self.config
            )
            if jnp.isnan(costs).any() or jnp.isinf(costs).any():
                print(f"NAN/INF in costs at opt step {idx}!")
                print(f"Costs sample: {costs[:10]}")

            
        action = self.base_plan[0]
        return np.array(action)
