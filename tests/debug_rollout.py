import jax
import jax.numpy as jnp
from jsbsim_gym.mppi_jax import JaxMPPIConfig, load_nominal_weights, f16_kinematics_step

def test_rollout():
    W, B, poly_powers, throttle_force_coeffs = load_nominal_weights()
    state = jnp.array([
        0.0, 0.0, 5000.0,     # p_N, p_E, h
        900.0, 0.0, 0.0,      # u, v, w
        0.0, 0.0, 0.0,        # p, q, r
        0.0, 0.0, 0.0         # phi, theta, psi
    ])
    action = jnp.array([0.0, 0.0, 0.0, 0.5])
    
    for i in range(40):
        state = f16_kinematics_step(state, action, W, B, poly_powers, throttle_force_coeffs)
        if jnp.isnan(state).any() or jnp.isinf(state).any():
            print(f"FAILED AT STEP {i}")
            print(f"State: {state}")
            break
        print(f"Step {i}: u={state[3]:.1f}, w={state[5]:.1f}, q={state[7]:.3f}, theta={state[10]:.3f}")

if __name__ == '__main__':
    test_rollout()
