import time
import numpy as np
from jsbsim_gym.data_collection_env import DataCollectionEnv
from jsbsim_gym.mppi_jax import JaxMPPIController

def render_state(state, width):
    y = state['p_E']
    y_norm = 2.0 * y / max(width, 1.0)
    
    # Text visualization of the canyon
    viz = "Wall |"
    pos = int(np.clip((y_norm + 1.0) * 15, 0, 30))
    for i in range(31):
        if i == pos:
            viz += "X"
        else:
            viz += " "
    viz += "| Wall"
    return viz

def main():
    print("Initializing MPPI Engine...")
    controller = JaxMPPIController()
    
    # Warm up JAX JIT
    print("Compiling JAX JIT... (this takes a moment)")
    fake_state = {
        'p_N': 0.0, 'p_E': 0.0, 'h': 5000.0,
        'u': 900.0, 'v': 0.0, 'w': 0.0,
        'p': 0.0, 'q': 0.0, 'r': 0.0,
        'phi': 0.0, 'theta': 0.0, 'psi': 0.0
    }
    _ = controller.get_action(fake_state)
    print("JIT compilation finished.")

    env = DataCollectionEnv(render_mode='human')
    env.reset()
    state = env.get_full_state_dict()
    env.render()
    
    print("\nStarting Flight...")
    print(f"{'Step':<5} | {'p_N':<8} | {'p_E':<8} | {'h':<8} | {'W_c':<6} | Canyon Vis")
    print("-" * 70)
    
    for step in range(500):
        t0 = time.time()
        action = controller.get_action(state)
        t_plan = time.time() - t0
        
        # DEBUG
        if np.isnan(action).any():
            print(f"NAN ACTION AT STEP {step}: {action}")
            print(f"Current State: {state}")
            break
            
        # Action is exactly [roll, pitch, yaw, throttle]
        _, state_next, done = env.step_collect(action)
        env.render()
        state = state_next
        
        if step % 5 == 0:
            width, _ = env.canyon.get_geometry(state['p_N'])
            viz = render_state(state, width)
            print(f"{step:<5} | {state['p_N']:<8.0f} | {state['p_E']:<8.0f} | {state['h']:<8.0f} | {width:<6.0f} | {viz}")
            
        if done:
            print(f"Crashed or terminated at step {step}.")
            break
            
    env.close()

if __name__ == "__main__":
    main()
