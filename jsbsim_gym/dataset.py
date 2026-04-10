import os
import numpy as np
import pandas as pd
from jsbsim_gym.data_collection_env import DataCollectionEnv
from jsbsim_gym.controllers import PersistentExcitationController

def generate_canonical_dataset():
    num_episodes = 100
    steps_per_episode = 300
    
    env = DataCollectionEnv(render_mode=None)
    controller = PersistentExcitationController(target_h=2000.0, target_p_e=0.0)
    
    dataset = []
    
    for episode in range(num_episodes):
        print(f"Collecting episode {episode+1}/{num_episodes} ...")
        
        env.reset()
        
        # Varied initial conditions for dataset richness
        # Set varied altitude target
        target_h = np.random.uniform(1500.0, 3500.0)
        target_pe = np.random.uniform(-500.0, 500.0)
        controller = PersistentExcitationController(target_h=target_h, target_p_e=target_pe)
        
        env.wind_state = np.random.randn(3) * 5.0
        
        for step in range(steps_per_episode):
            state_curr = env.get_full_state_dict()
            action = controller.get_action(state_curr, env.time_sec)
            
            state_t, state_t_plus_1, done = env.step_collect(action)
            
            row = {
                'p_N': state_t['p_N'],
                'p_E': state_t['p_E'],
                'h': state_t['h'],
                'V': state_t['V'],
                'u': state_t['u'],
                'v': state_t['v'],
                'w': state_t['w'],
                'alpha': state_t['alpha'],
                'beta': state_t['beta'],
                'phi': state_t['phi'],
                'theta': state_t['theta'],
                'psi': state_t['psi'],
                'p': state_t['p'],
                'q': state_t['q'],
                'r': state_t['r'],
                'delta_t': state_t['delta_t'],
                'delta_e': state_t['delta_e'],
                'delta_a': state_t['delta_a'],
                'delta_r': state_t['delta_r'],
                'qbar': state_t['qbar'],
                'alpha_dot': state_t['alpha_dot'],
                'wind_u': state_t['wind_u'],
                'wind_v': state_t['wind_v'],
                'wind_w': state_t['wind_w'],
                'canyon_width': state_t['canyon_width'],
                'canyon_width_grad': state_t['canyon_width_grad'],
            }
            
            # Append next_ state variables
            for key, val in state_t_plus_1.items():
                row[f'next_{key}'] = val
                
            dataset.append(row)
            
            if done:
                print(f"  Episode {episode+1} crashed/terminated early at step {step}")
                break
                
    env.close()
    
    print("Formatting DataFrame...")
    df = pd.DataFrame(dataset)
    print(f"Dataframe dimension: {df.shape}")
    
    parquet_path = "f16_dataset.parquet"
    print(f"Exporting to {parquet_path}...")
    df.to_parquet(parquet_path, engine='pyarrow')
    print("Export Complete.")

if __name__ == "__main__":
    generate_canonical_dataset()
