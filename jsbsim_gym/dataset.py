import os
import argparse
import numpy as np
import pandas as pd
from jsbsim_gym.data_collection_env import DataCollectionEnv
from jsbsim_gym.controllers import PersistentExcitationController


def _episode_speed_schedule(episode_idx, rng):
    base_levels = np.asarray([300.0, 380.0, 460.0, 540.0, 620.0, 700.0, 780.0, 860.0], dtype=np.float64)
    shift = int(episode_idx % base_levels.size)
    jitter = rng.uniform(-20.0, 20.0, size=base_levels.shape[0])
    return np.clip(np.roll(base_levels, shift) + jitter, 260.0, 900.0)


def _episode_aoa_schedule(episode_idx, rng):
    base_levels = np.asarray([-2.0, 0.5, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], dtype=np.float64)
    shift = int((2 * episode_idx) % base_levels.size)
    jitter = rng.uniform(-0.8, 0.8, size=base_levels.shape[0])
    return np.clip(np.roll(base_levels, shift) + jitter, -4.0, 20.0)


def _coverage_report(df):
    speed_fps = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
    aoa_deg = np.degrees(df["alpha"])

    speed_bins = np.linspace(260.0, 900.0, 9)
    aoa_bins = np.linspace(-10.0, 20.0, 9)
    hist, _, _ = np.histogram2d(speed_fps.to_numpy(), aoa_deg.to_numpy(), bins=[speed_bins, aoa_bins])
    occupied = int(np.count_nonzero(hist))
    total = int(hist.size)
    occupancy = occupied / max(total, 1)

    speed_hist, _ = np.histogram(speed_fps.to_numpy(), bins=speed_bins)
    aoa_hist, _ = np.histogram(aoa_deg.to_numpy(), bins=aoa_bins)
    speed_bin_coverage = int(np.count_nonzero(speed_hist)) / max(int(speed_hist.size), 1)
    aoa_bin_coverage = int(np.count_nonzero(aoa_hist)) / max(int(aoa_hist.size), 1)

    summary = {
        "speed_min_fps": float(speed_fps.min()),
        "speed_max_fps": float(speed_fps.max()),
        "speed_mean_fps": float(speed_fps.mean()),
        "aoa_min_deg": float(aoa_deg.min()),
        "aoa_max_deg": float(aoa_deg.max()),
        "aoa_mean_deg": float(aoa_deg.mean()),
        "coverage_grid_shape": [int(hist.shape[0]), int(hist.shape[1])],
        "occupied_bins": occupied,
        "total_bins": total,
        "occupancy_ratio": float(occupancy),
        "speed_bin_coverage": float(speed_bin_coverage),
        "aoa_bin_coverage": float(aoa_bin_coverage),
    }
    return summary

def generate_canonical_dataset(
    num_episodes=100,
    steps_per_episode=300,
    output_path="f16_dataset.parquet",
    coverage_output_path="f16_dataset_coverage.json",
    seed=1234,
):
    env = DataCollectionEnv(render_mode=None)
    
    dataset = []
    rng = np.random.default_rng(seed)
    
    for episode in range(num_episodes):
        print(f"Collecting episode {episode+1}/{num_episodes} ...")

        init_speed_fps = float(rng.uniform(280.0, 880.0))
        init_alpha_deg = float(rng.uniform(-6.0, 18.0))
        init_theta_deg = float(rng.uniform(-4.0, 16.0))
        env.simulation.set_property_value('ic/u-fps', init_speed_fps)
        env.simulation.set_property_value('ic/alpha-deg', init_alpha_deg)
        env.simulation.set_property_value('ic/theta-deg', init_theta_deg)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        
        # Varied initial conditions for dataset richness
        # Set varied altitude target
        target_h = float(rng.uniform(1500.0, 3500.0))
        target_pe = float(rng.uniform(-500.0, 500.0))
        speed_schedule_fps = _episode_speed_schedule(episode, rng)
        aoa_schedule_deg = _episode_aoa_schedule(episode, rng)
        controller = PersistentExcitationController(
            target_h=target_h,
            target_p_e=target_pe,
            speed_schedule_fps=speed_schedule_fps,
            aoa_schedule_deg=aoa_schedule_deg,
            schedule_segment_sec=10.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        
        env.wind_state = rng.normal(size=3) * 5.0
        
        for step in range(steps_per_episode):
            state_curr = env.get_full_state_dict()
            action = controller.get_action(state_curr, env.time_sec)
            targets = controller.last_targets
            
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
                'target_speed_fps': float(targets['target_speed_fps']),
                'target_aoa_deg': float(targets['target_aoa_deg']),
                'episode_index': int(episode),
                'step_index': int(step),
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

    coverage = _coverage_report(df)
    print("Coverage summary (speed/AoA):")
    print(coverage)
    if coverage["speed_bin_coverage"] < 0.75 or coverage["aoa_bin_coverage"] < 0.75:
        print(
            "WARNING: speed/AoA bin coverage is low. Increase episodes, segment schedule variety, "
            "or perturbation amplitudes for better envelope coverage."
        )
    
    parquet_path = str(output_path)
    print(f"Exporting to {parquet_path}...")
    df.to_parquet(parquet_path, engine='pyarrow')
    pd.DataFrame([coverage]).to_json(str(coverage_output_path), orient="records", indent=2)
    print("Export Complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate canonical F-16 dataset for uncertainty and MPPI fitting.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of rollouts to collect.")
    parser.add_argument("--steps", type=int, default=300, help="Max steps per rollout.")
    parser.add_argument("--output", default="f16_dataset.parquet", help="Output parquet path.")
    parser.add_argument(
        "--coverage-output",
        default="output/f16_dataset_coverage.json",
        help="Coverage summary JSON output path.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for collection scheduling.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_canonical_dataset(
        num_episodes=int(args.episodes),
        steps_per_episode=int(args.steps),
        output_path=args.output,
        coverage_output_path=args.coverage_output,
        seed=int(args.seed),
    )
