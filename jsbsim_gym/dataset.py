import os
import argparse
import numpy as np
import pandas as pd
from jsbsim_gym.data_collection_env import DataCollectionEnv
from jsbsim_gym.controllers import PersistentExcitationController

EXTREME_MANEUVER_FIRST_STEP = 20
EXTREME_MANEUVER_BURST_STEPS = 6
EXTREME_MANEUVER_RECOVERY_STEPS = 12


def _episode_profile(episode_idx):
    profiles = ("baseline", "high_alpha", "high_beta", "high_qbar")
    return profiles[int(episode_idx) % len(profiles)]


def _episode_speed_schedule(episode_idx, rng, profile):
    if profile == "high_qbar":
        base_levels = np.asarray([620.0, 700.0, 780.0, 860.0, 940.0, 980.0, 900.0, 820.0], dtype=np.float64)
        jitter = rng.uniform(-35.0, 35.0, size=base_levels.shape[0])
        clip_low, clip_high = 560.0, 1000.0
    elif profile == "high_alpha":
        base_levels = np.asarray([320.0, 400.0, 480.0, 560.0, 640.0, 720.0, 680.0, 600.0], dtype=np.float64)
        jitter = rng.uniform(-30.0, 30.0, size=base_levels.shape[0])
        clip_low, clip_high = 260.0, 860.0
    elif profile == "high_beta":
        base_levels = np.asarray([160.0, 200.0, 240.0, 280.0, 320.0, 360.0, 300.0, 220.0], dtype=np.float64)
        jitter = rng.uniform(-24.0, 24.0, size=base_levels.shape[0])
        clip_low, clip_high = 140.0, 420.0
    else:
        base_levels = np.asarray([300.0, 380.0, 460.0, 540.0, 620.0, 700.0, 780.0, 860.0], dtype=np.float64)
        jitter = rng.uniform(-25.0, 25.0, size=base_levels.shape[0])
        clip_low, clip_high = 240.0, 920.0
    shift = int(episode_idx % base_levels.size)
    return np.clip(np.roll(base_levels, shift) + jitter, clip_low, clip_high)


def _episode_aoa_schedule(episode_idx, rng, profile):
    if profile == "high_alpha":
        base_levels = np.asarray([8.0, 11.0, 14.0, 17.0, 20.0, 22.0, 18.0, 13.0], dtype=np.float64)
        jitter = rng.uniform(-1.2, 1.2, size=base_levels.shape[0])
        clip_low, clip_high = 2.0, 24.0
    elif profile == "high_qbar":
        base_levels = np.asarray([-4.0, -1.0, 2.0, 5.0, 8.0, 10.0, 7.0, 4.0], dtype=np.float64)
        jitter = rng.uniform(-1.0, 1.0, size=base_levels.shape[0])
        clip_low, clip_high = -6.0, 14.0
    elif profile == "high_beta":
        base_levels = np.asarray([4.0, 8.0, 12.0, 16.0, 20.0, 18.0, 14.0, 10.0], dtype=np.float64)
        jitter = rng.uniform(-1.0, 1.0, size=base_levels.shape[0])
        clip_low, clip_high = 2.0, 22.0
    else:
        base_levels = np.asarray([-2.0, 0.5, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], dtype=np.float64)
        jitter = rng.uniform(-0.9, 0.9, size=base_levels.shape[0])
        clip_low, clip_high = -6.0, 22.0
    shift = int((2 * episode_idx) % base_levels.size)
    return np.clip(np.roll(base_levels, shift) + jitter, clip_low, clip_high)


def _episode_beta_schedule(episode_idx, rng, profile):
    if profile == "high_beta":
        base_levels = np.asarray([-16.0, -12.0, -8.0, -4.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float64)
        jitter = rng.uniform(-1.5, 1.5, size=base_levels.shape[0])
    elif profile == "high_qbar":
        base_levels = np.asarray([-7.0, -4.0, -2.0, 0.0, 2.0, 4.0, 7.0, 0.0], dtype=np.float64)
        jitter = rng.uniform(-1.0, 1.0, size=base_levels.shape[0])
    elif profile == "high_alpha":
        base_levels = np.asarray([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 0.0], dtype=np.float64)
        jitter = rng.uniform(-1.0, 1.0, size=base_levels.shape[0])
    else:
        base_levels = np.asarray([-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 0.0], dtype=np.float64)
        jitter = rng.uniform(-1.0, 1.0, size=base_levels.shape[0])
    shift = int((3 * episode_idx) % base_levels.size)
    return np.clip(np.roll(base_levels, shift) + jitter, -18.0, 18.0)


def _extreme_corner_actions():
    actions = []
    for roll_cmd in (-1.0, 1.0):
        for pitch_cmd in (-1.0, 1.0):
            for yaw_cmd in (-1.0, 1.0):
                for throttle_cmd in (0.0, 1.0):
                    actions.append((roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd))
    return np.asarray(actions, dtype=np.float32)


def _episode_extreme_action_schedule(episode_idx, steps_per_episode):
    schedule = np.full((int(steps_per_episode), 4), np.nan, dtype=np.float32)
    schedule_ids = np.full((int(steps_per_episode),), -1, dtype=np.int32)
    corners = _extreme_corner_actions()
    start_step = int(EXTREME_MANEUVER_FIRST_STEP + (episode_idx % 11))
    corner_cursor = int((episode_idx * 5) % corners.shape[0])
    step = start_step
    while step < int(steps_per_episode):
        corner_idx = int(corner_cursor % corners.shape[0])
        step_hi = min(step + int(EXTREME_MANEUVER_BURST_STEPS), int(steps_per_episode))
        schedule[step:step_hi, :] = corners[corner_idx][None, :]
        schedule_ids[step:step_hi] = corner_idx
        corner_cursor += 1
        step = step_hi + int(EXTREME_MANEUVER_RECOVERY_STEPS)
    return schedule, schedule_ids


def _coverage_report(df):
    speed_fps = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
    aoa_deg = np.degrees(df["alpha"])
    beta_deg = np.degrees(df["beta"])
    qbar_psf = df["qbar"].to_numpy(dtype=float)

    speed_bins = np.linspace(240.0, 980.0, 10)
    aoa_bins = np.linspace(-10.0, 26.0, 10)
    beta_bins = np.linspace(-8.0, 8.0, 9)
    qbar_bins = np.linspace(80.0, 900.0, 11)

    hist, _, _ = np.histogram2d(speed_fps.to_numpy(), aoa_deg.to_numpy(), bins=[speed_bins, aoa_bins])
    occupied = int(np.count_nonzero(hist))
    total = int(hist.size)
    occupancy = occupied / max(total, 1)

    speed_hist, _ = np.histogram(speed_fps.to_numpy(), bins=speed_bins)
    aoa_hist, _ = np.histogram(aoa_deg.to_numpy(), bins=aoa_bins)
    beta_hist, _ = np.histogram(beta_deg.to_numpy(), bins=beta_bins)
    qbar_hist, _ = np.histogram(qbar_psf, bins=qbar_bins)

    speed_bin_coverage = int(np.count_nonzero(speed_hist)) / max(int(speed_hist.size), 1)
    aoa_bin_coverage = int(np.count_nonzero(aoa_hist)) / max(int(aoa_hist.size), 1)
    beta_bin_coverage = int(np.count_nonzero(beta_hist)) / max(int(beta_hist.size), 1)
    qbar_bin_coverage = int(np.count_nonzero(qbar_hist)) / max(int(qbar_hist.size), 1)

    state_cube = np.stack([aoa_deg.to_numpy(), beta_deg.to_numpy(), qbar_psf], axis=1)
    state_hist, _ = np.histogramdd(state_cube, bins=[aoa_bins, beta_bins, qbar_bins])
    state_occupied = int(np.count_nonzero(state_hist))
    state_total = int(state_hist.size)
    state_occupancy = state_occupied / max(state_total, 1)

    summary = {
        "speed_min_fps": float(speed_fps.min()),
        "speed_max_fps": float(speed_fps.max()),
        "speed_mean_fps": float(speed_fps.mean()),
        "speed_std_fps": float(speed_fps.std()),
        "aoa_min_deg": float(aoa_deg.min()),
        "aoa_max_deg": float(aoa_deg.max()),
        "aoa_mean_deg": float(aoa_deg.mean()),
        "aoa_std_deg": float(aoa_deg.std()),
        "aoa_span_deg": float(aoa_deg.max() - aoa_deg.min()),
        "beta_min_deg": float(beta_deg.min()),
        "beta_max_deg": float(beta_deg.max()),
        "beta_mean_deg": float(beta_deg.mean()),
        "beta_std_deg": float(beta_deg.std()),
        "beta_span_deg": float(beta_deg.max() - beta_deg.min()),
        "qbar_min_psf": float(np.nanmin(qbar_psf)),
        "qbar_max_psf": float(np.nanmax(qbar_psf)),
        "qbar_mean_psf": float(np.nanmean(qbar_psf)),
        "qbar_std_psf": float(np.nanstd(qbar_psf)),
        "qbar_span_psf": float(np.nanmax(qbar_psf) - np.nanmin(qbar_psf)),
        "coverage_grid_shape": [int(hist.shape[0]), int(hist.shape[1])],
        "occupied_bins": occupied,
        "total_bins": total,
        "occupancy_ratio": float(occupancy),
        "speed_bin_coverage": float(speed_bin_coverage),
        "aoa_bin_coverage": float(aoa_bin_coverage),
        "beta_bin_coverage": float(beta_bin_coverage),
        "qbar_bin_coverage": float(qbar_bin_coverage),
        "aoa_beta_qbar_grid_shape": [int(state_hist.shape[0]), int(state_hist.shape[1]), int(state_hist.shape[2])],
        "aoa_beta_qbar_occupied_bins": state_occupied,
        "aoa_beta_qbar_total_bins": state_total,
        "aoa_beta_qbar_occupancy_ratio": float(state_occupancy),
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
        profile = _episode_profile(episode)
        print(f"Collecting episode {episode+1}/{num_episodes} | profile={profile} ...")

        if profile == "high_qbar":
            init_speed_fps = float(rng.uniform(620.0, 960.0))
            init_alpha_deg = float(rng.uniform(-4.0, 10.0))
            init_theta_deg = float(rng.uniform(-2.0, 8.0))
        elif profile == "high_alpha":
            init_speed_fps = float(rng.uniform(280.0, 760.0))
            init_alpha_deg = float(rng.uniform(8.0, 22.0))
            init_theta_deg = float(rng.uniform(2.0, 18.0))
        elif profile == "high_beta":
            init_speed_fps = float(rng.uniform(170.0, 360.0))
            init_alpha_deg = float(rng.uniform(6.0, 18.0))
            init_theta_deg = float(rng.uniform(-2.0, 12.0))
        else:
            init_speed_fps = float(rng.uniform(260.0, 900.0))
            init_alpha_deg = float(rng.uniform(-8.0, 20.0))
            init_theta_deg = float(rng.uniform(-5.0, 16.0))
        env.simulation.set_property_value('ic/u-fps', init_speed_fps)
        env.simulation.set_property_value('ic/alpha-deg', init_alpha_deg)
        env.simulation.set_property_value('ic/theta-deg', init_theta_deg)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        
        # Varied initial conditions for dataset richness
        # Set varied altitude target
        if profile == "high_qbar":
            target_h = float(rng.uniform(1200.0, 2600.0))
            target_pe = float(rng.uniform(-700.0, 700.0))
        elif profile == "high_alpha":
            target_h = float(rng.uniform(1800.0, 3600.0))
            target_pe = float(rng.uniform(-650.0, 650.0))
        elif profile == "high_beta":
            target_h = float(rng.uniform(1200.0, 2800.0))
            target_pe = float(rng.uniform(-2000.0, 2000.0))
        else:
            target_h = float(rng.uniform(1500.0, 3500.0))
            target_pe = float(rng.uniform(-600.0, 600.0))

        speed_schedule_fps = _episode_speed_schedule(episode, rng, profile)
        aoa_schedule_deg = _episode_aoa_schedule(episode, rng, profile)
        beta_schedule_deg = _episode_beta_schedule(episode, rng, profile)
        schedule_segment_sec = 3.0
        schedule_phase_offset_sec = float(rng.uniform(0.0, schedule_segment_sec * 8.0))
        excitation_scale = {
            "baseline": 1.0,
            "high_alpha": 1.15,
            "high_beta": 1.90,
            "high_qbar": 1.20,
        }[profile]
        controller = PersistentExcitationController(
            target_h=target_h,
            target_p_e=target_pe,
            speed_schedule_fps=speed_schedule_fps,
            aoa_schedule_deg=aoa_schedule_deg,
            beta_schedule_deg=beta_schedule_deg,
            schedule_segment_sec=schedule_segment_sec,
            schedule_phase_offset_sec=schedule_phase_offset_sec,
            excitation_scale=excitation_scale,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        extreme_action_schedule, extreme_action_ids = _episode_extreme_action_schedule(
            episode_idx=episode,
            steps_per_episode=steps_per_episode,
        )

        if profile == "high_beta":
            env.wind_theta = float(rng.uniform(0.08, 0.30))
            env.wind_sigma = float(rng.uniform(100.0, 160.0))
            env.wind_state = rng.normal(size=3) * np.array([14.0, 95.0, 14.0], dtype=np.float64)
        elif profile == "high_qbar":
            env.wind_theta = float(rng.uniform(0.45, 0.95))
            env.wind_sigma = float(rng.uniform(12.0, 22.0))
            env.wind_state = rng.normal(size=3) * np.array([6.0, 10.0, 6.0], dtype=np.float64)
        elif profile == "high_alpha":
            env.wind_theta = float(rng.uniform(0.40, 0.90))
            env.wind_sigma = float(rng.uniform(14.0, 24.0))
            env.wind_state = rng.normal(size=3) * np.array([8.0, 12.0, 8.0], dtype=np.float64)
        else:
            env.wind_theta = float(rng.uniform(0.50, 1.10))
            env.wind_sigma = float(rng.uniform(10.0, 20.0))
            env.wind_state = rng.normal(size=3) * np.array([6.0, 8.0, 6.0], dtype=np.float64)
        
        for step in range(steps_per_episode):
            state_curr = env.get_full_state_dict()
            action = controller.get_action(state_curr, env.time_sec)
            extreme_combo_index = int(extreme_action_ids[step])
            if extreme_combo_index >= 0:
                action = extreme_action_schedule[step].astype(np.float32).tolist()
            targets = controller.last_targets
            
            state_t, state_t_plus_1, done = env.step_collect(action)
            
            row = {
                'p_N': state_t['p_N'],
                'p_E': state_t['p_E'],
                'h': state_t['h'],
                'V': state_t['V'],
                'mach': state_t['mach'],
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
                'mass_slugs': state_t['mass_slugs'],
                'alpha_dot': state_t['alpha_dot'],
                'wind_u': state_t['wind_u'],
                'wind_v': state_t['wind_v'],
                'wind_w': state_t['wind_w'],
                'canyon_width': state_t['canyon_width'],
                'canyon_width_grad': state_t['canyon_width_grad'],
                'target_speed_fps': float(targets['target_speed_fps']),
                'target_aoa_deg': float(targets['target_aoa_deg']),
                'target_beta_deg': float(targets['target_beta_deg']),
                'collection_profile': str(profile),
                'episode_index': int(episode),
                'step_index': int(step),
                'extreme_override': int(extreme_combo_index >= 0),
                'extreme_combo_index': int(extreme_combo_index),
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
    print("Coverage summary (speed/AoA/beta/qbar):")
    print(coverage)
    if (
        coverage["speed_bin_coverage"] < 0.70
        or coverage["aoa_bin_coverage"] < 0.70
        or coverage["beta_bin_coverage"] < 0.55
        or coverage["qbar_bin_coverage"] < 0.70
        or coverage["aoa_beta_qbar_occupancy_ratio"] < 0.08
        or coverage["aoa_span_deg"] < 16.0
        or coverage["beta_span_deg"] < 6.0
        or coverage["qbar_span_psf"] < 300.0
    ):
        print(
            "WARNING: state-envelope coverage is low. Increase episodes or tune profile/excitation settings "
            "to expand AoA, sideslip, and dynamic-pressure variation."
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
