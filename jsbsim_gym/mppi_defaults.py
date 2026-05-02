from __future__ import annotations

import numpy as np


MPPI_DEFAULT_HORIZON = 50
MPPI_DEFAULT_NUM_SAMPLES = 10000
MPPI_DEFAULT_OPTIMIZATION_STEPS = 1
MPPI_DEFAULT_REPLAN_INTERVAL = 1
MPPI_DEFAULT_LAMBDA = 10.0
MPPI_DEFAULT_GAMMA = 0.05
MPPI_DEFAULT_ACTION_NOISE_STD = (0.2, 0.2, 0.1, 0.1)
MPPI_DEFAULT_ACTION_LOW = (-1.0, -1.0, -0.0, 0.0)
MPPI_DEFAULT_ACTION_HIGH = (1.0, 1.0, 0.0, 1.0)
MPPI_DEFAULT_CONTOUR_WEIGHT = 1.0
MPPI_DEFAULT_LAG_WEIGHT = 0.05
MPPI_DEFAULT_PROGRESS_REWARD_WEIGHT = 25.0
MPPI_DEFAULT_VIRTUAL_SPEED_WEIGHT = 0.015
MPPI_DEFAULT_TERRAIN_COLLISION_PENALTY = 1.0e6
MPPI_DEFAULT_TERRAIN_REPULSION_SCALE = 1.0e5
MPPI_DEFAULT_TERRAIN_DECAY_RATE_FT_INV = 0.03
MPPI_DEFAULT_TERRAIN_SAFE_CLEARANCE_FT = 40.0 * 3.28084
MPPI_DEFAULT_CONTROL_RATE_WEIGHTS = (15.0, 20.0, 5.0, 2.0)
MPPI_DEFAULT_NZ_MIN_G = -1.0
MPPI_DEFAULT_NZ_MAX_G = 9.0
MPPI_DEFAULT_NZ_PENALTY_WEIGHT = 0.0
MPPI_DEFAULT_ALPHA_LIMIT_RAD = float(np.deg2rad(25.0))
MPPI_DEFAULT_ALPHA_PENALTY_WEIGHT = 0.0
MPPI_DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS = 120.0
MPPI_DEFAULT_VIRTUAL_SPEED_MIN_FPS = 0.0
MPPI_DEFAULT_VIRTUAL_SPEED_MAX_FPS = 1200.0
MPPI_DEFAULT_VIRTUAL_SPEED_TRIM_FPS = 800.0
MPPI_DEFAULT_DEBUG_RENDER_PLANS = True
MPPI_DEFAULT_DEBUG_NUM_TRAJECTORIES = 1000
MPPI_DEFAULT_SEED = 42

MPPI_SMOOTH_DEFAULT_ACTION_NOISE_STD = (0.14, 0.22, 0.12, 0.10)
MPPI_SMOOTH_DEFAULT_DELTA_NOISE_STD = (0.08, 0.12, 0.08, 0.06)
MPPI_SMOOTH_DEFAULT_DELTA_ACTION_BOUNDS = (0.18, 0.26, 0.14, 0.10)
MPPI_SMOOTH_DEFAULT_NOISE_SMOOTHING_KERNEL = (0.10, 0.20, 0.40, 0.20, 0.10)
MPPI_SMOOTH_DEFAULT_SEED = 101      


def default_mppi_config_kwargs() -> dict:
    """Single source of truth for MPPI config defaults."""

    return {
        "horizon": int(MPPI_DEFAULT_HORIZON),
        "num_samples": int(MPPI_DEFAULT_NUM_SAMPLES),
        "optimization_steps": MPPI_DEFAULT_OPTIMIZATION_STEPS,
        "replan_interval": int(MPPI_DEFAULT_REPLAN_INTERVAL),
        "lambda_": float(MPPI_DEFAULT_LAMBDA),
        "gamma_": float(MPPI_DEFAULT_GAMMA),
        "action_noise_std": tuple(float(x) for x in MPPI_DEFAULT_ACTION_NOISE_STD),
        "action_low": tuple(float(x) for x in MPPI_DEFAULT_ACTION_LOW),
        "action_high": tuple(float(x) for x in MPPI_DEFAULT_ACTION_HIGH),
        "contour_weight": float(MPPI_DEFAULT_CONTOUR_WEIGHT),
        "lag_weight": float(MPPI_DEFAULT_LAG_WEIGHT),
        "progress_reward_weight": float(MPPI_DEFAULT_PROGRESS_REWARD_WEIGHT),
        "virtual_speed_weight": float(MPPI_DEFAULT_VIRTUAL_SPEED_WEIGHT),
        "terrain_collision_penalty": float(MPPI_DEFAULT_TERRAIN_COLLISION_PENALTY),
        "terrain_repulsion_scale": float(MPPI_DEFAULT_TERRAIN_REPULSION_SCALE),
        "terrain_decay_rate_ft_inv": float(MPPI_DEFAULT_TERRAIN_DECAY_RATE_FT_INV),
        "terrain_safe_clearance_ft": float(MPPI_DEFAULT_TERRAIN_SAFE_CLEARANCE_FT),
        "control_rate_weights": tuple(float(x) for x in MPPI_DEFAULT_CONTROL_RATE_WEIGHTS),
        "nz_min_g": float(MPPI_DEFAULT_NZ_MIN_G),
        "nz_max_g": float(MPPI_DEFAULT_NZ_MAX_G),
        "nz_penalty_weight": float(MPPI_DEFAULT_NZ_PENALTY_WEIGHT),
        "alpha_limit_rad": float(MPPI_DEFAULT_ALPHA_LIMIT_RAD),
        "alpha_penalty_weight": float(MPPI_DEFAULT_ALPHA_PENALTY_WEIGHT),
        "virtual_speed_noise_std_fps": float(MPPI_DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS),
        "virtual_speed_min_fps": float(MPPI_DEFAULT_VIRTUAL_SPEED_MIN_FPS),
        "virtual_speed_max_fps": float(MPPI_DEFAULT_VIRTUAL_SPEED_MAX_FPS),
        "virtual_speed_trim_fps": float(MPPI_DEFAULT_VIRTUAL_SPEED_TRIM_FPS),
        "debug_render_plans": bool(MPPI_DEFAULT_DEBUG_RENDER_PLANS),
        "debug_num_trajectories": int(MPPI_DEFAULT_DEBUG_NUM_TRAJECTORIES),
        "seed": int(MPPI_DEFAULT_SEED),
    }
