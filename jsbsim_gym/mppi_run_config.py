from __future__ import annotations

import json
from pathlib import Path

from jsbsim_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from jsbsim_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
MPPI_TUNING_JSON_PATH = OUTPUT_ROOT / "canyon_mppi" / "mppi_optuna_best.json"
KTS_TO_FPS = 1.68781


def load_mppi_optuna_params(
    summary_json_path: Path = MPPI_TUNING_JSON_PATH,
    study_name: str = "mppi_canyon_tuning",
):
    params = {}
    source = None

    summary_path = Path(summary_json_path)
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            best_params = payload.get("best_params", {}) if isinstance(payload, dict) else {}
            if isinstance(best_params, dict) and best_params:
                params = dict(best_params)
                source = str(summary_path)
        except Exception:
            pass

    if params:
        return params, source

    tuning_db_candidates = [
        REPO_ROOT / "optuna" / "mppi_tuning.db",
        REPO_ROOT / "mppi_tuning.db",
    ]
    tuning_db_path = next((candidate for candidate in tuning_db_candidates if candidate.exists()), None)
    if tuning_db_path is None:
        return {}, None

    try:
        import optuna

        storage = f"sqlite:///{tuning_db_path.as_posix()}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        return dict(study.best_params), f"{tuning_db_path}::{study_name}"
    except Exception:
        return {}, None


def build_mppi_base_config_kwargs(
    *,
    optuna_params,
    target_speed_fps,
    target_altitude_ft,
    min_altitude_ft,
    max_altitude_ft,
    wall_margin_ft,
    horizon=40,
    num_samples=4000,
    optimization_steps=3,
    terrain_collision_height_ft=None,
):
    min_altitude_ft = float(min_altitude_ft)
    max_altitude_ft = float(max_altitude_ft)
    if terrain_collision_height_ft is None:
        terrain_collision_height_ft = max(min_altitude_ft + 40.0, 160.0)

    return dict(
        horizon=int(horizon),
        num_samples=int(num_samples),
        optimization_steps=int(optimization_steps),
        lambda_=optuna_params.get("lambda_", 10.0),
        gamma_=optuna_params.get("gamma_", 0.015),
        progress_gain=optuna_params.get("progress_gain", 10.20),
        speed_gain=optuna_params.get("speed_gain", 1.00),
        low_altitude_gain=optuna_params.get("low_altitude_gain", 1.40),
        centerline_gain=optuna_params.get("centerline_gain", 0.60),
        offcenter_penalty_gain=optuna_params.get("offcenter_penalty_gain", 0.30),
        target_speed_fps=float(target_speed_fps),
        target_altitude_ft=float(target_altitude_ft),
        min_altitude_ft=min_altitude_ft,
        max_altitude_ft=max_altitude_ft,
        terrain_collision_height_ft=float(terrain_collision_height_ft),
        wall_margin_ft=float(wall_margin_ft),
        terrain_crash_penalty=max(float(optuna_params.get("terrain_crash_penalty", 250.0)), 250.0),
        early_termination_penalty_gain=120.0,
        max_step_reward_abs=15.0,
    )


def build_mppi_controller(
    controller_tag,
    *,
    optuna_params,
    config_base_kwargs,
    reference_trajectory=None,
    canyon_north_samples_ft=None,
    canyon_width_samples_ft=None,
    canyon_center_east_samples_ft=None,
    canyon_centerline_heading_rad_samples=None,
):
    if controller_tag == "smooth_mppi":
        action_noise_std_roll = optuna_params.get("action_noise_std_roll", optuna_params.get("delta_roll_bound", 0.14))
        action_noise_std_pitch = optuna_params.get("action_noise_std_pitch", optuna_params.get("delta_pitch_bound", 0.22))
        action_noise_std_yaw = optuna_params.get("action_noise_std_yaw", 0.12)
        action_noise_std_throttle = optuna_params.get("action_noise_std_throttle", 0.10)
        config = JaxSmoothMPPIConfig(
            **config_base_kwargs,
            action_noise_std=(
                action_noise_std_roll,
                action_noise_std_pitch,
                action_noise_std_yaw,
                action_noise_std_throttle,
            ),
            delta_noise_std=(
                action_noise_std_roll * 0.6,
                action_noise_std_pitch * 0.6,
                action_noise_std_yaw * 0.6,
                action_noise_std_throttle * 0.6,
            ),
            delta_action_bounds=(
                action_noise_std_roll,
                action_noise_std_pitch,
                action_noise_std_yaw,
                action_noise_std_throttle,
            ),
            noise_smoothing_kernel=(0.10, 0.20, 0.40, 0.20, 0.10),
            smoothness_penalty_weight=optuna_params.get("smoothness_penalty_weight", 0.35),
            action_diff_weight=optuna_params.get("action_diff_weight", 0.8),
            action_l2_weight=optuna_params.get("action_l2_weight", 0.1),
        )
        controller_cls = JaxSmoothMPPIController
    elif controller_tag == "mppi":
        action_noise_std_roll = optuna_params.get("action_noise_std_roll", 0.7)
        action_noise_std_pitch = optuna_params.get("action_noise_std_pitch", 0.7)
        action_noise_std_yaw = optuna_params.get("action_noise_std_yaw", 0.7)
        action_noise_std_throttle = optuna_params.get("action_noise_std_throttle", 0.80)
        config = JaxMPPIConfig(
            **config_base_kwargs,
            action_noise_std=(
                action_noise_std_roll,
                action_noise_std_pitch,
                action_noise_std_yaw,
                action_noise_std_throttle,
            ),
            action_diff_weight=optuna_params.get("action_diff_weight", 0.6),
            action_l2_weight=optuna_params.get("action_l2_weight", 0.1),
        )
        controller_cls = JaxMPPIController
    else:
        raise ValueError(f"Unsupported MPPI controller tag: {controller_tag}")

    controller = controller_cls(
        config=config,
        canyon_north_samples_ft=canyon_north_samples_ft,
        canyon_width_samples_ft=canyon_width_samples_ft,
        canyon_center_east_samples_ft=canyon_center_east_samples_ft,
        canyon_centerline_heading_rad_samples=canyon_centerline_heading_rad_samples,
    )
    return controller, config
