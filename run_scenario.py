import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# On macOS, JAX can auto-select experimental MPS and stall during first MPPI JIT.
# Default to CPU for this script unless the user explicitly overrides it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_env import OBS_ALTITUDE_ERROR_FT, OBS_PHI, OBS_P, OBS_Q, OBS_R, OBS_THETA
from jsbsim_gym.canyon_artifacts import CanyonRunRecorder
from jsbsim_gym.mppi_jax import (
    JaxMPPIConfig,
    JaxMPPIController,
    JaxSmoothMPPIConfig,
    JaxSmoothMPPIController,
)
from jsbsim_gym.simple_controller import (
    SimpleCanyonController,
    SimpleCanyonControllerConfig,
    with_default_simple_controller_optuna_gains,
)

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
REPO_ROOT = Path(__file__).resolve().parent
DEM_PATH = REPO_ROOT / DEM_PATH
OUTPUT_ROOT = REPO_ROOT / "output"


def render_state(lateral_error_ft, width_ft):
    y_norm = 2.0 * lateral_error_ft / max(width_ft, 1.0)

    viz = "Wall |"
    pos = int(np.clip((y_norm + 1.0) * 15, 0, 30))
    for i in range(31):
        viz += "X" if i == pos else " "
    viz += "| Wall"
    return viz


def _wrap_heading_deg(angle_rad):
    return float(np.mod(np.degrees(float(angle_rad)), 360.0))


def to_mppi_state(env, state, altitude_ref_ft):
    p_n_ft = float(state["p_N"])
    p_e_ft = float(state["p_E"])

    canyon = getattr(env.unwrapped, "canyon", None)
    if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
        try:
            lat_deg = float(env.unwrapped.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(env.unwrapped.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = canyon.get_local_from_latlon(lat_deg, lon_deg)
            p_n_ft = float(local_north_ft)
            p_e_ft = float(local_east_ft)
        except Exception:
            pass

    return {
        "p_N": p_n_ft,
        "p_E": p_e_ft,
        "h": float(state["h"] - altitude_ref_ft),
        "u": float(state["u"]),
        "v": float(state["v"]),
        "w": float(state["w"]),
        "p": float(state["p"]),
        "q": float(state["q"]),
        "r": float(state["r"]),
        "phi": float(state["phi"]),
        "theta": float(state["theta"]),
        "psi": float(state["psi"]),
    }


class AltitudeHoldController:
    """Straight-flight + altitude-hold controller for visualization/debug runs."""

    def __init__(self):
        self.altitude_error_integral = 0.0

    def get_action(self, obs):
        altitude_error_ft = float(obs[OBS_ALTITUDE_ERROR_FT])
        roll_angle = float(obs[OBS_PHI])
        roll_rate = float(obs[OBS_P])
        pitch_rate = float(obs[OBS_Q])
        yaw_rate = float(obs[OBS_R])
        pitch_angle = float(obs[OBS_THETA])

        self.altitude_error_integral += altitude_error_ft * (1.0 / 30.0)
        self.altitude_error_integral = np.clip(self.altitude_error_integral, -2000.0, 2000.0)

        roll_cmd = np.clip(-1.6 * roll_angle - 0.45 * roll_rate, -0.35, 0.35)
        yaw_cmd = np.clip(-0.25 * yaw_rate, -0.15, 0.15)

        pitch_cmd = np.clip(
            0.0022 * altitude_error_ft
            + 0.70 * pitch_angle
            - 0.30 * pitch_rate
            + 0.00006 * self.altitude_error_integral,
            -0.50,
            0.50,
        )

        throttle_cmd = np.clip(0.64 - 0.00008 * altitude_error_ft, 0.50, 0.95)
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)




def parse_args():
    parser = argparse.ArgumentParser(description="Run canyon scenarios across available controllers.")
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi", "simple", "altitude_hold"],
        default="mppi",
        help="Controller variant to run.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable interactive rendering window (default is headless).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for run artifacts. The final subdirectory is controller-specific.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum control steps to run.",
    )
    parser.add_argument(
        "--target-speed-kts",
        type=float,
        default=450.0,
        help="Target flight speed in knots (kts).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mppi_canyon_tuning",
        help="Name of the Optuna study to load tuned parameters from.",
    )
    return parser.parse_args()





def main():
    args = parse_args()
    output_subdirs = {
        "mppi": "canyon_mppi",
        "smooth_mppi": "canyon_smooth_mppi",
        "simple": "canyon_simple",
        "altitude_hold": "canyon_altitude_hold",
    }
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.output_root) / output_subdirs[args.controller]
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "render" if args.render else "headless"
    controller_tag = str(args.controller)

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode="human" if args.render else "rgb_array",
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
        dem_bbox=DEM_BBOX,
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=DEM_START_PIXEL,
        dem_start_heading_mode="follow_canyon",
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=250.0,
        entry_altitude_ft=250.0,
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=10.0,
        entry_speed_kts=args.target_speed_kts,
        wind_sigma=1.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    obs, _ = env.reset(seed=3)

    state = env.unwrapped.get_full_state_dict()

    altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
    initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)
    start_path_north_ft = float(initial_controller_state["p_N"])

    target_altitude_ft = float(env.unwrapped.target_altitude_ft - altitude_ref_ft)
    min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
    max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)

    canyon = env.unwrapped.canyon
    if hasattr(canyon, "north_samples_ft") and hasattr(canyon, "width_samples_ft"):
        north_samples_ft = canyon.north_samples_ft
        width_samples_ft = canyon.width_samples_ft
        center_east_samples_ft = canyon.center_east_samples_ft
        centerline_heading_samples_rad = getattr(canyon, "centerline_heading_samples_rad", np.zeros_like(center_east_samples_ft))
    else:
        # Fallback for procedural or other types if they don't have these attributes
        north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
        width_samples_ft = np.ones(256, dtype=np.float32) * 1000.0
        center_east_samples_ft = np.zeros(256, dtype=np.float32)
        centerline_heading_samples_rad = np.zeros(256, dtype=np.float32)

    mppi_target_altitude_ft = target_altitude_ft

    optuna_params = {}
    tuning_db_candidates = [
        REPO_ROOT / "optuna" / "mppi_tuning.db",
        REPO_ROOT / "mppi_tuning.db",
    ]
    tuning_db_path = next((candidate for candidate in tuning_db_candidates if candidate.exists()), None)
    if tuning_db_path is not None:
        try:
            import optuna
            storage = f"sqlite:///{tuning_db_path.as_posix()}"
            study = optuna.load_study(study_name=args.study_name, storage=storage)
            optuna_params = study.best_params
            print(
                f"Auto-loaded tuned parameters from {args.study_name} "
                f"at {tuning_db_path}! (Reward: {study.best_value:.2f})"
            )
            if "target_alt_tune_ft" in optuna_params:
                mppi_target_altitude_ft = float(optuna_params["target_alt_tune_ft"])
        except Exception as e:
            print(f"Note: Could not load tuning DB {tuning_db_path} ({e})")

    config_base_kwargs = dict(
        horizon=30,
        num_samples=4000,
        optimization_steps=3,
        lambda_=optuna_params.get("lambda_", 2.0),
        progress_gain=optuna_params.get("progress_gain", 0.60),
        speed_gain=optuna_params.get("speed_gain", 0.20),
        low_altitude_gain=optuna_params.get("low_altitude_gain", 2.50),
        centerline_gain=optuna_params.get("centerline_gain", 2.50),
        offcenter_penalty_gain=optuna_params.get("offcenter_penalty_gain", 4.00),
        heading_alignment_gain=optuna_params.get("heading_alignment_gain", 0.90),
        heading_alignment_scale_rad=optuna_params.get("heading_alignment_scale_rad", 0.80),
        alive_bonus=optuna_params.get("alive_bonus", 0.15),
        target_speed_fps=args.target_speed_kts * 1.68781,
        target_altitude_ft=mppi_target_altitude_ft,
        min_altitude_ft=min_altitude_ft,
        max_altitude_ft=max_altitude_ft,
        terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
        wall_margin_ft=float(env.unwrapped.wall_margin_ft),
        terrain_crash_penalty=optuna_params.get("terrain_crash_penalty", 45.0),
        wall_crash_penalty=optuna_params.get("wall_crash_penalty", 32.0),
        altitude_violation_penalty=8.0,
        early_termination_penalty_gain=120.0,
        time_limit_bonus=35.0,
        max_step_reward_abs=15.0,
        angular_rate_penalty_gain=optuna_params.get("angular_rate_penalty_gain", 0.0),
        angular_rate_threshold_deg_s=0.0,
    )

    smooth_kwargs = {}
    if "gamma_" in optuna_params:
        smooth_kwargs["gamma_"] = optuna_params["gamma_"]

    if controller_tag == "simple":
        simple_config = SimpleCanyonControllerConfig(
            target_speed_fps=args.target_speed_kts * 1.68781,
        )
        simple_config, simple_tuning_source, simple_tuned_keys = with_default_simple_controller_optuna_gains(simple_config)
        if simple_tuned_keys:
            print(
                f"Auto-loaded simple-controller tuned gains from {simple_tuning_source} "
                f"({len(simple_tuned_keys)} parameters)."
            )
        else:
            print("Note: No tuned simple-controller gains found; using built-in defaults.")
        controller = SimpleCanyonController(
            env=env,
            config=simple_config,
        )
        controller.reset(initial_controller_state)
    elif controller_tag == "altitude_hold":
        controller = AltitudeHoldController()
    else:
        if controller_tag == "smooth_mppi":
            drb = optuna_params.get("delta_roll_bound", 0.14)
            dpb = optuna_params.get("delta_pitch_bound", 0.22)
            config = JaxSmoothMPPIConfig(
                **config_base_kwargs,
                **smooth_kwargs,
                action_noise_std=(drb, dpb, 0.12, 0.10),
                delta_noise_std=(drb * 0.6, dpb * 0.6, 0.08, 0.06),
                delta_action_bounds=(drb, dpb, 0.14, 0.10),
                noise_smoothing_kernel=(0.10, 0.20, 0.40, 0.20, 0.10),
                smoothness_penalty_weight=optuna_params.get("smoothness_penalty_weight", 0.35),
                action_diff_weight=optuna_params.get("action_diff_weight", 0.8),
                action_l2_weight=optuna_params.get("action_l2_weight", 0.1),
            )
            controller_cls = JaxSmoothMPPIController
        else:
            config = JaxMPPIConfig(
                **config_base_kwargs,
                action_noise_std=(0.14, 0.22, 0.12, 0.10),
                action_diff_weight=optuna_params.get("action_diff_weight", 0.6),
                action_l2_weight=optuna_params.get("action_l2_weight", 0.1),
            )
            controller_cls = JaxMPPIController

        controller = controller_cls(
            config=config,
            canyon_north_samples_ft=north_samples_ft,
            canyon_width_samples_ft=width_samples_ft,
            canyon_center_east_samples_ft=center_east_samples_ft,
            canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
        )

    recorder = CanyonRunRecorder(
        env=env,
        dem_path=DEM_PATH,
        dem_bbox=DEM_BBOX,
        dem_start_pixel=DEM_START_PIXEL,
        output_dir=output_dir,
        file_stem=f"canyon_{controller_tag}_{mode_tag}",
        title_prefix=f"{controller_tag.upper()} Trajectory Overlay",
        fps=30,
    )
    recorder.initialize()
    recorder.set_centerline_profile(north_samples_ft, center_east_samples_ft)
    termination_reason = "running"

    print(f"Initializing {controller_tag} canyon controller...")
    if controller_tag in {"mppi", "smooth_mppi"}:
        print("Compiling JAX JIT... (this takes a moment)")
        _ = controller.get_action(initial_controller_state)
        print("JIT compilation finished.")

    print("\nStarting Canyon Flight...")
    print(
        f"{'Step':<5} | {'p_N_rel':<8} | {'LatErr':<8} | {'h_rel':<8} | "
        f"{'V':<6} | {'W_c':<6} | {'Plan(ms)':<8} | Canyon Vis"
    )
    print("-" * 110)

    try:
        for step in range(int(args.max_steps)):
            controller_state = to_mppi_state(env, state, altitude_ref_ft)

            t0 = time.time()
            if controller_tag == "altitude_hold":
                action = controller.get_action(obs)
            else:
                action = controller.get_action(controller_state)
            plan_ms = (time.time() - t0) * 1000.0

            planner_debug = None
            debug_getter = getattr(controller, "get_render_debug", None)
            if callable(debug_getter):
                raw_debug = debug_getter()
                if raw_debug is not None:
                    planner_debug = {
                        "candidate_xy": np.asarray(
                            raw_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "candidate_h_ft": np.asarray(
                            raw_debug.get("candidate_h_ft", np.zeros((0, 0), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "final_xy": np.asarray(
                            raw_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                        "final_h_ft": np.asarray(
                            raw_debug.get("final_h_ft", np.zeros((0,), dtype=np.float32)),
                            dtype=np.float32,
                        ).copy(),
                    }

            if controller_tag == "simple":
                guidance = dict(getattr(controller, "last_guidance", {}) or {})
                lookahead_north_ft = float(guidance.get("lookahead_north_ft", controller_state["p_N"]))
                lookahead_center_east_ft = float(guidance.get("lookahead_center_east_ft", controller_state["p_E"]))

                if planner_debug is None:
                    planner_debug = {
                        "candidate_xy": np.zeros((0, 0, 2), dtype=np.float32),
                        "candidate_h_ft": np.zeros((0, 0), dtype=np.float32),
                        "final_xy": np.zeros((0, 2), dtype=np.float32),
                        "final_h_ft": np.zeros((0,), dtype=np.float32),
                    }

                planner_debug["lookahead_xy"] = np.asarray(
                    [[lookahead_north_ft, lookahead_center_east_ft]],
                    dtype=np.float32,
                )
                planner_debug["lookahead_h_ft"] = np.asarray(
                    # Match the centerline overlay reference height so the marker
                    # visually sits on the blue centerline ribbon.
                    [0.0],
                    dtype=np.float32,
                )

            heading_deg = _wrap_heading_deg(controller_state["psi"])
            heading_cmd_deg = heading_deg
            if controller_tag == "simple":
                simple_guidance = dict(getattr(controller, "last_guidance", {}) or {})
                heading_cmd_deg = float(simple_guidance.get("centerline_heading_deg", heading_deg))
            else:
                get_hdg_cmd = getattr(controller, "get_centerline_heading_rad", None)
                if callable(get_hdg_cmd):
                    try:
                        heading_cmd_deg = _wrap_heading_deg(get_hdg_cmd(controller_state["p_N"]))
                    except Exception:
                        heading_cmd_deg = heading_deg

            hud_debug = {
                "heading_deg": float(heading_deg),
                "heading_cmd_deg": float(heading_cmd_deg),
                "action_cmd": np.asarray(action, dtype=np.float32).copy(),
            }

            if np.isnan(action).any() or np.isinf(action).any():
                print(f"Invalid action at step {step}: {action}")
                break

            set_hud_commands = getattr(env.unwrapped, "set_hud_commands", None)
            if callable(set_hud_commands):
                set_hud_commands(heading_cmd_deg=heading_cmd_deg)

            obs, _, terminated, truncated, info = env.step(action)
            termination_reason = info.get("termination_reason", "running")
            state = env.unwrapped.get_full_state_dict()
            recorder.record_step(planner_debug=planner_debug, hud_debug=hud_debug)

            width_ft = float(info.get("canyon_width_ft", np.nan))
            lateral_ft = float(info.get("lateral_error_ft", np.nan))

            if step % 5 == 0:
                speed_fps = float(
                    np.sqrt(
                        float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2
                    )
                )
                rel_north_ft = float(controller_state["p_N"] - start_path_north_ft)
                rel_alt_ft = float(controller_state["h"])
                viz = render_state(lateral_ft, width_ft) if np.isfinite(lateral_ft) and np.isfinite(width_ft) else "Wall |                               | Wall"
                print(
                    f"{step:<5} | {rel_north_ft:<8.0f} | {lateral_ft:<8.0f} | {rel_alt_ft:<8.0f} | "
                    f"{speed_fps:<6.0f} | {width_ft:<6.0f} | {plan_ms:<8.1f} | {viz}"
                )

            if terminated or truncated:
                print(f"Episode ended at step {step}: {termination_reason}")
                break

            if args.render:
                time.sleep(1.0 / 30.0)
    finally:
        env.close()

    artifacts = recorder.finalize(termination_reason)
    print(f"Saved video: {artifacts['video_path']}")
    print(f"Saved trajectory overlay: {artifacts['overlay_path']}")
    print(f"Saved trajectory CSV: {artifacts['trajectory_csv_path']}")


if __name__ == "__main__":
    main()
