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
from jsbsim_gym.canyon_artifacts import CanyonRunRecorder
from jsbsim_gym.mppi_jax import (
    JaxMPPIConfig,
    JaxMPPIController,
    JaxSmoothMPPIConfig,
    JaxSmoothMPPIController,
)
from jsbsim_gym.simple_controller import SimpleCanyonController

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
OUTPUT_DIR = Path("output/canyon_mppi")


def render_state(lateral_error_ft, width_ft):
    y_norm = 2.0 * lateral_error_ft / max(width_ft, 1.0)

    viz = "Wall |"
    pos = int(np.clip((y_norm + 1.0) * 15, 0, 30))
    for i in range(31):
        viz += "X" if i == pos else " "
    viz += "| Wall"
    return viz


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


def build_canyon_profile(canyon, start_pixel=None):
    if hasattr(canyon, "north_samples_ft") and hasattr(canyon, "width_samples_ft"):
        north_samples_ft = np.asarray(canyon.north_samples_ft, dtype=np.float32)
        width_samples_ft = np.asarray(canyon.width_samples_ft, dtype=np.float32)
        if north_samples_ft.size >= 2 and width_samples_ft.size == north_samples_ft.size:
            center_east_samples_ft = np.zeros_like(north_samples_ft, dtype=np.float32)
            if hasattr(canyon, "center_east_samples_ft"):
                center_east_samples_ft = np.asarray(canyon.center_east_samples_ft, dtype=np.float32)
                if center_east_samples_ft.size != north_samples_ft.size:
                    center_east_samples_ft = np.zeros_like(north_samples_ft, dtype=np.float32)

            if (
                start_pixel is not None
                and hasattr(canyon, "ordered_dem_msl_m")
                and hasattr(canyon, "east_samples_ft")
                and hasattr(canyon, "get_pixel_info")
            ):
                try:
                    px, py = start_pixel
                    start_info = canyon.get_pixel_info(px, py)
                    rows = int(canyon.rows)
                    cols = int(canyon.cols)
                    dem_ordered = np.asarray(canyon.ordered_dem_msl_m, dtype=np.float32)
                    east_axis_ft = np.asarray(canyon.east_samples_ft, dtype=np.float32)

                    start_row = int(start_info["row_ordered"])
                    start_col = int(start_info["pixel_x"])
                    start_row = int(np.clip(start_row, 0, rows - 1))
                    start_col = int(np.clip(start_col, 0, cols - 1))

                    tracked_cols = np.asarray(
                        getattr(canyon, "center_col_samples", np.full(rows, start_col, dtype=np.int32)),
                        dtype=np.int32,
                    ).copy()
                    if tracked_cols.size != rows:
                        tracked_cols = np.full(rows, start_col, dtype=np.int32)
                    tracked_cols = np.clip(tracked_cols, 0, cols - 1)

                    search_radius_px = min(140, max(30, cols // 18))

                    tracked_cols[start_row] = start_col

                    col = start_col
                    for row in range(start_row + 1, rows):
                        lo = max(0, col - search_radius_px)
                        hi = min(cols - 1, col + search_radius_px)
                        row_profile = dem_ordered[row, lo : hi + 1]
                        candidate_cols = np.arange(lo, hi + 1, dtype=np.int32)
                        jump_penalty = 0.04 * ((candidate_cols - col).astype(np.float32) ** 2)
                        selected = int(np.argmin(row_profile + jump_penalty))
                        col = int(candidate_cols[selected])
                        tracked_cols[row] = col

                    col = start_col
                    for row in range(start_row - 1, -1, -1):
                        lo = max(0, col - search_radius_px)
                        hi = min(cols - 1, col + search_radius_px)
                        row_profile = dem_ordered[row, lo : hi + 1]
                        candidate_cols = np.arange(lo, hi + 1, dtype=np.int32)
                        jump_penalty = 0.04 * ((candidate_cols - col).astype(np.float32) ** 2)
                        selected = int(np.argmin(row_profile + jump_penalty))
                        col = int(candidate_cols[selected])
                        tracked_cols[row] = col

                    center_east_samples_ft = east_axis_ft[tracked_cols]
                except Exception:
                    pass

            if center_east_samples_ft.size >= 9:
                smooth_window = min(51, center_east_samples_ft.size // 2 * 2 + 1)
                smooth_window = max(9, smooth_window)
                kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
                center_east_samples_ft = np.convolve(center_east_samples_ft, kernel, mode="same").astype(np.float32)

            centerline_heading_samples_rad = np.zeros_like(center_east_samples_ft, dtype=np.float32)
            lookahead_rows = min(45, max(12, center_east_samples_ft.size // 18))
            for i in range(center_east_samples_ft.size):
                j = min(i + lookahead_rows, center_east_samples_ft.size - 1)
                if j == i:
                    j = max(i - lookahead_rows, 0)

                dn = float(north_samples_ft[j] - north_samples_ft[i])
                de = float(center_east_samples_ft[j] - center_east_samples_ft[i])

                if abs(dn) + abs(de) > 1e-6:
                    centerline_heading_samples_rad[i] = np.arctan2(de, dn)
                elif i > 0:
                    centerline_heading_samples_rad[i] = centerline_heading_samples_rad[i - 1]

            heading_unwrapped = np.unwrap(centerline_heading_samples_rad.astype(np.float64))
            if heading_unwrapped.size >= 9:
                heading_kernel = np.ones(21, dtype=np.float64) / 21.0
                heading_unwrapped = np.convolve(heading_unwrapped, heading_kernel, mode="same")
            centerline_heading_samples_rad = np.arctan2(
                np.sin(heading_unwrapped),
                np.cos(heading_unwrapped),
            ).astype(np.float32)

            return (
                north_samples_ft,
                width_samples_ft,
                center_east_samples_ft,
                centerline_heading_samples_rad,
            )

    # Fallback for non-profile canyons: sample local width over a long span.
    north_samples_ft = np.linspace(0.0, 24000.0, 256, dtype=np.float32)
    width_samples_ft = np.empty_like(north_samples_ft)
    for i, north_ft in enumerate(north_samples_ft):
        width_ft, _ = canyon.get_geometry(float(north_ft))
        width_samples_ft[i] = float(width_ft)
    center_east_samples_ft = np.zeros_like(north_samples_ft, dtype=np.float32)
    centerline_heading_samples_rad = np.zeros_like(north_samples_ft, dtype=np.float32)
    return (
        north_samples_ft,
        width_samples_ft,
        center_east_samples_ft,
        centerline_heading_samples_rad,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MPPI canyon controller.")
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi"],
        default="mppi",
        help="Controller variant to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for saved run artifacts (video and trajectory plot).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum control steps to run.",
    )
    return parser.parse_args()


def apply_safety_envelope(
    action,
    controller_state,
    lateral_error_ft,
    heading_error_rad,
    canyon_width_ft,
    wall_margin_ft,
    target_altitude_ft,
):
    corrected = np.asarray(action, dtype=np.float32).copy()

    p_rate = float(controller_state["p"])
    q_rate = float(controller_state["q"])
    h_rel_ft = float(controller_state["h"])
    speed_fps = float(
        np.sqrt(
            float(controller_state["u"]) ** 2
            + float(controller_state["v"]) ** 2
            + float(controller_state["w"]) ** 2
        )
    )

    usable_half_ft = max(0.5 * float(canyon_width_ft) - float(wall_margin_ft), 80.0)
    lateral_norm = abs(float(lateral_error_ft)) / max(usable_half_ft, 1.0)

    if lateral_norm > 0.72:
        roll_guard = np.clip(
            -0.0036 * float(lateral_error_ft)
            - 0.85 * float(heading_error_rad)
            - 0.24 * p_rate,
            -1.0,
            1.0,
        )
        yaw_guard = np.clip(-0.24 * float(controller_state["r"]) - 0.22 * float(heading_error_rad), -0.45, 0.45)
        roll_blend = float(np.clip((lateral_norm - 0.72) / 0.25, 0.25, 1.00))
        yaw_blend = float(np.clip((lateral_norm - 0.72) / 0.30, 0.20, 0.85))
        corrected[0] = (1.0 - roll_blend) * corrected[0] + roll_blend * roll_guard
        corrected[2] = (1.0 - yaw_blend) * corrected[2] + yaw_blend * yaw_guard

    altitude_error_ft = h_rel_ft - float(target_altitude_ft)
    altitude_deficit_ft = max(-altitude_error_ft, 0.0)
    if h_rel_ft < float(target_altitude_ft) + 120.0:
        # In this setup, more negative elevator commands produce climb.
        climb_guard = np.clip(0.0030 * altitude_error_ft + 0.28 * q_rate, -1.0, 0.55)
        pitch_blend = float(np.clip((float(target_altitude_ft) + 120.0 - h_rel_ft) / 240.0, 0.25, 0.92))
        corrected[1] = (1.0 - pitch_blend) * corrected[1] + pitch_blend * climb_guard

        throttle_guard = np.clip(
            0.62 + 0.0015 * (460.0 - speed_fps) + 0.0008 * altitude_deficit_ft,
            0.45,
            1.0,
        )
        corrected[3] = max(corrected[3], throttle_guard)

    return np.clip(corrected, -1.0, 1.0).astype(np.float32)


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "headless" if args.headless else "render"
    controller_tag = str(args.controller)

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode="rgb_array" if args.headless else "human",
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
        wind_sigma=1.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    obs, info = env.reset(seed=3)
    del obs, info

    state = env.unwrapped.get_full_state_dict()

    altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))
    initial_controller_state = to_mppi_state(env, state, altitude_ref_ft)
    start_path_north_ft = float(initial_controller_state["p_N"])

    target_altitude_ft = float(env.unwrapped.target_altitude_ft - altitude_ref_ft)
    min_altitude_ft = float(env.unwrapped.min_altitude_ft - altitude_ref_ft)
    max_altitude_ft = float(env.unwrapped.max_altitude_ft - altitude_ref_ft)

    (
        north_samples_ft,
        width_samples_ft,
        center_east_samples_ft,
        centerline_heading_samples_rad,
    ) = build_canyon_profile(env.unwrapped.canyon, start_pixel=DEM_START_PIXEL)

    mppi_target_altitude_ft = max(target_altitude_ft, 700.0)

    config_base_kwargs = dict(
        horizon=45,
        num_samples=768,
        optimization_steps=3,
        lambda_=2.0,
        progress_gain=0.60,
        speed_gain=0.20,
        low_altitude_gain=2.50,
        centerline_gain=0.95,
        offcenter_penalty_gain=0.85,
        heading_alignment_gain=0.90,
        heading_alignment_scale_rad=0.80,
        alive_bonus=0.15,
        target_altitude_ft=mppi_target_altitude_ft,
        min_altitude_ft=min_altitude_ft,
        max_altitude_ft=max_altitude_ft,
        terrain_collision_height_ft=max(min_altitude_ft + 40.0, 160.0),
        wall_margin_ft=float(env.unwrapped.wall_margin_ft),
        terrain_crash_penalty=45.0,
        wall_crash_penalty=32.0,
        altitude_violation_penalty=8.0,
        early_termination_penalty_gain=120.0,
        time_limit_bonus=35.0,
        max_step_reward_abs=5.0,
        angular_rate_penalty_gain=0.0,
        angular_rate_threshold_deg_s=0.0,
    )

    if controller_tag == "smooth_mppi":
        config = JaxSmoothMPPIConfig(
            **config_base_kwargs,
            action_noise_std=(0.14, 0.22, 0.12, 0.10),
            delta_noise_std=(0.08, 0.12, 0.08, 0.06),
            delta_action_bounds=(0.18, 0.26, 0.14, 0.10),
            noise_smoothing_kernel=(0.10, 0.20, 0.40, 0.20, 0.10),
            smoothness_penalty_weight=0.35,
            action_diff_weight=0.8,
            action_l2_weight=0.1,
        )
        controller_cls = JaxSmoothMPPIController
    else:
        config = JaxMPPIConfig(
            **config_base_kwargs,
            action_noise_std=(0.14, 0.22, 0.12, 0.10),
            action_diff_weight=0.6,
            action_l2_weight=0.1,
        )
        controller_cls = JaxMPPIController

    controller = controller_cls(
        config=config,
        canyon_north_samples_ft=north_samples_ft,
        canyon_width_samples_ft=width_samples_ft,
        canyon_center_east_samples_ft=center_east_samples_ft,
        canyon_centerline_heading_rad_samples=centerline_heading_samples_rad,
    )
    safety_controller = SimpleCanyonController(
        env,
        target_speed_fps=360.0,
        target_clearance_ft=700.0,
        lookahead_rows=40,
        terrain_lookahead_ft=(500.0, 1000.0, 1600.0, 2200.0),
        use_dem_centerline=True,
    )
    safety_controller.reset(state)

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
    termination_reason = "running"

    print(f"Initializing {controller_tag} canyon controller...")
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

            width_ft = float(controller.get_canyon_width_ft(controller_state["p_N"]))
            lateral_ft = float(controller.get_lateral_error_ft(controller_state["p_N"], controller_state["p_E"]))
            centerline_heading_rad = float(controller.get_centerline_heading_rad(controller_state["p_N"]))
            heading_error_rad = float(
                np.arctan2(
                    np.sin(float(controller_state["psi"]) - centerline_heading_rad),
                    np.cos(float(controller_state["psi"]) - centerline_heading_rad),
                )
            )
            safety_action = safety_controller.get_action(state)
            action = apply_safety_envelope(
                action,
                controller_state,
                lateral_ft,
                heading_error_rad,
                width_ft,
                env.unwrapped.wall_margin_ft,
                mppi_target_altitude_ft,
            )

            usable_half_ft = max(0.5 * width_ft - float(env.unwrapped.wall_margin_ft), 80.0)
            lateral_norm = abs(lateral_ft) / max(usable_half_ft, 1.0)
            altitude_recovery_risk = float(
                np.clip((mppi_target_altitude_ft + 100.0 - float(controller_state["h"])) / 260.0, 0.0, 1.0)
            )
            wall_recovery_risk = float(np.clip((lateral_norm - 0.62) / 0.45, 0.0, 1.0))
            blend = max(altitude_recovery_risk, wall_recovery_risk)
            if blend > 0.0:
                action = ((1.0 - blend) * action + blend * safety_action).astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            if np.isnan(action).any() or np.isinf(action).any():
                print(f"Invalid action at step {step}: {action}")
                break

            _, _, terminated, truncated, info = env.step(action)
            termination_reason = info.get("termination_reason", "running")
            state = env.unwrapped.get_full_state_dict()
            recorder.record_step(planner_debug=planner_debug)

            if step % 5 == 0:
                speed_fps = float(
                    np.sqrt(
                        float(state["u"]) ** 2 + float(state["v"]) ** 2 + float(state["w"]) ** 2
                    )
                )
                rel_north_ft = float(controller_state["p_N"] - start_path_north_ft)
                rel_alt_ft = float(controller_state["h"])
                viz = render_state(lateral_ft, width_ft)
                print(
                    f"{step:<5} | {rel_north_ft:<8.0f} | {lateral_ft:<8.0f} | {rel_alt_ft:<8.0f} | "
                    f"{speed_fps:<6.0f} | {width_ft:<6.0f} | {plan_ms:<8.1f} | {viz}"
                )

            if terminated or truncated:
                print(f"Episode ended at step {step}: {termination_reason}")
                break

            if not args.headless:
                time.sleep(1.0 / 30.0)
    finally:
        env.close()

    artifacts = recorder.finalize(termination_reason)
    print(f"Saved video: {artifacts['video_path']}")
    print(f"Saved trajectory overlay: {artifacts['overlay_path']}")
    print(f"Saved trajectory CSV: {artifacts['trajectory_csv_path']}")


if __name__ == "__main__":
    main()
