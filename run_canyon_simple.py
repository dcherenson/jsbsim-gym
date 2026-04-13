import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_artifacts import CanyonRunRecorder
from jsbsim_gym.simple_controller import (
    SimpleCanyonController,
    SimpleCanyonControllerConfig,
    with_default_simple_controller_optuna_gains,
)

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
OUTPUT_DIR = Path("output/canyon_simple")


def render_state(lateral_error_ft, width_ft):
    y_norm = 2.0 * lateral_error_ft / max(width_ft, 1.0)
    viz = "Wall |"
    pos = int(np.clip((y_norm + 1.0) * 15, 0, 30))
    for i in range(31):
        viz += "X" if i == pos else " "
    viz += "| Wall"
    return viz


def parse_args():
    parser = argparse.ArgumentParser(description="Run the simple canyon controller.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for saved run artifacts (video and trajectory plot).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "headless" if args.headless else "render"
    mode_label = mode_tag
    video_path = output_dir / f"canyon_simple_{mode_tag}.mp4"
    overlay_path = output_dir / f"canyon_simple_{mode_tag}_trajectory_overlay.png"
    trajectory_path = output_dir / f"canyon_simple_{mode_tag}_trajectory.csv"

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
        dem_start_heading_deg=None,
        dem_render_mesh=True,
        dem_render_stride=4,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=900.0,
        entry_altitude_ft=900.0,
        min_altitude_ft=-500.0,
        max_altitude_ft=5000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=10.0,
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    obs, info = env.reset(seed=3)
    del obs, info

    state = env.unwrapped.get_full_state_dict()

    config = SimpleCanyonControllerConfig(
        target_speed_fps=300.0,
        target_clearance_ft=1100.0,
        lookahead_rows=40,
        use_dem_centerline=True,
        use_terrain_following=True,
    )
    config, simple_tuning_source, simple_tuned_keys = with_default_simple_controller_optuna_gains(config)
    if simple_tuned_keys:
        print(
            f"Auto-loaded simple-controller tuned gains from {simple_tuning_source} "
            f"({len(simple_tuned_keys)} parameters)."
        )
    else:
        print("Note: No tuned simple-controller gains found; using built-in defaults.")

    controller = SimpleCanyonController(
        env=env,
        config=config,
    )
    controller.reset(state)

    recorder = CanyonRunRecorder(
        env=env,
        dem_path=DEM_PATH,
        dem_bbox=DEM_BBOX,
        dem_start_pixel=DEM_START_PIXEL,
        output_dir=output_dir,
        file_stem=f"canyon_simple_{mode_tag}",
        title_prefix="Simple Controller Trajectory Overlay",
        fps=30,
    )

    termination_reason = "running"
    recorder.initialize()

    print(f"Starting Simple Canyon Controller ({mode_label})...")
    print(
        f"{'Step':<5} | {'North':<8} | {'LatErr':<8} | {'LatDot':<8} | {'HeadErr':<8} | "
        f"{'phi':<7} | {'roll_d':<7} | {'roll_u':<7} | {'clr':<8} | {'clr_e':<8} | {'clr_d':<8} | {'V':<6} | {'margin':<8}"
    )
    print("-" * 145)

    min_margin_ft = float("inf")
    max_abs_lat_ft = 0.0

    try:
        for step in range(1200):
            action = controller.get_action(state)
            guidance = controller.last_guidance

            _, _, terminated, truncated, info = env.step(action)
            state = env.unwrapped.get_full_state_dict()
            termination_reason = info.get("termination_reason", "running")
            recorder.record_step()

            margin_ft = float(guidance.get("margin_to_wall_ft", 0.0))
            min_margin_ft = min(min_margin_ft, margin_ft)
            max_abs_lat_ft = max(max_abs_lat_ft, abs(float(guidance.get("lateral_error_ft", 0.0))))

            if step < 40 or step % 10 == 0:
                lateral_ft = float(guidance.get("lateral_error_ft", info.get("lateral_error_ft", 0.0)))
                lateral_rate_fps = float(guidance.get("lateral_error_rate_fps", 0.0))
                heading_deg = float(guidance.get("heading_error_deg", np.degrees(guidance.get("heading_error_rad", 0.0))))
                phi_deg = float(np.degrees(state["phi"]))
                roll_des_deg = float(guidance.get("roll_des_deg", 0.0))
                roll_cmd = float(guidance.get("roll_cmd", action[0]))
                clearance_ft = float(guidance.get("terrain_clearance_ft", info.get("terrain_clearance_ft", 0.0)))
                clearance_error_ft = float(guidance.get("clearance_error_ft", 0.0))
                clearance_rate_fps = float(guidance.get("clearance_rate_fps", 0.0))
                speed_fps = float(guidance.get("speed_fps", np.sqrt(state["u"] ** 2 + state["v"] ** 2 + state["w"] ** 2)))
                north_ft = float(guidance.get("local_north_ft", state["p_N"]))
                print(
                    f"{step:<5} | {north_ft:<8.0f} | {lateral_ft:<8.1f} | {lateral_rate_fps:<8.1f} | {heading_deg:<8.2f} | "
                    f"{phi_deg:<7.2f} | {roll_des_deg:<7.2f} | {roll_cmd:<7.3f} | {clearance_ft:<8.1f} | {clearance_error_ft:<8.1f} | {clearance_rate_fps:<8.1f} | {speed_fps:<6.1f} | {margin_ft:<8.1f}"
                )

            if terminated or truncated:
                print(
                    f"Episode ended at step {step}: {termination_reason} | "
                    f"min_margin_ft={min_margin_ft:.1f} | max_abs_lat_ft={max_abs_lat_ft:.1f}"
                )
                break
        else:
            print(
                f"Completed full episode without termination | min_margin_ft={min_margin_ft:.1f} "
                f"| max_abs_lat_ft={max_abs_lat_ft:.1f}"
            )
    finally:
        env.close()

    artifacts = recorder.finalize(termination_reason)

    print(f"Saved video: {artifacts['video_path']}")
    print(f"Saved trajectory overlay: {artifacts['overlay_path']}")
    print(f"Saved trajectory CSV: {artifacts['trajectory_csv_path']}")


if __name__ == "__main__":
    main()
