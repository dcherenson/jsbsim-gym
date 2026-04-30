from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0.
from jsbsim_gym.canyon import DEMCanyon
from jsbsim_gym.canyon_artifacts import latlon_to_pixel
from jsbsim_gym.mppi_run_config import build_mppi_base_config_kwargs, build_mppi_controller
from jsbsim_gym.nominal_trajectory import (
    build_nominal_reference_from_dyn,
    load_nominal_initial_conditions_from_dyn,
)

DEM_PATH = REPO_ROOT / "data" / "dem" / "black-canyon-gunnison_USGS10m.tif"
DYN_PATH = REPO_ROOT / "air-racing-optimization" / "final_results" / "dyn.asb"


def test_build_nominal_reference_from_dyn():
    canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=38.52,
        north=38.62,
        west=-107.78,
        east=-107.65,
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=(1400, 950),
    )
    start_info = canyon.get_pixel_info(1400, 950)

    reference = build_nominal_reference_from_dyn(
        DYN_PATH,
        canyon=canyon,
        altitude_ref_ft=start_info["elevation_msl_ft"],
        resample_spacing_ft=12.0,
    )

    assert int(len(reference["north_ft"])) > 100
    assert int(len(reference["time_s"])) == int(len(reference["north_ft"]))
    assert np.all(np.diff(reference["time_s"]) > 0.0)
    assert np.all(np.isfinite(reference["east_ft"]))
    assert np.all(np.isfinite(reference["heading_rad"]))
    assert np.all(np.isfinite(reference["altitude_ft"]))
    assert np.all(np.isfinite(reference["speed_fps"]))
    assert np.all(np.isfinite(reference["phi_rad"]))
    assert np.all(np.isfinite(reference["theta_rad"]))
    assert np.all(np.isfinite(reference["psi_rad"]))
    assert reference["reference_states_ft_rad"].shape[1] == 6
    assert int(reference["reference_states_ft_rad"].shape[0]) == int(len(reference["time_s"]))
    assert np.all(reference["width_ft"] > 0.0)
    assert int(len(reference["display_north_ft"])) > 100
    assert np.all(np.isfinite(reference["display_east_ft"]))
    assert np.all(np.isfinite(reference["display_altitude_ft"]))


def test_load_nominal_initial_conditions_from_dyn():
    canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=38.52,
        north=38.62,
        west=-107.78,
        east=-107.65,
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=(1400, 950),
    )

    ic = load_nominal_initial_conditions_from_dyn(
        DYN_PATH,
        canyon=canyon,
    )

    assert 0 <= int(ic["start_pixel"][0]) < int(canyon.cols)
    assert 0 <= int(ic["start_pixel"][1]) < int(canyon.rows)
    assert np.isfinite(float(ic["entry_altitude_ft"]))
    assert np.isfinite(float(ic["speed_kts"]))
    assert np.isfinite(float(ic["heading_deg"]))
    assert np.isfinite(float(ic["roll_deg"]))
    assert np.isfinite(float(ic["pitch_deg"]))
    assert np.isfinite(float(ic["alpha_deg"]))
    assert np.isfinite(float(ic["beta_deg"]))


def test_build_mppi_controller_from_nominal_reference():
    canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=38.52,
        north=38.62,
        west=-107.78,
        east=-107.65,
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=(1400, 950),
    )
    start_info = canyon.get_pixel_info(1400, 950)
    reference = build_nominal_reference_from_dyn(
        DYN_PATH,
        canyon=canyon,
        altitude_ref_ft=start_info["elevation_msl_ft"],
        resample_spacing_ft=12.0,
    )
    ic = load_nominal_initial_conditions_from_dyn(DYN_PATH, canyon=canyon)

    controller, _ = build_mppi_controller(
        "mppi",
        config_base_kwargs=build_mppi_base_config_kwargs(horizon=6, num_samples=8, optimization_steps=1),
        reference_trajectory=reference,
        terrain_north_samples_ft=np.asarray(canyon.north_samples_ft, dtype=np.float32),
        terrain_east_samples_ft=np.asarray(canyon.east_samples_ft, dtype=np.float32),
        terrain_elevation_ft=np.asarray(canyon.ordered_dem_msl_m, dtype=np.float32) * 3.28084
        - float(start_info["elevation_msl_ft"]),
    )

    speed_fps = float(ic["speed_fps"])
    alpha_rad = float(np.deg2rad(ic["alpha_deg"]))
    state = {
        "p_N": float(ic["local_north_ft"]),
        "p_E": float(ic["local_east_ft"]),
        "h": float(ic["entry_altitude_ft"]),
        "u": float(speed_fps),
        "v": 0.0,
        "w": float(np.tan(alpha_rad) * speed_fps),
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "phi": float(np.deg2rad(ic["roll_deg"])),
        "theta": float(np.deg2rad(ic["pitch_deg"])),
        "psi": float(np.deg2rad(ic["heading_deg"])),
        "ny": 0.0,
        "nz": 1.0,
    }
    action = np.asarray(controller.get_action(state), dtype=np.float32)
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


def test_canyon_env_reset_uses_explicit_nominal_start_pixel():
    canyon = DEMCanyon(
        dem_path=DEM_PATH,
        south=38.52,
        north=38.62,
        west=-107.78,
        east=-107.65,
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=(1400, 950),
    )
    ic = load_nominal_initial_conditions_from_dyn(
        DYN_PATH,
        canyon=canyon,
    )

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode=None,
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
        dem_bbox=(38.52, 38.62, -107.78, -107.65),
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=tuple(ic["start_pixel"]),
        dem_start_heading_mode="follow_canyon",
        dem_start_heading_deg=float(ic["heading_deg"]),
        dem_render_mesh=False,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=500.0,
        entry_altitude_ft=float(ic["entry_altitude_ft"]),
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=10.0,
        entry_speed_kts=float(ic["speed_kts"]),
        entry_roll_deg=float(ic["roll_deg"]),
        entry_pitch_deg=float(ic["pitch_deg"]),
        entry_alpha_deg=float(ic["alpha_deg"]),
        entry_beta_deg=float(ic["beta_deg"]),
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )
    try:
        env.reset(seed=3)
        actual_start_pixel = tuple(int(v) for v in env.unwrapped.dem_start_pixel)
        assert actual_start_pixel == tuple(int(v) for v in ic["start_pixel"])

        lat_deg = float(env.unwrapped.simulation.get_property_value("position/lat-gc-deg"))
        lon_deg = float(env.unwrapped.simulation.get_property_value("position/long-gc-deg"))
        px, py = latlon_to_pixel(
            lat_deg,
            lon_deg,
            38.52,
            38.62,
            -107.78,
            -107.65,
            env.unwrapped.canyon.rows,
            env.unwrapped.canyon.cols,
        )

        assert abs(px - float(ic["start_pixel"][0])) < 1e-6
        assert abs(py - float(ic["start_pixel"][1])) < 1e-6
    finally:
        env.close()
