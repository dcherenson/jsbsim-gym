from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsbsim_gym.canyon import DEMCanyon
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
    assert np.all(np.diff(reference["north_ft"]) > 0.0)
    assert np.all(np.isfinite(reference["east_ft"]))
    assert np.all(np.isfinite(reference["heading_rad"]))
    assert np.all(np.isfinite(reference["altitude_ft"]))
    assert np.all(np.isfinite(reference["speed_fps"]))
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
