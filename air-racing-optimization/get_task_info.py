from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

from task_config import get_active_task_config
from terrain_model.load_raw_data import lat_lon_to_north_east, terrain_data


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsbsim_gym.canyon import DEMCanyon  # noqa: E402


FT_TO_M = 0.3048


def _ordered_row_to_original_row(row_ordered: float, rows: int, fly_direction: str) -> float:
    if fly_direction == "south_to_north":
        return float(rows - 1) - float(row_ordered)
    return float(row_ordered)


def _pixel_to_lat_lon(pixel_x: float, pixel_y: float, rows: int, cols: int, dem_bbox):
    south, north, west, east = dem_bbox
    lat = north - ((float(pixel_y) + 0.5) / float(rows)) * (north - south)
    lon = west + ((float(pixel_x) + 0.5) / float(cols)) * (east - west)
    return float(lat), float(lon)


def _course_info_from_dem(task_config):
    canyon = DEMCanyon(
        dem_path=task_config.dem_path,
        south=task_config.dem_bbox[0],
        north=task_config.dem_bbox[1],
        west=task_config.dem_bbox[2],
        east=task_config.dem_bbox[3],
        valley_rel_elev=task_config.valley_rel_elev,
        smoothing_window=task_config.smoothing_window,
        min_width_ft=task_config.min_width_ft,
        max_width_ft=task_config.max_width_ft,
        fly_direction=task_config.fly_direction,
        dem_start_pixel=task_config.start_pixel,
    )

    start_info = canyon.get_pixel_info(*task_config.start_pixel)
    north_start, east_start = lat_lon_to_north_east(start_info["lat_deg"], start_info["lon_deg"])

    if task_config.finish_pixel is not None:
        finish_info = canyon.get_pixel_info(*task_config.finish_pixel)
    else:
        span_ft = float(task_config.span_ft if task_config.span_ft is not None else canyon.get_total_length_ft())
        target_local_north_ft = np.clip(
            start_info["local_north_ft"] + span_ft,
            0.0,
            canyon.get_total_length_ft(),
        )
        row_axis = np.arange(canyon.rows, dtype=np.float32)
        target_row_ordered = float(np.interp(target_local_north_ft, canyon.north_samples_ft, row_axis))
        target_center_east_ft = float(
            np.interp(target_local_north_ft, canyon.north_samples_ft, canyon.center_east_samples_ft)
        )
        target_col = float(np.interp(target_center_east_ft, canyon.east_samples_ft, np.arange(canyon.cols)))
        target_row_original = _ordered_row_to_original_row(
            target_row_ordered,
            canyon.rows,
            canyon.fly_direction,
        )
        finish_lat, finish_lon = _pixel_to_lat_lon(
            pixel_x=target_col,
            pixel_y=target_row_original,
            rows=canyon.rows,
            cols=canyon.cols,
            dem_bbox=task_config.dem_bbox,
        )
        finish_info = {
            "lat_deg": finish_lat,
            "lon_deg": finish_lon,
            "row_ordered": target_row_ordered,
            "pixel_x": target_col,
            "pixel_y": target_row_original,
            "elevation_msl_m": float(canyon.get_elevation_msl_ft_from_latlon(finish_lat, finish_lon) * FT_TO_M),
        }

    north_end, east_end = lat_lon_to_north_east(finish_info["lat_deg"], finish_info["lon_deg"])
    initial_heading_deg = float(canyon.get_heading_for_pixel(*task_config.start_pixel))

    return {
        "task_name": task_config.name,
        "north_start": float(north_start),
        "east_start": float(east_start),
        "north_end": float(north_end),
        "east_end": float(east_end),
        "start_lat_deg": float(start_info["lat_deg"]),
        "start_lon_deg": float(start_info["lon_deg"]),
        "end_lat_deg": float(finish_info["lat_deg"]),
        "end_lon_deg": float(finish_info["lon_deg"]),
        "start_terrain_elev_m": float(start_info["elevation_msl_m"]),
        "end_terrain_elev_m": float(finish_info["elevation_msl_m"]),
        "initial_track_deg": initial_heading_deg,
        "initial_altitude_agl_m": float(task_config.initial_altitude_agl_ft * FT_TO_M),
    }


def _course_info_from_legacy_latlon(task_config):
    if task_config.legacy_start_latlon is None or task_config.legacy_end_latlon is None:
        raise ValueError("Legacy task config is missing start/end lat/lon.")

    north_start, east_start = lat_lon_to_north_east(*task_config.legacy_start_latlon)
    north_end, east_end = lat_lon_to_north_east(*task_config.legacy_end_latlon)

    return {
        "task_name": task_config.name,
        "north_start": float(north_start),
        "east_start": float(east_start),
        "north_end": float(north_end),
        "east_end": float(east_end),
        "start_lat_deg": float(task_config.legacy_start_latlon[0]),
        "start_lon_deg": float(task_config.legacy_start_latlon[1]),
        "end_lat_deg": float(task_config.legacy_end_latlon[0]),
        "end_lon_deg": float(task_config.legacy_end_latlon[1]),
        "start_terrain_elev_m": 0.0,
        "end_terrain_elev_m": 0.0,
        "initial_track_deg": float(task_config.initial_heading_deg),
        "initial_altitude_agl_m": float(task_config.initial_altitude_agl_ft * FT_TO_M),
    }


print("Loading task-specific info and terrain data...")

task_config = get_active_task_config()
if task_config.terrain_source == "dem_clip":
    course_info = _course_info_from_dem(task_config)
else:
    course_info = _course_info_from_legacy_latlon(task_config)

north_start_index = np.argmin(np.abs(terrain_data["north_edges"] - course_info["north_start"]))
east_start_index = np.argmin(np.abs(terrain_data["east_edges"] - course_info["east_start"]))
north_end_index = np.argmin(np.abs(terrain_data["north_edges"] - course_info["north_end"]))
east_end_index = np.argmin(np.abs(terrain_data["east_edges"] - course_info["east_end"]))

i_lims = np.sort(np.array([north_start_index, north_end_index]))
j_lims = np.sort(np.array([east_start_index, east_end_index]))

padding_distance = float(task_config.crop_padding_m)
dx_north = float(np.mean(np.diff(terrain_data["north_edges"])))
dx_east = float(np.mean(np.diff(terrain_data["east_edges"])))

i_lims += np.round(np.array([-padding_distance, padding_distance]) / dx_north).astype(int)
j_lims += np.round(np.array([-padding_distance, padding_distance]) / dx_east).astype(int)

i_lims = np.clip(i_lims, 0, terrain_data["elev"].shape[0] - 1)
j_lims = np.clip(j_lims, 0, terrain_data["elev"].shape[1] - 1)

terrain_data_zoomed = {
    "elev": terrain_data["elev"][i_lims[0] : i_lims[1], j_lims[0] : j_lims[1]],
    "north_edges": terrain_data["north_edges"][i_lims[0] : i_lims[1]],
    "east_edges": terrain_data["east_edges"][j_lims[0] : j_lims[1]],
    "cache_tag": (
        f"{terrain_data.get('cache_tag', 'terrain')}"
        f"_crop_i{i_lims[0]}_{i_lims[1]}_j{j_lims[0]}_{j_lims[1]}"
    ),
    "dx_north": dx_north,
    "dx_east": dx_east,
    "north_start": course_info["north_start"],
    "east_start": course_info["east_start"],
    "north_end": course_info["north_end"],
    "east_end": course_info["east_end"],
    "north_start_index": np.argmin(
        np.abs(terrain_data["north_edges"][i_lims[0] : i_lims[1]] - course_info["north_start"])
    ),
    "east_start_index": np.argmin(
        np.abs(terrain_data["east_edges"][j_lims[0] : j_lims[1]] - course_info["east_start"])
    ),
    "north_end_index": np.argmin(
        np.abs(terrain_data["north_edges"][i_lims[0] : i_lims[1]] - course_info["north_end"])
    ),
    "east_end_index": np.argmin(
        np.abs(terrain_data["east_edges"][j_lims[0] : j_lims[1]] - course_info["east_end"])
    ),
}

terrain_cost_heuristic = terrain_data_zoomed["elev"] - ndimage.gaussian_filter(
    terrain_data_zoomed["elev"],
    (
        task_config.terrain_cost_sigma_m / dx_north,
        task_config.terrain_cost_sigma_m / dx_east,
    ),
    truncate=2,
)
terrain_cost_std = max(float(np.std(terrain_cost_heuristic)), 1e-6)
terrain_cost_heuristic = np.exp(
    (terrain_cost_heuristic - float(np.mean(terrain_cost_heuristic)))
    / terrain_cost_std
    * float(task_config.terrain_cost_weight)
)
