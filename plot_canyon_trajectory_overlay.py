#!/usr/bin/env python3
"""Plot aircraft trajectory over DEM elevation for heading diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from run_scenario import AltitudeHoldController


def latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, rows, cols):
    x = ((lon_deg - west) / (east - west)) * float(cols) - 0.5
    y = ((north - lat_deg) / (north - south)) * float(rows) - 0.5
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240, help="Max control steps to simulate")
    parser.add_argument("--seed", type=int, default=3, help="Environment reset seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dem/plots/black_canyon_trajectory_overlay.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dem_path = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
    south, north, west, east = 38.52, 38.62, -107.78, -107.65
    start_pixel = (1400, 950)  # plot-space x,y provided by user

    dem = iio.imread(dem_path).astype(np.float32)
    if dem.ndim == 3:
        dem = dem[..., 0]
    dem[(~np.isfinite(dem)) | (dem < -1e20)] = np.nan

    rows, cols = dem.shape

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode=None,
        canyon_mode="dem",
        dem_path=str(dem_path),
        dem_bbox=(south, north, west, east),
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=start_pixel,
        dem_start_heading_mode="follow_canyon",
        dem_render_mesh=True,
        dem_render_stride=2,
        dem_render_vertical_exaggeration=1.0,
        dem_render_padding_ft=6000.0,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=250.0,
        entry_altitude_ft=250.0,
        min_altitude_ft=60.0,
        max_altitude_ft=1500.0,
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    controller = AltitudeHoldController()

    obs, info = env.reset(seed=args.seed)

    track_x = []
    track_y = []

    sim = env.unwrapped.simulation

    def sample_position():
        lat_deg = float(sim.get_property_value("position/lat-gc-deg"))
        lon_deg = float(sim.get_property_value("position/long-gc-deg"))
        px, py = latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, rows, cols)
        track_x.append(px)
        track_y.append(py)

    sample_position()

    termination_reason = "running"
    for _ in range(args.steps):
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        sample_position()
        termination_reason = info.get("termination_reason", "running")
        if terminated or truncated:
            break

    env.close()

    vmin = float(np.nanpercentile(dem, 2.0))
    vmax = float(np.nanpercentile(dem, 98.0))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(dem, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    ax.plot(track_x, track_y, color="red", linewidth=2.0, label="Aircraft trajectory")
    ax.scatter([track_x[0]], [track_y[0]], c="lime", s=70, edgecolors="black", linewidths=0.5, label="Trajectory start")
    ax.scatter([track_x[-1]], [track_y[-1]], c="red", marker="x", s=80, label="Trajectory end")

    ax.scatter(
        [start_pixel[0]],
        [start_pixel[1]],
        c="cyan",
        marker="+",
        s=120,
        linewidths=2.0,
        label="Configured start pixel",
    )

    if len(track_x) > 8:
        dx = track_x[8] - track_x[0]
        dy = track_y[8] - track_y[0]
        ax.arrow(
            track_x[0],
            track_y[0],
            dx,
            dy,
            color="white",
            width=0.6,
            head_width=12.0,
            head_length=14.0,
            length_includes_head=True,
            alpha=0.8,
        )

    ax.set_xlim(0, cols - 1)
    ax.set_ylim(rows - 1, 0)
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    ax.set_title(f"Black Canyon Trajectory Overlay | steps={len(track_x)-1}, end={termination_reason}")
    ax.legend(loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Saved overlay: {args.output}")
    print(f"DEM shape: {rows}x{cols}")
    print(f"Configured start pixel (x,y): {start_pixel}")
    print(f"First sampled pixel: ({track_x[0]:.1f}, {track_y[0]:.1f})")
    print(f"Last sampled pixel: ({track_x[-1]:.1f}, {track_y[-1]:.1f})")


if __name__ == "__main__":
    main()
