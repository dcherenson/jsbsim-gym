#!/usr/bin/env python3
"""Plot downloaded DEM GeoTIFFs for quick visual verification.

Examples:
  uv run python plot_downloaded_dem.py
  uv run python plot_downloaded_dem.py data/dem/grand-canyon_USGS10m.tif --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem_files",
        nargs="*",
        type=Path,
        help="GeoTIFF DEM files to preview. If omitted, uses data/dem/*.tif",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dem/plots"),
        help="Directory for generated preview PNGs",
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=2.0,
        help="Lower percentile for elevation color clipping",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=98.0,
        help="Upper percentile for elevation color clipping",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving PNGs",
    )
    return parser.parse_args()


def compute_hillshade(z: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    finite = np.isfinite(z)
    if not np.any(finite):
        return np.full_like(z, np.nan, dtype=np.float32)

    fill = float(np.nanmedian(z))
    z_filled = np.where(finite, z, fill)

    gy, gx = np.gradient(z_filled)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)

    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, 0.0, 1.0).astype(np.float32)
    shaded[~finite] = np.nan
    return shaded


def load_dem(path: Path) -> np.ndarray:
    arr = iio.imread(path).astype(np.float32)

    if arr.ndim == 3:
        # Some TIFFs can carry singleton channels; use first channel for DEM.
        arr = arr[..., 0]

    # Guard against common no-data sentinels in floating rasters.
    arr[(~np.isfinite(arr)) | (arr < -1e20)] = np.nan
    return arr


def render_preview(dem_path: Path, output_dir: Path, clip_lower: float, clip_upper: float, show: bool) -> Path:
    dem = load_dem(dem_path)
    finite = np.isfinite(dem)
    if not np.any(finite):
        raise RuntimeError(f"No finite elevation values in {dem_path}")

    vmin = float(np.nanpercentile(dem, clip_lower))
    vmax = float(np.nanpercentile(dem, clip_upper))
    hill = compute_hillshade(dem)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(dem, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("Elevation (m)")
    axes[0].set_xlabel("X pixels")
    axes[0].set_ylabel("Y pixels")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(hill, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Hillshade")
    axes[1].set_xlabel("X pixels")
    axes[1].set_ylabel("Y pixels")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    z_min = float(np.nanmin(dem))
    z_max = float(np.nanmax(dem))
    fig.suptitle(f"{dem_path.name} | min={z_min:.1f} m, max={z_max:.1f} m")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dem_path.stem}_preview.png"
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()

    dem_files = args.dem_files
    if not dem_files:
        dem_files = sorted(Path("data/dem").glob("*.tif"))

    if not dem_files:
        print("No DEM files found. Provide files explicitly or place .tif files in data/dem/.", file=sys.stderr)
        raise SystemExit(2)

    for dem_path in dem_files:
        if not dem_path.exists():
            print(f"Skipping missing file: {dem_path}", file=sys.stderr)
            continue

        out_path = render_preview(
            dem_path=dem_path,
            output_dir=args.output_dir,
            clip_lower=args.clip_lower,
            clip_upper=args.clip_upper,
            show=args.show,
        )
        print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()
