
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from terrain_model.load_raw_data import terrain_data, north_east_to_normalized_coordinates, lat_lon_to_north_east
from scipy import fft, ndimage, interpolate
import numpy as np
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import interpolate
from typing import Tuple
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pickle
import casadi as cas
from time import perf_counter

_default_resolution = (100, 300)


def _cache_name(resolution: Tuple[float, float], terrain_data) -> str:
    cache_tag = str(terrain_data.get("cache_tag", "terrain")).replace("/", "_")
    return f"{cache_tag}_{resolution[0]}_{resolution[1]}"


def precompute_and_save_model(
        resolution: Tuple[float, float] = _default_resolution,
        terrain_data=terrain_data
):

    name = _cache_name(resolution, terrain_data)
    file = Path(__file__).parent / "precomputed_models" / f"{name}.pkl"
    if not file.exists():
        start_time = perf_counter()
        print(f"Pre-computing interpolant: '{name}'...", flush=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        north_zoom_factor = resolution[0] / terrain_data["elev"].shape[0]
        east_zoom_factor = resolution[1] / terrain_data["elev"].shape[1]

        print(
            f"  Resampling terrain from {terrain_data['elev'].shape} to {resolution}...",
            flush=True,
        )
        resample_start = perf_counter()
        elev_resampled = ndimage.zoom(
            terrain_data["elev"],
            zoom=(north_zoom_factor, east_zoom_factor),
            order=3,
        )
        print(
            f"  Terrain resample complete in {perf_counter() - resample_start:.1f} s.",
            flush=True,
        )

        north_edges = np.linspace(
            terrain_data["north_edges"].min(),
            terrain_data["north_edges"].max(),
            elev_resampled.shape[0]
        )

        east_edges = np.linspace(
            terrain_data["east_edges"].min(),
            terrain_data["east_edges"].max(),
            elev_resampled.shape[1]
        )
        print("  Building CasADi bspline interpolant...", flush=True)
        interpolant_start = perf_counter()
        interpolant = cas.interpolant(
            "interpolant",
            "bspline",
            [north_edges, east_edges],
            elev_resampled.ravel(order="F"),
        )
        print(
            f"  Interpolant build complete in {perf_counter() - interpolant_start:.1f} s.",
            flush=True,
        )

        with open(file, "wb+") as f:
            pickle.dump(
                interpolant,
                f
            )
        print(
            f"Saved interpolant cache to '{file.name}' in {perf_counter() - start_time:.1f} s.",
            flush=True,
        )


def get_elevation_interpolated_north_east(
        query_points_north: np.ndarray,
        query_points_east: np.ndarray,
        resolution=_default_resolution,
        terrain_data=terrain_data,
):
    query_points_north = np.reshape(np.array(query_points_north), -1)
    query_points_east = np.reshape(np.array(query_points_east), -1)

    name = _cache_name(resolution, terrain_data)
    file = Path(__file__).parent / "precomputed_models" / f"{name}.pkl"
    if not file.exists():
        precompute_and_save_model(resolution=resolution, terrain_data=terrain_data)
    else:
        print(f"Using cached interpolant: '{name}'.", flush=True)

    with open(file, "rb") as f:
        interpolant = pickle.load(f)

    return interpolant(np.stack([
        query_points_north,
        query_points_east
    ], axis=1).T).T


def get_elevation_interpolated_lat_lon(
        query_points_lat: np.ndarray,
        query_points_lon: np.ndarray,
        resolution=_default_resolution,
        terrain_data=terrain_data,
):
    query_points_lat = np.reshape(np.array(query_points_lat), -1)
    query_points_lon = np.reshape(np.array(query_points_lon), -1)

    query_points_north, query_points_east = lat_lon_to_north_east(
        query_points_lat,
        query_points_lon,
    )

    return get_elevation_interpolated_north_east(
        query_points_north,
        query_points_east,
        resolution=resolution,
        terrain_data=terrain_data,
    )


if __name__ == '__main__':
    from aerosandbox.tools.code_benchmarking import Timer

    for res in [
        (100, 300),
        (150, 450),
        (200, 600),
        (300, 900),
        (600, 1800),
        (1200, 3600),
        (1800, 4800),
        (2400, 7200),
        (3600, 10800),
    ]:
        with Timer(str(res)):
            precompute_and_save_model(resolution=res)
