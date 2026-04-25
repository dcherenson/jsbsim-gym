from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from task_config import get_active_task_config


METERS_PER_DEG_LAT = 1e6 / 9

terrain_folder = Path(__file__).parent / "raw_data"


def assert_close_to_integer_and_round(x, atol=1e-3):
    rounded_x = np.round(x)
    assert np.allclose(x, rounded_x, atol=atol)
    return int(rounded_x)


def _clean_dem_array(dem: np.ndarray) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim == 3:
        dem = dem[..., 0]
    if dem.ndim != 2 or dem.size == 0:
        raise ValueError(f"Expected a 2D DEM array, got shape {dem.shape!r}")

    dem[(~np.isfinite(dem)) | (dem < -1e20)] = np.nan
    finite = np.isfinite(dem)
    if not np.any(finite):
        raise ValueError("DEM contains no finite elevation values.")

    fill_value = float(np.nanmedian(dem[finite]))
    return np.where(finite, dem, fill_value).astype(np.float32)


def _load_dem_clip(task_config):
    if task_config.dem_path is None or task_config.dem_bbox is None:
        raise ValueError("DEM-backed tasks require both dem_path and dem_bbox.")
    if not task_config.dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {task_config.dem_path}")

    south, north, west, east = task_config.dem_bbox
    if not (south < north and west < east):
        raise ValueError("Invalid DEM bbox; expected (south, north, west, east) with south<north and west<east.")

    dem_original = _clean_dem_array(iio.imread(task_config.dem_path))
    rows, cols = dem_original.shape

    # GeoTIFF rasters are typically stored north-to-south in row order.
    # Flip vertically so increasing row index corresponds to increasing northing.
    elev = np.flipud(dem_original)

    lat_step = (north - south) / float(rows)
    lon_step = (east - west) / float(cols)
    lat_edges = south + (np.arange(rows, dtype=np.float64) + 0.5) * lat_step
    lon_edges = west + (np.arange(cols, dtype=np.float64) + 0.5) * lon_step

    return {
        "elev": elev,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "cache_tag": task_config.cache_tag,
        "terrain_source": "dem_clip",
        "dem_path": str(task_config.dem_path),
        "dem_bbox": tuple(task_config.dem_bbox),
    }


def _load_legacy_usgs_tiles():
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Legacy USGS tile loading requires rasterio. "
            "Either install rasterio or use the DEM-backed canyon configuration."
        ) from exc

    terrain_files = sorted(terrain_folder.glob("*.tif"))
    if not terrain_files:
        raise FileNotFoundError(
            f"No legacy terrain tiles found in {terrain_folder}. "
            "Use the DEM-backed canyon config or add raw USGS tiles there."
        )

    tile_data = []

    for terrain_file in terrain_files:
        with rasterio.open(terrain_file) as dataset:
            elev = dataset.read(1)
            transform = dataset.transform
            dataset_width = dataset.width
            dataset_height = dataset.height

        lat_edges = np.linspace(
            transform[5],
            transform[5] + dataset_width * transform[4],
            elev.shape[0],
        )
        lon_edges = np.linspace(
            transform[2],
            transform[2] + dataset_height * transform[0],
            elev.shape[1],
        )

        lat_argsort = np.argsort(lat_edges)
        lon_argsort = np.argsort(lon_edges)

        lat_edges = lat_edges[lat_argsort]
        lon_edges = lon_edges[lon_argsort]
        elev = elev[lat_argsort, :][:, lon_argsort]

        if elev.shape == (3612, 3612):
            lat_edges = np.linspace(
                assert_close_to_integer_and_round(transform[5] + 6 * transform[4]),
                assert_close_to_integer_and_round(transform[5] + 3606 * transform[4]),
                3600,
            )
            lon_edges = np.linspace(
                assert_close_to_integer_and_round(transform[2] + 6 * transform[0]),
                assert_close_to_integer_and_round(transform[2] + 3606 * transform[0]),
                3600,
            )
            elev = elev[6:3606, 6:3606]
        else:
            raise ValueError(
                "Legacy raw tile loading currently expects 1-arcsecond USGS 1deg x 1deg grids "
                "with 6 extra datapoints on each axis."
            )

        tile_data.append(
            {
                "elev": elev,
                "lat_edges": lat_edges,
                "lon_edges": lon_edges,
                "min_lat": assert_close_to_integer_and_round(lat_edges.min()),
                "min_lon": assert_close_to_integer_and_round(lon_edges.min()),
            }
        )

    unique_lats = np.unique([tile["min_lat"] for tile in tile_data])
    unique_lons = np.unique([tile["min_lon"] for tile in tile_data])

    tile_array = np.empty((len(unique_lats), len(unique_lons)), dtype="O")

    for tile in tile_data:
        tile_array[
            np.where(unique_lats == tile["min_lat"])[0][0],
            np.where(unique_lons == tile["min_lon"])[0][0],
        ] = tile

    elev = np.empty((3599 * len(unique_lats) + 1, 3599 * len(unique_lons) + 1))
    elev[:] = np.nan

    for i in range(tile_array.shape[0]):
        for j in range(tile_array.shape[1]):
            tile = tile_array[i, j]
            elev[
                i * 3599 : i * 3599 + 3600,
                j * 3599 : j * 3599 + 3600,
            ] = tile["elev"]

    lat_edges = np.unique([tile["lat_edges"] for tile in sorted(tile_data, key=lambda tile: tile["min_lat"])])
    lon_edges = np.unique([tile["lon_edges"] for tile in sorted(tile_data, key=lambda tile: tile["min_lon"])])

    if not (
        (len(lat_edges) == elev.shape[0])
        and (len(lon_edges) == elev.shape[1])
        and (np.sort(lat_edges) == lat_edges).all()
        and (np.sort(lon_edges) == lon_edges).all()
    ):
        raise ValueError(
            "Something went wrong while merging the legacy raw tiles; the merged grid is inconsistent."
        )

    return {
        "elev": _clean_dem_array(elev),
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "cache_tag": "legacy_riffe",
        "terrain_source": "legacy_usgs_tiles",
    }


task_config = get_active_task_config()
if task_config.terrain_source == "dem_clip":
    terrain_data = _load_dem_clip(task_config)
else:
    terrain_data = _load_legacy_usgs_tiles()

datum_lat = float(np.mean(terrain_data["lat_edges"]))
datum_lon = float(np.mean(terrain_data["lon_edges"]))
cos_lat = float(np.cos(np.deg2rad(datum_lat)))

terrain_data["north_edges"] = (terrain_data["lat_edges"] - datum_lat) * METERS_PER_DEG_LAT
terrain_data["east_edges"] = (terrain_data["lon_edges"] - datum_lon) * METERS_PER_DEG_LAT * cos_lat


def lat_lon_to_north_east(lat, lon):
    return (
        (np.asarray(lat) - datum_lat) * METERS_PER_DEG_LAT,
        (np.asarray(lon) - datum_lon) * METERS_PER_DEG_LAT * cos_lat,
    )


def north_east_to_lat_lon(north, east):
    return (
        np.asarray(north) / METERS_PER_DEG_LAT + datum_lat,
        np.asarray(east) / METERS_PER_DEG_LAT / cos_lat + datum_lon,
    )


def lat_lon_to_normalized_coordinates(lat, lon):
    return (
        (np.asarray(lat) - terrain_data["lat_edges"].min())
        / (terrain_data["lat_edges"].max() - terrain_data["lat_edges"].min()),
        (np.asarray(lon) - terrain_data["lon_edges"].min())
        / (terrain_data["lon_edges"].max() - terrain_data["lon_edges"].min()),
    )


def north_east_to_normalized_coordinates(north, east):
    return (
        (np.asarray(north) - terrain_data["north_edges"].min())
        / (terrain_data["north_edges"].max() - terrain_data["north_edges"].min()),
        (np.asarray(east) - terrain_data["east_edges"].min())
        / (terrain_data["east_edges"].max() - terrain_data["east_edges"].min()),
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(
        terrain_data["elev"],
        cmap="terrain",
        origin="lower",
        extent=[
            terrain_data["east_edges"].min(),
            terrain_data["east_edges"].max(),
            terrain_data["north_edges"].min(),
            terrain_data["north_edges"].max(),
        ],
    )
    p.equal()
    plt.colorbar(label="Elevation [m]")
    plt.xlabel("Position East of Datum [m]")
    plt.ylabel("Position North of Datum [m]")
    plt.title(f"Terrain Overview ({terrain_data['cache_tag']})")
    p.show_plot(rotate_axis_labels=False)
