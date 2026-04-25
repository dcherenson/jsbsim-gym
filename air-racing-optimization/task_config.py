from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os
from typing import Literal


TerrainSource = Literal["dem_clip", "legacy_usgs_tiles"]

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TaskConfig:
    name: str
    terrain_source: TerrainSource
    dem_path: Path | None = None
    dem_bbox: tuple[float, float, float, float] | None = None
    fly_direction: str = "south_to_north"
    start_pixel: tuple[int, int] | None = None
    finish_pixel: tuple[int, int] | None = None
    span_ft: float | None = None
    crop_padding_m: float = 3000.0
    valley_rel_elev: float = 0.08
    smoothing_window: int = 11
    min_width_ft: float = 140.0
    max_width_ft: float = 2200.0
    terrain_cost_sigma_m: float = 10000.0
    terrain_cost_weight: float = 5.0
    initial_altitude_agl_ft: float = 220.0
    legacy_start_latlon: tuple[float, float] | None = None
    legacy_end_latlon: tuple[float, float] | None = None
    initial_heading_deg: float | None = None

    @property
    def cache_tag(self) -> str:
        return self.name.replace("-", "_").replace(" ", "_")


BLACK_CANYON_DEM = TaskConfig(
    name="black_canyon_dem",
    terrain_source="dem_clip",
    dem_path=REPO_ROOT / "data/dem/black-canyon-gunnison_USGS10m.tif",
    dem_bbox=(38.52, 38.62, -107.78, -107.65),
    start_pixel=(1400, 950),
    span_ft=9000.0,
    crop_padding_m=3000.0,
    valley_rel_elev=0.08,
    smoothing_window=11,
    min_width_ft=140.0,
    max_width_ft=2200.0,
    terrain_cost_sigma_m=10000.0,
    terrain_cost_weight=5.0,
    initial_altitude_agl_ft=220.0,
)

LEGACY_RIFFE = TaskConfig(
    name="legacy_riffe",
    terrain_source="legacy_usgs_tiles",
    crop_padding_m=3000.0,
    terrain_cost_sigma_m=10000.0,
    terrain_cost_weight=5.0,
    initial_altitude_agl_ft=1201.11,
    legacy_start_latlon=(
        46 + 32 / 60 + 53.84 / 3600,
        -(122 + 28 / 60 + 8.98 / 3600),
    ),
    legacy_end_latlon=(
        46 + 16 / 60 + 37.56 / 3600,
        -(121 + 34 / 60 + 38.70 / 3600),
    ),
    initial_heading_deg=115.68559831535447,
)

TASK_PRESETS: dict[str, TaskConfig] = {
    "black_canyon_dem": BLACK_CANYON_DEM,
    "black-canyon-dem": BLACK_CANYON_DEM,
    "black_canyon": BLACK_CANYON_DEM,
    "legacy_riffe": LEGACY_RIFFE,
    "legacy-riffe": LEGACY_RIFFE,
    "riffe": LEGACY_RIFFE,
}


def _parse_csv(raw: str, expected_len: int, cast, name: str) -> tuple:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{name} must have {expected_len} comma-separated values, got: {raw!r}")
    return tuple(cast(part) for part in parts)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def get_active_task_config() -> TaskConfig:
    requested_name = os.getenv("AIR_RACING_TASK", BLACK_CANYON_DEM.name).strip().lower()
    base = TASK_PRESETS.get(requested_name, BLACK_CANYON_DEM)

    dem_path = os.getenv("AIR_RACING_DEM_PATH")
    dem_bbox = os.getenv("AIR_RACING_DEM_BBOX")
    start_pixel = os.getenv("AIR_RACING_DEM_START_PIXEL")
    finish_pixel = os.getenv("AIR_RACING_DEM_FINISH_PIXEL")
    span_ft = os.getenv("AIR_RACING_DEM_SPAN_FT")
    crop_padding_m = os.getenv("AIR_RACING_CROP_PADDING_M")
    fly_direction = os.getenv("AIR_RACING_FLY_DIRECTION")

    overrides = {}
    if dem_path:
        overrides["dem_path"] = _resolve_path(dem_path)
    if dem_bbox:
        overrides["dem_bbox"] = _parse_csv(dem_bbox, 4, float, "AIR_RACING_DEM_BBOX")
        overrides["terrain_source"] = "dem_clip"
    if start_pixel:
        overrides["start_pixel"] = _parse_csv(start_pixel, 2, int, "AIR_RACING_DEM_START_PIXEL")
        overrides["terrain_source"] = "dem_clip"
    if finish_pixel:
        overrides["finish_pixel"] = _parse_csv(finish_pixel, 2, int, "AIR_RACING_DEM_FINISH_PIXEL")
        overrides["terrain_source"] = "dem_clip"
    if span_ft:
        overrides["span_ft"] = float(span_ft)
        overrides["terrain_source"] = "dem_clip"
    if crop_padding_m:
        overrides["crop_padding_m"] = float(crop_padding_m)
    if fly_direction:
        overrides["fly_direction"] = fly_direction.strip()

    return replace(base, **overrides) if overrides else base
