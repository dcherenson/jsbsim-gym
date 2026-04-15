#!/usr/bin/env python3
"""Download clipped USGS 3DEP DEMs from OpenTopography.

Examples:
  OT_API_KEY=... uv run python download_canyon_dem.py --preset grand-canyon
  OT_API_KEY=... uv run python download_canyon_dem.py --preset black-canyon-gunnison
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


API_ENDPOINT = "https://portal.opentopography.org/API/usgsdem"

PRESETS = {
    "grand-canyon": {
        "dataset": "USGS10m",
        "south": 36.20,
        "north": 36.35,
        "west": -112.25,
        "east": -111.95,
    },
    "black-canyon-gunnison": {
        "dataset": "USGS10m",
        "south": 38.52,
        "north": 38.62,
        "west": -107.78,
        "east": -107.65,
    },
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def approx_bbox_area_km2(south: float, north: float, west: float, east: float) -> float:
    mean_lat_rad = math.radians(0.5 * (south + north))
    lat_km = abs(north - south) * 111.32
    lon_km = abs(east - west) * 111.32 * math.cos(mean_lat_rad)
    return lat_km * lon_km


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Named canyon bbox preset")
    parser.add_argument("--dataset", choices=["USGS1m", "USGS10m", "USGS30m"], default=None)
    parser.add_argument("--south", type=float, default=None)
    parser.add_argument("--north", type=float, default=None)
    parser.add_argument("--west", type=float, default=None)
    parser.add_argument("--east", type=float, default=None)
    parser.add_argument("--output-format", choices=["GTiff", "AAIGrid", "HFA"], default="GTiff")
    parser.add_argument("--output", type=Path, default=None, help="Output file path")
    parser.add_argument("--api-key", default=os.getenv("OT_API_KEY"), help="OpenTopography API key")
    parser.add_argument("--print-url", action="store_true", help="Print request URL")
    parser.add_argument("--dry-run", action="store_true", help="Validate args and print request only")
    return parser.parse_args()


def merge_with_preset(args: argparse.Namespace) -> dict[str, float | str]:
    merged: dict[str, float | str] = {}
    if args.preset:
        merged.update(PRESETS[args.preset])

    for key in ("south", "north", "west", "east"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value

    if args.dataset is not None:
        merged["dataset"] = args.dataset

    for key in ("dataset", "south", "north", "west", "east"):
        if key not in merged:
            fail(f"Missing required parameter '{key}'. Use --preset or pass explicit bounds.")

    return merged


def default_output_path(args: argparse.Namespace, params: dict[str, float | str]) -> Path:
    ext_by_format = {"GTiff": ".tif", "AAIGrid": ".asc", "HFA": ".img"}
    ext = ext_by_format[args.output_format]

    if args.output is not None:
        return args.output

    label = args.preset or "custom"
    dataset = str(params["dataset"])
    return Path("data") / "dem" / f"{label}_{dataset}{ext}"


def fetch_dem(url: str, output_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "jsbsim-gym-dem-downloader/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get_content_type()

            if content_type == "application/json":
                payload = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(payload)
                    fail(f"API returned JSON instead of DEM: {parsed}")
                except json.JSONDecodeError:
                    fail(f"API returned non-binary response: {payload[:400]}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            byte_count = 0
            with output_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    out_f.write(chunk)

            if byte_count == 0:
                fail("Download succeeded but response was empty.")

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        fail(f"HTTP {exc.code} from OpenTopography: {body[:800]}")
    except urllib.error.URLError as exc:
        fail(f"Network error while contacting OpenTopography: {exc}")


def main() -> None:
    args = parse_args()
    params = merge_with_preset(args)

    if not args.api_key:
        fail("No API key provided. Pass --api-key or set OT_API_KEY.")

    south = float(params["south"])
    north = float(params["north"])
    west = float(params["west"])
    east = float(params["east"])

    if south >= north:
        fail("south must be less than north")
    if west >= east:
        fail("west must be less than east")

    area_km2 = approx_bbox_area_km2(south, north, west, east)
    dataset = str(params["dataset"])
    if dataset == "USGS1m" and area_km2 > 250.0:
        print(
            "WARNING: USGS1m requests are limited to 250 km^2 by OpenTopography. "
            f"Current bbox is ~{area_km2:.1f} km^2.",
            file=sys.stderr,
        )

    query_params = {
        "datasetName": dataset,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": args.output_format,
        "API_Key": args.api_key,
    }

    url = f"{API_ENDPOINT}?{urllib.parse.urlencode(query_params)}"
    output_path = default_output_path(args, params)

    print(f"Dataset: {dataset}")
    print(f"BBox: south={south}, north={north}, west={west}, east={east}")
    print(f"Approx area: {area_km2:.2f} km^2")
    print(f"Output: {output_path}")

    if args.print_url or args.dry_run:
        print(f"Request URL: {url}")

    if args.dry_run:
        return

    fetch_dem(url, output_path)
    print("Download complete.")


if __name__ == "__main__":
    main()
