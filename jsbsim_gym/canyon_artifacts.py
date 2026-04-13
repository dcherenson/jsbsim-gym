from pathlib import Path

import imageio.v2 as iio
import imageio.v3 as iio_v3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


FT_TO_M = 1.0 / 3.28084
RENDER_SCALE = 1e-3


def latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, rows, cols):
    x = ((lon_deg - west) / (east - west)) * float(cols) - 0.5
    y = ((north - lat_deg) / (north - south)) * float(rows) - 0.5
    return x, y


def capture_frame(env):
    frame = env.render()
    if isinstance(frame, np.ndarray):
        return frame

    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is not None and hasattr(viewer, "get_frame"):
        try:
            return viewer.get_frame()
        except Exception:
            return None
    return None


def _centerline_pixel_coords(canyon, dem_bbox, dem_rows, dem_cols,
                              north_samples_ft=None, center_east_samples_ft=None):
    """Convert centerline local-ft arrays to DEM pixel coordinates.

    Returns arrays (cx, cy) in pixel space suitable for matplotlib plotting.
    """
    if north_samples_ft is None or center_east_samples_ft is None:
        return None, None
    if not hasattr(canyon, "north_samples_ft") or not hasattr(canyon, "east_samples_ft"):
        return None, None

    n_arr = np.asarray(north_samples_ft, dtype=np.float64)
    e_arr = np.asarray(center_east_samples_ft, dtype=np.float64)
    if n_arr.size < 2 or e_arr.size != n_arr.size:
        return None, None

    # Map local north (ft) → ordered-row index
    canyon_north = np.asarray(canyon.north_samples_ft, dtype=np.float64)
    canyon_east_axis = np.asarray(canyon.east_samples_ft, dtype=np.float64)

    # Ordered row index (float) for each sample in the profile
    row_ordered = np.interp(n_arr, canyon_north, np.arange(len(canyon_north), dtype=np.float64))
    # East-ft → column index (float)
    col_float = np.interp(e_arr, canyon_east_axis, np.arange(len(canyon_east_axis), dtype=np.float64))

    # Convert ordered-row back to original-image row
    if getattr(canyon, "fly_direction", "south_to_north") == "south_to_north":
        row_original = float(canyon.rows - 1) - row_ordered
    else:
        row_original = row_ordered

    return col_float, row_original


def save_canyon_overlay_plot(
    dem_path,
    dem_bbox,
    dem_start_pixel,
    track_x,
    track_y,
    termination_reason,
    output_path,
    title_prefix,
    centerline_x=None,
    centerline_y=None,
):
    dem = iio_v3.imread(Path(dem_path)).astype(np.float32)
    if dem.ndim == 3:
        dem = dem[..., 0]
    dem[(~np.isfinite(dem)) | (dem < -1e20)] = np.nan

    rows, cols = dem.shape
    vmin = float(np.nanpercentile(dem, 2.0))
    vmax = float(np.nanpercentile(dem, 98.0))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(dem, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    # Draw centerline if available
    if centerline_x is not None and centerline_y is not None:
        cl_x = np.asarray(centerline_x)
        cl_y = np.asarray(centerline_y)
        valid = np.isfinite(cl_x) & np.isfinite(cl_y)
        ax.plot(cl_x[valid], cl_y[valid], color="cyan", linewidth=1.5,
                linestyle="--", alpha=0.85, label="MPPI centerline")

    ax.plot(track_x, track_y, color="red", linewidth=2.0, label="Aircraft trajectory")
    ax.scatter([track_x[0]], [track_y[0]], c="lime", s=70, edgecolors="black", linewidths=0.5, label="Trajectory start")
    ax.scatter([track_x[-1]], [track_y[-1]], c="red", marker="x", s=80, label="Trajectory end")
    ax.scatter(
        [dem_start_pixel[0]],
        [dem_start_pixel[1]],
        c="cyan",
        marker="+",
        s=120,
        linewidths=2.0,
        label="Configured start pixel",
    )

    ax.set_xlim(0, cols - 1)
    ax.set_ylim(rows - 1, 0)
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    ax.set_title(f"{title_prefix} | steps={len(track_x)-1}, end={termination_reason}")
    ax.legend(loc="lower left")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_trajectory_csv(output_path, track_x, track_y, track_lat, track_lon):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_arr = np.column_stack(
        [
            np.arange(len(track_x), dtype=np.int32),
            np.asarray(track_x, dtype=np.float64),
            np.asarray(track_y, dtype=np.float64),
            np.asarray(track_lat, dtype=np.float64),
            np.asarray(track_lon, dtype=np.float64),
        ]
    )
    np.savetxt(
        output_path,
        track_arr,
        delimiter=",",
        header="step,pixel_x,pixel_y,lat_deg,lon_deg",
        comments="",
    )


class CanyonRunRecorder:
    def __init__(
        self,
        env,
        dem_path,
        dem_bbox,
        dem_start_pixel,
        output_dir,
        file_stem,
        title_prefix,
        fps=30,
    ):
        self.env = env
        self.dem_path = Path(dem_path)
        self.dem_bbox = tuple(dem_bbox)
        self.dem_start_pixel = tuple(dem_start_pixel)
        self.title_prefix = str(title_prefix)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = output_dir / f"{file_stem}.mp4"
        self.overlay_path = output_dir / f"{file_stem}_trajectory_overlay.png"
        self.trajectory_csv_path = output_dir / f"{file_stem}_trajectory.csv"

        self._writer = iio.get_writer(self.video_path, format="ffmpeg", fps=int(fps))
        self._closed = False

        self._sim = self.env.unwrapped.simulation
        self._rows = int(self.env.unwrapped.canyon.rows)
        self._cols = int(self.env.unwrapped.canyon.cols)

        self.track_x = []
        self.track_y = []
        self.track_lat = []
        self.track_lon = []

        # Centerline profile (set via set_centerline_profile)
        self._centerline_north_ft = None
        self._centerline_east_ft = None
        self._centerline_overlay_x = None
        self._centerline_overlay_y = None

    def set_centerline_profile(self, north_samples_ft, center_east_samples_ft):
        """Provide the MPPI canyon centerline so it can be drawn in video frames
        and the trajectory overlay."""
        self._centerline_north_ft = np.asarray(north_samples_ft, dtype=np.float32).copy()
        self._centerline_east_ft = np.asarray(center_east_samples_ft, dtype=np.float32).copy()

        # Precompute pixel coords for the 2D overlay plot
        canyon = getattr(self.env.unwrapped, "canyon", None)
        if canyon is not None:
            cx, cy = _centerline_pixel_coords(
                canyon, self.dem_bbox, self._rows, self._cols,
                self._centerline_north_ft, self._centerline_east_ft,
            )
            self._centerline_overlay_x = cx
            self._centerline_overlay_y = cy

    @staticmethod
    def _project_world_points(world_points, view, projection, width, height):
        if world_points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        homogeneous = np.concatenate(
            [
                np.asarray(world_points, dtype=np.float32),
                np.ones((world_points.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        clip = (projection @ view @ homogeneous.T).T
        w = np.maximum(clip[:, 3:4], 1e-6)
        ndc = clip[:, :3] / w

        pixels = np.empty((world_points.shape[0], 2), dtype=np.float32)
        pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * float(width - 1)
        pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height - 1)

        valid = np.logical_and.reduce(
            [
                clip[:, 3] > 1e-6,
                ndc[:, 2] > -1.5,
                ndc[:, 2] < 1.5,
            ]
        )
        pixels[~valid] = np.nan
        return pixels

    def _trajectory_world_points(self, xy_ft, h_ft):
        env_unwrapped = self.env.unwrapped
        xy_ft = np.asarray(xy_ft, dtype=np.float32)
        h_ft = np.asarray(h_ft, dtype=np.float32)
        if xy_ft.ndim != 2 or xy_ft.shape[1] != 2:
            return np.zeros((0, 3), dtype=np.float32)

        if h_ft.ndim != 1 or h_ft.shape[0] != xy_ft.shape[0]:
            default_h_ft = float(self._sim.get_property_value("position/h-sl-ft"))
            h_ft = np.full((xy_ft.shape[0],), default_h_ft, dtype=np.float32)

        base_elev_ft = float(getattr(env_unwrapped, "dem_render_base_elev_ft", 0.0))
        start_elev_ft = float(getattr(env_unwrapped, "dem_start_elev_ft", 0.0))

        north_ft = xy_ft[:, 0]
        east_ft = xy_ft[:, 1]
        h_msl_ft = h_ft + start_elev_ft

        world = np.empty((xy_ft.shape[0], 3), dtype=np.float32)
        world[:, 0] = -east_ft * FT_TO_M * RENDER_SCALE
        world[:, 1] = (h_msl_ft - base_elev_ft) * FT_TO_M * RENDER_SCALE
        world[:, 2] = north_ft * FT_TO_M * RENDER_SCALE
        return world

    def _overlay_planner_debug(self, frame, planner_debug):
        if planner_debug is None:
            return frame

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            return frame

        candidate_xy = np.asarray(
            planner_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        final_xy = np.asarray(
            planner_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        if candidate_xy.size == 0 and final_xy.size == 0:
            return frame

        candidate_h = np.asarray(
            planner_debug.get("candidate_h_ft", np.zeros((0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        final_h = np.asarray(
            planner_debug.get("final_h_ft", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )

        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        for idx, traj_xy in enumerate(candidate_xy):
            traj_h = candidate_h[idx] if idx < len(candidate_h) else np.zeros((traj_xy.shape[0],), dtype=np.float32)
            world_points = self._trajectory_world_points(traj_xy, traj_h)
            pixels = self._project_world_points(world_points, view, projection, width, height)
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            pixels = pixels[2:]
            if len(pixels) >= 2:
                draw.line([tuple(point) for point in pixels], fill=(84, 180, 255, 52), width=1)

        if len(final_xy) >= 2:
            world_points = self._trajectory_world_points(final_xy, final_h)
            pixels = self._project_world_points(world_points, view, projection, width, height)
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            pixels = pixels[2:]
            if len(pixels) >= 2:
                draw.line([tuple(point) for point in pixels], fill=(255, 184, 44, 255), width=4)
                draw.line([tuple(point) for point in pixels], fill=(255, 236, 180, 176), width=2)

        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _overlay_centerline(self, frame):
        """Draw the MPPI centerline as a cyan ribbon in the 3D video frame."""
        if self._centerline_north_ft is None or self._centerline_east_ft is None:
            return frame

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            return frame

        env_unwrapped = self.env.unwrapped
        start_elev_ft = float(getattr(env_unwrapped, "dem_start_elev_ft", 0.0))

        # Get current aircraft p_N to only draw nearby centerline
        cur_p_N = float(self._sim.get_property_value("position/lat-gc-rad"))
        # Use local north via canyon for a better reference
        canyon = getattr(env_unwrapped, "canyon", None)
        cur_local_north = None
        if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
            try:
                lat_deg = float(self._sim.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self._sim.get_property_value("position/long-gc-deg"))
                cur_local_north, _ = canyon.get_local_from_latlon(lat_deg, lon_deg)
            except Exception:
                pass

        n_arr = self._centerline_north_ft
        e_arr = self._centerline_east_ft

        # Subsample for performance — draw every Nth point
        stride = max(1, len(n_arr) // 600)
        n_sub = n_arr[::stride]
        e_sub = e_arr[::stride]

        # Only draw within a window around the aircraft
        if cur_local_north is not None:
            window_ft = 3000.0
            mask = np.abs(n_sub - cur_local_north) < window_ft
            n_sub = n_sub[mask]
            e_sub = e_sub[mask]

        if len(n_sub) < 2:
            return frame

        # Altitude: place centerline at the DEM surface elevation (start_elev_ft relative)
        # Use a constant altitude reference near ground for visibility
        h_ref = np.full_like(n_sub, 0.0)  # h=0 means DEM start elevation
        xy = np.column_stack([n_sub, e_sub])
        world_points = self._trajectory_world_points(xy, h_ref)
        
        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)
        pixels = self._project_world_points(world_points, view, projection, width, height)

        valid = np.all(np.isfinite(pixels), axis=1)
        pixels_valid = pixels[valid]
        if len(pixels_valid) < 2:
            return frame

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        # Draw a bright cyan centerline
        draw.line([tuple(p) for p in pixels_valid], fill=(0, 255, 255, 200), width=3)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _sample_position(self):
        south, north, west, east = self.dem_bbox
        lat_deg = float(self._sim.get_property_value("position/lat-gc-deg"))
        lon_deg = float(self._sim.get_property_value("position/long-gc-deg"))
        px, py = latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, self._rows, self._cols)
        self.track_x.append(px)
        self.track_y.append(py)
        self.track_lat.append(lat_deg)
        self.track_lon.append(lon_deg)

    def initialize(self):
        frame = capture_frame(self.env)
        if frame is not None:
            self._writer.append_data(frame)
        self._sample_position()

    def record_step(self, planner_debug=None):
        frame = capture_frame(self.env)
        if frame is not None:
            frame = self._overlay_centerline(frame)
            frame = self._overlay_planner_debug(frame, planner_debug)
            self._writer.append_data(frame)
        self._sample_position()

    def close_writer(self):
        if not self._closed:
            self._writer.close()
            self._closed = True

    def finalize(self, termination_reason):
        self.close_writer()

        if len(self.track_x) >= 2:
            save_canyon_overlay_plot(
                dem_path=self.dem_path,
                dem_bbox=self.dem_bbox,
                dem_start_pixel=self.dem_start_pixel,
                track_x=self.track_x,
                track_y=self.track_y,
                termination_reason=termination_reason,
                output_path=self.overlay_path,
                title_prefix=self.title_prefix,
                centerline_x=self._centerline_overlay_x,
                centerline_y=self._centerline_overlay_y,
            )
            save_trajectory_csv(
                output_path=self.trajectory_csv_path,
                track_x=self.track_x,
                track_y=self.track_y,
                track_lat=self.track_lat,
                track_lon=self.track_lon,
            )

        return {
            "video_path": self.video_path,
            "overlay_path": self.overlay_path,
            "trajectory_csv_path": self.trajectory_csv_path,
        }
