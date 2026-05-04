from pathlib import Path

import imageio.v2 as iio
import imageio.v3 as iio_v3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    reference_x=None,
    reference_y=None,
    reference_label="Reference trajectory",
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

    # Draw reference trajectory if available
    if reference_x is not None and reference_y is not None:
        cl_x = np.asarray(reference_x)
        cl_y = np.asarray(reference_y)
        valid = np.isfinite(cl_x) & np.isfinite(cl_y)
        ax.plot(cl_x[valid], cl_y[valid], color="cyan", linewidth=1.5,
                linestyle="--", alpha=0.85, label=str(reference_label))

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
        self._capture_enabled = True
        self._capture_failure_reason = None

        self._sim = self.env.unwrapped.simulation
        self._rows = int(self.env.unwrapped.canyon.rows)
        self._cols = int(self.env.unwrapped.canyon.cols)

        self.track_x = []
        self.track_y = []
        self.track_lat = []
        self.track_lon = []

        # Reference profile shown in overlays/video.
        self._reference_north_ft = None
        self._reference_east_ft = None
        self._reference_altitude_ft = None
        self._reference_overlay_x = None
        self._reference_overlay_y = None
        self._reference_label = "Reference trajectory"

    def _capture_frame_or_disable(self):
        if not self._capture_enabled:
            return None

        try:
            frame = capture_frame(self.env)
        except Exception as exc:
            self._capture_enabled = False
            self._capture_failure_reason = f"{type(exc).__name__}: {exc}"
            print(
                "Recorder warning: frame capture disabled for this run "
                f"because rendering failed ({self._capture_failure_reason})."
            )
            return None

        return frame

    def set_reference_profile(
        self,
        north_samples_ft,
        east_samples_ft,
        altitude_samples_ft=None,
        label="Reference trajectory",
    ):
        """Provide a reference trajectory to draw in video frames and overlays."""
        self._reference_north_ft = np.asarray(north_samples_ft, dtype=np.float32).copy()
        self._reference_east_ft = np.asarray(east_samples_ft, dtype=np.float32).copy()
        if altitude_samples_ft is None:
            self._reference_altitude_ft = None
        else:
            self._reference_altitude_ft = np.asarray(altitude_samples_ft, dtype=np.float32).copy()
        self._reference_label = str(label)

        # Precompute pixel coords for the 2D overlay plot
        canyon = getattr(self.env.unwrapped, "canyon", None)
        if canyon is not None:
            cx, cy = _centerline_pixel_coords(
                canyon, self.dem_bbox, self._rows, self._cols,
                self._reference_north_ft, self._reference_east_ft,
            )
            self._reference_overlay_x = cx
            self._reference_overlay_y = cy

    def set_centerline_profile(self, north_samples_ft, center_east_samples_ft):
        """Backward-compatible wrapper for legacy centerline overlays."""
        self.set_reference_profile(
            north_samples_ft=north_samples_ft,
            east_samples_ft=center_east_samples_ft,
            altitude_samples_ft=None,
            label="MPPI centerline",
        )

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
        lookahead_xy = np.asarray(
            planner_debug.get("lookahead_xy", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        )

        candidate_h = np.asarray(
            planner_debug.get("candidate_h_ft", np.zeros((0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        final_h = np.asarray(
            planner_debug.get("final_h_ft", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        lookahead_h = np.asarray(
            planner_debug.get("lookahead_h_ft", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        gk_trajectories = np.asarray(
            planner_debug.get("gk_trajectories", np.zeros((0, 0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        gk_h_ft = np.asarray(
            planner_debug.get("gk_h_ft", np.zeros((0, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        pid_error_xy = np.asarray(
            planner_debug.get("pid_error_xy", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        pid_error_h = np.asarray(
            planner_debug.get("pid_error_h_ft", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )
        failure_mask = np.asarray(
            planner_debug.get("failure_mask", np.zeros((0,), dtype=bool)),
            dtype=bool,
        )

        if (
            candidate_xy.size == 0
            and final_xy.size == 0
            and lookahead_xy.size == 0
            and gk_trajectories.size == 0
            and pid_error_xy.size == 0
        ):
            return frame

        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        if gk_trajectories.size > 0:
            traj_len = int(gk_trajectories.shape[1]) if gk_trajectories.ndim == 3 else 0
            s_t = int(planner_debug.get("s_t", 0))
            plan_start_t = int(planner_debug.get("plan_start_t", 0))
            local_m = s_t - plan_start_t
            m_slice = int(np.clip(local_m, 0, max(traj_len - 1, 0)))

            for idx, traj_xy in enumerate(gk_trajectories):
                traj_h = gk_h_ft[idx] if idx < len(gk_h_ft) else np.zeros((traj_xy.shape[0],), dtype=np.float32)
                world_points = self._trajectory_world_points(traj_xy, traj_h)
                pixels = self._project_world_points(world_points, view, projection, width, height)
                valid_mask = np.all(np.isfinite(pixels), axis=1)

                nom_pixels = pixels[1 : m_slice + 2][valid_mask[1 : m_slice + 2]]
                if len(nom_pixels) >= 2:
                    draw.line([tuple(point) for point in nom_pixels], fill=(34, 193, 114, 72), width=1)

                is_failed = bool(failure_mask[idx]) if idx < len(failure_mask) else False
                backup_color = (255, 30, 30, 96) if is_failed else (52, 152, 219, 72)
                back_pixels = pixels[m_slice + 2 :][valid_mask[m_slice + 2 :]]
                if len(back_pixels) >= 2:
                    draw.line([tuple(point) for point in back_pixels], fill=backup_color, width=1)

                switch_idx = local_m + 1
                if 0 <= switch_idx < len(pixels) and valid_mask[switch_idx]:
                    x_pix, y_pix = pixels[switch_idx]
                    draw.ellipse((x_pix - 3, y_pix - 3, x_pix + 3, y_pix + 3), fill=(255, 235, 59, 210))
        else:
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

        if len(lookahead_xy) >= 1:
            world_points = self._trajectory_world_points(lookahead_xy, lookahead_h)
            pixels = self._project_world_points(world_points, view, projection, width, height)
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            for x_pix, y_pix in pixels:
                draw.ellipse((x_pix - 7, y_pix - 7, x_pix + 7, y_pix + 7), fill=(28, 32, 18, 220))
                draw.ellipse((x_pix - 5, y_pix - 5, x_pix + 5, y_pix + 5), fill=(160, 255, 74, 255))
                draw.ellipse((x_pix - 2, y_pix - 2, x_pix + 2, y_pix + 2), fill=(250, 255, 230, 255))

        if len(pid_error_xy) >= 2:
            world_points = self._trajectory_world_points(pid_error_xy, pid_error_h)
            pixels = self._project_world_points(world_points, view, projection, width, height)
            valid_mask = np.all(np.isfinite(pixels), axis=1)
            valid_pixels = pixels[valid_mask]
            
            if len(valid_pixels) >= 2:
                # Draw the error lines (horizontal cross-track, vertical altitude)
                draw.line([tuple(point) for point in valid_pixels], fill=(255, 50, 50, 255), width=2)
                
                # Draw small dots at the airplane and closest point on path
                for idx, (x_pix, y_pix) in enumerate(valid_pixels):
                    if idx == 0:
                        # Airplane position
                        draw.ellipse((x_pix - 4, y_pix - 4, x_pix + 4, y_pix + 4), fill=(50, 200, 255, 255))
                    elif idx == len(valid_pixels) - 1:
                        # Path closest point
                        draw.ellipse((x_pix - 4, y_pix - 4, x_pix + 4, y_pix + 4), fill=(255, 50, 50, 255))

        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _overlay_reference_trajectory(self, frame):
        """Draw the configured reference trajectory as a cyan ribbon in the 3D video frame."""
        if self._reference_north_ft is None or self._reference_east_ft is None:
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

        n_arr = self._reference_north_ft
        e_arr = self._reference_east_ft
        h_arr = self._reference_altitude_ft

        # Subsample for performance — draw every Nth point
        stride = max(1, len(n_arr) // 600)
        n_sub = n_arr[::stride]
        e_sub = e_arr[::stride]
        h_sub = None if h_arr is None else h_arr[::stride]

        # Only draw within a window around the aircraft
        if cur_local_north is not None:
            window_ft = 3000.0
            mask = np.abs(n_sub - cur_local_north) < window_ft
            n_sub = n_sub[mask]
            e_sub = e_sub[mask]
            if h_sub is not None:
                h_sub = h_sub[mask]

        if len(n_sub) < 2:
            return frame

        # If no altitude profile is provided, keep the legacy near-ground ribbon behavior.
        if h_sub is None:
            h_ref = np.full_like(n_sub, 0.0)
        else:
            h_ref = np.asarray(h_sub, dtype=np.float32)
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
        draw.line([tuple(p) for p in pixels_valid], fill=(0, 255, 255, 200), width=3)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _overlay_flight_hud(self, frame, hud_debug):
        if hud_debug is None:
            return frame

        action_cmd = np.asarray(hud_debug.get("action_cmd", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)

        if action_cmd.size < 4:
            padded = np.zeros((4,), dtype=np.float32)
            padded[: action_cmd.size] = action_cmd
            action_cmd = padded

        aileron_cmd = float(action_cmd[0])
        elevator_cmd = float(action_cmd[1])
        rudder_cmd = float(action_cmd[2])
        throttle_cmd = float(action_cmd[3])

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()

        frame_h, frame_w = frame.shape[0], frame.shape[1]

        if bool(hud_debug.get("gatekeeper_active", False)):
            using_backup = bool(hud_debug.get("using_backup", False))
            is_reverting = bool(hud_debug.get("is_reverting", False))
            s_t = int(hud_debug.get("s_t", 0))
            m_star = int(hud_debug.get("m_star", 0))
            q_bar_star = float(hud_debug.get("q_bar_star", 0.0))
            epsilon = float(hud_debug.get("epsilon", 0.0))

            status_x0 = 16
            status_y0 = 16
            status_x1 = min(frame_w - 8, status_x0 + 246)
            status_y1 = min(frame_h - 8, status_y0 + 104)
            draw.rounded_rectangle(
                (status_x0, status_y0, status_x1, status_y1),
                radius=12,
                fill=(8, 12, 18, 176),
                outline=(180, 210, 235, 190),
                width=2,
            )

            mode_text = "BACKUP" if using_backup else "NOMINAL"
            mode_color = (255, 87, 34, 255) if using_backup else (76, 175, 80, 255)
            draw.text((status_x0 + 12, status_y0 + 10), mode_text, fill=mode_color, font=font)
            if is_reverting:
                draw.text((status_x0 + 94, status_y0 + 10), "REVERT", fill=(255, 235, 59, 255), font=font)

            draw.text(
                (status_x0 + 12, status_y0 + 34),
                f"Switch: {s_t} (T{m_star:+d})",
                fill=(228, 236, 244, 245),
                font=font,
            )

            prob_color = (255, 255, 255, 255)
            if q_bar_star > 0.5 * epsilon:
                prob_color = (255, 235, 59, 255)
            if q_bar_star > 0.8 * epsilon:
                prob_color = (255, 152, 0, 255)
            if q_bar_star > 1.0 * epsilon:
                prob_color = (255, 30, 30, 255)

            draw.text(
                (status_x0 + 12, status_y0 + 58),
                f"P(fail) Bound: {q_bar_star:.3f}",
                fill=prob_color,
                font=font,
            )
            draw.text(
                (status_x0 + 12, status_y0 + 80),
                f"Threshold eps: {epsilon:.2f}",
                fill=(180, 180, 180, 255),
                font=font,
            )

        panel_w = int(min(300, max(240, frame_w * 0.24)))
        panel_h = 136
        panel_x0 = int(max(16, frame_w - panel_w - 16))
        panel_y0 = int(max(16, frame_h - panel_h - 12))
        panel_x1 = min(frame_w - 8, panel_x0 + panel_w)
        panel_y1 = min(frame_h - 8, panel_y0 + panel_h)
        panel_w = panel_x1 - panel_x0
        panel_h = panel_y1 - panel_y0

        draw.rounded_rectangle(
            (panel_x0, panel_y0, panel_x1, panel_y1),
            radius=14,
            fill=(8, 12, 18, 172),
            outline=(180, 210, 235, 190),
            width=2,
        )

        draw.text((panel_x0 + 12, panel_y0 + 8), "Control Cmd", fill=(208, 230, 246, 245))

        # Horizontal command bars: aileron and rudder in [-1, 1].
        h_label_x = panel_x0 + 14
        h_bar_x0 = panel_x0 + 46
        h_bar_x1 = panel_x0 + panel_w - 88
        if h_bar_x1 < h_bar_x0 + 96:
            h_bar_x1 = h_bar_x0 + 96

        h_rows = [
            ("AIL", aileron_cmd, (114, 203, 255, 220), panel_y0 + 40),
            ("RUD", rudder_cmd, (247, 187, 119, 230), panel_y0 + 74),
        ]
        h_bar_h = 14
        for label, value, color, cy in h_rows:
            by0 = cy - h_bar_h // 2
            by1 = cy + h_bar_h // 2
            draw.text((h_label_x, cy - 12), label, fill=(218, 230, 242, 245))
            draw.rounded_rectangle(
                (h_bar_x0, by0, h_bar_x1, by1),
                radius=5,
                fill=(22, 28, 36, 190),
                outline=(122, 142, 162, 180),
                width=1,
            )

            center_x = 0.5 * (h_bar_x0 + h_bar_x1)
            draw.line((center_x, by0 - 1, center_x, by1 + 1), fill=(180, 192, 205, 210), width=1)

            value_clamped = float(np.clip(value, -1.0, 1.0))
            x_val = center_x + value_clamped * (0.5 * (h_bar_x1 - h_bar_x0))
            if x_val >= center_x:
                draw.rectangle((center_x, by0 + 2, x_val, by1 - 2), fill=color)
            else:
                draw.rectangle((x_val, by0 + 2, center_x, by1 - 2), fill=color)
            draw.text((h_bar_x1 + 6, cy - 10), f"{value_clamped:+.2f}", fill=(218, 230, 242, 245))

        # Vertical command meters: elevator in [-1, 1], throttle in [0, 1].
        meter_top = panel_y0 + 28
        meter_bot = panel_y0 + panel_h - 18
        meter_w = 18

        ele_x0 = panel_x0 + panel_w - 62
        ele_x1 = ele_x0 + meter_w
        thr_x0 = panel_x0 + panel_w - 36
        thr_x1 = thr_x0 + meter_w

        draw.text((ele_x0 - 2, panel_y0 + 10), "ELE", fill=(208, 230, 246, 245))
        draw.text((thr_x0 - 2, panel_y0 + 10), "THR", fill=(245, 226, 170, 245))

        draw.rounded_rectangle(
            (ele_x0, meter_top, ele_x1, meter_bot),
            radius=6,
            fill=(22, 28, 36, 190),
            outline=(122, 142, 162, 180),
            width=1,
        )
        draw.rounded_rectangle(
            (thr_x0, meter_top, thr_x1, meter_bot),
            radius=6,
            fill=(22, 28, 36, 190),
            outline=(122, 142, 162, 180),
            width=1,
        )

        # Elevator: symmetric scale [-1, +1] with center line.
        ele_center_y = 0.5 * (meter_top + meter_bot)
        draw.line((ele_x0 - 1, ele_center_y, ele_x1 + 1, ele_center_y), fill=(180, 192, 205, 210), width=1)
        elevator_clamped = float(np.clip(elevator_cmd, -1.0, 1.0))
        ele_half = max(1.0, 0.5 * (meter_bot - meter_top - 4))
        ele_y = ele_center_y - elevator_clamped * ele_half
        if ele_y <= ele_center_y:
            draw.rectangle((ele_x0 + 2, ele_y, ele_x1 - 2, ele_center_y), fill=(129, 233, 164, 225))
        else:
            draw.rectangle((ele_x0 + 2, ele_center_y, ele_x1 - 2, ele_y), fill=(129, 233, 164, 225))
        draw.text((ele_x0 - 8, meter_bot + 2), f"{elevator_clamped:+.2f}", fill=(218, 230, 242, 245))

        # Throttle: unilateral scale [0, 1] from bottom to top.
        throttle_clamped = float(np.clip(throttle_cmd, 0.0, 1.0))
        fill_h = (meter_bot - meter_top - 4) * throttle_clamped
        thr_fill_y = meter_bot - 2 - fill_h
        draw.rectangle((thr_x0 + 2, thr_fill_y, thr_x1 - 2, meter_bot - 2), fill=(255, 205, 86, 235))
        draw.text((thr_x0 - 2, meter_bot + 2), f"{throttle_clamped:.2f}", fill=(245, 226, 170, 245))

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
        frame = self._capture_frame_or_disable()
        if frame is not None:
            self._writer.append_data(frame)
        self._sample_position()

    def record_step(self, planner_debug=None, hud_debug=None):
        frame = self._capture_frame_or_disable()
        if frame is not None:
            frame = self._overlay_reference_trajectory(frame)
            frame = self._overlay_planner_debug(frame, planner_debug)
            frame = self._overlay_flight_hud(frame, hud_debug)
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
                reference_x=self._reference_overlay_x,
                reference_y=self._reference_overlay_y,
                reference_label=self._reference_label,
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
