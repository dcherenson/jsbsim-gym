from pathlib import Path
import csv

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


def save_gatekeeper_rollout_csv(
    output_path,
    trajectories,
    step_index,
    failure_mask=None,
    s_t=None,
    plan_start_t=None,
    using_backup=None,
    is_reverting=None,
):
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2 or trajectories.shape[0] == 0:
        return

    failure_mask = np.asarray(
        failure_mask if failure_mask is not None else np.zeros((trajectories.shape[0],), dtype=bool),
        dtype=bool,
    )
    rollout_idx = np.repeat(np.arange(trajectories.shape[0], dtype=np.int32), trajectories.shape[1])
    point_idx = np.tile(np.arange(trajectories.shape[1], dtype=np.int32), trajectories.shape[0])
    flat = trajectories.reshape(-1, 2)
    valid = np.isfinite(flat[:, 0]) & np.isfinite(flat[:, 1])

    csv_arr = np.column_stack(
        [
            np.full(flat.shape[0], int(step_index), dtype=np.int32),
            rollout_idx,
            point_idx,
            flat[:, 0],
            flat[:, 1],
            valid.astype(np.int32),
            np.repeat(failure_mask.astype(np.int32), trajectories.shape[1]),
            np.full(flat.shape[0], -1 if s_t is None else int(s_t), dtype=np.int32),
            np.full(flat.shape[0], -1 if plan_start_t is None else int(plan_start_t), dtype=np.int32),
            np.full(flat.shape[0], -1 if using_backup is None else int(bool(using_backup)), dtype=np.int32),
            np.full(flat.shape[0], -1 if is_reverting is None else int(bool(is_reverting)), dtype=np.int32),
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        csv_arr,
        delimiter=",",
        header="step,rollout_idx,point_idx,north_ft,east_ft,is_valid,failed_rollout,s_t,plan_start_t,using_backup,is_reverting",
        comments="",
    )


def save_planner_debug_csv(output_path, planner_debug, step_index):
    if planner_debug is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "key", "index", "value"])

        for key in sorted(planner_debug.keys()):
            value = planner_debug[key]
            arr = np.asarray(value)

            if arr.ndim == 0:
                writer.writerow([int(step_index), key, "", arr.item()])
                continue

            for idx in np.ndindex(arr.shape):
                writer.writerow([int(step_index), key, ",".join(str(i) for i in idx), arr[idx]])


def save_gatekeeper_rollout_plot(
    output_path,
    trajectories,
    step_index,
    failure_mask=None,
    s_t=None,
    plan_start_t=None,
    using_backup=None,
    is_reverting=None,
):
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2 or trajectories.shape[0] == 0:
        return

    failure_mask = np.asarray(
        failure_mask if failure_mask is not None else np.zeros((trajectories.shape[0],), dtype=bool),
        dtype=bool,
    )

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for idx, traj_xy in enumerate(trajectories):
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        valid = np.all(np.isfinite(traj_xy), axis=1)
        if not np.any(valid):
            continue

        valid_idx = np.flatnonzero(valid)
        segment_breaks = np.where(np.diff(valid_idx) > 1)[0] + 1
        color = "crimson" if idx < len(failure_mask) and bool(failure_mask[idx]) else "deepskyblue"
        alpha = 0.18 if color == "deepskyblue" else 0.22
        for segment in np.split(valid_idx, segment_breaks):
            if segment.size < 2:
                continue
            segment_xy = traj_xy[segment]
            ax.plot(segment_xy[:, 1], segment_xy[:, 0], color=color, alpha=alpha, linewidth=1.0)

    first = np.asarray(trajectories[:, 0, :], dtype=np.float32)
    last = np.asarray(trajectories[:, -1, :], dtype=np.float32)
    first_valid = np.all(np.isfinite(first), axis=1)
    last_valid = np.all(np.isfinite(last), axis=1)
    if np.any(first_valid):
        ax.scatter(
            first[first_valid, 1],
            first[first_valid, 0],
            c="lime",
            s=12,
            alpha=0.6,
            edgecolors="none",
            label="Rollout start",
        )
    if np.any(last_valid):
        ax.scatter(
            last[last_valid, 1],
            last[last_valid, 0],
            c="orange",
            s=12,
            alpha=0.6,
            edgecolors="none",
            label="Rollout end",
        )

    title_bits = [f"Gatekeeper rollouts | step={int(step_index)} | N={int(trajectories.shape[0])}"]
    if s_t is not None:
        title_bits.append(f"s_t={int(s_t)}")
    if plan_start_t is not None:
        title_bits.append(f"plan_start_t={int(plan_start_t)}")
    if using_backup is not None:
        title_bits.append("backup" if bool(using_backup) else "nominal")
    if is_reverting is not None:
        title_bits.append("reverting" if bool(is_reverting) else "stable")

    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
        save_stepwise_gatekeeper_artifacts=True,
    ):
        self.env = env
        self.dem_path = Path(dem_path)
        self.dem_bbox = tuple(dem_bbox)
        self.dem_start_pixel = tuple(dem_start_pixel)
        self.title_prefix = str(title_prefix)
        self._save_stepwise_gatekeeper_artifacts = bool(save_stepwise_gatekeeper_artifacts)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = output_dir / f"{file_stem}.mp4"
        self.gif_path = output_dir / f"{file_stem}.gif"
        self.overlay_path = output_dir / f"{file_stem}_trajectory_overlay.png"
        self.trajectory_csv_path = output_dir / f"{file_stem}_trajectory.csv"
        self.gk_rollout_plot_dir = output_dir / f"{file_stem}_gatekeeper_rollout_plots"
        self.gk_rollout_csv_dir = output_dir / f"{file_stem}_gatekeeper_rollout_csv"
        self.planner_debug_csv_dir = output_dir / f"{file_stem}_planner_debug_csv"

        self._writer = iio.get_writer(self.video_path, format="ffmpeg", fps=int(fps))
        self._gif_writer = iio.get_writer(self.gif_path, format="gif", fps=int(fps))
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
        self._step_index = 0

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
        if self._reference_north_ft is None or self._reference_east_ft is None:
            return frame

        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            return frame

        n_arr = np.asarray(self._reference_north_ft, dtype=np.float32).reshape(-1)
        e_arr = np.asarray(self._reference_east_ft, dtype=np.float32).reshape(-1)
        if n_arr.size < 2 or e_arr.size != n_arr.size:
            return frame

        h_arr = None
        if self._reference_altitude_ft is not None:
            h_ref = np.asarray(self._reference_altitude_ft, dtype=np.float32).reshape(-1)
            if h_ref.size == n_arr.size:
                h_arr = h_ref

        cur_n = None
        cur_e = None
        env_unwrapped = self.env.unwrapped
        canyon = getattr(env_unwrapped, "canyon", None)
        if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
            try:
                lat_deg = float(self._sim.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self._sim.get_property_value("position/long-gc-deg"))
                cur_n, cur_e = canyon.get_local_from_latlon(lat_deg, lon_deg)
                cur_n = float(cur_n)
                cur_e = float(cur_e)
            except Exception:
                cur_n = None
                cur_e = None
        if cur_n is None or cur_e is None:
            try:
                state = env_unwrapped.get_full_state_dict()
                cur_n = float(state.get("p_N", np.nan))
                cur_e = float(state.get("p_E", np.nan))
            except Exception:
                cur_n = np.nan
                cur_e = np.nan
            if not (np.isfinite(cur_n) and np.isfinite(cur_e)):
                return frame

        valid = np.isfinite(n_arr) & np.isfinite(e_arr)
        if not np.any(valid):
            return frame

        valid_idx = np.flatnonzero(valid)
        dn = n_arr[valid_idx] - float(cur_n)
        de = e_arr[valid_idx] - float(cur_e)
        nearest_valid = int(np.argmin(dn * dn + de * de))
        start_idx = int(valid_idx[nearest_valid])
        end_idx = int(min(start_idx + 150, n_arr.size - 1))
        if end_idx <= start_idx:
            return frame

        seg_n = n_arr[start_idx : end_idx + 1]
        seg_e = e_arr[start_idx : end_idx + 1]
        if h_arr is None:
            seg_h = np.full_like(seg_n, 0.0, dtype=np.float32)
        else:
            seg_h = h_arr[start_idx : end_idx + 1]
            if seg_h.size != seg_n.size:
                seg_h = np.full_like(seg_n, 0.0, dtype=np.float32)

        xy = np.column_stack([seg_n, seg_e])
        world_points = self._trajectory_world_points(xy, seg_h)
        height, width = frame.shape[0], frame.shape[1]
        view = np.asarray(viewer.transform.inv_matrix, dtype=np.float32)
        projection = np.asarray(viewer.projection, dtype=np.float32)
        pixels = self._project_world_points(world_points, view, projection, width, height)
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
        if pixels.shape[0] < 2:
            return frame

        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.line([tuple(p) for p in pixels], fill=(255, 165, 0, 230), width=3)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _overlay_flight_hud(self, frame, hud_debug):
        return frame

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
            self._gif_writer.append_data(frame)
        self._sample_position()

    def record_step(self, planner_debug=None, hud_debug=None):
        frame = self._capture_frame_or_disable()
        if frame is not None:
            frame = self._overlay_reference_trajectory(frame)
            frame = self._overlay_planner_debug(frame, planner_debug)
            frame = self._overlay_flight_hud(frame, hud_debug)
            self._writer.append_data(frame)
            self._gif_writer.append_data(frame)

        if planner_debug is not None and self._save_stepwise_gatekeeper_artifacts:
            save_planner_debug_csv(
                output_path=self.planner_debug_csv_dir / f"step_{self._step_index:04d}.csv",
                planner_debug=planner_debug,
                step_index=self._step_index,
            )

            gk_trajectories = np.asarray(
                planner_debug.get("gk_trajectories", np.zeros((0, 0, 2), dtype=np.float32)),
                dtype=np.float32,
            )
            if gk_trajectories.ndim == 3 and gk_trajectories.shape[-1] == 2 and gk_trajectories.shape[0] > 0:
                save_gatekeeper_rollout_csv(
                    output_path=self.gk_rollout_csv_dir / f"step_{self._step_index:04d}.csv",
                    trajectories=gk_trajectories,
                    step_index=self._step_index,
                    failure_mask=planner_debug.get("failure_mask", None),
                    s_t=planner_debug.get("s_t", None),
                    plan_start_t=planner_debug.get("plan_start_t", None),
                    using_backup=planner_debug.get("using_backup", None),
                    is_reverting=planner_debug.get("is_reverting", None),
                )
                save_gatekeeper_rollout_plot(
                    output_path=self.gk_rollout_plot_dir / f"step_{self._step_index:04d}.png",
                    trajectories=gk_trajectories,
                    step_index=self._step_index,
                    failure_mask=planner_debug.get("failure_mask", None),
                    s_t=planner_debug.get("s_t", None),
                    plan_start_t=planner_debug.get("plan_start_t", None),
                    using_backup=planner_debug.get("using_backup", None),
                    is_reverting=planner_debug.get("is_reverting", None),
                )

        self._sample_position()
        self._step_index += 1

    def close_writer(self):
        if not self._closed:
            self._writer.close()
            self._gif_writer.close()
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
            "gif_path": self.gif_path,
            "overlay_path": self.overlay_path,
            "trajectory_csv_path": self.trajectory_csv_path,
            "gk_rollout_plot_dir": self.gk_rollout_plot_dir if self._save_stepwise_gatekeeper_artifacts else None,
            "gk_rollout_csv_dir": self.gk_rollout_csv_dir if self._save_stepwise_gatekeeper_artifacts else None,
            "planner_debug_csv_dir": self.planner_debug_csv_dir if self._save_stepwise_gatekeeper_artifacts else None,
        }
