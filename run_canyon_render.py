import time
from pathlib import Path

import gymnasium as gym
import imageio as iio
import imageio.v3 as iio_v3
import matplotlib.pyplot as plt
import numpy as np

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon_env import OBS_ALTITUDE_ERROR_FT, OBS_PHI, OBS_P, OBS_Q, OBS_R, OBS_THETA


DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)
OVERLAY_OUTPUT = Path("data/dem/plots/black_canyon_trajectory_overlay.png")


class AltitudeHoldController:
    """Straight-flight + altitude-hold controller for canyon visualization."""

    def __init__(self):
        self.altitude_error_integral = 0.0

    def get_action(self, obs):
        altitude_error_ft = float(obs[OBS_ALTITUDE_ERROR_FT])
        roll_angle = float(obs[OBS_PHI])
        roll_rate = float(obs[OBS_P])
        pitch_rate = float(obs[OBS_Q])
        yaw_rate = float(obs[OBS_R])
        pitch_angle = float(obs[OBS_THETA])

        # Integral term improves altitude hold over long horizons.
        self.altitude_error_integral += altitude_error_ft * (1.0 / 30.0)
        self.altitude_error_integral = np.clip(self.altitude_error_integral, -2000.0, 2000.0)

        # Keep wings level and damp yaw to maintain straight flight.
        roll_cmd = np.clip(-1.6 * roll_angle - 0.45 * roll_rate, -0.35, 0.35)
        yaw_cmd = np.clip(-0.25 * yaw_rate, -0.15, 0.15)

        pitch_cmd = np.clip(
            0.0022 * altitude_error_ft
            + 0.70 * pitch_angle
            - 0.30 * pitch_rate
            + 0.00006 * self.altitude_error_integral,
            -0.50,
            0.50,
        )

        throttle_cmd = np.clip(0.64 - 0.00008 * altitude_error_ft, 0.50, 0.95)

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


def latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, rows, cols):
    x = ((lon_deg - west) / (east - west)) * float(cols) - 0.5
    y = ((north - lat_deg) / (north - south)) * float(rows) - 0.5
    return x, y


def save_overlay_plot(track_x, track_y, termination_reason):
    dem = iio_v3.imread(DEM_PATH).astype(np.float32)
    if dem.ndim == 3:
        dem = dem[..., 0]
    dem[(~np.isfinite(dem)) | (dem < -1e20)] = np.nan

    rows, cols = dem.shape
    vmin = float(np.nanpercentile(dem, 2.0))
    vmax = float(np.nanpercentile(dem, 98.0))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(dem, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    ax.plot(track_x, track_y, color="red", linewidth=2.0, label="Aircraft trajectory")
    ax.scatter([track_x[0]], [track_y[0]], c="lime", s=70, edgecolors="black", linewidths=0.5, label="Trajectory start")
    ax.scatter([track_x[-1]], [track_y[-1]], c="red", marker="x", s=80, label="Trajectory end")

    ax.scatter(
        [DEM_START_PIXEL[0]],
        [DEM_START_PIXEL[1]],
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

    OVERLAY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLAY_OUTPUT, dpi=150)
    plt.close(fig)

    print(f"Saved overlay: {OVERLAY_OUTPUT}")
    print(f"First sampled pixel: ({track_x[0]:.1f}, {track_y[0]:.1f})")
    print(f"Last sampled pixel: ({track_x[-1]:.1f}, {track_y[-1]:.1f})")


def main():
    # With 500 ft canyon walls, 250 ft keeps the aircraft near the canyon midline.
    target_altitude_ft = 250.0

    env = gym.make(
        "JSBSimCanyon-v0",
        render_mode="rgb_array",
        canyon_mode="dem",
        dem_path=str(DEM_PATH),
        dem_bbox=DEM_BBOX,
        dem_valley_rel_elev=0.08,
        dem_smoothing_window=11,
        dem_min_width_ft=140.0,
        dem_max_width_ft=2200.0,
        dem_start_pixel=DEM_START_PIXEL,
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
        target_altitude_ft=target_altitude_ft,
        entry_altitude_ft=target_altitude_ft,
        min_altitude_ft=60.0,
        max_altitude_ft=1500.0,
        wind_sigma=0.0,
        canyon_span_ft=9000.0,
        # Keep spacing below cylinder diameter (2 * 8 ft) so segments overlap
        # slightly and the canyon walls appear continuous.
        canyon_segment_spacing_ft=12.0,
    )

    controller = AltitudeHoldController()

    mp4_writer = iio.get_writer("canyon.mp4", format="ffmpeg", fps=30)
    gif_writer = iio.get_writer("canyon.gif", format="gif", fps=8)

    obs, info = env.reset(seed=3)
    print(
        "Canyon source:",
        f"mode={env.unwrapped.canyon_mode}",
        f"class={type(env.unwrapped.canyon).__name__}",
        f"mesh={env.unwrapped.dem_render_mesh}",
        f"stride={env.unwrapped.dem_render_stride}",
        f"start_pixel={env.unwrapped.dem_start_pixel}",
        f"start_heading_deg={env.unwrapped.dem_start_heading_deg}",
        f"proxy_bounds={env.unwrapped.dem_use_proxy_canyon_bounds}",
        f"mesh_padding_ft={env.unwrapped.dem_render_padding_ft}",
    )

    print("Running canyon rendering demo...")

    south, north, west, east = DEM_BBOX
    rows = int(env.unwrapped.canyon.rows)
    cols = int(env.unwrapped.canyon.cols)
    sim = env.unwrapped.simulation

    track_x = []
    track_y = []
    termination_reason = "running"

    def sample_position():
        lat_deg = float(sim.get_property_value("position/lat-gc-deg"))
        lon_deg = float(sim.get_property_value("position/long-gc-deg"))
        px, py = latlon_to_pixel(lat_deg, lon_deg, south, north, west, east, rows, cols)
        track_x.append(px)
        track_y.append(py)

    # Warm-up frame avoids occasional black first frame on some GL backends.
    _ = env.render()
    sample_position()

    try:
        for step in range(1200):
            render_data = env.render()
            mp4_writer.append_data(render_data)
            if step % 6 == 0:
                gif_writer.append_data(render_data[::2, ::2, :])

            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            sample_position()
            termination_reason = info.get("termination_reason", "running")

            if step % 20 == 0:
                print(
                    f"step={step:4d} | p_N={info['progress_ft']:7.1f} ft/step "
                    f"| width={info['canyon_width_ft']:7.1f} ft "
                    f"| lateral={info['lateral_error_ft']:7.1f} ft"
                )

            if terminated or truncated:
                print(f"Episode ended at step {step}: {info['termination_reason']}")
                break

            time.sleep(1 / 30)
    finally:
        mp4_writer.close()
        gif_writer.close()
        env.close()

    if track_x:
        save_overlay_plot(track_x, track_y, termination_reason)


if __name__ == "__main__":
    main()
