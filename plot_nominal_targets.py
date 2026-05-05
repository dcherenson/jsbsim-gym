"""
plot_nominal_targets.py
=======================
Plot roll (phi), alpha, and speed targets from a nominal dyn.asb trajectory.

Usage
-----
    uv run python plot_nominal_targets.py [--dyn-path PATH] [--end-fraction FRAC] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEM_PATH = Path("data/dem/black-canyon-gunnison_USGS10m.tif")
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
KTS_PER_FPS = 1.0 / 1.68781


def main():
    parser = argparse.ArgumentParser(description="Plot nominal trajectory targets (phi, alpha, speed).")
    parser.add_argument(
        "--dyn-path",
        type=Path,
        default=Path("air-racing-optimization/final_results/f16dyn_crude.asb"),
        help="Path to the Aerosandbox dyn.asb file.",
    )
    parser.add_argument(
        "--end-fraction",
        type=float,
        default=1.0,
        help="Fraction [0,1] of trajectory to plot (default 1.0 = full).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: next to dyn.asb as nominal_targets.png).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load reference
    # ------------------------------------------------------------------ #
    from jsbsim_gym.canyon import DEMCanyon
    from jsbsim_gym.nominal_trajectory import build_nominal_reference_from_dyn

    canyon = DEMCanyon(
        dem_path=str(DEM_PATH),
        south=DEM_BBOX[0],
        north=DEM_BBOX[1],
        west=DEM_BBOX[2],
        east=DEM_BBOX[3],
        valley_rel_elev=0.08,
        smoothing_window=11,
        min_width_ft=140.0,
        max_width_ft=2200.0,
        fly_direction="south_to_north",
        dem_start_pixel=(1400, 950),
    )

    ref = build_nominal_reference_from_dyn(
        args.dyn_path,
        canyon=canyon,
        altitude_ref_ft=0.0,
        end_fraction=float(np.clip(args.end_fraction, 0.0, 1.0)),
    )

    time_s   = np.asarray(ref["time_s"],   dtype=np.float64)
    # phi_rad is unwrapped — wrap back to (-π, π] for display
    phi_deg  = np.degrees(np.arctan2(
        np.sin(np.asarray(ref["phi_rad"],   dtype=np.float64)),
        np.cos(np.asarray(ref["phi_rad"],   dtype=np.float64)),
    ))
    alpha_deg = np.degrees(np.asarray(ref["alpha_rad"], dtype=np.float64))
    speed_kts = np.asarray(ref["speed_fps"], dtype=np.float64) * KTS_PER_FPS

    # Relative time from zero
    t = time_s - time_s[0]

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    # — Roll —
    axs[0].plot(t, phi_deg, color="tab:purple", linewidth=1.8)
    axs[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4, linestyle="--")
    axs[0].set_ylabel("Roll φ (deg)")
    axs[0].set_title(
        f"Nominal Trajectory Targets — {args.dyn_path.name}"
        + (f"  [end={args.end_fraction:.2f}]" if args.end_fraction < 1.0 else ""),
        fontsize=12,
    )
    axs[0].grid(True, alpha=0.25)
    axs[0].set_ylim(-90, 90)

    # — Alpha —
    axs[1].plot(t, alpha_deg, color="tab:orange", linewidth=1.8)
    axs[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4, linestyle="--")
    axs[1].set_ylabel("Alpha α (deg)")
    axs[1].grid(True, alpha=0.25)

    # — Speed —
    axs[2].plot(t, speed_kts, color="tab:blue", linewidth=1.8)
    axs[2].set_ylabel("Speed (kts)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True, alpha=0.25)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_path = args.out
    if out_path is None:
        out_path = args.dyn_path.parent / "nominal_targets.png"

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(t)} samples, t=[{t[0]:.1f}, {t[-1]:.1f}] s)")
    total_distance = np.trapezoid(speed_kts, t) * 0.3048 / KTS_PER_FPS
    print(f"Total trajectory length in meters: {total_distance}")


if __name__ == "__main__":
    main()
