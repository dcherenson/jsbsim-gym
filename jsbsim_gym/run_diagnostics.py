"""
run_diagnostics.py
==================
Post-run diagnostic helpers: CSV writing and matplotlib plot generation for all
controller types used in run_scenario.py.

Functions
---------
save_pid_traj_diagnostics
save_simple_controller_diagnostics
save_mppi_tracking_diagnostics
save_mppi_plan_diagnostics
_append_state_fields
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# PID trajectory controller
# ---------------------------------------------------------------------------

def save_pid_traj_diagnostics(output_dir, file_stem, rows, termination_reason):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    csv_path = output_dir / f"{file_stem}_diagnostics.csv"
    plot_path = output_dir / f"{file_stem}_diagnostics.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True, constrained_layout=True)

    # 1. Guidance Errors (cross-track and altitude)
    axs[0].plot(time_s, [r["e_xtrk"] for r in rows], label="Xtrack Err (ft)", color="tab:red", linewidth=2.0)
    axs[0].plot(time_s, [r["e_z"] for r in rows], label="Alt Err (ft)", color="tab:blue", linewidth=2.0)
    axs[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[0].set_ylabel("Errors (ft)")
    axs[0].set_title(f"PID Traj Controller Diagnostics | end={termination_reason} | steps={len(rows)}")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.25)

    # 2. Guidance Commands (phi and alpha) + nominal trajectory reference
    ax1 = axs[1]
    ax1.plot(time_s, [np.degrees(r["phi_cmd"]) for r in rows],
             label="Phi Cmd (deg)", color="tab:purple", linewidth=1.8)
    ax1.plot(time_s, [np.degrees(r["phi"]) for r in rows],
             label="Phi Actual", color="tab:green", linewidth=1.4, alpha=0.8)
    ax1.plot(time_s, [np.degrees(r.get("phi_ref", float("nan"))) for r in rows],
             label="Phi Ref (traj)", color="tab:purple", linewidth=1.2,
             linestyle="--", alpha=0.55)
    ax1.plot(time_s, [np.degrees(r["alpha_cmd"]) for r in rows],
             label="Alpha Cmd (deg)", color="tab:orange", linewidth=1.8)
    ax1.plot(time_s, [np.degrees(r["alpha"]) for r in rows],
             label="Alpha Actual", color="tab:brown", linewidth=1.4, alpha=0.8)
    ax1.plot(time_s, [np.degrees(r.get("alpha_ref", float("nan"))) for r in rows],
             label="Alpha Ref (traj)", color="tab:orange", linewidth=1.2,
             linestyle="--", alpha=0.55)
    ax1.set_ylabel("Angles (deg)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # 3. Rate Commands (p and q)
    axs[2].plot(time_s, [np.degrees(r["p_cmd"]) for r in rows], label="p Cmd (deg/s)", color="tab:green")
    axs[2].plot(time_s, [np.degrees(r["p"]) for r in rows], label="p Actual", color="tab:purple", alpha=0.7)
    axs[2].plot(time_s, [np.degrees(r["q_cmd"]) for r in rows], label="q Cmd (deg/s)", color="tab:blue")
    axs[2].plot(time_s, [np.degrees(r["q"]) for r in rows], label="q Actual", color="tab:orange", alpha=0.7)
    axs[2].set_ylabel("Rates (deg/s)")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.25)

    # 4. Actuator Commands
    axs[3].plot(time_s, [r["elevator_cmd"] for r in rows], label="Elevator Cmd", color="tab:red")
    axs[3].plot(time_s, [r["aileron_cmd"] for r in rows], label="Aileron Cmd", color="tab:blue")
    axs[3].plot(time_s, [r["rudder_cmd"] for r in rows], label="Rudder Cmd", color="tab:green")
    axs[3].set_ylabel("Actuator Surface [-1, 1]")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.25)

    # 5. Speed and Throttle
    axs[4].plot(time_s, [r["v_opt_val"] for r in rows], label="V Opt (fps)", color="tab:orange")
    axs[4].plot(time_s, [r["V"] for r in rows], label="V Actual (fps)", color="tab:brown")
    axs[4].plot(time_s, [(r["throttle_cmd"] * 100) for r in rows], label="Throttle Cmd (*100)", color="tab:gray", linestyle="--")
    axs[4].set_ylabel("Speed / Throttle")
    axs[4].set_xlabel("Time (s)")
    axs[4].legend(loc="best")
    axs[4].grid(True, alpha=0.25)

    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return csv_path, plot_path


# ---------------------------------------------------------------------------
# Simple trajectory controller
# ---------------------------------------------------------------------------

def save_simple_controller_diagnostics(output_dir, file_stem, rows, termination_reason):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    csv_path = output_dir / f"{file_stem}_diagnostics.csv"
    plot_path = output_dir / f"{file_stem}_diagnostics.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    xtrack_ft = np.asarray([row["lateral_error_ft"] for row in rows], dtype=np.float64)
    xtrack_norm = np.asarray([row["lateral_error_norm"] for row in rows], dtype=np.float64)
    heading_error_deg = np.asarray([row["heading_error_deg"] for row in rows], dtype=np.float64)
    roll_cmd = np.asarray([row["roll_cmd"] for row in rows], dtype=np.float64)
    roll_des_deg = np.asarray([row["roll_des_deg"] for row in rows], dtype=np.float64)
    phi_deg = np.asarray([row["phi_deg"] for row in rows], dtype=np.float64)
    track_accel_cmd_fps2 = np.asarray([row["track_accel_cmd_fps2"] for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    axs[0].plot(time_s, xtrack_ft, color="tab:red", linewidth=2.0)
    axs[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axs[0].set_ylabel("Xtrack (ft)")
    axs[0].set_title(f"Simple Controller Diagnostics | end={termination_reason} | steps={len(rows)}")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(time_s, xtrack_norm, color="tab:orange", linewidth=2.0, label="xtrack norm")
    axs[1].axhline(1.0, color="black", linewidth=1.0, alpha=0.4, linestyle="--")
    axs[1].axhline(-1.0, color="black", linewidth=1.0, alpha=0.4, linestyle="--")
    axs[1].plot(time_s, heading_error_deg, color="tab:blue", linewidth=1.5, label="heading error (deg)")
    axs[1].set_ylabel("Norm / Deg")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(time_s, roll_des_deg, color="tab:purple", linewidth=2.0, label="roll des (deg)")
    axs[2].plot(time_s, phi_deg, color="tab:green", linewidth=1.5, label="phi (deg)")
    axs[2].set_ylabel("Bank (deg)")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.25)

    axs[3].plot(time_s, roll_cmd, color="tab:brown", linewidth=2.0, label="roll cmd")
    axs[3].plot(time_s, track_accel_cmd_fps2, color="tab:gray", linewidth=1.5, label="track accel (fps^2)")
    axs[3].set_ylabel("Cmd / Accel")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.25)

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return csv_path, plot_path


# ---------------------------------------------------------------------------
# MPPI tracking diagnostics
# ---------------------------------------------------------------------------

def save_mppi_tracking_diagnostics(output_dir, file_stem, rows, termination_reason, controller_label):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{file_stem}_tracking_diagnostics.csv"
    plot_path = output_dir / f"{file_stem}_tracking_diagnostics.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    progress_s_ft = np.asarray([row["progress_s_ft"] for row in rows], dtype=np.float64)
    virtual_speed_fps = np.asarray([row["virtual_speed_fps"] for row in rows], dtype=np.float64)
    contour_error_ft = np.asarray([row["contour_error_ft"] for row in rows], dtype=np.float64)
    lag_error_ft = np.asarray([row["lag_error_ft"] for row in rows], dtype=np.float64)
    position_error_ft = np.asarray([row["position_error_ft"] for row in rows], dtype=np.float64)
    altitude_error_ft = np.asarray([row["altitude_error_ft"] for row in rows], dtype=np.float64)
    terrain_clearance_ft = np.asarray([row.get("terrain_clearance_ft", np.nan) for row in rows], dtype=np.float64)
    terrain_safe_clearance_ft = np.asarray([row.get("terrain_safe_clearance_ft", np.nan) for row in rows], dtype=np.float64)
    alpha_deg = np.asarray([row.get("alpha_deg", np.nan) for row in rows], dtype=np.float64)
    alpha_limit_deg = np.asarray([row.get("alpha_limit_deg", np.nan) for row in rows], dtype=np.float64)
    nz_g = np.asarray([row.get("nz_g", np.nan) for row in rows], dtype=np.float64)
    nz_min_g = np.asarray([row.get("nz_min_g", np.nan) for row in rows], dtype=np.float64)
    nz_max_g = np.asarray([row.get("nz_max_g", np.nan) for row in rows], dtype=np.float64)
    aileron_cmd = np.asarray([row.get("aileron_cmd", np.nan) for row in rows], dtype=np.float64)
    elevator_cmd = np.asarray([row.get("elevator_cmd", np.nan) for row in rows], dtype=np.float64)
    rudder_cmd = np.asarray([row.get("rudder_cmd", np.nan) for row in rows], dtype=np.float64)
    throttle_cmd = np.asarray([row.get("throttle_cmd", np.nan) for row in rows], dtype=np.float64)
    rudder_pos_norm = np.asarray([row.get("rudder_pos_norm", np.nan) for row in rows], dtype=np.float64)
    rudder_pos_rad = np.asarray([row.get("rudder_pos_rad", np.nan) for row in rows], dtype=np.float64)
    rudder_pos_deg = np.degrees(rudder_pos_rad)
    aileron_rate = np.asarray([row.get("aileron_rate", np.nan) for row in rows], dtype=np.float64)
    elevator_rate = np.asarray([row.get("elevator_rate", np.nan) for row in rows], dtype=np.float64)
    rudder_rate = np.asarray([row.get("rudder_rate", np.nan) for row in rows], dtype=np.float64)
    throttle_rate = np.asarray([row.get("throttle_rate", np.nan) for row in rows], dtype=np.float64)
    contour_cost_est = np.asarray([row.get("contour_cost_est", np.nan) for row in rows], dtype=np.float64)
    lag_cost_est = np.asarray([row.get("lag_cost_est", np.nan) for row in rows], dtype=np.float64)
    progress_reward_est = np.asarray([row.get("progress_reward_est", np.nan) for row in rows], dtype=np.float64)
    virtual_speed_cost_est = np.asarray([row.get("virtual_speed_cost_est", np.nan) for row in rows], dtype=np.float64)
    contouring_cost_est = np.asarray([row.get("contouring_cost_est", np.nan) for row in rows], dtype=np.float64)
    terrain_cost_est = np.asarray([row.get("terrain_cost_est", np.nan) for row in rows], dtype=np.float64)
    rate_cost_est = np.asarray([row.get("rate_cost_est", np.nan) for row in rows], dtype=np.float64)
    limit_cost_est = np.asarray([row.get("limit_cost_est", np.nan) for row in rows], dtype=np.float64)
    total_stage_cost_est = np.asarray([row.get("total_stage_cost_est", np.nan) for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(4, 2, figsize=(14, 13), sharex=True, constrained_layout=True)
    axs = axs.reshape(-1)

    axs[0].plot(time_s, contour_error_ft, color="tab:blue", linewidth=2.0, label="contour")
    axs[0].plot(time_s, np.abs(lag_error_ft), color="tab:orange", linewidth=1.5, alpha=0.9, label="|lag|")
    axs[0].plot(time_s, position_error_ft, color="tab:green", linewidth=1.2, alpha=0.9, label="position")
    axs[0].set_ylabel("Feet")
    axs[0].set_title("Contouring Errors")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(time_s, progress_s_ft, color="tab:purple", linewidth=2.0, label="s")
    axs[1].plot(time_s, virtual_speed_fps, color="tab:brown", linewidth=1.5, alpha=0.9, label="v_s")
    axs[1].plot(time_s, altitude_error_ft, color="tab:red", linewidth=1.2, alpha=0.8, label="altitude err")
    axs[1].set_ylabel("Feet / ft/s")
    axs[1].set_title("Virtual Progress")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(time_s, terrain_clearance_ft, color="tab:green", linewidth=2.0, label="terrain clr")
    if np.isfinite(terrain_clearance_ft).any():
        axs[2].plot(
            time_s,
            terrain_safe_clearance_ft,
            color="black",
            linewidth=1.0,
            alpha=0.6,
            linestyle="--",
            label="safe clr",
        )
    axs[2].set_ylabel("Feet")
    axs[2].set_title("Terrain Clearance")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.25)

    axs[3].plot(time_s, nz_g, color="tab:orange", linewidth=1.8, label="nz")
    axs[3].plot(time_s, alpha_deg, color="tab:red", linewidth=1.8, label="alpha")
    if np.isfinite(nz_min_g).any():
        axs[3].plot(time_s, nz_min_g, color="black", linewidth=1.0, alpha=0.5, linestyle="--")
    if np.isfinite(nz_max_g).any():
        axs[3].plot(time_s, nz_max_g, color="black", linewidth=1.0, alpha=0.5, linestyle="--")
    if np.isfinite(alpha_limit_deg).any():
        axs[3].plot(time_s, alpha_limit_deg, color="tab:red", linewidth=1.0, alpha=0.5, linestyle="--")
    axs[3].set_ylabel("g / deg")
    axs[3].set_title("Structural Limits")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.25)

    axs[4].plot(time_s, contour_cost_est, color="tab:blue", linewidth=1.6, label="contour")
    axs[4].plot(time_s, lag_cost_est, color="tab:orange", linewidth=1.6, label="lag")
    axs[4].plot(time_s, progress_reward_est, color="tab:green", linewidth=1.6, label="-w_v v_s")
    axs[4].plot(time_s, virtual_speed_cost_est, color="tab:purple", linewidth=1.6, label="R_vs v_s^2")
    axs[4].plot(time_s, contouring_cost_est, color="black", linewidth=1.8, alpha=0.9, label="L_cont")
    axs[4].set_ylabel("Cost")
    axs[4].set_title("Contouring Cost Terms")
    axs[4].legend(loc="best")
    axs[4].grid(True, alpha=0.25)

    axs[5].plot(time_s, aileron_rate, color="tab:blue", linewidth=1.5, label="ail rate")
    axs[5].plot(time_s, elevator_rate, color="tab:orange", linewidth=1.5, label="ele rate")
    axs[5].plot(time_s, rudder_rate, color="tab:green", linewidth=1.5, label="rud rate")
    axs[5].plot(time_s, throttle_rate, color="tab:brown", linewidth=1.5, label="thr rate")
    axs[5].set_ylabel("Delta cmd")
    axs[5].set_title("Control Rates")
    axs[5].legend(loc="best")
    axs[5].grid(True, alpha=0.25)

    axs[6].plot(time_s, aileron_cmd, color="tab:blue", linewidth=1.5, label="ail")
    axs[6].plot(time_s, elevator_cmd, color="tab:orange", linewidth=1.5, label="ele")
    axs[6].plot(time_s, rudder_cmd, color="tab:green", linewidth=1.5, label="rud")
    axs[6].plot(time_s, throttle_cmd, color="tab:brown", linewidth=1.5, label="thr")
    if np.isfinite(rudder_pos_norm).any():
        axs[6].plot(
            time_s,
            rudder_pos_norm,
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label="rud pos norm",
        )
    axs[6].set_xlabel("Time (s)")
    axs[6].set_ylabel("Cmd")
    axs[6].set_title("Action Commands")
    axs[6].legend(loc="best")
    axs[6].grid(True, alpha=0.25)

    if np.isfinite(rudder_pos_deg).any():
        ax6b = axs[6].twinx()
        ax6b.plot(
            time_s,
            rudder_pos_deg,
            color="tab:gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
        )
        ax6b.set_ylabel("Rudder (deg)")

    axs[7].plot(time_s, terrain_cost_est, color="tab:green", linewidth=1.6, label="terrain")
    axs[7].plot(time_s, rate_cost_est, color="tab:orange", linewidth=1.6, label="rate")
    axs[7].plot(time_s, limit_cost_est, color="tab:red", linewidth=1.6, label="limit")
    axs[7].plot(time_s, total_stage_cost_est, color="black", linewidth=1.8, alpha=0.9, label="total")
    axs[7].set_xlabel("Time (s)")
    axs[7].set_ylabel("Cost")
    axs[7].set_title("Stage Cost Terms")
    axs[7].legend(loc="best")
    axs[7].grid(True, alpha=0.25)

    fig.suptitle(
        f"{controller_label.upper()} Tracking Diagnostics | end={termination_reason} | steps={len(rows)}",
        fontsize=13,
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return csv_path, plot_path


# ---------------------------------------------------------------------------
# MPPI plan diagnostics
# ---------------------------------------------------------------------------

def save_mppi_plan_diagnostics(output_dir, file_stem, rows, action_plans, virtual_speed_plans):
    if not rows:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{file_stem}_plan_diagnostics.csv"
    npz_path = output_dir / f"{file_stem}_plan_diagnostics.npz"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    action_plan_arr = np.asarray(action_plans, dtype=np.float32)
    virtual_speed_plan_arr = np.asarray(virtual_speed_plans, dtype=np.float32)
    nominal_action_arr = np.asarray(
        [
            [
                float(row["nominal_aileron_cmd"]),
                float(row["nominal_elevator_cmd"]),
                float(row["nominal_rudder_cmd"]),
                float(row["nominal_throttle_cmd"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    applied_action_arr = np.asarray(
        [
            [
                float(row["applied_aileron_cmd"]),
                float(row["applied_elevator_cmd"]),
                float(row["applied_rudder_cmd"]),
                float(row["applied_throttle_cmd"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        npz_path,
        action_plan=action_plan_arr,
        virtual_speed_plan=virtual_speed_plan_arr,
        call_index=np.asarray([int(row["call_index"]) for row in rows], dtype=np.int32),
        step=np.asarray([int(row["step"]) for row in rows], dtype=np.int32),
        time_s=np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32),
        progress_s_ft=np.asarray([float(row["progress_s_ft"]) for row in rows], dtype=np.float32),
        controller_step_index=np.asarray([int(row["controller_step_index"]) for row in rows], dtype=np.int32),
        using_backup=np.asarray([bool(row["using_backup"]) for row in rows], dtype=np.bool_),
        nominal_action=nominal_action_arr,
        applied_action=applied_action_arr,
    )
    return csv_path, npz_path


# ---------------------------------------------------------------------------
# Shared row-building helper
# ---------------------------------------------------------------------------

# Keys mirrored from run_scenario.MPPI_STATE_KEYS
_MPPI_STATE_KEYS = (
    "p_N", "p_E", "h", "u", "v", "w",
    "p", "q", "r", "phi", "theta", "psi", "ny", "nz",
)


def _append_state_fields(row_dict, *, prefix, state_dict):
    for key in _MPPI_STATE_KEYS:
        row_dict[f"{prefix}_{key}"] = float(state_dict.get(key, np.nan))
