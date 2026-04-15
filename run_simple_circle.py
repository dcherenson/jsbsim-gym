import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jsbsim_gym.env import JSBSimEnv
from jsbsim_gym.simple_controller import (
    SimpleCanyonControllerConfig,
    SimpleTrajectoryController,
    apply_simple_controller_optuna_params,
    build_reference_trajectory,
    with_default_simple_controller_optuna_gains,
)


KTS_TO_FPS = 1.68781
FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
G_FTPS2 = 32.174
TURN_SPEED_SAFETY_FACTOR = 0.85


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a circular tracking scenario to test simple-controller gains.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable interactive rendering.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum number of control steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Environment seed.",
    )
    parser.add_argument(
        "--radius-ft",
        type=float,
        default=2800.0,
        help="Circle radius in feet.",
    )
    parser.add_argument(
        "--direction",
        choices=["cw", "ccw"],
        default="cw",
        help="Circle direction: clockwise (cw) or counter-clockwise (ccw).",
    )
    parser.add_argument(
        "--target-speed-kts",
        type=float,
        default=450.0,
        help=(
            "Requested target speed in knots for speed hold. "
            "The script will cap this to a turn-feasible speed when needed."
        ),
    )
    parser.add_argument(
        "--target-altitude-ft",
        type=float,
        default=None,
        help="Target altitude in feet. Default uses initial altitude.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Render pacing when --render is enabled.",
    )
    parser.add_argument(
        "--no-tuned-gains",
        action="store_true",
        help="Use built-in SimpleCanyonControllerConfig defaults only.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file containing best_params to override gains.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/simple_controller/circle_tracking"),
        help="Directory for summary, csv, and plot outputs.",
    )
    return parser.parse_args()


def _max_turn_feasible_speed_kts(radius_ft, roll_max_rad, safety_factor=TURN_SPEED_SAFETY_FACTOR):
    radius_ft = max(float(radius_ft), 1.0)
    roll_max_rad = float(np.clip(roll_max_rad, np.deg2rad(5.0), np.deg2rad(85.0)))
    safety_factor = float(np.clip(safety_factor, 0.2, 1.0))
    max_speed_fps = np.sqrt(G_FTPS2 * radius_ft * np.tan(roll_max_rad))
    return float((safety_factor * max_speed_fps) / KTS_TO_FPS)


def _load_config(args):
    config = SimpleCanyonControllerConfig(
        target_speed_fps=float(args.target_speed_kts) * KTS_TO_FPS,
        use_dem_centerline=False,
    )

    source = "defaults"
    tuned_keys = []
    if not args.no_tuned_gains:
        config, tuned_source, tuned_keys = with_default_simple_controller_optuna_gains(config)
        if tuned_source is not None:
            source = tuned_source

    if args.config_json is not None:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        params = payload.get("best_params", payload) if isinstance(payload, dict) else {}
        config, file_tuned_keys = apply_simple_controller_optuna_params(config, params)
        tuned_keys = sorted(set(list(tuned_keys) + list(file_tuned_keys)))
        source = str(args.config_json)

    return config, source, tuned_keys


def _extract_state(env, start_n_m, start_e_m):
    sim = env.simulation
    state = env.state
    n_ft = float((state[0] - start_n_m) * M_TO_FT)
    e_ft = float((state[1] - start_e_m) * M_TO_FT)

    u_fps = float(sim.get_property_value("velocities/u-fps"))
    v_fps = float(sim.get_property_value("velocities/v-fps"))
    w_fps = float(sim.get_property_value("velocities/w-fps"))

    return {
        "n_ft": n_ft,
        "e_ft": e_ft,
        "h_ft": float(sim.get_property_value("position/h-sl-ft")),
        "u_fps": u_fps,
        "v_fps": v_fps,
        "w_fps": w_fps,
        "ny": float(sim.get_property_value("accelerations/Ny")),
        "nz": float(sim.get_property_value("accelerations/Nz")),
        "p_rad_s": float(state[6]),
        "q_rad_s": float(state[7]),
        "r_rad_s": float(state[8]),
        "phi_rad": float(state[9]),
        "theta_rad": float(state[10]),
        "psi_rad": float(state[11]),
        "beta_rad": float(state[5]),
    }


def _to_controller_state(state):
    return {
        "p_N": float(state["n_ft"]),
        "p_E": float(state["e_ft"]),
        "h": float(state["h_ft"]),
        "u": float(state["u_fps"]),
        "v": float(state["v_fps"]),
        "w": float(state["w_fps"]),
        "p": float(state["p_rad_s"]),
        "q": float(state["q_rad_s"]),
        "r": float(state["r_rad_s"]),
        "phi": float(state["phi_rad"]),
        "theta": float(state["theta_rad"]),
        "psi": float(state["psi_rad"]),
        "beta": float(state["beta_rad"]),
        "ny": float(state["ny"]),
        "nz": float(state["nz"]),
    }


def _build_circle_reference_trajectory(center_n_ft, center_e_ft, radius_ft, direction, num_points=2048):
    if direction == "cw":
        theta = np.linspace(0.0, -2.0 * np.pi, int(num_points), endpoint=False, dtype=np.float64)
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False, dtype=np.float64)

    north_ft = float(center_n_ft) + float(radius_ft) * np.cos(theta)
    east_ft = float(center_e_ft) + float(radius_ft) * np.sin(theta)
    return build_reference_trajectory(
        north_ft=north_ft,
        east_ft=east_ft,
        closed_loop=True,
    )


def _save_outputs(output_dir, rows, summary, center_n_ft, center_e_ft, radius_ft):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "circle_tracking_trajectory.csv"
    summary_path = output_dir / "circle_tracking_summary.json"
    plot_path = output_dir / "circle_tracking_plot.png"

    fieldnames = [
        "step",
        "n_ft",
        "e_ft",
        "ref_n_ft",
        "ref_e_ft",
        "lateral_error_ft",
        "heading_error_deg",
        "altitude_error_ft",
        "speed_fps",
        "roll_cmd",
        "pitch_cmd",
        "yaw_cmd",
        "throttle_cmd",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    traj_n = np.array([row["n_ft"] for row in rows], dtype=np.float64)
    traj_e = np.array([row["e_ft"] for row in rows], dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    ref_n = center_n_ft + radius_ft * np.cos(theta)
    ref_e = center_e_ft + radius_ft * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.plot(ref_e, ref_n, "--", color="tab:blue", linewidth=2.0, label="Reference circle")
    ax.plot(traj_e, traj_n, color="tab:red", linewidth=2.0, label="Aircraft track")
    if len(traj_n) > 0:
        ax.scatter([traj_e[0]], [traj_n[0]], color="limegreen", edgecolors="black", s=70, zorder=3, label="Start")
        ax.scatter([traj_e[-1]], [traj_n[-1]], color="crimson", marker="x", s=80, zorder=3, label="End")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.set_title("Simple Controller Circle Tracking")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return csv_path, summary_path, plot_path


def main():
    args = _parse_args()
    config, config_source, tuned_keys = _load_config(args)

    requested_speed_kts = float(args.target_speed_kts)
    max_turn_speed_kts = _max_turn_feasible_speed_kts(
        radius_ft=float(args.radius_ft),
        roll_max_rad=float(config.roll_max_rad),
    )
    effective_speed_kts = float(min(requested_speed_kts, max_turn_speed_kts))
    speed_was_limited = effective_speed_kts + 1e-9 < requested_speed_kts
    config.target_speed_fps = effective_speed_kts * KTS_TO_FPS

    env = JSBSimEnv(render_mode="human" if args.render else None)
    rows = []
    termination_reason = "running"

    try:
        env.simulation.set_property_value("ic/u-fps", float(config.target_speed_fps))
        if args.target_altitude_ft is not None:
            env.simulation.set_property_value("ic/h-sl-ft", float(args.target_altitude_ft))
        env.reset(seed=int(args.seed))
        # Disable accidental task completion from JSBSimEnv's random goal logic.
        env.goal[:] = 1.0e12

        start_n_m = float(env.state[0])
        start_e_m = float(env.state[1])

        init_state = _extract_state(env, start_n_m, start_e_m)
        target_altitude_ft = (
            float(args.target_altitude_ft)
            if args.target_altitude_ft is not None
            else float(init_state["h_ft"])
        )

        psi0 = float(init_state["psi_rad"])
        right_n = -np.sin(psi0)
        right_e = np.cos(psi0)

        if args.direction == "cw":
            center_n_ft = init_state["n_ft"] - float(args.radius_ft) * right_n
            center_e_ft = init_state["e_ft"] - float(args.radius_ft) * right_e
        else:
            center_n_ft = init_state["n_ft"] + float(args.radius_ft) * right_n
            center_e_ft = init_state["e_ft"] + float(args.radius_ft) * right_e

        circle_reference = _build_circle_reference_trajectory(
            center_n_ft=center_n_ft,
            center_e_ft=center_e_ft,
            radius_ft=float(args.radius_ft),
            direction=args.direction,
        )

        controller = SimpleTrajectoryController(
            config=config,
            target_altitude_ft=float(target_altitude_ft),
            wall_margin_ft=0.0,
            altitude_reference_offset_ft=0.0,
            reference_trajectory=circle_reference,
        )
        controller.reset(
            state_dict=_to_controller_state(init_state),
            target_altitude_ft=float(target_altitude_ft),
        )

        print("Starting simple-controller circle tracking test")
        print(f"Gain source: {config_source}")
        if tuned_keys:
            print(f"Applied tuned keys: {', '.join(tuned_keys)}")
        print(
            f"Circle: radius_ft={args.radius_ft:.1f}, direction={args.direction}, "
            f"target_speed_kts={requested_speed_kts:.1f}, target_altitude_ft={target_altitude_ft:.1f}"
        )
        if speed_was_limited:
            print(
                f"Speed capped for turn feasibility: requested={requested_speed_kts:.1f} kts, "
                f"effective={effective_speed_kts:.1f} kts (radius={args.radius_ft:.1f} ft, "
                f"roll_max={np.degrees(config.roll_max_rad):.1f} deg)"
            )
        print(
            f"{'step':<5} | {'lat_err':<8} | {'head_err':<9} | {'alt_err':<8} | {'speed':<7} | {'thr':<5}"
        )
        print("-" * 72)

        for step in range(int(args.max_steps)):
            state = _extract_state(env, start_n_m, start_e_m)
            action = controller.get_action(_to_controller_state(state))
            diag = dict(controller.last_guidance)

            if not np.isfinite(action).all():
                termination_reason = "invalid_action"
                print(f"Invalid action at step {step}: {action}")
                break

            _, _, done, truncated, _ = env.step(action)

            rows.append(
                {
                    "step": int(step),
                    "n_ft": float(state["n_ft"]),
                    "e_ft": float(state["e_ft"]),
                    "ref_n_ft": float(diag.get("ontrack_north_ft", state["n_ft"])),
                    "ref_e_ft": float(diag.get("ontrack_center_east_ft", state["e_ft"])),
                    "lateral_error_ft": float(diag.get("lateral_error_ft", 0.0)),
                    "heading_error_deg": float(diag.get("heading_error_deg", 0.0)),
                    "altitude_error_ft": float(diag.get("altitude_error_ft", 0.0)),
                    "speed_fps": float(diag.get("speed_fps", 0.0)),
                    "roll_cmd": float(action[0]),
                    "pitch_cmd": float(action[1]),
                    "yaw_cmd": float(action[2]),
                    "throttle_cmd": float(action[3]),
                }
            )

            if step % 10 == 0:
                print(
                    f"{step:<5} | {diag.get('lateral_error_ft', 0.0):<8.1f} | {diag.get('heading_error_deg', 0.0):<9.2f} | "
                    f"{diag.get('altitude_error_ft', 0.0):<8.1f} | {diag.get('speed_fps', 0.0):<7.1f} | {action[3]:<5.2f}"
                )

            if args.render:
                env.render()
                time.sleep(max(1.0 / max(float(args.fps), 1.0), 0.0))

            if done:
                termination_reason = "terminated"
                print(f"Episode terminated at step {step}")
                break
            if truncated:
                termination_reason = "time_limit"
                print(f"Episode truncated at step {step}")
                break
        else:
            termination_reason = "max_steps"
    finally:
        env.close()

    if len(rows) == 0:
        print("No trajectory rows collected; nothing to save.")
        return

    lat_err = np.array([row["lateral_error_ft"] for row in rows], dtype=np.float64)
    head_err = np.array([row["heading_error_deg"] for row in rows], dtype=np.float64)
    alt_err = np.array([row["altitude_error_ft"] for row in rows], dtype=np.float64)

    summary = {
        "termination_reason": termination_reason,
        "steps": int(len(rows)),
        "circle": {
            "radius_ft": float(args.radius_ft),
            "direction": str(args.direction),
            "center_n_ft": float(center_n_ft),
            "center_e_ft": float(center_e_ft),
        },
        "tracking_metrics": {
            "mean_abs_lateral_error_ft": float(np.mean(np.abs(lat_err))),
            "rms_lateral_error_ft": float(np.sqrt(np.mean(np.square(lat_err)))),
            "mean_abs_heading_error_deg": float(np.mean(np.abs(head_err))),
            "rms_heading_error_deg": float(np.sqrt(np.mean(np.square(head_err)))),
            "mean_abs_altitude_error_ft": float(np.mean(np.abs(alt_err))),
        },
        "controller": {
            "requested_target_speed_kts": float(requested_speed_kts),
            "effective_target_speed_kts": float(effective_speed_kts),
            "max_turn_feasible_speed_kts": float(max_turn_speed_kts),
            "target_altitude_ft": float(target_altitude_ft),
            "gain_source": str(config_source),
            "gains": {
                k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in vars(config).items()
            },
        },
    }

    csv_path, summary_path, plot_path = _save_outputs(
        output_dir=Path(args.output_dir),
        rows=rows,
        summary=summary,
        center_n_ft=center_n_ft,
        center_e_ft=center_e_ft,
        radius_ft=float(args.radius_ft),
    )

    print("\nCircle tracking run complete")
    print(f"Saved trajectory CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(
        "RMS lateral error (ft): "
        f"{summary['tracking_metrics']['rms_lateral_error_ft']:.2f}"
    )


if __name__ == "__main__":
    main()
