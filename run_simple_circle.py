import argparse
import csv
import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jsbsim_gym.env import JSBSimEnv
from jsbsim_gym.mppi_run_config import (
    KTS_TO_FPS,
    apply_mppi_optuna_params,
    build_mppi_base_config_kwargs,
    build_mppi_controller,
    with_default_mppi_optuna_params,
)


FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
G_FTPS2 = 32.174
TURN_SPEED_SAFETY_FACTOR = 0.85
CONTROL_HZ = 30.0
REPO_ROOT = Path(__file__).resolve().parent
CIRCLE_MPPI_TUNING_JSON_PATH = REPO_ROOT / "output" / "circle_mppi" / "circle_mppi_optuna_best.json"
CIRCLE_MPPI_TUNING_STUDY_NAME = "circle_mppi_contouring_tuning"
CIRCLE_MPPI_TUNING_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'circle_mppi_tuning.db').as_posix()}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a circular tracking scenario to test contouring MPPI controllers.",
    )
    parser.add_argument(
        "--controller",
        choices=["mppi", "smooth_mppi"],
        default="mppi",
        help="MPPI controller variant.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable interactive rendering.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
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
            "Requested target speed in knots for the nominal circle profile. "
            "The script caps this to a turn-feasible speed when needed."
        ),
    )
    parser.add_argument(
        "--target-altitude-ft",
        type=float,
        default=None,
        help="Target altitude in feet MSL. Default uses initial altitude.",
    )
    parser.add_argument(
        "--reference-points-per-lap",
        type=int,
        default=2048,
        help="Reference trajectory samples per lap.",
    )
    parser.add_argument(
        "--reference-laps",
        type=int,
        default=None,
        help="Optional explicit number of laps in the nominal reference.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Render pacing when --render is enabled.",
    )
    parser.add_argument(
        "--no-tuned-params",
        "--no-tuned-gains",
        dest="no_tuned_params",
        action="store_true",
        help="Do not auto-load tuned MPPI params from default artifacts.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file containing best_params to override MPPI params.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory override.",
    )
    return parser.parse_args()


def _max_turn_feasible_speed_kts(radius_ft, roll_max_rad, safety_factor=TURN_SPEED_SAFETY_FACTOR):
    radius_ft = max(float(radius_ft), 1.0)
    roll_max_rad = float(np.clip(roll_max_rad, np.deg2rad(5.0), np.deg2rad(85.0)))
    safety_factor = float(np.clip(safety_factor, 0.2, 1.0))
    max_speed_fps = np.sqrt(G_FTPS2 * radius_ft * np.tan(roll_max_rad))
    return float((safety_factor * max_speed_fps) / KTS_TO_FPS)


def _print_mppi_config(controller_tag, config):
    if not is_dataclass(config):
        print(f"{controller_tag.upper()} config: {config}")
        return

    print(f"\n{controller_tag.upper()} effective configuration:")
    for field in fields(config):
        print(f"  {field.name}: {getattr(config, field.name)}")


def _config_to_dict(config):
    if is_dataclass(config):
        return {field.name: getattr(config, field.name) for field in fields(config)}
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {"value": config}


def _load_mppi_config_kwargs(args):
    config_base_kwargs = build_mppi_base_config_kwargs()
    source = "defaults"
    tuned_keys = []

    if args.controller == "mppi" and not args.no_tuned_params:
        config_base_kwargs, mppi_tuning_source, mppi_tuned_keys = with_default_mppi_optuna_params(
            config_base_kwargs,
            summary_json_path=CIRCLE_MPPI_TUNING_JSON_PATH,
            study_name=CIRCLE_MPPI_TUNING_STUDY_NAME,
            storage=CIRCLE_MPPI_TUNING_STORAGE,
        )
        if mppi_tuned_keys:
            source = str(mppi_tuning_source)
            tuned_keys = list(mppi_tuned_keys)

    if args.config_json is not None:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        params = payload.get("best_params", payload) if isinstance(payload, dict) else {}
        config_base_kwargs, file_tuned_keys = apply_mppi_optuna_params(config_base_kwargs, params)
        tuned_keys = sorted(set(list(tuned_keys) + list(file_tuned_keys)))
        source = str(args.config_json)

    # Keep the controller identical to canyon MPPI contouring, but disable terrain costs.
    config_base_kwargs["terrain_collision_penalty"] = 0.0
    config_base_kwargs["terrain_repulsion_scale"] = 0.0

    return config_base_kwargs, source, tuned_keys


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


def _build_circle_nominal_reference(
    *,
    start_n_ft,
    start_e_ft,
    center_n_ft,
    center_e_ft,
    radius_ft,
    direction,
    target_altitude_ft,
    target_speed_fps,
    max_steps,
    control_hz,
    points_per_lap,
    laps_override,
):
    radius_ft = float(max(radius_ft, 1.0))
    points_per_lap = int(max(points_per_lap, 64))

    circumference_ft = float(2.0 * np.pi * radius_ft)
    if laps_override is None:
        distance_budget_ft = float(max_steps) * (float(target_speed_fps) / float(max(control_hz, 1.0)))
        laps = int(max(1, np.ceil(distance_budget_ft / max(circumference_ft, 1.0)) + 1))
    else:
        laps = int(max(laps_override, 1))

    total_points = int(max(2, points_per_lap * laps))
    theta0 = float(np.arctan2(float(start_e_ft) - float(center_e_ft), float(start_n_ft) - float(center_n_ft)))
    angle_progress = np.linspace(0.0, 2.0 * np.pi * float(laps), total_points, endpoint=False, dtype=np.float64)
    if direction == "cw":
        theta = theta0 - angle_progress
    else:
        theta = theta0 + angle_progress

    north_ft = float(center_n_ft) + radius_ft * np.cos(theta)
    east_ft = float(center_e_ft) + radius_ft * np.sin(theta)
    altitude_ft = np.full_like(north_ft, float(target_altitude_ft), dtype=np.float64)

    d_north = np.gradient(north_ft)
    d_east = np.gradient(east_ft)
    heading_rad = np.unwrap(np.arctan2(d_east, d_north))

    reference_states_ft_rad = np.column_stack(
        [
            north_ft,
            east_ft,
            altitude_ft,
            np.zeros_like(north_ft),
            np.zeros_like(north_ft),
            heading_rad,
        ]
    ).astype(np.float32)

    speed_fps = np.full((total_points,), float(target_speed_fps), dtype=np.float32)
    time_s = np.arange(total_points, dtype=np.float32) / float(control_hz)

    return {
        "reference_states_ft_rad": reference_states_ft_rad,
        "speed_fps": speed_fps,
        "north_ft": north_ft.astype(np.float32),
        "east_ft": east_ft.astype(np.float32),
        "altitude_ft": altitude_ft.astype(np.float32),
        "heading_rad": heading_rad.astype(np.float32),
        "time_s": time_s,
        "circle_laps": int(laps),
        "circle_circumference_ft": circumference_ft,
    }


def _build_flat_terrain_grid(reference_north_ft, reference_east_ft, margin_ft=6000.0, samples_per_axis=64):
    margin_ft = float(max(margin_ft, 1000.0))
    samples_per_axis = int(max(samples_per_axis, 8))

    min_n = float(np.min(reference_north_ft)) - margin_ft
    max_n = float(np.max(reference_north_ft)) + margin_ft
    min_e = float(np.min(reference_east_ft)) - margin_ft
    max_e = float(np.max(reference_east_ft)) + margin_ft

    north_axis = np.linspace(min_n, max_n, samples_per_axis, dtype=np.float32)
    east_axis = np.linspace(min_e, max_e, samples_per_axis, dtype=np.float32)
    terrain = np.zeros((samples_per_axis, samples_per_axis), dtype=np.float32)
    return north_axis, east_axis, terrain


def _save_outputs(output_dir, rows, summary, center_n_ft, center_e_ft, radius_ft, controller_tag):
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
        "progress_s_ft",
        "virtual_speed_fps",
        "contour_error_ft",
        "lag_error_ft",
        "position_error_ft",
        "altitude_error_ft",
        "aileron_cmd",
        "elevator_cmd",
        "rudder_cmd",
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
    ax.set_title(f"{controller_tag.upper()} Circle Tracking")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return csv_path, summary_path, plot_path


def main():
    args = _parse_args()
    config_base_kwargs, config_source, tuned_keys = _load_mppi_config_kwargs(args)

    requested_speed_kts = float(args.target_speed_kts)
    max_turn_speed_kts = _max_turn_feasible_speed_kts(
        radius_ft=float(args.radius_ft),
        roll_max_rad=np.deg2rad(70.0),
    )
    effective_speed_kts = float(min(requested_speed_kts, max_turn_speed_kts))
    speed_was_limited = effective_speed_kts + 1e-9 < requested_speed_kts
    target_speed_fps = float(effective_speed_kts * KTS_TO_FPS)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.output_root) / f"circle_{args.controller}"

    env = JSBSimEnv(render_mode="human" if args.render else None)
    rows = []
    termination_reason = "running"

    try:
        env.simulation.set_property_value("ic/u-fps", target_speed_fps)
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

        nominal_reference = _build_circle_nominal_reference(
            start_n_ft=float(init_state["n_ft"]),
            start_e_ft=float(init_state["e_ft"]),
            center_n_ft=float(center_n_ft),
            center_e_ft=float(center_e_ft),
            radius_ft=float(args.radius_ft),
            direction=str(args.direction),
            target_altitude_ft=float(target_altitude_ft),
            target_speed_fps=float(target_speed_fps),
            max_steps=int(args.max_steps),
            control_hz=CONTROL_HZ,
            points_per_lap=int(args.reference_points_per_lap),
            laps_override=args.reference_laps,
        )
        terrain_north_ft, terrain_east_ft, terrain_elevation_ft = _build_flat_terrain_grid(
            reference_north_ft=np.asarray(nominal_reference["north_ft"], dtype=np.float32),
            reference_east_ft=np.asarray(nominal_reference["east_ft"], dtype=np.float32),
            margin_ft=max(2.0 * float(args.radius_ft), 5000.0),
            samples_per_axis=64,
        )

        controller, config = build_mppi_controller(
            args.controller,
            config_base_kwargs=config_base_kwargs,
            reference_trajectory=nominal_reference,
            terrain_north_samples_ft=terrain_north_ft,
            terrain_east_samples_ft=terrain_east_ft,
            terrain_elevation_ft=terrain_elevation_ft,
        )

        _print_mppi_config(args.controller, config)

        print("Compiling JAX JIT... (this takes a moment)", flush=True)
        _ = controller.get_action(_to_controller_state(init_state))
        controller.reset(seed=int(args.seed))
        print("JIT compilation finished.", flush=True)

        print("Starting contouring-MPPI circle tracking test")
        print(f"Controller: {args.controller}")
        print(f"Config source: {config_source}")
        if tuned_keys:
            print(f"Applied tuned keys: {', '.join(sorted(tuned_keys))}")
        else:
            print("No tuned overrides applied; using base config defaults.")
        print(
            f"Circle: radius_ft={args.radius_ft:.1f}, direction={args.direction}, "
            f"target_speed_kts={requested_speed_kts:.1f}, target_altitude_ft={target_altitude_ft:.1f}, "
            f"reference_laps={int(nominal_reference['circle_laps'])}"
        )
        if speed_was_limited:
            print(
                f"Speed capped for turn feasibility: requested={requested_speed_kts:.1f} kts, "
                f"effective={effective_speed_kts:.1f} kts (radius={args.radius_ft:.1f} ft)"
            )
        print(
            f"{'step':<5} | {'prog_s':<9} | {'contour':<8} | {'lag':<8} | {'pos':<8} | {'alt':<8} | {'v_s':<7}"
        )
        print("-" * 78)

        for step in range(int(args.max_steps)):
            state = _extract_state(env, start_n_m, start_e_m)
            controller_state = _to_controller_state(state)
            action = np.asarray(controller.get_action(controller_state), dtype=np.float32)

            if not np.isfinite(action).all():
                termination_reason = "invalid_action"
                print(f"Invalid action at step {step}: {action}")
                break

            _, _, done, truncated, _ = env.step(action)

            post_state = _extract_state(env, start_n_m, start_e_m)
            tracking = dict(controller.get_tracking_metrics(_to_controller_state(post_state)))

            rows.append(
                {
                    "step": int(step),
                    "n_ft": float(post_state["n_ft"]),
                    "e_ft": float(post_state["e_ft"]),
                    "ref_n_ft": float(tracking.get("reference_north_ft", np.nan)),
                    "ref_e_ft": float(tracking.get("reference_east_ft", np.nan)),
                    "progress_s_ft": float(tracking.get("progress_s_ft", np.nan)),
                    "virtual_speed_fps": float(tracking.get("virtual_speed_fps", np.nan)),
                    "contour_error_ft": float(tracking.get("contour_error_ft", np.nan)),
                    "lag_error_ft": float(tracking.get("lag_error_ft", np.nan)),
                    "position_error_ft": float(tracking.get("position_error_ft", np.nan)),
                    "altitude_error_ft": float(tracking.get("altitude_error_ft", np.nan)),
                    "aileron_cmd": float(action[0]),
                    "elevator_cmd": float(action[1]),
                    "rudder_cmd": float(action[2]),
                    "throttle_cmd": float(action[3]),
                }
            )

            if step % 10 == 0:
                print(
                    f"{step:<5} | {tracking.get('progress_s_ft', np.nan):<9.1f} | "
                    f"{tracking.get('contour_error_ft', np.nan):<8.1f} | {tracking.get('lag_error_ft', np.nan):<8.1f} | "
                    f"{tracking.get('position_error_ft', np.nan):<8.1f} | {tracking.get('altitude_error_ft', np.nan):<8.1f} | "
                    f"{tracking.get('virtual_speed_fps', np.nan):<7.1f}"
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

    contour_err = np.array([row["contour_error_ft"] for row in rows], dtype=np.float64)
    lag_err = np.array([row["lag_error_ft"] for row in rows], dtype=np.float64)
    pos_err = np.array([row["position_error_ft"] for row in rows], dtype=np.float64)
    alt_err = np.array([row["altitude_error_ft"] for row in rows], dtype=np.float64)

    summary = {
        "termination_reason": termination_reason,
        "steps": int(len(rows)),
        "circle": {
            "radius_ft": float(args.radius_ft),
            "direction": str(args.direction),
            "center_n_ft": float(center_n_ft),
            "center_e_ft": float(center_e_ft),
            "reference_laps": int(nominal_reference["circle_laps"]),
            "reference_circumference_ft": float(nominal_reference["circle_circumference_ft"]),
        },
        "tracking_metrics": {
            "mean_contour_error_ft": float(np.nanmean(contour_err)),
            "rms_contour_error_ft": float(np.sqrt(np.nanmean(np.square(contour_err)))),
            "mean_abs_lag_error_ft": float(np.nanmean(np.abs(lag_err))),
            "rms_lag_error_ft": float(np.sqrt(np.nanmean(np.square(lag_err)))),
            "mean_position_error_ft": float(np.nanmean(pos_err)),
            "rms_position_error_ft": float(np.sqrt(np.nanmean(np.square(pos_err)))),
            "mean_abs_altitude_error_ft": float(np.nanmean(np.abs(alt_err))),
        },
        "controller": {
            "tag": str(args.controller),
            "requested_target_speed_kts": float(requested_speed_kts),
            "effective_target_speed_kts": float(effective_speed_kts),
            "max_turn_feasible_speed_kts": float(max_turn_speed_kts),
            "target_altitude_ft": float(target_altitude_ft),
            "config_source": str(config_source),
            "terrain_costs_disabled": True,
            "applied_keys": [str(k) for k in sorted(set(tuned_keys))],
            "config": {
                k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in _config_to_dict(config).items()
            },
        },
    }

    csv_path, summary_path, plot_path = _save_outputs(
        output_dir=Path(output_dir),
        rows=rows,
        summary=summary,
        center_n_ft=center_n_ft,
        center_e_ft=center_e_ft,
        radius_ft=float(args.radius_ft),
        controller_tag=str(args.controller),
    )

    print("\nCircle tracking run complete")
    print(f"Saved trajectory CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(f"RMS contour error (ft): {summary['tracking_metrics']['rms_contour_error_ft']:.2f}")


if __name__ == "__main__":
    main()
