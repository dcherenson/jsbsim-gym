from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsbsim_gym.canyon_env  # Registers JSBSimCanyon-v0
from jsbsim_gym.canyon import DEMCanyon
from jsbsim_gym.mppi_support import f16_kinematics_step_with_load_factors, load_nominal_weights
from jsbsim_gym.nominal_trajectory import load_nominal_initial_conditions_from_dyn

STATE_KEYS = (
    "p_N",
    "p_E",
    "h",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "phi",
    "theta",
    "psi",
    "ny",
    "nz",
)
ANGLE_KEYS = {"phi", "theta", "psi"}

REPO_ROOT = Path(__file__).resolve().parent
DEM_PATH = REPO_ROOT / "data/dem/black-canyon-gunnison_USGS10m.tif"
DEM_BBOX = (38.52, 38.62, -107.78, -107.65)
DEM_START_PIXEL = (1400, 950)

M_TO_FT = 3.28084
DEFAULT_INITIAL_SPEED_KTS = 450.0
DEFAULT_INITIAL_ALTITUDE_FT = 500.0
DEFAULT_INITIAL_HEADING_DEG = None
DEFAULT_INITIAL_ROLL_DEG = None
DEFAULT_INITIAL_PITCH_DEG = None
DEFAULT_INITIAL_ALPHA_DEG = None
DEFAULT_INITIAL_BETA_DEG = None


DEFAULT_NOMINAL_DYN_PATH = REPO_ROOT / "air-racing-optimization/final_results/dyn.asb"
if not DEFAULT_NOMINAL_DYN_PATH.exists():
    DEFAULT_NOMINAL_DYN_PATH = None


def _fraction_0_to_1(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a floating-point value in [0, 1], got {value!r}.") from exc
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise argparse.ArgumentTypeError(f"Expected a floating-point value in [0, 1], got {value!r}.")
    return fraction


def _to_mppi_state(env, state, altitude_ref_ft):
    p_n_ft = float(state["p_N"])
    p_e_ft = float(state["p_E"])

    canyon = getattr(env.unwrapped, "canyon", None)
    if canyon is not None and hasattr(canyon, "get_local_from_latlon"):
        try:
            lat_deg = float(env.unwrapped.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(env.unwrapped.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = canyon.get_local_from_latlon(lat_deg, lon_deg)
            p_n_ft = float(local_north_ft)
            p_e_ft = float(local_east_ft)
        except Exception:
            pass

    return {
        "p_N": p_n_ft,
        "p_E": p_e_ft,
        "h": float(state["h"] - altitude_ref_ft),
        "u": float(state["u"]),
        "v": float(state["v"]),
        "w": float(state["w"]),
        "p": float(state["p"]),
        "q": float(state["q"]),
        "r": float(state["r"]),
        "phi": float(state["phi"]),
        "theta": float(state["theta"]),
        "psi": float(state["psi"]),
        "ny": float(state.get("ny", 0.0)),
        "nz": float(state.get("nz", 1.0)),
    }


def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _extract_state_columns(rows: list[dict[str, str]], prefix: str) -> np.ndarray | None:
    needed = [f"{prefix}_{key}" for key in STATE_KEYS]
    if not rows or any(col not in rows[0] for col in needed):
        return None
    out = np.zeros((len(rows), len(STATE_KEYS)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, key in enumerate(STATE_KEYS):
            out[i, j] = float(row[f"{prefix}_{key}"])
    return out


def _load_tracking_csv(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No rows found in {path}.")

    required_columns = [
        "step",
        "time_s",
        "aileron_cmd",
        "elevator_cmd",
        "rudder_cmd",
        "throttle_cmd",
    ]
    missing = [col for col in required_columns if col not in rows[0]]
    if missing:
        raise ValueError("Tracking CSV missing required columns: " + ", ".join(missing))

    step = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int32)
    time_s = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    actions = np.asarray(
        [
            [
                float(row["aileron_cmd"]),
                float(row["elevator_cmd"]),
                float(row["rudder_cmd"]),
                float(row["throttle_cmd"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    logged_pre = _extract_state_columns(rows, "pre")
    logged_post = _extract_state_columns(rows, "post")

    return {
        "rows": rows,
        "step": step,
        "time_s": time_s,
        "actions": actions,
        "logged_pre": logged_pre,
        "logged_post": logged_post,
    }


def _build_initial_conditions(nominal_dyn_path: Path | None, nominal_start_fraction: float) -> dict:
    initial = {
        "dem_start_pixel": DEM_START_PIXEL,
        "initial_speed_kts": float(DEFAULT_INITIAL_SPEED_KTS),
        "initial_altitude_ft": float(DEFAULT_INITIAL_ALTITUDE_FT),
        "initial_heading_deg": DEFAULT_INITIAL_HEADING_DEG,
        "initial_roll_deg": DEFAULT_INITIAL_ROLL_DEG,
        "initial_pitch_deg": DEFAULT_INITIAL_PITCH_DEG,
        "initial_alpha_deg": DEFAULT_INITIAL_ALPHA_DEG,
        "initial_beta_deg": DEFAULT_INITIAL_BETA_DEG,
        "nominal_source": None,
    }

    if nominal_dyn_path is None:
        return initial

    nominal_canyon = DEMCanyon(
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
        dem_start_pixel=DEM_START_PIXEL,
    )
    nominal_initial_conditions = load_nominal_initial_conditions_from_dyn(
        nominal_dyn_path,
        canyon=nominal_canyon,
        progress_fraction=nominal_start_fraction,
    )
    initial["dem_start_pixel"] = tuple(nominal_initial_conditions["start_pixel"])
    initial["initial_speed_kts"] = float(nominal_initial_conditions["speed_kts"])
    initial["initial_altitude_ft"] = float(nominal_initial_conditions["entry_altitude_ft"])
    initial["initial_heading_deg"] = float(nominal_initial_conditions["heading_deg"])
    initial["initial_roll_deg"] = float(nominal_initial_conditions["roll_deg"])
    initial["initial_pitch_deg"] = float(nominal_initial_conditions["pitch_deg"])
    initial["initial_alpha_deg"] = float(nominal_initial_conditions["alpha_deg"])
    initial["initial_beta_deg"] = float(nominal_initial_conditions["beta_deg"])
    initial["nominal_source"] = {
        "dyn_path": str(nominal_dyn_path),
        "progress_fraction": float(nominal_start_fraction),
        "sample_index": int(nominal_initial_conditions["sample_index"]),
        "sample_count": int(nominal_initial_conditions["sample_count"]),
    }
    return initial


def _replay_jsbsim(
    actions: np.ndarray,
    *,
    nominal_dyn_path: Path | None,
    nominal_start_fraction: float,
    seed: int,
    wind_sigma: float,
):
    initial = _build_initial_conditions(nominal_dyn_path=nominal_dyn_path, nominal_start_fraction=nominal_start_fraction)

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
        dem_start_pixel=initial["dem_start_pixel"],
        dem_start_heading_mode="follow_canyon",
        dem_start_heading_deg=initial["initial_heading_deg"],
        dem_render_mesh=True,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=30.0,
        wall_visual_offset_ft=40.0,
        wall_radius_ft=8.0,
        wall_height_ft=500.0,
        target_altitude_ft=500.0,
        entry_altitude_ft=initial["initial_altitude_ft"],
        min_altitude_ft=-500.0,
        max_altitude_ft=3000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=10.0,
        entry_speed_kts=initial["initial_speed_kts"],
        entry_roll_deg=initial["initial_roll_deg"],
        entry_pitch_deg=initial["initial_pitch_deg"],
        entry_alpha_deg=initial["initial_alpha_deg"],
        entry_beta_deg=initial["initial_beta_deg"],
        wind_sigma=float(wind_sigma),
        canyon_span_ft=9000.0,
        canyon_segment_spacing_ft=12.0,
    )

    pre_states = []
    post_states = []
    terminated_early = False
    termination_reason = "running"

    try:
        env.reset(seed=int(seed))
        state = env.unwrapped.get_full_state_dict()
        altitude_ref_ft = float(getattr(env.unwrapped, "dem_start_elev_ft", 0.0))

        for action in actions:
            controller_pre = _to_mppi_state(env, state, altitude_ref_ft)
            pre_states.append([float(controller_pre[k]) for k in STATE_KEYS])

            _, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            state = env.unwrapped.get_full_state_dict()
            controller_post = _to_mppi_state(env, state, altitude_ref_ft)
            post_states.append([float(controller_post[k]) for k in STATE_KEYS])

            if terminated or truncated:
                terminated_early = True
                termination_reason = str(info.get("termination_reason", "terminated" if terminated else "truncated"))
                break
    finally:
        env.close()

    return {
        "pre_states": np.asarray(pre_states, dtype=np.float32),
        "post_states": np.asarray(post_states, dtype=np.float32),
        "terminated_early": bool(terminated_early),
        "termination_reason": termination_reason,
        "initial": initial,
    }


def _compute_errors(true_states: np.ndarray, pred_states: np.ndarray) -> np.ndarray:
    err = pred_states - true_states
    for idx, key in enumerate(STATE_KEYS):
        if key in ANGLE_KEYS:
            err[:, idx] = _angle_diff_rad(pred_states[:, idx], true_states[:, idx])
    return err


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _norm3(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, :3], axis=1)


def _alpha_deg_from_state_seq(state_seq: np.ndarray) -> np.ndarray:
    u = np.asarray(state_seq[:, 3], dtype=np.float64)
    w = np.asarray(state_seq[:, 5], dtype=np.float64)
    return np.rad2deg(np.arctan2(w, np.maximum(u, 1.0)))


def _beta_deg_from_state_seq(state_seq: np.ndarray) -> np.ndarray:
    u = np.asarray(state_seq[:, 3], dtype=np.float64)
    v = np.asarray(state_seq[:, 4], dtype=np.float64)
    w = np.asarray(state_seq[:, 5], dtype=np.float64)
    vmag = np.sqrt(u * u + v * v + w * w)
    return np.rad2deg(np.arcsin(np.clip(v / np.maximum(vmag, 1.0), -1.0, 1.0)))


def run_comparison(
    tracking_csv: Path,
    output_dir: Path,
    prefix: str,
    *,
    nominal_dyn_path: Path | None,
    nominal_start_fraction: float,
    seed: int,
    wind_sigma: float,
):
    data = _load_tracking_csv(tracking_csv)
    step = data["step"]
    time_s = data["time_s"]
    actions = data["actions"]

    replay = _replay_jsbsim(
        actions,
        nominal_dyn_path=nominal_dyn_path,
        nominal_start_fraction=nominal_start_fraction,
        seed=seed,
        wind_sigma=wind_sigma,
    )
    pre_states_true = replay["pre_states"]
    post_states_true = replay["post_states"]

    if post_states_true.shape[0] < 1:
        raise RuntimeError("JSBSim replay produced no steps.")

    n = int(post_states_true.shape[0])
    actions = actions[:n]
    step = step[:n]
    time_s = time_s[:n]

    W, B, poly_powers, throttle_force_coeffs = load_nominal_weights()
    W = jnp.asarray(W, dtype=jnp.float32)
    B = jnp.asarray(B, dtype=jnp.float32)
    poly_powers = jnp.asarray(poly_powers, dtype=jnp.int32)
    throttle_force_coeffs = jnp.asarray(throttle_force_coeffs, dtype=jnp.float32)

    post_states_poly_rollout = np.zeros_like(post_states_true)

    rollout_state = pre_states_true[0].copy()
    for i in range(n):
        rollout_state = np.asarray(
            f16_kinematics_step_with_load_factors(
                jnp.asarray(rollout_state, dtype=jnp.float32),
                jnp.asarray(actions[i], dtype=jnp.float32),
                W,
                B,
                poly_powers,
                throttle_force_coeffs,
            ),
            dtype=np.float32,
        )
        post_states_poly_rollout[i] = rollout_state

    err_rollout = _compute_errors(post_states_true, post_states_poly_rollout)
    rollout_pos_err_ft = _norm3(err_rollout)

    summary = {
        "tracking_csv": str(tracking_csv),
        "num_actions_from_tracking": int(data["actions"].shape[0]),
        "num_replayed_steps": int(n),
        "jsbsim_replay_terminated_early": bool(replay["terminated_early"]),
        "jsbsim_replay_termination_reason": str(replay["termination_reason"]),
        "replay_seed": int(seed),
        "replay_wind_sigma": float(wind_sigma),
        "nominal_dyn_path": str(nominal_dyn_path) if nominal_dyn_path is not None else None,
        "nominal_start_fraction": float(nominal_start_fraction),
        "initial_conditions": replay["initial"],
        "rollout_position_rmse_ft": _rmse(rollout_pos_err_ft),
        "rollout_position_mae_ft": _mae(rollout_pos_err_ft),
        "state_metrics": {},
    }

    if data["logged_pre"] is not None and data["logged_pre"].shape[0] >= 1:
        init_diff = _compute_errors(data["logged_pre"][:1], pre_states_true[:1])[0]
        summary["logged_vs_replay_initial_state_diff"] = {
            key: float(init_diff[idx]) for idx, key in enumerate(STATE_KEYS)
        }

    if data["logged_post"] is not None and data["logged_post"].shape[0] >= n:
        replay_vs_logged = _compute_errors(data["logged_post"][:n], post_states_true)
        summary["logged_vs_replay_position_rmse_ft"] = _rmse(_norm3(replay_vs_logged))

    for idx, key in enumerate(STATE_KEYS):
        key_scale = "rad" if key in ANGLE_KEYS else "native"
        summary["state_metrics"][key] = {
            "units": key_scale,
            "rollout_rmse": _rmse(err_rollout[:, idx]),
            "rollout_mae": _mae(err_rollout[:, idx]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}_summary.json"
    per_step_csv_path = output_dir / f"{prefix}_per_step.csv"
    plot_path = output_dir / f"{prefix}_comparison.png"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = ["step", "time_s", "aileron_cmd", "elevator_cmd", "rudder_cmd", "throttle_cmd"]
    for key in STATE_KEYS:
        fieldnames.extend(
            [
                f"jsbsim_replay_post_{key}",
                f"poly_rollout_post_{key}",
                f"rollout_error_{key}",
            ]
        )

    with per_step_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row = {
                "step": int(step[i]),
                "time_s": float(time_s[i]),
                "aileron_cmd": float(actions[i, 0]),
                "elevator_cmd": float(actions[i, 1]),
                "rudder_cmd": float(actions[i, 2]),
                "throttle_cmd": float(actions[i, 3]),
            }
            for j, key in enumerate(STATE_KEYS):
                row[f"jsbsim_replay_post_{key}"] = float(post_states_true[i, j])
                row[f"poly_rollout_post_{key}"] = float(post_states_poly_rollout[i, j])
                row[f"rollout_error_{key}"] = float(err_rollout[i, j])
            writer.writerow(row)

    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True, constrained_layout=True)

    alpha_true_deg = _alpha_deg_from_state_seq(post_states_true)
    alpha_poly_deg = _alpha_deg_from_state_seq(post_states_poly_rollout)
    beta_true_deg = _beta_deg_from_state_seq(post_states_true)
    beta_poly_deg = _beta_deg_from_state_seq(post_states_poly_rollout)
    roll_true_deg = np.rad2deg(np.asarray(post_states_true[:, STATE_KEYS.index("phi")], dtype=np.float64))
    roll_poly_deg = np.rad2deg(np.asarray(post_states_poly_rollout[:, STATE_KEYS.index("phi")], dtype=np.float64))
    nz_true = np.asarray(post_states_true[:, STATE_KEYS.index("nz")], dtype=np.float64)
    nz_poly = np.asarray(post_states_poly_rollout[:, STATE_KEYS.index("nz")], dtype=np.float64)
    speed_true_fps = np.linalg.norm(np.asarray(post_states_true[:, 3:6], dtype=np.float64), axis=1)
    speed_poly_fps = np.linalg.norm(np.asarray(post_states_poly_rollout[:, 3:6], dtype=np.float64), axis=1)

    axs[0, 0].plot(time_s, nz_true, color="black", linewidth=1.8, label="JSBSim replay")
    axs[0, 0].plot(time_s, nz_poly, color="tab:red", linewidth=1.4, alpha=0.9, label="poly rollout")
    axs[0, 0].set_title("N_z Rollout")
    axs[0, 0].set_ylabel("g")
    axs[0, 0].grid(True, alpha=0.25)
    axs[0, 0].legend(loc="best")

    axs[0, 1].plot(time_s, alpha_true_deg, color="black", linewidth=1.8, label="JSBSim replay")
    axs[0, 1].plot(time_s, alpha_poly_deg, color="tab:red", linewidth=1.4, alpha=0.9, label="poly rollout")
    axs[0, 1].set_title("Alpha Rollout")
    axs[0, 1].set_ylabel("deg")
    axs[0, 1].grid(True, alpha=0.25)
    axs[0, 1].legend(loc="best")

    axs[1, 0].plot(time_s, beta_true_deg, color="black", linewidth=1.8, label="JSBSim replay")
    axs[1, 0].plot(time_s, beta_poly_deg, color="tab:red", linewidth=1.4, alpha=0.9, label="poly rollout")
    axs[1, 0].set_title("Beta Rollout")
    axs[1, 0].set_ylabel("deg")
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].grid(True, alpha=0.25)
    axs[1, 0].legend(loc="best")

    axs[1, 1].plot(time_s, roll_true_deg, color="black", linewidth=1.8, label="JSBSim replay")
    axs[1, 1].plot(time_s, roll_poly_deg, color="tab:red", linewidth=1.4, alpha=0.9, label="poly rollout")
    axs[1, 1].set_title("Roll Angle Rollout")
    axs[1, 1].set_ylabel("deg")
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].grid(True, alpha=0.25)
    axs[1, 1].legend(loc="best")

    axs[2, 0].plot(time_s, speed_true_fps, color="black", linewidth=1.8, label="JSBSim replay")
    axs[2, 0].plot(time_s, speed_poly_fps, color="tab:red", linewidth=1.4, alpha=0.9, label="poly rollout")
    axs[2, 0].set_title("Speed Rollout")
    axs[2, 0].set_ylabel("ft/s")
    axs[2, 0].set_xlabel("time (s)")
    axs[2, 0].grid(True, alpha=0.25)
    axs[2, 0].legend(loc="best")

    axs[2, 1].axis("off")

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary_path, per_step_csv_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay JSBSim with the exact logged MPPI action sequence, then compare polynomial dynamics to that replay."
        )
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        required=True,
        help="Path to canyon_*_tracking_diagnostics.csv produced by run_scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to tracking CSV directory.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mppi_jsbsim_vs_poly",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--nominal-dyn-path",
        type=Path,
        default=DEFAULT_NOMINAL_DYN_PATH,
        help=(
            "Nominal dyn.asb path used for replay initial conditions. "
            "Defaults to air-racing-optimization/final_results/dyn.asb when present."
        ),
    )
    parser.add_argument(
        "--nominal-start-fraction",
        type=_fraction_0_to_1,
        default=0.0,
        help="Replay start fraction along the nominal dyn trajectory in [0,1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Environment seed used for replay.",
    )
    parser.add_argument(
        "--wind-sigma",
        type=float,
        default=0.0,
        help="Wind sigma for replay environment (should match original run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_csv = args.tracking_csv.expanduser().resolve()
    if not tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else tracking_csv.parent

    nominal_dyn_path = None
    if args.nominal_dyn_path is not None:
        nominal_dyn_path = args.nominal_dyn_path.expanduser().resolve()
        if not nominal_dyn_path.exists():
            raise FileNotFoundError(f"nominal dyn.asb not found: {nominal_dyn_path}")

    summary_path, per_step_csv_path, plot_path = run_comparison(
        tracking_csv=tracking_csv,
        output_dir=output_dir,
        prefix=str(args.prefix),
        nominal_dyn_path=nominal_dyn_path,
        nominal_start_fraction=float(args.nominal_start_fraction),
        seed=int(args.seed),
        wind_sigma=float(args.wind_sigma),
    )
    print(f"Saved comparison summary JSON: {summary_path}")
    print(f"Saved per-step comparison CSV: {per_step_csv_path}")
    print(f"Saved comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
