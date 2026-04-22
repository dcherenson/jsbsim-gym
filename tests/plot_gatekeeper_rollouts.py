from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TMP_CACHE_ROOT = Path("/tmp/jsbsim_gym_poly_rollouts")
_TMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_TMP_CACHE_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
(_TMP_CACHE_ROOT / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE_ROOT / "xdg-cache"))

import jax
from jax import lax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jsbsim_gym.env import JSBSimEnv, RADIUS
from jsbsim_gym.mppi_jax import f16_kinematics_step_with_load_factors, load_nominal_weights
from jsbsim_gym.uncertainty import RuntimeUncertaintySampler, sample_empirical_jax


M_TO_FT = 3.28084
RADIUS_FT = RADIUS * M_TO_FT
KTS_TO_FPS = 1.68781
AIR_DENSITY_SLUG_FT3 = 0.0023769
STANDARD_SPEED_OF_SOUND_FPS = 1116.45
DEFAULT_MASS_LBS = 17400.0 + 230.0 + 2000.0
DEFAULT_MASS_SLUGS = DEFAULT_MASS_LBS / 32.174
ROLL_OUT_STEPS = 40

UNCERTAINTY_ARTIFACT_PATH = REPO_ROOT / "f16_uncertainty_model.pkl"
NOMINAL_WEIGHTS_PATH = REPO_ROOT / "jsbsim_gym" / "nominal_coeff_weights.npz"

TRUTH_COLOR = "black"
NOMINAL_COLOR = "tab:blue"
SAMPLED_COLOR = "tab:orange"
SAMPLED_ALPHA = 0.22


@dataclass(frozen=True)
class ManeuverCase:
    slug: str
    label: str
    maneuver: str
    entry_speed_kts: float
    entry_altitude_ft: float
    entry_roll_deg: float = 0.0
    entry_pitch_deg: float = 0.0
    entry_heading_deg: float = 0.0
    entry_alpha_deg: float | None = None
    entry_beta_deg: float | None = None


DEFAULT_CASES: tuple[ManeuverCase, ...] = (
    ManeuverCase(
        slug="pitch_doublet",
        label="Pitch Doublet",
        maneuver="pitch_doublet",
        entry_speed_kts=450.0,
        entry_altitude_ft=5000.0,
    ),
    ManeuverCase(
        slug="roll_reversal_fast",
        label="Fast Roll Reversal",
        maneuver="roll_reversal",
        entry_speed_kts=520.0,
        entry_altitude_ft=8000.0,
        entry_roll_deg=10.0,
    ),
    ManeuverCase(
        slug="rudder_pulse_slow",
        label="Slow Rudder Pulse",
        maneuver="rudder_pulse",
        entry_speed_kts=320.0,
        entry_altitude_ft=6500.0,
        entry_pitch_deg=2.0,
    ),
    ManeuverCase(
        slug="combined_turn",
        label="Combined Turn Entry",
        maneuver="combined_turn",
        entry_speed_kts=430.0,
        entry_altitude_ft=7000.0,
        entry_roll_deg=-15.0,
        entry_pitch_deg=-1.5,
    ),
)

CASE_BY_SLUG = {case.slug: case for case in DEFAULT_CASES}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare empty-airspace JSBSim truth against the polynomial model with sampled uncertainty.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASE_BY_SLUG.keys()),
        default=None,
        help="Optional case selector. Repeat to run a subset of maneuver cases.",
    )
    parser.add_argument(
        "--num-sampled-rollouts",
        type=int,
        default=12,
        help="Number of sampled-uncertainty model rollouts to overlay per case.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=7,
        help="Base seed for sampled uncertainty rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "poly_model_rollout_compare",
        help="Directory where plots and summary JSON are written.",
    )
    return parser.parse_args()


def _ensure_required_files():
    required_paths = (
        UNCERTAINTY_ARTIFACT_PATH,
        NOMINAL_WEIGHTS_PATH,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts for polynomial rollout comparison:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def _selected_cases(case_slugs: list[str] | None):
    if not case_slugs:
        return list(DEFAULT_CASES)
    return [CASE_BY_SLUG[slug] for slug in case_slugs]


def _make_action_plan(segments, *, base_action=(0.0, 0.0, 0.0, 0.55), horizon=ROLL_OUT_STEPS):
    plan = np.repeat(np.asarray(base_action, dtype=np.float32)[None, :], int(horizon), axis=0)
    for start, stop, action in segments:
        lo = max(int(start), 0)
        hi = min(int(stop), int(horizon))
        if lo >= hi:
            continue
        plan[lo:hi] = np.asarray(action, dtype=np.float32)
    return np.clip(plan, np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def _build_action_plan(case: ManeuverCase):
    if case.maneuver == "pitch_doublet":
        return _make_action_plan(
            [
                (4, 10, (0.0, 0.42, 0.0, 0.58)),
                (10, 16, (0.0, -0.42, 0.0, 0.58)),
                (16, 26, (0.0, 0.18, 0.0, 0.56)),
            ]
        )
    if case.maneuver == "roll_reversal":
        return _make_action_plan(
            [
                (3, 11, (0.60, 0.0, 0.0, 0.56)),
                (11, 20, (-0.70, 0.0, 0.0, 0.56)),
                (20, 28, (0.40, 0.0, 0.0, 0.54)),
            ]
        )
    if case.maneuver == "rudder_pulse":
        return _make_action_plan(
            [
                (5, 11, (0.0, 0.0, 0.45, 0.54)),
                (11, 17, (0.0, 0.0, -0.45, 0.54)),
                (17, 26, (0.12, 0.0, 0.25, 0.55)),
            ]
        )
    if case.maneuver == "combined_turn":
        return _make_action_plan(
            [
                (3, 10, (0.35, 0.20, 0.0, 0.68)),
                (10, 24, (0.45, 0.12, 0.08, 0.66)),
                (24, 34, (-0.28, -0.10, -0.08, 0.52)),
            ]
        )
    raise ValueError(f"Unknown maneuver id: {case.maneuver}")


def _make_env():
    return JSBSimEnv(render_mode=None)


def _set_initial_conditions(env: JSBSimEnv, case: ManeuverCase):
    sim = env.simulation
    sim.set_property_value("ic/h-sl-ft", float(case.entry_altitude_ft))
    sim.set_property_value("ic/vt-fps", float(case.entry_speed_kts) * KTS_TO_FPS)
    sim.set_property_value("ic/u-fps", float(case.entry_speed_kts) * KTS_TO_FPS)
    sim.set_property_value("ic/phi-deg", float(case.entry_roll_deg))
    sim.set_property_value("ic/theta-deg", float(case.entry_pitch_deg))
    sim.set_property_value("ic/psi-true-deg", float(case.entry_heading_deg))
    if case.entry_alpha_deg is not None:
        sim.set_property_value("ic/alpha-deg", float(case.entry_alpha_deg))
    if case.entry_beta_deg is not None:
        sim.set_property_value("ic/beta-deg", float(case.entry_beta_deg))


def _get_full_state_dict(env: JSBSimEnv):
    sim = env.simulation
    lat_rad = float(sim.get_property_value("position/lat-gc-rad"))
    lon_rad = float(sim.get_property_value("position/long-gc-rad"))
    h_ft = float(sim.get_property_value("position/h-sl-ft"))
    u = float(sim.get_property_value("velocities/u-fps"))
    v = float(sim.get_property_value("velocities/v-fps"))
    w = float(sim.get_property_value("velocities/w-fps"))
    v_total = float(sim.get_property_value("velocities/vt-fps"))
    mach = float(sim.get_property_value("velocities/mach"))
    alpha = float(sim.get_property_value("aero/alpha-rad"))
    beta = float(sim.get_property_value("aero/beta-rad"))
    phi = float(sim.get_property_value("attitude/phi-rad"))
    theta = float(sim.get_property_value("attitude/theta-rad"))
    psi = float(sim.get_property_value("attitude/psi-rad"))
    p = float(sim.get_property_value("velocities/p-rad_sec"))
    q = float(sim.get_property_value("velocities/q-rad_sec"))
    r = float(sim.get_property_value("velocities/r-rad_sec"))
    qbar = float(sim.get_property_value("aero/qbar-psf"))
    ny = float(sim.get_property_value("accelerations/Ny"))
    nz = float(sim.get_property_value("accelerations/Nz"))
    mass_slugs = float(sim.get_property_value("inertia/mass-slugs"))
    if not np.isfinite(mass_slugs) or mass_slugs <= 0.0:
        mass_slugs = DEFAULT_MASS_SLUGS

    return {
        "p_N": lat_rad * RADIUS_FT,
        "p_E": lon_rad * RADIUS_FT,
        "h": h_ft,
        "V": v_total,
        "mach": mach,
        "u": u,
        "v": v,
        "w": w,
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
        "theta": theta,
        "psi": psi,
        "p": p,
        "q": q,
        "r": r,
        "qbar": qbar,
        "ny": ny,
        "nz": nz,
        "mass_slugs": mass_slugs,
    }


def _step_truth_env(env: JSBSimEnv, action: np.ndarray):
    roll_cmd, pitch_cmd, yaw_cmd, throttle = np.asarray(action, dtype=np.float32)
    sim = env.simulation
    sim.set_property_value("fcs/aileron-cmd-norm", float(roll_cmd))
    sim.set_property_value("fcs/elevator-cmd-norm", float(pitch_cmd))
    sim.set_property_value("fcs/rudder-cmd-norm", float(yaw_cmd))
    sim.set_property_value("fcs/throttle-cmd-norm", float(throttle))

    for _ in range(int(env.down_sample)):
        sim.set_property_value("propulsion/tank/contents-lbs", 1000.0)
        sim.set_property_value("propulsion/tank[1]/contents-lbs", 1000.0)
        sim.set_property_value("gear/gear-cmd-norm", 0.0)
        sim.set_property_value("gear/gear-pos-norm", 0.0)
        sim.run()


def _state_dict_to_flat14(state_dict):
    return np.asarray(
        [
            float(state_dict["p_N"]),
            float(state_dict["p_E"]),
            float(state_dict["h"]),
            float(state_dict["u"]),
            float(state_dict["v"]),
            float(state_dict["w"]),
            float(state_dict["p"]),
            float(state_dict["q"]),
            float(state_dict["r"]),
            float(state_dict["phi"]),
            float(state_dict["theta"]),
            float(state_dict["psi"]),
            float(state_dict.get("ny", 0.0)),
            float(state_dict.get("nz", 1.0)),
        ],
        dtype=np.float32,
    )


def _trace_from_state_traj(state_traj: np.ndarray):
    state_traj = np.asarray(state_traj, dtype=np.float32)
    speed_fps = np.linalg.norm(state_traj[:, 3:6], axis=1)
    return {
        "north_ft": state_traj[:, 0],
        "east_ft": state_traj[:, 1],
        "altitude_ft": state_traj[:, 2],
        "speed_fps": speed_fps,
        "phi_deg": np.degrees(state_traj[:, 9]),
        "theta_deg": np.degrees(state_traj[:, 10]),
        "psi_deg": np.degrees(state_traj[:, 11]),
        "ny_g": state_traj[:, 12],
        "nz_g": state_traj[:, 13],
    }


def _rollout_truth(case: ManeuverCase, action_plan: np.ndarray):
    env = _make_env()
    try:
        _set_initial_conditions(env, case)
        env.reset(seed=0)
        state_rows = [_state_dict_to_flat14(_get_full_state_dict(env))]
        for action in np.asarray(action_plan, dtype=np.float32):
            _step_truth_env(env, action)
            state_rows.append(_state_dict_to_flat14(_get_full_state_dict(env)))
        state_traj = np.asarray(state_rows, dtype=np.float32)
        return _trace_from_state_traj(state_traj), state_traj[0]
    finally:
        env.close()


def _build_nominal_rollout_fn(W, B, poly_powers):
    def rollout_fn(x0, action_plan):
        def step_fn(state, action):
            next_state = f16_kinematics_step_with_load_factors(state, action, W, B, poly_powers)
            return next_state, state

        final_state, state_steps = lax.scan(step_fn, x0, action_plan, length=action_plan.shape[0])
        return jnp.concatenate([state_steps, final_state[None, :]], axis=0)

    return jax.jit(rollout_fn)


def _build_sampled_rollout_fn(W, B, poly_powers, uncertainty_sampler: RuntimeUncertaintySampler):
    active_feature_names = uncertainty_sampler.configure_active_features()
    active_feature_set = set(active_feature_names)
    uncertainty_data = uncertainty_sampler.to_jax()
    width_ft_default = float(np.median(uncertainty_sampler.dataset["canyon_width"].to_numpy(dtype=np.float32, copy=True)))
    width_grad_default = 0.0

    def feature_fn(state, action, prev_action):
        u = state[3]
        v = state[4]
        w = state[5]
        v_sq = u * u + v * v + w * w
        v_total = jnp.sqrt(jnp.maximum(v_sq, 1.0))

        feature_map = {
            "p": state[6],
            "q": state[7],
            "r": state[8],
            "delta_t": action[3],
            "delta_e": action[1],
            "delta_a": action[0],
            "delta_r": action[2],
            "prev_delta_t": prev_action[3],
            "prev_delta_e": prev_action[1],
            "prev_delta_a": prev_action[0],
            "prev_delta_r": prev_action[2],
            "canyon_width": jnp.asarray(width_ft_default, dtype=jnp.float32),
            "canyon_width_grad": jnp.asarray(width_grad_default, dtype=jnp.float32),
        }
        if "alpha" in active_feature_set:
            feature_map["alpha"] = jnp.arctan2(w, jnp.maximum(u, 1.0))
        if "beta" in active_feature_set:
            feature_map["beta"] = jnp.arcsin(jnp.clip(v / v_total, -1.0, 1.0))
        if "mach" in active_feature_set:
            feature_map["mach"] = v_total / STANDARD_SPEED_OF_SOUND_FPS
        if "qbar" in active_feature_set:
            feature_map["qbar"] = jnp.maximum(0.5 * AIR_DENSITY_SLUG_FT3 * v_sq, 1.0)
        return jnp.asarray([feature_map[name] for name in active_feature_names], dtype=jnp.float32)

    def rollout_fn(x0, action_plan, rng_key):
        def step_fn(carry, subkey):
            state, step_idx, prev_action = carry
            action = action_plan[step_idx]
            features = feature_fn(state, action, prev_action)
            noise = sample_empirical_jax(features, 0, subkey, uncertainty_data)
            next_state = f16_kinematics_step_with_load_factors(state, action, W, B + noise, poly_powers)
            return (next_state, step_idx + 1, action), (state, noise)

        keys = jax.random.split(rng_key, int(action_plan.shape[0]))
        final_carry, (state_steps, noise_steps) = lax.scan(
            step_fn,
            (x0, jnp.asarray(0, dtype=jnp.int32), action_plan[0]),
            keys,
            length=action_plan.shape[0],
        )
        final_state = final_carry[0]
        state_traj = jnp.concatenate([state_steps, final_state[None, :]], axis=0)
        return state_traj, noise_steps

    return jax.jit(rollout_fn), width_ft_default, width_grad_default


def _compute_metrics(truth_trace, nominal_trace, sampled_traces):
    sampled_alt = np.stack([trace["altitude_ft"] for trace in sampled_traces], axis=0)
    sampled_phi = np.stack([trace["phi_deg"] for trace in sampled_traces], axis=0)
    sampled_theta = np.stack([trace["theta_deg"] for trace in sampled_traces], axis=0)
    sampled_speed = np.stack([trace["speed_fps"] for trace in sampled_traces], axis=0)

    sampled_mean = {
        "altitude_ft": np.mean(sampled_alt, axis=0),
        "phi_deg": np.mean(sampled_phi, axis=0),
        "theta_deg": np.mean(sampled_theta, axis=0),
        "speed_fps": np.mean(sampled_speed, axis=0),
        "north_ft": np.mean(np.stack([trace["north_ft"] for trace in sampled_traces], axis=0), axis=0),
        "east_ft": np.mean(np.stack([trace["east_ft"] for trace in sampled_traces], axis=0), axis=0),
    }

    truth_terminal = np.array(
        [
            truth_trace["north_ft"][-1],
            truth_trace["east_ft"][-1],
            truth_trace["altitude_ft"][-1],
        ],
        dtype=np.float32,
    )
    nominal_terminal = np.array(
        [
            nominal_trace["north_ft"][-1],
            nominal_trace["east_ft"][-1],
            nominal_trace["altitude_ft"][-1],
        ],
        dtype=np.float32,
    )
    sampled_terminal = np.array(
        [
            sampled_mean["north_ft"][-1],
            sampled_mean["east_ft"][-1],
            sampled_mean["altitude_ft"][-1],
        ],
        dtype=np.float32,
    )

    return {
        "nominal_terminal_position_error_ft": float(np.linalg.norm(nominal_terminal[:2] - truth_terminal[:2])),
        "sampled_mean_terminal_position_error_ft": float(np.linalg.norm(sampled_terminal[:2] - truth_terminal[:2])),
        "nominal_terminal_altitude_error_ft": float(abs(nominal_terminal[2] - truth_terminal[2])),
        "sampled_mean_terminal_altitude_error_ft": float(abs(sampled_terminal[2] - truth_terminal[2])),
        "nominal_altitude_rmse_ft": float(np.sqrt(np.mean((nominal_trace["altitude_ft"] - truth_trace["altitude_ft"]) ** 2))),
        "sampled_mean_altitude_rmse_ft": float(np.sqrt(np.mean((sampled_mean["altitude_ft"] - truth_trace["altitude_ft"]) ** 2))),
        "nominal_phi_rmse_deg": float(np.sqrt(np.mean((nominal_trace["phi_deg"] - truth_trace["phi_deg"]) ** 2))),
        "sampled_mean_phi_rmse_deg": float(np.sqrt(np.mean((sampled_mean["phi_deg"] - truth_trace["phi_deg"]) ** 2))),
        "nominal_theta_rmse_deg": float(np.sqrt(np.mean((nominal_trace["theta_deg"] - truth_trace["theta_deg"]) ** 2))),
        "sampled_mean_theta_rmse_deg": float(np.sqrt(np.mean((sampled_mean["theta_deg"] - truth_trace["theta_deg"]) ** 2))),
        "nominal_speed_rmse_fps": float(np.sqrt(np.mean((nominal_trace["speed_fps"] - truth_trace["speed_fps"]) ** 2))),
        "sampled_mean_speed_rmse_fps": float(np.sqrt(np.mean((sampled_mean["speed_fps"] - truth_trace["speed_fps"]) ** 2))),
    }


def _plot_comparison(case_results, output_path: Path):
    n_cases = len(case_results)
    fig, axes = plt.subplots(n_cases, 4, figsize=(20, 4.6 * n_cases), squeeze=False, constrained_layout=True)
    steps = np.arange(ROLL_OUT_STEPS + 1, dtype=np.int32)

    for row_idx, result in enumerate(case_results):
        case = result["case"]
        truth = result["truth_trace"]
        nominal = result["nominal_trace"]
        sampled = result["sampled_traces"]

        panels = (
            ("north_ft", "east_ft", "Plan View", "North (ft)", "East (ft)"),
            ("altitude_ft", None, "Altitude", "Step", "Altitude (ft)"),
            ("phi_deg", None, "Bank Angle", "Step", "Phi (deg)"),
            ("theta_deg", None, "Pitch Angle", "Step", "Theta (deg)"),
        )

        for col_idx, (key_x, key_y, title, xlabel, ylabel) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            if key_y is None:
                for trace in sampled:
                    ax.plot(steps, trace[key_x], color=SAMPLED_COLOR, linewidth=1.0, alpha=SAMPLED_ALPHA)
                ax.plot(steps, truth[key_x], color=TRUTH_COLOR, linewidth=2.4, label="Truth JSBSim")
                ax.plot(steps, nominal[key_x], color=NOMINAL_COLOR, linewidth=2.2, linestyle="--", label="Nominal polynomial")
                sampled_mean = np.mean(np.stack([trace[key_x] for trace in sampled], axis=0), axis=0)
                ax.plot(steps, sampled_mean, color=SAMPLED_COLOR, linewidth=2.4, label="Sampled uncertainty mean")
            else:
                for trace in sampled:
                    ax.plot(trace[key_x], trace[key_y], color=SAMPLED_COLOR, linewidth=1.0, alpha=SAMPLED_ALPHA)
                ax.plot(truth[key_x], truth[key_y], color=TRUTH_COLOR, linewidth=2.4, label="Truth JSBSim")
                ax.plot(nominal[key_x], nominal[key_y], color=NOMINAL_COLOR, linewidth=2.2, linestyle="--", label="Nominal polynomial")
                sampled_mean_x = np.mean(np.stack([trace[key_x] for trace in sampled], axis=0), axis=0)
                sampled_mean_y = np.mean(np.stack([trace[key_y] for trace in sampled], axis=0), axis=0)
                ax.plot(sampled_mean_x, sampled_mean_y, color=SAMPLED_COLOR, linewidth=2.4, label="Sampled uncertainty mean")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.98,
                    (
                        f"{case.label}\n"
                        f"{case.entry_speed_kts:.0f} kt | {case.entry_altitude_ft:.0f} ft | "
                        f"phi0={case.entry_roll_deg:.0f} deg | theta0={case.entry_pitch_deg:.0f} deg"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10.5,
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
                )
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best")

    fig.suptitle("Polynomial Model vs JSBSim Truth in Empty-Airspace Maneuvers", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_control_plans(case_results, output_path: Path):
    n_cases = len(case_results)
    fig, axes = plt.subplots(n_cases, 4, figsize=(18, 3.2 * n_cases), squeeze=False, constrained_layout=True)
    steps = np.arange(ROLL_OUT_STEPS, dtype=np.int32)
    labels = ("Roll Cmd", "Pitch Cmd", "Yaw Cmd", "Throttle Cmd")

    for row_idx, result in enumerate(case_results):
        action_plan = result["action_plan"]
        case = result["case"]
        for col_idx in range(4):
            ax = axes[row_idx][col_idx]
            ax.step(steps, action_plan[:, col_idx], where="post", color="tab:purple", linewidth=2.0)
            ax.set_xlabel("Step")
            ax.set_ylabel(labels[col_idx])
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(labels[col_idx])
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.94,
                    case.label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10.5,
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
                )

    fig.suptitle("Open-Loop Maneuver Command Sequences (40 Steps)", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    _ensure_required_files()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    W, B, poly_powers = load_nominal_weights()
    uncertainty_sampler = RuntimeUncertaintySampler(str(UNCERTAINTY_ARTIFACT_PATH))
    nominal_rollout_fn = _build_nominal_rollout_fn(W, B, poly_powers)
    sampled_rollout_fn, width_ft_default, width_grad_default = _build_sampled_rollout_fn(W, B, poly_powers, uncertainty_sampler)

    case_results = []
    for case_idx, case in enumerate(_selected_cases(args.case)):
        action_plan = _build_action_plan(case)
        truth_trace, initial_state = _rollout_truth(case, action_plan)

        x0 = jnp.asarray(initial_state, dtype=jnp.float32)
        action_plan_jax = jnp.asarray(action_plan, dtype=jnp.float32)

        nominal_state_traj = np.asarray(nominal_rollout_fn(x0, action_plan_jax), dtype=np.float32)
        nominal_trace = _trace_from_state_traj(nominal_state_traj)

        sampled_traces = []
        for sample_idx in range(int(args.num_sampled_rollouts)):
            seed = int(args.base_seed + case_idx * 1000 + sample_idx)
            rng_key = jax.random.PRNGKey(seed)
            sampled_state_traj, _ = sampled_rollout_fn(x0, action_plan_jax, rng_key)
            sampled_traces.append(_trace_from_state_traj(np.asarray(sampled_state_traj, dtype=np.float32)))

        metrics = _compute_metrics(truth_trace, nominal_trace, sampled_traces)
        case_results.append(
            {
                "case": case,
                "action_plan": action_plan,
                "truth_trace": truth_trace,
                "nominal_trace": nominal_trace,
                "sampled_traces": sampled_traces,
                "metrics": metrics,
            }
        )
        print(
            f"[{case.slug}] nominal alt RMSE={metrics['nominal_altitude_rmse_ft']:.1f} ft, "
            f"sampled alt RMSE={metrics['sampled_mean_altitude_rmse_ft']:.1f} ft, "
            f"nominal phi RMSE={metrics['nominal_phi_rmse_deg']:.1f} deg, "
            f"sampled phi RMSE={metrics['sampled_mean_phi_rmse_deg']:.1f} deg"
        )

    comparison_plot_path = output_dir / "poly_model_vs_truth_rollouts.png"
    controls_plot_path = output_dir / "poly_model_rollout_commands.png"
    summary_path = output_dir / "poly_model_rollout_summary.json"

    _plot_comparison(case_results, comparison_plot_path)
    _plot_control_plans(case_results, controls_plot_path)

    summary_payload = {
        "steps_per_rollout": int(ROLL_OUT_STEPS),
        "num_sampled_rollouts_per_case": int(args.num_sampled_rollouts),
        "uncertainty_context": {
            "fixed_canyon_width_ft": float(width_ft_default),
            "fixed_canyon_width_grad": float(width_grad_default),
        },
        "cases": [
            {
                **asdict(result["case"]),
                "metrics": result["metrics"],
            }
            for result in case_results
        ],
        "comparison_plot": str(comparison_plot_path),
        "controls_plot": str(controls_plot_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved comparison plot: {comparison_plot_path}")
    print(f"Saved control plot: {controls_plot_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
