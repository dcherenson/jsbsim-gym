import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
G_FTPS2 = 32.174
SIMPLE_TUNING_JSON_PATH = REPO_ROOT / "output" / "simple_controller" / "simple_controller_optuna_best.json"
SIMPLE_TUNING_STUDY_NAME = "simple_controller_gain_tuning"
SIMPLE_TUNING_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'simple_controller_tuning.db').as_posix()}"
SIMPLE_TUNING_STORAGE_FALLBACKS = (
    SIMPLE_TUNING_STORAGE,
    f"sqlite:///{(REPO_ROOT / 'simple_controller_tuning.db').as_posix()}",
    "sqlite:///simple_controller_tuning.db",
)


@dataclass
class SimpleCanyonControllerConfig:
    target_speed_fps: float = 450.0 * 1.68781
    lookahead_rows: int = 20
    dt: float = 1.0 / 30.0
    use_dem_centerline: bool = True

    # Lookahead guidance -> desired track-normal acceleration.
    track_accel_lateral_gain: float = -1.0
    track_accel_heading_gain: float = -1.0
    track_accel_lateral_rate_gain: float = -1.0
    track_accel_max_fps2: float = 300.0

    # Roll alignment loop.
    roll_p_gain: float = 2.8
    roll_rate_damping: float = -0.06
    roll_max_rad: float = 100.0

    # Pitch / normal-acceleration control.
    pitch_nz_gain: float = -0.4
    # Positive elevator command pitches the airframe down, so q damping is positive.
    pitch_q_damping: float = 3.0
    nz_altitude_gain: float = 0.05
    nz_vrate_gain: float = 0.2
    nz_altitude_max_bias: float = 4.0
    nz_min_cmd: float = -1000.0
    nz_max_cmd: float = 1000.0

    # Yaw / sideslip coordination.
    # Positive rudder command drives Ny and r negative, so both feedback gains are positive.
    yaw_ny_gain: float = 0.4
    yaw_rate_damping: float = 0.25
    yaw_max_cmd: float = 100.0

    # Throttle / speed hold.
    throttle_base: float = 0.1
    throttle_speed_gain: float = 0.1
    throttle_max: float = 1.0


def _sqlite_storage_to_path(storage_url: str):
    prefix = "sqlite:///"
    if not isinstance(storage_url, str) or not storage_url.startswith(prefix):
        return None
    return Path(storage_url[len(prefix):])


def load_simple_controller_optuna_params(
    summary_json_path: Path = SIMPLE_TUNING_JSON_PATH,
    study_name: str = SIMPLE_TUNING_STUDY_NAME,
    storage: str = SIMPLE_TUNING_STORAGE,
):
    """Load best tuned SimpleCanyonController params from JSON, then SQLite fallback."""
    params = {}
    source = None

    summary_path = Path(summary_json_path)
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            best_params = payload.get("best_params", {}) if isinstance(payload, dict) else {}
            if isinstance(best_params, dict) and best_params:
                params = dict(best_params)
                source = str(summary_path)
        except Exception:
            pass

    if params:
        return params, source

    storage_candidates = []
    if isinstance(storage, str) and storage:
        storage_candidates.append(storage)
    for fallback in SIMPLE_TUNING_STORAGE_FALLBACKS:
        if fallback not in storage_candidates:
            storage_candidates.append(fallback)

    for storage_url in storage_candidates:
        sqlite_path = _sqlite_storage_to_path(storage_url)
        if sqlite_path is None or not sqlite_path.exists():
            continue

        try:
            import optuna

            study = optuna.load_study(study_name=study_name, storage=storage_url)
            return dict(study.best_params), f"{storage_url}::{study_name}"
        except Exception:
            continue

    return {}, None


def apply_simple_controller_optuna_params(config: SimpleCanyonControllerConfig, params: dict):
    """Apply tuned Optuna params onto a controller config and return applied keys."""
    if not isinstance(params, dict) or not params:
        return config, []

    field_defs = SimpleCanyonControllerConfig.__dataclass_fields__
    field_names = set(field_defs.keys())
    overrides = {}

    for key, value in params.items():
        if key in field_names:
            field_type = field_defs[key].type
            if field_type is int:
                overrides[key] = int(value)
            elif field_type is float:
                overrides[key] = float(value)
            elif field_type is bool:
                overrides[key] = bool(value)
            else:
                overrides[key] = value

    if not overrides:
        return config, []

    return replace(config, **overrides), sorted(overrides.keys())


def with_default_simple_controller_optuna_gains(
    config: SimpleCanyonControllerConfig,
    summary_json_path: Path = SIMPLE_TUNING_JSON_PATH,
    study_name: str = SIMPLE_TUNING_STUDY_NAME,
    storage: str = SIMPLE_TUNING_STORAGE,
):
    """Return config with best tuned gains applied when tuning artifacts are available."""
    params, source = load_simple_controller_optuna_params(
        summary_json_path=summary_json_path,
        study_name=study_name,
        storage=storage,
    )
    tuned_config, tuned_keys = apply_simple_controller_optuna_params(config, params)
    return tuned_config, source, tuned_keys


def build_reference_trajectory(
    north_ft,
    east_ft,
    heading_rad=None,
    width_ft=None,
    closed_loop=False,
):
    north_arr = np.asarray(north_ft, dtype=np.float32).reshape(-1)
    east_arr = np.asarray(east_ft, dtype=np.float32).reshape(-1)
    if north_arr.size < 2 or east_arr.size < 2:
        raise ValueError("Reference trajectory requires at least two points.")
    if north_arr.size != east_arr.size:
        raise ValueError("north_ft and east_ft must have the same length.")

    if heading_rad is None:
        if bool(closed_loop):
            dn = np.roll(north_arr, -1) - north_arr
            de = np.roll(east_arr, -1) - east_arr
            heading_arr = np.arctan2(de, dn).astype(np.float32)
        else:
            heading_arr = np.zeros_like(north_arr, dtype=np.float32)
            dn = np.diff(north_arr)
            de = np.diff(east_arr)
            heading_arr[:-1] = np.arctan2(de, dn)
            heading_arr[-1] = heading_arr[-2]
    else:
        heading_arr = np.asarray(heading_rad, dtype=np.float32).reshape(-1)
        if heading_arr.size != north_arr.size:
            raise ValueError("heading_rad must have the same length as north_ft/east_ft.")

    if width_ft is None:
        width_arr = np.full_like(north_arr, 600.0, dtype=np.float32)
    else:
        width_arr = np.asarray(width_ft, dtype=np.float32).reshape(-1)
        if width_arr.size != north_arr.size:
            raise ValueError("width_ft must have the same length as north_ft/east_ft.")

    return {
        "north_ft": north_arr,
        "east_ft": east_arr,
        "heading_rad": heading_arr,
        "width_ft": width_arr,
        "closed_loop": bool(closed_loop),
    }


def build_simple_trajectory_policy_jax(
    config: SimpleCanyonControllerConfig,
    reference_trajectory,
    target_altitude_ft,
    wall_margin_ft=0.0,
    altitude_reference_offset_ft=0.0,
):
    """Build a JAX policy closure that mirrors the simple trajectory controller.

    The gatekeeper rollout state uses the flat convention:
    [p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi, ny, nz]
    """

    reference = SimpleTrajectoryController._normalize_reference_trajectory(reference_trajectory)
    north_samples = jnp.asarray(reference["north_ft"], dtype=jnp.float32)
    east_samples = jnp.asarray(reference["east_ft"], dtype=jnp.float32)
    heading_samples = jnp.asarray(reference["heading_rad"], dtype=jnp.float32)
    width_samples = jnp.asarray(reference["width_ft"], dtype=jnp.float32)
    closed_loop = bool(reference["closed_loop"])
    sample_count = int(len(reference["north_ft"]))
    lookahead_rows = int(max(1, config.lookahead_rows))
    target_speed_fps = float(config.target_speed_fps)
    target_altitude_ft = float(target_altitude_ft)
    wall_margin_ft = float(wall_margin_ft)
    altitude_reference_offset_ft = float(altitude_reference_offset_ft)

    def _wrap_angle_rad(angle_rad):
        return jnp.arctan2(jnp.sin(angle_rad), jnp.cos(angle_rad))

    def _position_rates(state_flat):
        u, v, w = state_flat[3], state_flat[4], state_flat[5]
        phi, theta, psi = state_flat[9], state_flat[10], state_flat[11]
        c_psi = jnp.cos(psi)
        s_psi = jnp.sin(psi)
        c_theta = jnp.cos(theta)
        s_theta = jnp.sin(theta)
        c_phi = jnp.cos(phi)
        s_phi = jnp.sin(phi)
        p_n_dot = u * (c_theta * c_psi) + v * (s_phi * s_theta * c_psi - c_phi * s_psi) + w * (c_phi * s_theta * c_psi + s_phi * s_psi)
        p_e_dot = u * (c_theta * s_psi) + v * (s_phi * s_theta * s_psi + c_phi * c_psi) + w * (c_phi * s_theta * s_psi - s_phi * c_psi)
        h_dot = u * s_theta - v * s_phi * c_theta - w * c_phi * c_theta
        return p_n_dot, p_e_dot, h_dot

    def policy_fn(state_flat):
        p_n_ft = state_flat[0]
        p_e_ft = state_flat[1]
        h_ft = state_flat[2]
        u = state_flat[3]
        v = state_flat[4]
        w = state_flat[5]
        p_rate = state_flat[6]
        q_rate = state_flat[7]
        r_rate = state_flat[8]
        phi = state_flat[9]
        psi = state_flat[11]
        ny_current = state_flat[12]
        nz_current = state_flat[13]

        dn_all = north_samples - p_n_ft
        de_all = east_samples - p_e_ft
        ontrack_idx = jnp.argmin(dn_all * dn_all + de_all * de_all)
        if closed_loop:
            look_idx = jnp.mod(ontrack_idx + lookahead_rows, sample_count)
        else:
            look_idx = jnp.clip(ontrack_idx + lookahead_rows, 0, sample_count - 1)

        ontrack_north_ft = north_samples[ontrack_idx]
        ontrack_center_east_ft = east_samples[ontrack_idx]
        track_heading_rad = heading_samples[ontrack_idx]
        desired_heading_rad = heading_samples[look_idx]

        dn = p_n_ft - ontrack_north_ft
        de = p_e_ft - ontrack_center_east_ft
        lateral_error_ft = dn * (-jnp.sin(track_heading_rad)) + de * jnp.cos(track_heading_rad)
        heading_error_rad = _wrap_angle_rad(psi - desired_heading_rad)

        width_ft = width_samples[ontrack_idx]
        width_ft = jnp.where(jnp.logical_or(~jnp.isfinite(width_ft), width_ft <= 1.0), 600.0, width_ft)
        half_width_ft = 0.5 * width_ft
        usable_half_ft = jnp.maximum(half_width_ft - wall_margin_ft, 80.0)
        del usable_half_ft

        p_n_dot, p_e_dot, h_dot = _position_rates(state_flat)
        lateral_error_rate_fps = p_n_dot * (-jnp.sin(track_heading_rad)) + p_e_dot * jnp.cos(track_heading_rad)
        altitude_error_rate_fps = h_dot

        altitude_error_raw_ft = h_ft - target_altitude_ft
        altitude_error_offset_ft = (h_ft + altitude_reference_offset_ft) - target_altitude_ft
        altitude_error_ft = jnp.where(
            jnp.abs(altitude_error_raw_ft) <= jnp.abs(altitude_error_offset_ft),
            altitude_error_raw_ft,
            altitude_error_offset_ft,
        )

        speed_fps = jnp.sqrt(jnp.maximum(u * u + v * v + w * w, 1.0))
        track_accel_cmd_fps2 = jnp.clip(
            config.track_accel_lateral_gain * lateral_error_ft
            + config.track_accel_lateral_rate_gain * lateral_error_rate_fps
            + config.track_accel_heading_gain * speed_fps * jnp.sin(heading_error_rad),
            -config.track_accel_max_fps2,
            config.track_accel_max_fps2,
        )

        nz_altitude_bias = jnp.clip(
            -config.nz_altitude_gain * altitude_error_ft
            -config.nz_vrate_gain * altitude_error_rate_fps,
            -config.nz_altitude_max_bias,
            config.nz_altitude_max_bias,
        )
        vertical_load_target = 1.0 + nz_altitude_bias
        lateral_load_target = track_accel_cmd_fps2 / G_FTPS2
        nz_vector_mag = jnp.sqrt(vertical_load_target ** 2 + lateral_load_target ** 2)
        roll_des_rad = jnp.clip(
            jnp.arctan2(lateral_load_target, vertical_load_target),
            -config.roll_max_rad,
            config.roll_max_rad,
        )
        # Roll toward the desired acceleration vector first, then let pitch/Nz
        # build the full load demand as bank aligns. This avoids turning a
        # large lateral command into an immediate pull-up while nearly level.
        roll_alignment = jnp.maximum(0.0, jnp.cos(roll_des_rad - phi)) ** 2
        nz_des = jnp.clip(
            vertical_load_target + roll_alignment * (nz_vector_mag - vertical_load_target),
            config.nz_min_cmd,
            config.nz_max_cmd,
        )
        roll_cmd = jnp.clip(
            config.roll_p_gain * (roll_des_rad - phi) + config.roll_rate_damping * p_rate,
            -1.0,
            1.0,
        )
        pitch_cmd = jnp.clip(config.pitch_nz_gain * (nz_des - nz_current) + config.pitch_q_damping * q_rate, -1.0, 1.0)

        yaw_cmd = jnp.clip(
            config.yaw_rate_damping * r_rate + config.yaw_ny_gain * ny_current,
            -config.yaw_max_cmd,
            config.yaw_max_cmd,
        )

        throttle_cmd = jnp.clip(
            config.throttle_base + config.throttle_speed_gain * (target_speed_fps - speed_fps),
            0.0,
            config.throttle_max,
        )

        return jnp.asarray([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=jnp.float32)

    return policy_fn


class SimpleTrajectoryController:
    """Generic simple tracking controller for any 2D reference trajectory.

    Input state convention matches DataCollectionEnv state dict keys:
    p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi, beta, ny, nz.
    """

    def __init__(
        self,
        config: SimpleCanyonControllerConfig = None,
        target_altitude_ft=None,
        wall_margin_ft=0.0,
        altitude_reference_offset_ft=0.0,
        reference_trajectory=None,
    ):
        self.config = config if config is not None else SimpleCanyonControllerConfig()

        self.target_speed_fps = float(self.config.target_speed_fps)
        self.lookahead_rows = int(max(1, self.config.lookahead_rows))
        self.dt = float(max(self.config.dt, 1e-3))

        self.target_altitude_ft = None if target_altitude_ft is None else float(target_altitude_ft)
        self.wall_margin_ft = float(wall_margin_ft)
        self.altitude_reference_offset_ft = float(altitude_reference_offset_ft)

        self.reference_trajectory = None
        self._ontrack_idx = None
        self._prev_altitude_error_ft = 0.0
        self._prev_lateral_error_ft = 0.0
        self.last_guidance = {}

        if reference_trajectory is not None:
            self.set_reference_trajectory(reference_trajectory)

    @staticmethod
    def _wrap_angle_rad(angle_rad):
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    @staticmethod
    def _normalize_reference_trajectory(reference_trajectory):
        if reference_trajectory is None:
            return None
        if not isinstance(reference_trajectory, dict):
            raise TypeError("reference_trajectory must be a dict or None.")

        north_ft = reference_trajectory.get("north_ft", reference_trajectory.get("north_samples_ft"))
        east_ft = reference_trajectory.get(
            "east_ft",
            reference_trajectory.get("center_east_samples_ft", reference_trajectory.get("east_samples_ft")),
        )
        heading_rad = reference_trajectory.get(
            "heading_rad",
            reference_trajectory.get("centerline_heading_samples_rad", reference_trajectory.get("heading_samples_rad")),
        )
        width_ft = reference_trajectory.get("width_ft", reference_trajectory.get("width_samples_ft"))
        closed_loop = bool(reference_trajectory.get("closed_loop", False))

        if north_ft is None or east_ft is None:
            raise ValueError("reference_trajectory must define north_ft/east_ft arrays.")

        return build_reference_trajectory(
            north_ft=north_ft,
            east_ft=east_ft,
            heading_rad=heading_rad,
            width_ft=width_ft,
            closed_loop=closed_loop,
        )

    def set_reference_trajectory(self, reference_trajectory):
        self.reference_trajectory = self._normalize_reference_trajectory(reference_trajectory)
        self._ontrack_idx = None

    def reset(self, state_dict, target_altitude_ft=None, reference_trajectory=None):
        if target_altitude_ft is not None:
            self.target_altitude_ft = float(target_altitude_ft)
        elif self.target_altitude_ft is None:
            self.target_altitude_ft = float(state_dict["h"])

        if reference_trajectory is not None:
            self.set_reference_trajectory(reference_trajectory)

        self._ontrack_idx = None
        self._prev_altitude_error_ft = self._altitude_error_ft(state_dict)
        self._prev_lateral_error_ft = 0.0
        self.last_guidance = {}

    def _project_to_ontrack_index(self, local_north_ft, local_east_ft, reference):
        north_samples_ft = reference["north_ft"]
        east_samples_ft = reference["east_ft"]
        sample_count = int(len(north_samples_ft))
        if sample_count <= 0:
            return 0

        window_back = int(max(60, self.lookahead_rows * 8))
        window_fwd = int(max(120, self.lookahead_rows * 16))

        if self._ontrack_idx is None:
            dn = north_samples_ft - float(local_north_ft)
            de = east_samples_ft - float(local_east_ft)
            ontrack_idx = int(np.argmin(dn * dn + de * de))
        else:
            seed_idx = int(np.clip(self._ontrack_idx, 0, sample_count - 1))
            if reference["closed_loop"]:
                offsets = np.arange(-window_back, window_fwd + 1, dtype=np.int32)
                cand_idx = (seed_idx + offsets) % sample_count
                dn = north_samples_ft[cand_idx] - float(local_north_ft)
                de = east_samples_ft[cand_idx] - float(local_east_ft)
                ontrack_idx = int(cand_idx[int(np.argmin(dn * dn + de * de))])
            else:
                lo = max(0, seed_idx - window_back)
                hi = min(sample_count, seed_idx + window_fwd + 1)
                dn = north_samples_ft[lo:hi] - float(local_north_ft)
                de = east_samples_ft[lo:hi] - float(local_east_ft)
                rel_idx = int(np.argmin(dn * dn + de * de))
                ontrack_idx = int(np.clip(lo + rel_idx, 0, sample_count - 1))

        self._ontrack_idx = ontrack_idx
        return ontrack_idx

    def _compute_guidance(self, state_dict, reference, position_override=None):
        if position_override is None:
            local_north_ft = float(state_dict["p_N"])
            local_east_ft = float(state_dict["p_E"])
        else:
            local_north_ft = float(position_override["p_N"])
            local_east_ft = float(position_override["p_E"])

        north_samples_ft = reference["north_ft"]
        east_samples_ft = reference["east_ft"]
        heading_samples_rad = reference["heading_rad"]
        width_samples_ft = reference["width_ft"]

        sample_count = int(len(north_samples_ft))
        ontrack_idx = self._project_to_ontrack_index(
            local_north_ft=local_north_ft,
            local_east_ft=local_east_ft,
            reference=reference,
        )

        if reference["closed_loop"]:
            look_idx = int((ontrack_idx + self.lookahead_rows) % sample_count)
        else:
            look_idx = int(np.clip(ontrack_idx + self.lookahead_rows, 0, sample_count - 1))

        ontrack_north_ft = float(north_samples_ft[ontrack_idx])
        ontrack_center_east_ft = float(east_samples_ft[ontrack_idx])
        lookahead_north_ft = float(north_samples_ft[look_idx])
        lookahead_center_east_ft = float(east_samples_ft[look_idx])

        track_heading_rad = float(heading_samples_rad[ontrack_idx])
        desired_heading_rad = float(heading_samples_rad[look_idx])

        dn = float(local_north_ft - ontrack_north_ft)
        de = float(local_east_ft - ontrack_center_east_ft)
        lateral_error_ft = float(
            dn * (-np.sin(track_heading_rad)) + de * np.cos(track_heading_rad)
        )
        heading_error_rad = self._wrap_angle_rad(float(state_dict["psi"]) - desired_heading_rad)

        width_ft = float(width_samples_ft[ontrack_idx])
        if (not np.isfinite(width_ft)) or width_ft <= 1.0:
            width_ft = 600.0

        return {
            "lateral_error_ft": lateral_error_ft,
            "heading_error_rad": heading_error_rad,
            "centerline_heading_deg": float(np.rad2deg(desired_heading_rad)),
            "canyon_width_ft": width_ft,
            "local_north_ft": float(local_north_ft),
            "ontrack_idx": int(ontrack_idx),
            "ontrack_north_ft": float(ontrack_north_ft),
            "ontrack_center_east_ft": float(ontrack_center_east_ft),
            "lookahead_rows": int(self.lookahead_rows),
            "lookahead_north_ft": float(lookahead_north_ft),
            "lookahead_center_east_ft": float(lookahead_center_east_ft),
        }

    def _altitude_error_ft(self, state_dict):
        if self.target_altitude_ft is None:
            self.target_altitude_ft = float(state_dict["h"])

        h_raw_ft = float(state_dict["h"])
        h_with_offset_ft = h_raw_ft + self.altitude_reference_offset_ft
        if abs(h_raw_ft - self.target_altitude_ft) <= abs(h_with_offset_ft - self.target_altitude_ft):
            h_msl_ft = h_raw_ft
        else:
            h_msl_ft = h_with_offset_ft
        return float(h_msl_ft - self.target_altitude_ft)

    def get_action(self, state_dict, reference_trajectory=None, position_override=None):
        reference = self.reference_trajectory
        if reference_trajectory is not None:
            reference = self._normalize_reference_trajectory(reference_trajectory)
        if reference is None:
            raise ValueError("No reference trajectory provided to SimpleTrajectoryController.")

        guidance = self._compute_guidance(
            state_dict=state_dict,
            reference=reference,
            position_override=position_override,
        )

        lateral_error_ft = float(guidance["lateral_error_ft"])
        lateral_error_rate_fps = (lateral_error_ft - self._prev_lateral_error_ft) / self.dt
        self._prev_lateral_error_ft = lateral_error_ft

        heading_error_rad = float(guidance["heading_error_rad"])
        canyon_width_ft = float(guidance["canyon_width_ft"])

        half_width_ft = 0.5 * canyon_width_ft
        usable_half_ft = max(half_width_ft - self.wall_margin_ft, 80.0)
        lateral_norm = np.clip(lateral_error_ft / usable_half_ft, -2.5, 2.5)

        phi = float(state_dict["phi"])
        p_rate = float(state_dict["p"])
        q_rate = float(state_dict["q"])
        r_rate = float(state_dict["r"])
        ny_current = float(state_dict.get("ny", 0.0))
        nz_current = float(state_dict.get("nz", 1.0))

        altitude_error_ft = self._altitude_error_ft(state_dict)
        altitude_error_rate_fps = (altitude_error_ft - self._prev_altitude_error_ft) / self.dt
        self._prev_altitude_error_ft = altitude_error_ft

        speed_fps = float(
            np.sqrt(
                float(state_dict["u"]) ** 2
                + float(state_dict["v"]) ** 2
                + float(state_dict["w"]) ** 2
            )
        )

        track_accel_cmd_fps2 = np.clip(
            self.config.track_accel_lateral_gain * lateral_error_ft
            + self.config.track_accel_lateral_rate_gain * lateral_error_rate_fps
            + self.config.track_accel_heading_gain * speed_fps * np.sin(heading_error_rad),
            -self.config.track_accel_max_fps2,
            self.config.track_accel_max_fps2,
        )

        nz_altitude_bias = np.clip(
            -self.config.nz_altitude_gain * altitude_error_ft
            -self.config.nz_vrate_gain * altitude_error_rate_fps,
            -self.config.nz_altitude_max_bias,
            self.config.nz_altitude_max_bias,
        )
        vertical_load_target = 1.0 + nz_altitude_bias
        lateral_load_target = track_accel_cmd_fps2 / G_FTPS2
        nz_vector_mag = np.sqrt(vertical_load_target ** 2 + lateral_load_target ** 2)
        roll_des_rad = np.clip(
            np.arctan2(lateral_load_target, vertical_load_target),
            -self.config.roll_max_rad,
            self.config.roll_max_rad,
        )
        # Roll toward the desired acceleration vector first, then let pitch/Nz
        # build the full load demand as bank aligns. This avoids turning a
        # large lateral command into an immediate pull-up while nearly level.
        roll_alignment = max(0.0, np.cos(roll_des_rad - phi)) ** 2
        nz_des = np.clip(
            vertical_load_target + roll_alignment * (nz_vector_mag - vertical_load_target),
            self.config.nz_min_cmd,
            self.config.nz_max_cmd,
        )
        roll_cmd = np.clip(
            self.config.roll_p_gain * (roll_des_rad - phi) + self.config.roll_rate_damping * p_rate,
            -1.0,
            1.0,
        )

        pitch_cmd = np.clip(
            self.config.pitch_nz_gain * (nz_des - nz_current) + self.config.pitch_q_damping * q_rate,
            -1.0,
            1.0,
        )

        yaw_cmd = np.clip(
            self.config.yaw_rate_damping * r_rate
            + self.config.yaw_ny_gain * ny_current,
            -self.config.yaw_max_cmd,
            self.config.yaw_max_cmd,
        )

        throttle_cmd = np.clip(
            self.config.throttle_base
            + self.config.throttle_speed_gain * (self.target_speed_fps - speed_fps)
            ,
            0.0,
            self.config.throttle_max,
        )

        margin_ft = usable_half_ft - abs(lateral_error_ft)
        self.last_guidance = {
            **guidance,
            "lateral_norm": float(lateral_norm),
            "lateral_error_rate_fps": float(lateral_error_rate_fps),
            "altitude_error_ft": float(altitude_error_ft),
            "altitude_error_rate_fps": float(altitude_error_rate_fps),
            "speed_fps": float(speed_fps),
            "track_accel_cmd_fps2": float(track_accel_cmd_fps2),
            "heading_error_deg": float(np.degrees(heading_error_rad)),
            "roll_des_deg": float(np.degrees(roll_des_rad)),
            "nz_vector_mag": float(nz_vector_mag),
            "nz_des": float(nz_des),
            "vertical_load_target": float(vertical_load_target),
            "lateral_load_target": float(lateral_load_target),
            "roll_alignment": float(roll_alignment),
            "nz_current": float(nz_current),
            "ny_current": float(ny_current),
            "margin_to_wall_ft": float(margin_ft),
            "roll_cmd": float(roll_cmd),
            "pitch_cmd": float(pitch_cmd),
            "yaw_cmd": float(yaw_cmd),
            "throttle_cmd": float(throttle_cmd),
        }

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


class SimpleCanyonController:
    """Compatibility adapter that applies SimpleTrajectoryController to canyon centerlines."""

    def __init__(
        self,
        env,
        config: SimpleCanyonControllerConfig = None,
    ):
        self.env = env.unwrapped
        self.canyon = self.env.canyon
        self.config = config if config is not None else SimpleCanyonControllerConfig()
        self.use_dem_centerline = bool(self.config.use_dem_centerline)

        self.wall_margin_ft = float(getattr(self.env, "wall_margin_ft", 0.0))
        self.target_altitude_ft = float(getattr(self.env, "target_altitude_ft", 0.0))
        self.dem_start_elev_ft = float(getattr(self.env, "dem_start_elev_ft", 0.0))

        self._center_east_ft = None
        self._reference_heading_rad = None
        self.last_guidance = {}

        self._dem_reference = self._build_dem_reference_trajectory()
        self._core = SimpleTrajectoryController(
            config=self.config,
            target_altitude_ft=self.target_altitude_ft,
            wall_margin_ft=self.wall_margin_ft,
            altitude_reference_offset_ft=self.dem_start_elev_ft,
            reference_trajectory=self._dem_reference,
        )

    def _has_dem_centerline(self):
        return (
            hasattr(self.canyon, "get_local_from_latlon")
            and hasattr(self.canyon, "centerline_heading_samples_rad")
            and hasattr(self.canyon, "north_samples_ft")
            and hasattr(self.canyon, "center_east_samples_ft")
            and hasattr(self.canyon, "width_samples_ft")
        )

    def _build_dem_reference_trajectory(self):
        if not self.use_dem_centerline or not self._has_dem_centerline():
            return None
        return build_reference_trajectory(
            north_ft=np.asarray(self.canyon.north_samples_ft, dtype=np.float32),
            east_ft=np.asarray(self.canyon.center_east_samples_ft, dtype=np.float32),
            heading_rad=np.asarray(self.canyon.centerline_heading_samples_rad, dtype=np.float32),
            width_ft=np.asarray(self.canyon.width_samples_ft, dtype=np.float32),
            closed_loop=False,
        )

    def _build_fallback_reference_trajectory(self, state_dict):
        sample_count = 512
        step_ft = 60.0
        center_east_ft = float(
            getattr(self.env, "canyon_center_east_ft", state_dict["p_E"])
        )
        ref_heading_rad = float(state_dict["psi"])

        n_start_ft = float(state_dict["p_N"]) - 64.0 * step_ft
        north_samples_ft = n_start_ft + step_ft * np.arange(sample_count, dtype=np.float32)
        east_samples_ft = np.full(sample_count, center_east_ft, dtype=np.float32)
        heading_samples_rad = np.full(sample_count, ref_heading_rad, dtype=np.float32)
        width_samples_ft = np.full(
            sample_count,
            float(state_dict.get("canyon_width", 600.0)),
            dtype=np.float32,
        )

        return build_reference_trajectory(
            north_ft=north_samples_ft,
            east_ft=east_samples_ft,
            heading_rad=heading_samples_rad,
            width_ft=width_samples_ft,
            closed_loop=False,
        )

    def _current_local_position(self, state_dict):
        if hasattr(self.canyon, "get_local_from_latlon"):
            try:
                lat_deg = float(self.env.simulation.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self.env.simulation.get_property_value("position/long-gc-deg"))
                local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)
                return float(local_north_ft), float(local_east_ft)
            except Exception:
                pass
        return float(state_dict["p_N"]), float(state_dict["p_E"])

    def reset(self, state_dict):
        self._center_east_ft = float(
            getattr(self.env, "canyon_center_east_ft", state_dict["p_E"])
        )
        self._reference_heading_rad = float(state_dict["psi"])

        if self._dem_reference is not None:
            self._core.set_reference_trajectory(self._dem_reference)
        else:
            self._core.set_reference_trajectory(self._build_fallback_reference_trajectory(state_dict))

        self._core.reset(
            state_dict=state_dict,
            target_altitude_ft=self.target_altitude_ft,
        )
        self.last_guidance = {}

    def get_action(self, state_dict):
        if self._core.reference_trajectory is None:
            self._core.set_reference_trajectory(self._build_fallback_reference_trajectory(state_dict))

        local_north_ft, local_east_ft = self._current_local_position(state_dict)
        action = self._core.get_action(
            state_dict=state_dict,
            position_override={"p_N": local_north_ft, "p_E": local_east_ft},
        )
        self.last_guidance = dict(self._core.last_guidance)
        return action

    def get_canyon_width_ft(self, p_N):
        if hasattr(self.canyon, "get_geometry"):
            width, _ = self.canyon.get_geometry(p_N)
            return float(width)
        if self.last_guidance:
            return float(self.last_guidance.get("canyon_width_ft", 600.0))
        return 600.0

    def get_lateral_error_ft(self, p_N, p_E):
        if self.last_guidance:
            return float(self.last_guidance.get("lateral_error_ft", 0.0))

        if hasattr(self.canyon, "get_local_from_latlon") and hasattr(self.canyon, "center_east_samples_ft"):
            lat_deg = float(self.env.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(self.env.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)
            center_east_ft = float(
                np.interp(
                    local_north_ft,
                    self.canyon.north_samples_ft,
                    self.canyon.center_east_samples_ft,
                )
            )
            return float(local_east_ft - center_east_ft)
        return float(p_E - getattr(self.env, "canyon_center_east_ft", 0.0))
