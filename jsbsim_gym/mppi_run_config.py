from __future__ import annotations

import json
from pathlib import Path

from jsbsim_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from jsbsim_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController


KTS_TO_FPS = 1.68781
REPO_ROOT = Path(__file__).resolve().parents[1]
MPPI_TUNING_JSON_PATH = REPO_ROOT / "output" / "canyon_mppi" / "mppi_optuna_best.json"
MPPI_TUNING_STUDY_NAME = "mppi_nominal_tracking_tuning"
MPPI_TUNING_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'mppi_tuning.db').as_posix()}"
MPPI_TUNING_STORAGE_FALLBACKS = (
    MPPI_TUNING_STORAGE,
    f"sqlite:///{(REPO_ROOT / 'mppi_tuning.db').as_posix()}",
    "sqlite:///mppi_tuning.db",
)
MPPI_TUNABLE_TUPLE_SPECS = {
    "action_noise_std": 4,
    "state_tracking_weights": 6,
    "control_rate_weights": 4,
}
MPPI_TUNABLE_SCALAR_KEYS = frozenset(
    {
        "lambda_",
        "gamma_",
        "terrain_collision_penalty",
        "terrain_repulsion_scale",
    }
)
MPPI_TUNABLE_KEYS = frozenset(MPPI_TUNABLE_SCALAR_KEYS | set(MPPI_TUNABLE_TUPLE_SPECS))


def build_mppi_base_config_kwargs(
    *,
    horizon=40,
    num_samples=4000,
    optimization_steps=3,
):
    return dict(
        horizon=int(horizon),
        num_samples=int(num_samples),
        optimization_steps=int(optimization_steps),
        lambda_=1.0,
        gamma_=0.05,
        action_noise_std=(0.5, 0.5, 0.5, 0.5),
        state_tracking_weights=(0.05, 0.05, 0.05, 10.0, 10.0, 8.0),
        terrain_collision_penalty=0.0*1.0e6,
        terrain_repulsion_scale=0.0*1.0e5,
        terrain_decay_rate_ft_inv=0.03,
        terrain_safe_clearance_ft=40.0 * 3.28084,
        control_rate_weights=(15.0, 20.0, 5.0, 2.0),
        nz_limit_g=9.0,
        nz_penalty_weight=0.0*1.0e4,
        alpha_limit_rad=25.0 * 3.141592653589793 / 180.0,
        alpha_penalty_weight=0.0*1.0e6,
    )


def build_mppi_controller(
    controller_tag,
    *,
    config_base_kwargs,
    reference_trajectory,
    terrain_north_samples_ft,
    terrain_east_samples_ft,
    terrain_elevation_ft,
):
    if controller_tag == "smooth_mppi":
        config = JaxSmoothMPPIConfig(**config_base_kwargs)
        controller_cls = JaxSmoothMPPIController
    elif controller_tag == "mppi":
        config = JaxMPPIConfig(**config_base_kwargs)
        controller_cls = JaxMPPIController
    else:
        raise ValueError(f"Unsupported MPPI controller tag: {controller_tag}")

    controller = controller_cls(
        config=config,
        reference_trajectory=reference_trajectory,
        terrain_north_samples_ft=terrain_north_samples_ft,
        terrain_east_samples_ft=terrain_east_samples_ft,
        terrain_elevation_ft=terrain_elevation_ft,
    )
    return controller, config


def _sqlite_storage_to_path(storage_url: str):
    prefix = "sqlite:///"
    if not isinstance(storage_url, str) or not storage_url.startswith(prefix):
        return None
    return Path(storage_url[len(prefix):])


def _normalize_mppi_tunable_value(key: str, value):
    if key in MPPI_TUNABLE_TUPLE_SPECS:
        try:
            values = tuple(float(v) for v in value)
        except TypeError:
            return None
        if len(values) != MPPI_TUNABLE_TUPLE_SPECS[key]:
            return None
        return values
    if key in MPPI_TUNABLE_SCALAR_KEYS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _trial_params_to_effective_mppi_params(params: dict) -> dict:
    if not isinstance(params, dict):
        return {}

    def _maybe_tuple(prefix: str, names: tuple[str, ...]):
        values = []
        for suffix in names:
            key = f"{prefix}_{suffix}"
            if key not in params:
                return None
            values.append(float(params[key]))
        return tuple(values)

    effective = {}
    if "lambda_" in params:
        effective["lambda_"] = float(params["lambda_"])
    if "gamma_" in params:
        effective["gamma_"] = float(params["gamma_"])
    if "terrain_collision_penalty" in params:
        effective["terrain_collision_penalty"] = float(params["terrain_collision_penalty"])
    if "terrain_repulsion_scale" in params:
        effective["terrain_repulsion_scale"] = float(params["terrain_repulsion_scale"])

    action_noise_std = _maybe_tuple("action_noise_std", ("aileron", "elevator", "rudder", "throttle"))
    if action_noise_std is not None:
        effective["action_noise_std"] = action_noise_std

    state_tracking_weights = _maybe_tuple(
        "state_tracking_weight",
        ("north", "east", "altitude", "phi", "theta", "psi"),
    )
    if state_tracking_weights is not None:
        effective["state_tracking_weights"] = state_tracking_weights

    control_rate_weights = _maybe_tuple(
        "control_rate_weight",
        ("aileron", "elevator", "rudder", "throttle"),
    )
    if control_rate_weights is not None:
        effective["control_rate_weights"] = control_rate_weights

    return effective


def load_mppi_optuna_params(
    summary_json_path: Path = MPPI_TUNING_JSON_PATH,
    study_name: str = MPPI_TUNING_STUDY_NAME,
    storage: str = MPPI_TUNING_STORAGE,
):
    """Load best tuned MPPI params from JSON, then Optuna SQLite fallback."""
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
    for fallback in MPPI_TUNING_STORAGE_FALLBACKS:
        if fallback not in storage_candidates:
            storage_candidates.append(fallback)

    for storage_url in storage_candidates:
        sqlite_path = _sqlite_storage_to_path(storage_url)
        if sqlite_path is None or not sqlite_path.exists():
            continue

        try:
            import optuna

            study = optuna.load_study(study_name=study_name, storage=storage_url)
            best_trial = study.best_trial
            effective = best_trial.user_attrs.get("effective_params")
            if not isinstance(effective, dict) or not effective:
                effective = _trial_params_to_effective_mppi_params(best_trial.params)
            if isinstance(effective, dict) and effective:
                return dict(effective), f"{storage_url}::{study_name}"
        except Exception:
            continue

    return {}, None


def apply_mppi_optuna_params(config_base_kwargs: dict, params: dict):
    """Apply tuned Optuna params onto MPPI config kwargs and return applied keys."""
    updated = dict(config_base_kwargs)
    applied = []

    if not isinstance(params, dict) or not params:
        return updated, applied

    for key, value in params.items():
        normalized = _normalize_mppi_tunable_value(key, value)
        if normalized is None:
            continue
        updated[key] = normalized
        applied.append(key)

    return updated, applied


def with_default_mppi_optuna_params(
    config_base_kwargs: dict,
    summary_json_path: Path = MPPI_TUNING_JSON_PATH,
    study_name: str = MPPI_TUNING_STUDY_NAME,
    storage: str = MPPI_TUNING_STORAGE,
):
    params, source = load_mppi_optuna_params(
        summary_json_path=summary_json_path,
        study_name=study_name,
        storage=storage,
    )
    updated, applied = apply_mppi_optuna_params(config_base_kwargs, params)
    return updated, source, applied
