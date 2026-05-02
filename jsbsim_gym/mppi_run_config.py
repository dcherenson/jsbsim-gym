from __future__ import annotations

import json
from pathlib import Path

from jsbsim_gym.mppi_defaults import (
    MPPI_DEFAULT_HORIZON,
    MPPI_DEFAULT_NUM_SAMPLES,
    MPPI_DEFAULT_OPTIMIZATION_STEPS,
    default_mppi_config_kwargs,
)
from jsbsim_gym.mppi_jax import JaxMPPIConfig, JaxMPPIController
from jsbsim_gym.smooth_mppi_jax import JaxSmoothMPPIConfig, JaxSmoothMPPIController


KTS_TO_FPS = 1.68781
REPO_ROOT = Path(__file__).resolve().parents[1]
MPPI_TUNING_JSON_PATH = REPO_ROOT / "output" / "canyon_mppi" / "mppi_optuna_best.json"
MPPI_TUNING_STUDY_NAME = "mppi_nominal_contouring_tuning"
MPPI_TUNING_STORAGE = f"sqlite:///{(REPO_ROOT / 'optuna' / 'mppi_tuning.db').as_posix()}"
MPPI_TUNING_STORAGE_FALLBACKS = (
    MPPI_TUNING_STORAGE,
    f"sqlite:///{(REPO_ROOT / 'mppi_tuning.db').as_posix()}",
    "sqlite:///mppi_tuning.db",
)
MPPI_TUNABLE_TUPLE_SPECS = {
    "action_noise_std": 4,
    "control_rate_weights": 4,
}
MPPI_TUNABLE_SCALAR_KEYS = frozenset(
    {
        "lambda_",
        "gamma_",
        "contour_weight",
        "lag_weight",
        "progress_reward_weight",
        "virtual_speed_weight",
        "terrain_collision_penalty",
        "terrain_repulsion_scale",
        "terrain_decay_rate_ft_inv",
        "terrain_safe_clearance_ft",
        "nz_min_g",
        "nz_max_g",
        "nz_penalty_weight",
        "alpha_limit_rad",
        "alpha_penalty_weight",
    }
)
MPPI_TUNABLE_KEYS = frozenset(MPPI_TUNABLE_SCALAR_KEYS | set(MPPI_TUNABLE_TUPLE_SPECS))
MPPI_REQUIRED_CONTOURING_KEYS = frozenset(
    {
        "contour_weight",
        "lag_weight",
        "progress_reward_weight",
        "virtual_speed_weight",
    }
)


def build_mppi_base_config_kwargs(
):
    return default_mppi_config_kwargs()



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


def _is_valid_contouring_tuning_params(params: dict) -> bool:
    return isinstance(params, dict) and MPPI_REQUIRED_CONTOURING_KEYS.issubset(params.keys())


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
    if "contour_weight" in params:
        effective["contour_weight"] = float(params["contour_weight"])
    if "lag_weight" in params:
        effective["lag_weight"] = float(params["lag_weight"])
    elif "contour_weight" in params and "lag_ratio" in params:
        effective["lag_weight"] = float(params["contour_weight"]) * float(params["lag_ratio"])
    if "progress_reward_weight" in params:
        effective["progress_reward_weight"] = float(params["progress_reward_weight"])
    if "virtual_speed_weight" in params:
        effective["virtual_speed_weight"] = float(params["virtual_speed_weight"])
    if "terrain_collision_penalty" in params:
        effective["terrain_collision_penalty"] = float(params["terrain_collision_penalty"])
    if "terrain_repulsion_scale" in params:
        effective["terrain_repulsion_scale"] = float(params["terrain_repulsion_scale"])
    if "terrain_decay_rate_ft_inv" in params:
        effective["terrain_decay_rate_ft_inv"] = float(params["terrain_decay_rate_ft_inv"])
    if "terrain_safe_clearance_ft" in params:
        effective["terrain_safe_clearance_ft"] = float(params["terrain_safe_clearance_ft"])
    if "nz_min_g" in params:
        effective["nz_min_g"] = float(params["nz_min_g"])
    if "nz_max_g" in params:
        effective["nz_max_g"] = float(params["nz_max_g"])
    if "nz_penalty_weight" in params:
        effective["nz_penalty_weight"] = float(params["nz_penalty_weight"])
    if "alpha_limit_rad" in params:
        effective["alpha_limit_rad"] = float(params["alpha_limit_rad"])
    if "alpha_penalty_weight" in params:
        effective["alpha_penalty_weight"] = float(params["alpha_penalty_weight"])

    action_noise_std = _maybe_tuple("action_noise_std", ("aileron", "elevator", "rudder", "throttle"))
    if action_noise_std is not None:
        effective["action_noise_std"] = action_noise_std

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
            if _is_valid_contouring_tuning_params(best_params):
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
            if not _is_valid_contouring_tuning_params(effective):
                effective = _trial_params_to_effective_mppi_params(best_trial.params)
            if _is_valid_contouring_tuning_params(effective):
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
