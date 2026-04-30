from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsbsim_gym.mppi_run_config import (
    _trial_params_to_effective_mppi_params,
    apply_mppi_optuna_params,
    build_mppi_base_config_kwargs,
    load_mppi_optuna_params,
)


def test_apply_mppi_optuna_params():
    base = build_mppi_base_config_kwargs()
    params = {
        "lambda_": 0.8,
        "gamma_": 0.02,
        "action_noise_std": [0.11, 0.22, 0.33, 0.44],
        "state_tracking_weights": [0.01, 0.02, 0.03, 4.0, 5.0, 6.0],
        "terrain_collision_penalty": 2.5e6,
        "terrain_repulsion_scale": 4.0e4,
        "control_rate_weights": [12.0, 18.0, 3.0, 1.5],
        "ignored_key": 123.0,
    }

    updated, applied = apply_mppi_optuna_params(base, params)

    assert updated["lambda_"] == 0.8
    assert updated["gamma_"] == 0.02
    assert updated["action_noise_std"] == (0.11, 0.22, 0.33, 0.44)
    assert updated["state_tracking_weights"] == (0.01, 0.02, 0.03, 4.0, 5.0, 6.0)
    assert updated["terrain_collision_penalty"] == 2.5e6
    assert updated["terrain_repulsion_scale"] == 4.0e4
    assert updated["control_rate_weights"] == (12.0, 18.0, 3.0, 1.5)
    assert "ignored_key" not in updated
    assert set(applied) == {
        "lambda_",
        "gamma_",
        "action_noise_std",
        "state_tracking_weights",
        "terrain_collision_penalty",
        "terrain_repulsion_scale",
        "control_rate_weights",
    }


def test_load_mppi_optuna_params_from_json(tmp_path):
    payload = {
        "best_params": {
            "lambda_": 0.9,
            "gamma_": 0.03,
            "action_noise_std": [0.2, 0.3, 0.1, 0.05],
            "state_tracking_weights": [0.04, 0.04, 0.06, 8.0, 9.0, 7.0],
            "terrain_collision_penalty": 1.2e6,
            "terrain_repulsion_scale": 6.5e4,
            "control_rate_weights": [10.0, 14.0, 2.0, 1.0],
        }
    }
    summary_path = tmp_path / "mppi_optuna_best.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    params, source = load_mppi_optuna_params(
        summary_json_path=summary_path,
        study_name="unused",
        storage="",
    )

    assert params == payload["best_params"]
    assert source == str(summary_path)


def test_trial_params_to_effective_mppi_params():
    trial_params = {
        "lambda_": 0.7,
        "gamma_": 0.04,
        "action_noise_std_aileron": 0.21,
        "action_noise_std_elevator": 0.31,
        "action_noise_std_rudder": 0.12,
        "action_noise_std_throttle": 0.06,
        "state_tracking_weight_north": 0.05,
        "state_tracking_weight_east": 0.06,
        "state_tracking_weight_altitude": 0.07,
        "state_tracking_weight_phi": 3.0,
        "state_tracking_weight_theta": 4.0,
        "state_tracking_weight_psi": 5.0,
        "terrain_collision_penalty": 1.5e6,
        "terrain_repulsion_scale": 7.5e4,
        "control_rate_weight_aileron": 11.0,
        "control_rate_weight_elevator": 19.0,
        "control_rate_weight_rudder": 2.5,
        "control_rate_weight_throttle": 1.2,
    }

    effective = _trial_params_to_effective_mppi_params(trial_params)

    assert effective == {
        "lambda_": 0.7,
        "gamma_": 0.04,
        "action_noise_std": (0.21, 0.31, 0.12, 0.06),
        "state_tracking_weights": (0.05, 0.06, 0.07, 3.0, 4.0, 5.0),
        "terrain_collision_penalty": 1.5e6,
        "terrain_repulsion_scale": 7.5e4,
        "control_rate_weights": (11.0, 19.0, 2.5, 1.2),
    }
