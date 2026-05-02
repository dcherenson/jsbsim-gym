# MPPI Code Walkthrough (Full Stack)

This document explains the **setup and entry points** of the MPPI runtime, starting from [`run_scenario.py`](./run_scenario.py), as requested.

Scope in this part:
- File bootstrap and constants
- CLI parsing and start-fraction validation
- Environment and nominal-trajectory setup
- MPPI config loading and controller construction
- JIT warm-up and gatekeeper warm-up
- Main loop entry and MPPI diagnostics capture

## 1) `run_scenario.py`: Imports and Global Setup

### Lines 1-5
- `import argparse`: CLI parser for run options.
- `import csv`: writes diagnostics tables.
- `from dataclasses import fields, is_dataclass`: used to print config dataclasses generically.
- `import time`: timing instrumentation for controller and gatekeeper latency.
- `from pathlib import Path`: normalized path handling.

### Lines 7-12
- `import gymnasium as gym`: environment factory (`gym.make`).
- `import matplotlib.pyplot as plt`: diagnostics plotting.
- `import numpy as np`: numeric utilities and array operations.
- `import jax.numpy as jnp`: JAX arrays used in gatekeeper/MPPI glue paths.

### Lines 13-37
- Imports all controller/gatekeeper/runtime modules.
- Key MPPI entry imports:
  - `build_mppi_base_config_kwargs`
  - `build_mppi_controller`
  - `with_default_mppi_optuna_params`
  - `f16_kinematics_step_with_load_factors`
  - `build_nominal_reference_from_dyn`
  - `load_nominal_initial_conditions_from_dyn`

### Lines 39-55
- Declares run-time constants.
- `DEM_PATH/DEM_BBOX/DEM_START_PIXEL`: terrain source and start cell.
- `REPO_ROOT`, `OUTPUT_ROOT`: output directory roots.
- `M_TO_FT`, `G_FTPS2`: unit/physics constants used in conversions and debug.
- `DEFAULT_INITIAL_*`: fallback initial conditions if no nominal dyn path is provided.

---

## 2) CLI Input Validation Helpers

### `_fraction_0_to_1` (lines 57-64)
- Converts string CLI input into float.
- Throws `argparse.ArgumentTypeError` on parse failure.
- Enforces finite value in closed interval `[0, 1]`.
- Used by `--nominal-start-fraction` so bad values are rejected before runtime.

### `_set_mppi_nominal_start_progress` (lines 67-82)
- Takes controller and normalized progress fraction.
- Verifies controller has `params.path_s_np` and writable `_progress_s_ft`.
- Converts fraction into absolute path arclength:
  - `s = s0 + fraction * (s1 - s0)`
- Writes `controller._progress_s_ft = s`.
- Returns the assigned `s` (or `None` if controller does not support it).

Why this matters:
- It is the main bridge between your CLI fraction argument and the MPPI internal progress state.

---

## 3) MPPI-State Conversion Boundary

### `to_mppi_state` (lines 309-340)
- Converts simulator state dictionary to MPPI controller state layout.
- Starts from `p_N`, `p_E` from sim state.
- If canyon exposes `get_local_from_latlon`, recomputes local N/E from live latitude/longitude (more robust local frame alignment).
- Returns normalized dict:
  - position (`p_N`, `p_E`, `h`)
  - body velocities (`u`, `v`, `w`)
  - angular rates (`p`, `q`, `r`)
  - Euler angles (`phi`, `theta`, `psi`)
  - load factors (`ny`, `nz`)
- `h` is transformed to relative altitude (`state["h"] - altitude_ref_ft`), matching MPPI’s relative terrain frame.

Why this matters:
- This is the exact API contract between JSBSim environment state and MPPI internals.

---

## 4) Entry-Point Argument Parsing

### `parse_args` (lines 666-720)
- Defines runtime CLI switches:
  - `--controller`: `mppi|smooth_mppi|simple|altitude_hold`
  - `--render`
  - `--output-root`
  - `--output-dir`
  - `--max-steps`
  - `--gatekeeper`
  - `--gatekeeper-debug-timing`
  - `--nominal-dyn-path`
  - `--nominal-start-fraction` (validated by `_fraction_0_to_1`)
- Returns parsed `argparse.Namespace`.

---

## 5) Main Entry Flow

### `main` start (lines 722-737)
- Parses args.
- Enforces gatekeeper only for MPPI variants.
- Enforces that `--nominal-start-fraction > 0` requires `--nominal-dyn-path`.
- Seeds default initial condition values from global constants.

### Nominal initial condition override (lines 738-778)
- If `--nominal-dyn-path` is provided:
  - Constructs temporary `DEMCanyon`.
  - Calls `load_nominal_initial_conditions_from_dyn(..., progress_fraction=nominal_start_fraction)`.
  - Replaces all initial condition fields (start pixel, speed, altitude, heading, roll, pitch, alpha, beta) from nominal sample.
  - Prints detailed confirmation including sample index and progress fraction.

This is where the fraction argument actually drives simulation start state.

### Output directory routing (lines 780-793)
- Maps controller type to default subdir name.
- Appends `_gatekeeper` suffix when enabled.
- Honors explicit `--output-dir` override.
- Creates output directory.

### Environment construction and reset (lines 794-839)
- Creates `JSBSimCanyon-v0` with DEM-specific parameters and chosen start state.
- Resets with fixed seed (`seed=3`).
- Pulls initial full state from env.
- Computes altitude reference and converts to MPPI state via `to_mppi_state`.

### Nominal reference build (lines 841-854)
- If nominal dyn path provided:
  - Calls `build_nominal_reference_from_dyn(...)`.
  - Resamples by canyon segment spacing.
  - Prints sample count for confirmation.

### MPPI config and parameter loading (lines 856-866)
- Calls `build_mppi_base_config_kwargs()` (single defaults source).
- For `controller_tag == "mppi"`:
  - Calls `with_default_mppi_optuna_params(...)`.
  - Auto-loads tuned params from JSON/Optuna fallback if available.
  - Prints whether tuned params were loaded or defaults are used.

### Controller selection (lines 867-909)
- `simple`: builds and resets `SimpleCanyonController`.
- `altitude_hold`: uses fallback debug controller.
- `mppi|smooth_mppi`:
  - Validates nominal reference exists.
  - Validates DEM terrain grid attributes exist.
  - Calls `build_mppi_controller(...)` with:
    - effective config kwargs
    - nominal reference trajectory
    - terrain north/east axes
    - terrain elevation in relative feet frame
  - Prints full effective MPPI config via `_print_mppi_config`.
  - Initializes controller progress from `nominal_start_fraction` via `_set_mppi_nominal_start_progress`.

---

## 6) Recorder, JIT Warm-Up, and Gatekeeper Warm-Up

### Recorder setup (lines 910-933)
- Builds `CanyonRunRecorder`.
- Initializes it.
- Registers nominal reference profile if present, otherwise centerline profile.
- Initializes diagnostic row containers for simple/MPPI.

### MPPI JIT warm-up (lines 937-944)
- If MPPI variant:
  - Sets progress start fraction.
  - Calls `controller.get_action(initial_controller_state)` once to trigger JAX compilation.
  - Resets controller with deterministic seed.
  - Re-applies start progress.
- This removes first-step JIT latency from the active run.

### Gatekeeper warm-up (lines 945-974)
- Validates uncertainty artifact file exists.
- Builds gatekeeper bundle via `build_jsbsim_gatekeeper(...)`.
- Compiles gatekeeper path with one warm-up `update(...)`.
- Resets gatekeeper and controller after warm-up.

---

## 7) Main Simulation Loop Entry

### Loop skeleton (lines 1000-1012)
- Initializes `prev_applied_action`.
- Iterates `for step in range(max_steps)`.
- Recomputes controller state each step (`to_mppi_state`).
- Times planner call.
- For MPPI variants:
  - `nominal_action = controller.get_action(controller_state)`.
  - uses nominal action directly unless gatekeeper overrides.

### Gatekeeper override phase (lines 1013-1059)
- If gatekeeper enabled:
  - Updates nominal action in shared structure.
  - Builds/pads nominal action plan for gatekeeper horizon.
  - Calls gatekeeper `update(...)`.
  - If gatekeeper chooses backup mode, queries backup controller action.
  - Ticks gatekeeper and tracks mode transitions.

### Render-debug data capture (lines 1060-1133)
- Pulls planner trajectories from controller `get_render_debug()` if available.
- If gatekeeper active, overlays gatekeeper predicted trajectories and failure masks.
- Stores all of this in `planner_debug` for recorder overlays.

### HUD command synthesis (lines 1157-1207)
- Computes current and commanded headings.
- Emits mode labels (`MPPI/SMPPI/SIMPLE/HOLD`, `NOMINAL/BACKUP`).
- Pushes HUD commands to env renderer if supported.

---

## 8) Step Execution and MPPI Diagnostics Collection

### Step and post-step state (lines 1209-1214)
- Executes `env.step(action)`.
- Reads `termination_reason`.
- Refreshes full env state and converted controller state.
- Records current frame with debug overlays.

### MPPI metric extraction (lines 1222-1330)
- Gets controller config and tracking metrics:
  - progress `s`
  - virtual speed
  - contour/lag/position errors
  - contouring cost components
- Recomputes terrain/rate/limit cost estimates for diagnostics output.
- Appends one diagnostics row per step (`mppi_tracking_rows`), including commands, rates, timing, and mode.

### Periodic terminal prints (lines 1351-1369)
- Every 5 steps:
  - MPPI variants print compact table:
    - `Prog`, `CErr`, `Lag`, `PosErr`, `dH`, `Clr`, `Cost`, timings
  - simple/hold prints a separate compact state table.

### Loop termination conditions (lines 1371-1376)
- Breaks when env reports `terminated` or `truncated`.
- In render mode, sleeps to hold real-time frame rate.

---

## 9) Finalization and Artifact Export

### Cleanup and artifacts (lines 1377-1384)
- Always closes env in `finally`.
- Finalizes recorder (video, overlay image, trajectory CSV).

### Diagnostics exports (lines 1384-1404)
- For simple controller: writes simple diagnostics CSV + plot.
- For MPPI variants: writes tracking diagnostics CSV + multi-panel plot.

### MPPI summary statistics printout (lines 1405-1439)
- Computes mean contour, mean abs lag, mean/max position error, min clearance, mean limit cost, mean stage cost.
- Prints compact final summary for quick run-level comparison.

### Script entry point (lines 1442-1443)
- Standard Python executable guard:
  - calls `main()` when run as script.

---

## 10) Call Graph From `run_scenario` Into MPPI Core

Main setup and entry path:
1. `main()`
2. `build_mppi_base_config_kwargs()`
3. `with_default_mppi_optuna_params(...)`
4. `build_mppi_controller(...)`
5. `JaxMPPIController` or `JaxSmoothMPPIController` constructor
6. `controller.get_action(...)` in main loop
7. controller internals call rollout/cost functions from `mppi_support.py` and `_mppi_backend.py`

This document starts with `run_scenario` and now continues through the full MPPI stack.

---

## 11) `mppi_defaults.py`: Single Source of Default Values

File: [`jsbsim_gym/mppi_defaults.py`](./jsbsim_gym/mppi_defaults.py)

### Lines 6-35: scalar/tuple default constants
- Defines all base MPPI defaults used by config dataclasses.
- Key groups:
  - sampling/planning:
    - `MPPI_DEFAULT_HORIZON`
    - `MPPI_DEFAULT_NUM_SAMPLES`
    - `MPPI_DEFAULT_OPTIMIZATION_STEPS`
    - `MPPI_DEFAULT_REPLAN_INTERVAL`
    - `MPPI_DEFAULT_LAMBDA`
    - `MPPI_DEFAULT_GAMMA`
  - action noise and bounds:
    - `MPPI_DEFAULT_ACTION_NOISE_STD`
    - `MPPI_DEFAULT_ACTION_LOW`
    - `MPPI_DEFAULT_ACTION_HIGH`
  - contouring objective:
    - `MPPI_DEFAULT_CONTOUR_WEIGHT`
    - `MPPI_DEFAULT_LAG_WEIGHT`
    - `MPPI_DEFAULT_PROGRESS_REWARD_WEIGHT`
    - `MPPI_DEFAULT_VIRTUAL_SPEED_WEIGHT`
  - terrain objective:
    - `MPPI_DEFAULT_TERRAIN_COLLISION_PENALTY`
    - `MPPI_DEFAULT_TERRAIN_REPULSION_SCALE`
    - `MPPI_DEFAULT_TERRAIN_DECAY_RATE_FT_INV`
    - `MPPI_DEFAULT_TERRAIN_SAFE_CLEARANCE_FT`
  - regularization/limits:
    - `MPPI_DEFAULT_CONTROL_RATE_WEIGHTS`
    - `MPPI_DEFAULT_NZ_MIN_G`, `MPPI_DEFAULT_NZ_MAX_G`, `MPPI_DEFAULT_NZ_PENALTY_WEIGHT`
    - `MPPI_DEFAULT_ALPHA_LIMIT_RAD`, `MPPI_DEFAULT_ALPHA_PENALTY_WEIGHT`
  - virtual progress process:
    - `MPPI_DEFAULT_VIRTUAL_SPEED_NOISE_STD_FPS`
    - `MPPI_DEFAULT_VIRTUAL_SPEED_MIN_FPS`
    - `MPPI_DEFAULT_VIRTUAL_SPEED_MAX_FPS`
    - `MPPI_DEFAULT_VIRTUAL_SPEED_TRIM_FPS`
  - debug/repro:
    - `MPPI_DEFAULT_DEBUG_RENDER_PLANS`
    - `MPPI_DEFAULT_DEBUG_NUM_TRAJECTORIES`
    - `MPPI_DEFAULT_SEED`

### Lines 37-41: smooth MPPI defaults
- Separate default distributions for smooth-perturbation MPPI:
  - base action noise
  - delta-noise scale
  - per-channel delta bounds
  - temporal smoothing kernel
  - seed

### Lines 44-78: `default_mppi_config_kwargs()`
- Returns a dictionary with typed values (int/float/tuple/bool).
- This is the canonical default config payload consumed by `mppi_run_config`.
- Every key in this map is expected to align with controller config dataclass fields.

---

## 12) `mppi_run_config.py`: Runtime Wiring and Optuna Override Layer

File: [`jsbsim_gym/mppi_run_config.py`](./jsbsim_gym/mppi_run_config.py)

### Constants and tunable-key declarations (lines 16-57)
- `MPPI_TUNING_JSON_PATH`: default JSON source (`output/canyon_mppi/mppi_optuna_best.json`).
- `MPPI_TUNING_STORAGE` + fallbacks: SQLite study sources for fallback loading.
- `MPPI_TUNABLE_TUPLE_SPECS` + `MPPI_TUNABLE_SCALAR_KEYS`: whitelist of runtime-overridable keys.
- `MPPI_REQUIRED_CONTOURING_KEYS`: minimum key set needed to accept loaded tuning payload.

Important behavior:
- Only keys in `MPPI_TUNABLE_KEYS` are applied at runtime by `apply_mppi_optuna_params`.
- Non-whitelisted keys in JSON are silently ignored.

### `build_mppi_base_config_kwargs` (lines 60-62)
- Thin pass-through to `default_mppi_config_kwargs()`.

### `build_mppi_controller` (lines 66-91)
- Chooses class by `controller_tag`:
  - `mppi` -> `JaxMPPIController`
  - `smooth_mppi` -> `JaxSmoothMPPIController`
- Constructs matching config dataclass instance with `config_base_kwargs`.
- Passes reference trajectory and terrain grid arrays into controller constructor.

### Optuna payload normalization path

#### `_normalize_mppi_tunable_value` (lines 101-115)
- Enforces shape/type for each key:
  - tuple keys must have exact expected tuple length
  - scalar keys must cast to float

#### `_trial_params_to_effective_mppi_params` (lines 122-180)
- Converts Optuna trial parameter naming convention to runtime config keys.
- Handles:
  - scalar key passthrough
  - optional `lag_ratio` expansion to `lag_weight = contour_weight * lag_ratio`
  - per-channel tuple assembly:
    - `action_noise_std_{aileron,elevator,rudder,throttle}`
    - `control_rate_weight_{...}`

#### `load_mppi_optuna_params` (lines 183-231)
- Load priority:
  1. JSON file (`best_params`)
  2. Optuna SQLite study fallback(s)
- Rejects incomplete payloads via `_is_valid_contouring_tuning_params`.
- On DB fallback, prefers `best_trial.user_attrs["effective_params"]`; otherwise re-maps from `best_trial.params`.

#### `apply_mppi_optuna_params` (lines 234-249)
- Applies only normalized/whitelisted keys into config kwargs.
- Returns updated kwargs + list of keys actually applied.

#### `with_default_mppi_optuna_params` (lines 252-264)
- Combined helper used by `run_scenario`:
  - load params
  - apply params
  - return `(updated_kwargs, source, applied_keys)`.

---

## 13) `mppi_jax/controller.py`: Core MPPI Planner Loop

File: [`jsbsim_gym/mppi_jax/controller.py`](./jsbsim_gym/mppi_jax/controller.py)

### `JaxMPPIConfig` dataclass (lines 59-90)
- Runtime config schema mirrored from defaults.
- Carries sampling, cost, constraint, and debug parameters.

### `JaxMPPIController.__init__` (lines 93-158)
- Builds and stores:
  1. `self.params = build_nominal_params(...)`  
     Encodes reference trajectory and terrain into interpolable arrays.
  2. `self.cost_config = MPPICostConfig(...)`  
     Cost coefficients/bounds.
  3. JIT rollout callables:
     - `self._rollout_costs`
     - `self._rollout_positions`
  4. PRNG key from `seed`.
  5. Initial action plan and virtual-speed plan:
     - action plan starts at neutral surfaces + trim throttle
     - virtual speed plan initialized from reference first sample, clipped to min/max
  6. internal caches:
     - `_last_action`, `_cached_action`
     - `_last_virtual_speed_fps`, `_cached_virtual_speed_fps`
     - `_progress_s_ft` initialized to path start
     - replan bookkeeping fields

### `reset` (lines 160-179)
- Re-initializes RNG, action/virtual-speed plans, cached action/speed, progress, and debug state.

### `_shift_plan` (lines 184-193)
- Warm-start mechanism:
  - left-shifts previous action plan by one step.
  - fills last row with previous chosen action.
  - same process for virtual-speed plan.

### `_optimize` (lines 194-282): the MPPI algorithm body

At each optimization iteration:
1. Sample action noise:
   - shape: `(num_samples, horizon, 4)`
   - iid normal scaled by `action_noise_std`.
2. Sample virtual-speed noise:
   - shape: `(num_samples, horizon)`
   - iid normal scaled by `virtual_speed_noise_std_fps`.
3. Build candidate sequences:
   - `candidate_actions = clip(base_plan + noise, action bounds)`
   - `candidate_virtual_speed = clip(base_virtual_speed_plan + virtual speed noise, min/max)`.
4. Evaluate rollout cost batch:
   - `costs = self._rollout_costs(state, candidate_actions, candidate_virtual_speed, prev_action, progress_s_ft)`.
5. Add perturbation regularization:
   - `gamma * ||noise/sigma||^2` for action noise and virtual-speed noise.
6. Convert costs to MPPI weights:
   - `weights = softmax_weights(total_costs, lambda_)`.
7. Weighted update step:
   - `weighted_noise = Σ_i w_i * noise_i`
   - `weighted_virtual_speed_noise = Σ_i w_i * vs_noise_i`
   - `base_plan += weighted_noise` (clipped)
   - `base_virtual_speed_plan += weighted_virtual_speed_noise` (clipped).

After iterations:
- stores updated plans.
- if debug enabled, stores candidate/final rollout traces for visualization.
- chooses control to apply as first element of updated plan.
- chooses virtual speed similarly.
- advances internal progress:
  - `s <- clip(s + dt * virtual_speed_fps, path bounds)`.

### `get_tracking_metrics` (lines 294-328)
- Computes diagnostics against contouring reference at current progress:
  - 3D position error
  - lag projection along path tangent
  - contour component orthogonal to tangent
  - contour/lag/progress/virtual-speed cost estimates
- Returns values consumed by terminal printouts and CSV diagnostics.

### `get_action` (lines 333-351)
- Replan scheduling:
  - if not yet at replan interval:
    - returns cached action
    - still advances progress using cached virtual speed.
  - else:
    - `_shift_plan()`
    - `_optimize(current_state)`
    - cache/update bookkeepers
    - return optimized action.

---

## 14) `mppi_support.py`: Adaptation Layer Between Controller and Backend

File: [`jsbsim_gym/mppi_support.py`](./jsbsim_gym/mppi_support.py)

Purpose:
- Keep controller code clean by handling:
  - parameter preprocessing
  - trajectory/path interpolation
  - backend config translation
  - closure construction for rollout cost/positions

### Dataclasses (lines 55-97)
- `MPPICostConfig`: cost and constraints payload passed into backend.
- `NominalJSBSimParams`: packed JAX + NumPy arrays for reference path and terrain data.

### Utility wrappers (lines 107-159)
- `clip_action`, `make_trim_action_plan`, `make_trim_virtual_speed_plan`, and state conversion to JAX array.

### `_build_spatial_path` (lines 162-185)
- Takes reference 3D positions.
- Removes near-duplicate points.
- Computes arc-length coordinate `s`.
- Computes normalized path tangents by finite differencing.

### `build_nominal_params` (lines 188-244)
- Loads surrogate model weights `W,B,poly_powers`.
- Validates reference trajectory and speed arrays.
- Builds path interpolants (`s`, positions, tangents).
- Validates terrain grid shape/finite values.
- Packs both JAX arrays and NumPy arrays into one object.

### Reference interpolation helpers (lines 247-300)
- index-based and progress-based accessors for reference state/heading.
- `_interp_path_position_and_tangent_np` does scalar interpolation at progress `s`.

### Backend config translation (lines 303-331)
- `_backend_cost_config` builds backend-native `JaxMPPIConfig` from `MPPICostConfig`.

### Cost/rollout closure factories (lines 334-368)
- `build_rollout_cost_fn`:
  - builds backend rollout-state function
  - builds backend rollout-cost function
  - returns closure with controller-facing signature.
- `build_rollout_positions_fn`:
  - returns backend trajectory rollout callable for visualization/debug.

---

## 15) `_mppi_backend.py`: Dynamics, Rollout, and Cost Math

File: [`jsbsim_gym/_mppi_backend.py`](./jsbsim_gym/_mppi_backend.py)

This module is where the algorithm’s physics and per-stage objective are actually computed.

### 15.1 Config and metadata structures
- `JaxMPPIConfig` / `JaxSmoothMPPIConfig` (lines 81-239):
  - immutable backend config
  - pytree flatten/unflatten support so configs can flow through JAX transforms.

### 15.2 Surrogate-model loading and feature expansion
- `_build_polynomial_powers` (lines 242-252):
  - generates monomial exponents for polynomial feature map.
- `load_nominal_weights` (lines 255-307):
  - reads `nominal_coeff_weights.npz`
  - validates feature names, target names, model-space metadata, polynomial compatibility.
- `expand_poly` (lines 310-311):
  - evaluates polynomial basis for one feature vector.

### 15.3 Core helper math
- `softmax_weights` (lines 322-327):
  - numerically-stable MPPI weighting:
    - subtract min and max before exponentiation.
- `moments_to_angular_rate_derivatives` (lines 330-346):
  - rigid-body rotational dynamics from moments and inertia tensor.

### 15.4 Surrogate dynamics integration

#### `f16_kinematics_step` (lines 349-442)
- State transition for one `DT=1/30 s` step using aerodynamic-coefficient surrogate.
- Pipeline:
  1. Extract state/action.
  2. Build aerodynamic features (`alpha`, `beta`, `mach`, rates, controls).
  3. Predict aero coefficients (`C_X..C_N`) via polynomial regression.
  4. Convert to forces/moments using `qbar`, geometry, mass/inertia constants.
  5. Integrate translational and rotational equations.
  6. Compute inertial frame position derivatives and integrate.
  7. Clip derivatives for numerical robustness.

#### `f16_kinematics_step_with_load_factors` (lines 445-543)
- Same as above but returns extended state with:
  - `ny = Y/g`
  - `nz = -Z/g`
- MPPI cost uses these for load-factor limits.

### 15.5 Rollout engines
- `rollout_trajectory_with_load_factors` (lines 546-552):
  - uses `jax.lax.scan` to roll one action sequence.
- `rollout_trajectory_batch_with_load_factors` (lines 556-559):
  - vmaps the single-rollout function over batch dimension.

### 15.6 Smooth-noise operator
- `smooth_noise_batch` (lines 562-579):
  - temporally smooths sampled noise with 1D convolution kernel.
  - used only in smooth MPPI variant.

### 15.7 Terrain and path interpolation primitives
- `_terrain_elevation_ft_at` (lines 582-605):
  - bilinear interpolation on terrain grid.
- `_path_reference_at_s` (lines 608-620):
  - linear interpolation of position + tangent along arclength axis.

### 15.8 Per-rollout cost function

#### `single_rollout_cost_from_states` (lines 623-706)
- Inputs:
  - `state_seq`, `action_seq`, `virtual_speed_seq`
  - path and terrain arrays
  - cost config.
- Steps:
  1. Clip actions, build previous-action sequence for rate term.
  2. Build progress sequence:
     - `s_t = s_0 + DT * cumsum(virtual_speed_seq)`
     - clipped to path bounds.
  3. Interpolate path reference at each `s_t`.
  4. Compute position error decomposition:
     - lag error = projection on tangent
     - contour error = orthogonal residual.
  5. Stage contouring terms:
     - `contour_weight * ||contour||^2`
     - `lag_weight * lag^2`
     - progress-speed terms:
       - `-progress_reward_weight * v_s`
       - `virtual_speed_weight * v_s^2`
  6. Terrain term:
     - `hagl = h - terrain_elevation`
     - if `hagl <= 0`: collision penalty
     - else soft repulsion:
       - `terrain_repulsion_scale * exp(-decay * (hagl - safe_clearance))`
       - capped at collision penalty.
  7. Action-rate term:
     - weighted squared delta of commands.
  8. Limit term:
     - nz lower/upper violations squared
     - alpha-over-limit squared.
  9. Sum stage costs.
  10. Collision truncation behavior:
      - after first collision sample, future stage costs are zeroed.
  11. Return total rollout cost (sum of active stage costs).

### 15.9 Batched JIT wrappers
- `build_rollout_state_batch_fn` (lines 709-713): returns jitted state rollout function.
- `build_rollout_cost_from_states_fn` (lines 716-751): vmapped + jitted rollout-cost function.
- `build_rollout_positions_fn` (lines 754-763): returns full position traces including initial state (for debug plots).

---

## 16) `smooth_mppi_jax/controller.py`: How Smooth MPPI Differs

File: [`jsbsim_gym/smooth_mppi_jax/controller.py`](./jsbsim_gym/smooth_mppi_jax/controller.py)

### `JaxSmoothMPPIConfig` (lines 21-26)
- Inherits standard config, overrides/extends noise defaults:
  - action noise defaults
  - delta-noise std
  - delta bounds
  - smoothing kernel
  - seed

### `JaxSmoothMPPIController._optimize` (lines 47-141)
- Same MPPI structure as base controller with one key change:
  - samples **delta noise** (`raw_noise`)
  - applies temporal smoothing kernel via `smooth_noise_batch`
  - clips smoothed deltas by `delta_action_bounds`
  - applies these bounded smooth deltas to base plan.
- Everything else (rollout cost, softmax weights, weighted update, progress advance) matches base algorithm.

Interpretation:
- Base MPPI: iid control perturbations in time.
- Smooth MPPI: temporally correlated perturbations, usually giving smoother candidate control sequences.

---

## 17) End-to-End MPPI Stack: One Control Tick

Given `controller.get_action(controller_state)` in `run_scenario`:

1. `run_scenario` converts env state via `to_mppi_state`.
2. `JaxMPPIController.get_action` decides:
   - reuse cached action (between replans), or
   - run `_optimize`.
3. `_optimize`:
   - shift warm-start plan
   - sample action and virtual-speed noise
   - build candidate control sequences
   - call `self._rollout_costs` closure.
4. `self._rollout_costs` (from `mppi_support`) does:
   - `rollout_states = backend rollout over surrogate dynamics`
   - `cost = backend single_rollout_cost_from_states vmapped`.
5. Backend computes stage cost terms for each rollout and sums.
6. Controller computes softmax weights from total rollout costs and updates nominal plan.
7. Controller emits first action from updated plan.
8. Controller advances internal progress `s` using chosen virtual speed.
9. `run_scenario` applies action to env and logs diagnostics.

---

## 18) Symbol/Diagnostic Mapping

When terminal prints this header:
- `Prog`: `progress_s_ft`
- `CErr`: contour error magnitude (orthogonal-to-path)
- `Lag`: signed lag error along tangent
- `PosErr`: full 3D position error norm
- `dH`: altitude component of position error
- `Clr`: terrain clearance from env
- `Cost`: reconstructed stage cost estimate in `run_scenario` diagnostics logic

This is why `CErr` and `PosErr` differ:
- `PosErr` includes both contour and lag components.
- `CErr` intentionally excludes lag by projecting out tangent-direction error.

---

## 19) Practical Reading Order (If You Keep Digging)

Best order to read code in-place:
1. `run_scenario.py` (`main`, `to_mppi_state`, diagnostics accumulation)
2. `mppi_run_config.py` (how defaults+tuning become effective config)
3. `mppi_jax/controller.py` (`get_action`, `_optimize`)
4. `mppi_support.py` (parameter packing and closure adapters)
5. `_mppi_backend.py` (dynamics + cost math)
6. `smooth_mppi_jax/controller.py` (noise smoothing differences)
