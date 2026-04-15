"""
Distributionally Robust Stochastic Gatekeeper (DRS-GK)
=======================================================

Implements Algorithm 2 from:
  "Distributionally Robust Probabilistic Safety Verification
   for Nonlinear Systems with Arbitrary Distributions"

The algorithm selects a switching time s_t from {1,...,M} such that
the switched policy π_S satisfies the distributionally robust chance
constraint (DRCC) with probability ≥ 1-δ.

Parallelism
-----------
- Outer vmap over M switching times
- Inner vmap over N noise samples
- lax.scan over T timesteps (the rollout)

Usage
-----
See `DRSGatekeeper` (the stateful Python wrapper) at the bottom of this file.

TODOs
-----
- Implement specialized `h` (unsafe-set distance) and `h_c` (PCIS membership distance) if scenarios go beyond simple track-half-width boundaries.
- Optimization: Enable XLA-native pre-caching for feature vectors to reduce JIT overhead during massive sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time


import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import scipy.special as sc_special

try:
    from uncertain_racecar_gym.dynamics import VehicleState  # type: ignore
except Exception:
    @dataclass
    class VehicleState:
        x: float
        y: float
        yaw: float
        vx: float
        vy: float
        yaw_rate: float
        steer: float
        throttle: float
        brake: float
        progress: float
        lateral_error: float
        heading_error: float
        wheel_rotation: float = 0.0

try:
    from uncertain_racecar_gym.uncertainty_jax import JAXEmpiricalData, sample_empirical_jax  # type: ignore
except Exception:
    class JAXEmpiricalData(NamedTuple):
        pass

    def sample_empirical_jax(*args, **kwargs):
        raise RuntimeError(
            "JAX empirical uncertainty sampling is unavailable in this environment. "
            "Use the offline noise_sampler hook instead."
        )

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = jax.Array

# A JAX-compatible state type stored as a flat float32 vector.
# The exact semantics are scenario-defined by the supplied dynamics / policy /
# safety callbacks.  The JSBSim integration uses:
# [p_N, p_E, h, u, v, w, p, q, r, phi, theta, psi]
STATE_DIM = 12

# The JSBSim integration uses [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd].
ACTION_DIM = 4


# ---------------------------------------------------------------------------
# Hyperparameter / state containers
# ---------------------------------------------------------------------------


class GatekeeperParams(NamedTuple):
    """
    Static hyperparameters for DRS-GK (Algorithm 2).

    Attributes
    ----------
    M : int
        Number of candidate switching times {1, ..., M}.
    T : int
        Rollout horizon (number of steps).
    N : int
        Number of noise trajectory samples per candidate switching time.
    delta : float
        Incorrectness tolerance.  The chosen switching time satisfies the DRCC
        with probability ≥ 1-δ.
    epsilon : float
        Per-decision failure tolerance (safety level).
    beta : float
        Wasserstein ambiguity-set radius.
    alpha : float
        Probability with which the backup policy renders C probabilistically
        controlled-invariant (PCI).  Used in the feasibility threshold
        (ε - α) / (1 - α).
    p : int
        Order of the Wasserstein distance (typically 1 or 2).
    lipschitz_mode : str
        How to compute the Lipschitz constant L_H.
        'autodiff' — estimate via jax.grad through the rollout (expensive).
        'fixed'    — use `lipschitz_constant` directly (cheap).
    lipschitz_constant : float
        Fixed L_H value used when lipschitz_mode='fixed'.
    """

    M: int
    T: int
    N: int
    delta: float
    epsilon: float
    beta: float
    alpha: float
    p: int = 1
    lipschitz_mode: str = "fixed"   # 'fixed' | 'autodiff'
    lipschitz_constant: float = 0.0
    debug_timing: bool = False


class GatekeeperState(NamedTuple):
    """
    Mutable runtime state for DRS-GK.

    Attributes
    ----------
    t : int
        Current timestep (absolute, not within any rollout).
    s_prev : int
        Switching time from the previous iteration.  If no new feasible time
        is found the algorithm falls back to this value.
    predicted_trajectories : Optional[np.ndarray]
        Visualization data.
    failure_mask : Optional[np.ndarray]
        Boolean mask of failures for the predicted trajectories.
    m_star : int
        Distance to switching point.
    q_bar_star : float
        Estimated failure probability bound for the chosen switching time.
    """

    t: int
    s_prev: int
    using_backup: bool = False
    predicted_trajectories: Optional[np.ndarray] = None
    failure_mask: Optional[np.ndarray] = None
    m_star: int = 0
    q_bar_star: float = 0.0
    is_reverting: bool = False
    plan_start_t: int = 0

# ---------------------------------------------------------------------------
# Track-boundary observation model  (perception module for θ)
# ---------------------------------------------------------------------------
# Defined in a separate module for use within rollouts and the DRSGatekeeper
# class.  The estimate is generic despite the historical name.
from drs_gatekeeper.track_bounds import TrackBoundsEstimate  # noqa: F401

# State packing / unpacking helpers
# ---------------------------------------------------------------------------


def pack_state(x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake,
               progress, lateral_error, heading_error) -> Array:
    """Pack individual state components into a flat JAX array."""
    return jnp.array(
        [x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake,
         progress, lateral_error, heading_error],
        dtype=jnp.float32,
    )


def unpack_state(s: Array):
    """Unpack a flat state vector into named components."""
    return dict(
        x=s[0], y=s[1], yaw=s[2],
        vx=s[3], vy=s[4], yaw_rate=s[5],
        steer=s[6], throttle=s[7], brake=s[8],
        progress=s[9], lateral_error=s[10], heading_error=s[11],
    )


def jax_feature_extractor(state: Array, curvature: float, lr: float = 0.16) -> Array:
    """
    Extract a JAX-native feature vector from the flat state vector.
    Used to drive the empirical sampler inside the JAX kernel.
    """
    # state: [x, y, yaw, vx, vy, yaw_rate, steer, throttle, brake, progress, lateral_error, heading_error]
    vx = state[3]
    vy = state[4]
    yaw_rate = state[5]
    steer = state[6]
    throttle = state[7]
    brake = state[8]
    progress = state[9]

    # Proxies for telemetry-based features
    accel_y = vx * yaw_rate
    rear_slip_angle = jnp.arctan2(vy - lr * yaw_rate, jnp.maximum(jnp.abs(vx), 0.5))
    
    feat = jnp.zeros(21)
    feat = feat.at[0].set(curvature)
    feat = feat.at[1].set(progress)
    feat = feat.at[2].set(vx)
    feat = feat.at[3].set(vy)
    feat = feat.at[4].set(yaw_rate)
    feat = feat.at[5].set(steer)
    feat = feat.at[6].set(throttle)
    feat = feat.at[7].set(brake)
    feat = feat.at[13].set(rear_slip_angle)
    # Simplified gear/rpm proxies for regime logic
    feat = feat.at[14].set(jnp.clip(vx / 50.0, 0.0, 1.0))
    # History placeholders
    feat = feat.at[18].set(steer)
    feat = feat.at[19].set(throttle)
    feat = feat.at[20].set(brake)
    
    return feat


def rollout_single(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """
    Standard rollout with pre-sampled noise trajectories.
    """
    def step_fn(carry, inputs):
        state, step_idx = carry
        noise_step, env_param_step = inputs
        use_nominal = step_idx < switching_offset
        
        # Policy-based (PPO) or open-loop nominal (MPPI)
        if nominal_trajectory is not None:
             action_nominal = nominal_trajectory[step_idx]
        else:
             # This is where PPO's policy_fn is called during the rollout
             action_nominal = nominal_policy_fn(state)

        action = jax.tree.map(
            lambda n, b: jnp.where(use_nominal, n, b),
            action_nominal,
            backup_policy_fn(state),
        )
        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_param_step)
        # Return state for trajectory visualization
        xy = jnp.stack([state[0], state[1]], axis=0)
        return (next_state, step_idx + 1), (h_val, xy)

    final_carry, (h_values, xy_traj) = lax.scan(
        step_fn, (x0, 0), (noise_traj, env_params_traj), length=T
    )
    final_xy = jnp.stack([final_carry[0][0], final_carry[0][1]], axis=0)
    xy_traj = jnp.concatenate([xy_traj, final_xy[None, :]], axis=0)
    
    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), xy_traj


def rollout_single_empirical(
    x0: Array,
    switching_offset: Array,
    rng_key: Array,
    env_params: Array,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    empirical_feature_fn: Callable,
    empirical_sampler_fn: Callable,
    lr: float = 0.16,
    nominal_trajectory: Optional[Array] = None,
) -> Array:
    """
    Rollout with ONLINE empirical sampling.
    """
    def step_fn(carry, _):
        state, step_idx, key, prev_action = carry
        use_nominal = step_idx < switching_offset
        
        # Policy-based (PPO) or open-loop nominal (MPPI)
        if nominal_trajectory is not None:
             action_nominal = nominal_trajectory[step_idx]
        else:
             # This is where PPO's policy_fn is called during the empirical rollout
             action_nominal = nominal_policy_fn(state)

        action = jax.tree.map(
            lambda n, b: jnp.where(use_nominal, n, b),
            action_nominal,
            backup_policy_fn(state),
        )
        
        # Sample noise online (autoregressive)
        key, subkey = jax.random.split(key)
        feat = empirical_feature_fn(state, action, prev_action, step_idx)
        noise_step = empirical_sampler_fn(feat, gate_id, subkey, uncertainty_data)

        next_state = dynamics_fn(state, action, noise_step)
        h_val = safety_fn(next_state, env_params)
        # Return state for trajectory visualization
        xy = jnp.stack([state[0], state[1]], axis=0)
        return (next_state, step_idx + 1, key, action), (h_val, noise_step, xy)

    if nominal_trajectory is not None:
        initial_prev_action = nominal_trajectory[0]
    else:
        initial_prev_action = nominal_policy_fn(x0)

    final_carry, (h_values, noise_traj, xy_traj) = lax.scan(
        step_fn,
        (x0, 0, rng_key, initial_prev_action),
        None,
        length=T,
    )
    final_xy = jnp.stack([final_carry[0][0], final_carry[0][1]], axis=0)
    xy_traj = jnp.concatenate([xy_traj, final_xy[None, :]], axis=0)

    h_c_terminal = pcis_fn(final_carry[0])
    return jnp.minimum(h_c_terminal, jnp.min(h_values)), noise_traj, xy_traj


def rollout_with_grad(
    x0: Array,
    switching_offset: Array,
    noise_traj: Array,
    env_params_traj: Array,
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Compute H_T and its gradient w.r.t. noise_traj."""
    def h_only(w):
        # rollout_single returns (h_min, xy_traj)
        return rollout_single(
            x0, switching_offset, w, env_params_traj,
            dynamics_fn, nominal_policy_fn, backup_policy_fn,
            safety_fn, pcis_fn, T,
            nominal_trajectory=nominal_trajectory
        )
    # Use has_aux=True because h_only returns (scalar, auxiliary_data)
    (val, aux), grad = jax.value_and_grad(h_only, has_aux=True)(noise_traj)
    return (val, aux), grad


# Parallelised rollout over N samples × M switching times
# ---------------------------------------------------------------------------


def build_rollout_all(
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
) -> Callable:
    """
    Build and return a vmapped function that computes H_T for all (m, i) pairs.

    The returned function has signature::

        rollout_all(x0, switching_offsets, noise_batch, env_params_batch, compute_grad=False)
          -> H_matrix, Grad_matrix

    If compute_grad is False, Grad_matrix is None.
    """

    def rollout_all(
        x0: Array,              # [STATE_DIM]
        switching_offsets: Array,  # [M] int32
        noise_batch: Array,     # [M, N, T, noise_dim]
        env_params_batch: Array,   # [M, N, T, theta_dim]
        compute_grad: bool = False,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[Array, Optional[Array], Array]:

        def _single_val(m, w, th):
            return rollout_single(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        def _single_grad(m, w, th):
            return rollout_with_grad(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        # vmap over N trajectories, then over M switching times
        if not compute_grad:
            single_N = jax.vmap(_single_val, in_axes=(None, 0, 0))
            all_MN = jax.vmap(single_N, in_axes=(0, 0, 0))
            H_matrix, X_all = all_MN(switching_offsets, noise_batch, env_params_batch)
            return H_matrix, None, X_all
        else:
            single_N = jax.vmap(_single_grad, in_axes=(None, 0, 0))
            all_MN = jax.vmap(single_N, in_axes=(0, 0, 0))
            (H_matrix, X_all), Grad_matrix = all_MN(switching_offsets, noise_batch, env_params_batch)
            return H_matrix, Grad_matrix, X_all

    return rollout_all


def build_rollout_all_empirical(
    dynamics_fn: Callable,
    nominal_policy_fn: Callable,
    backup_policy_fn: Callable,
    safety_fn: Callable,
    pcis_fn: Callable,
    T: int,
    uncertainty_data: JAXEmpiricalData,
    gate_id: int,
    empirical_feature_fn: Callable,
    empirical_sampler_fn: Callable,
) -> Callable:
    """
    Build a JIT-able function that rollouts out all M x N candidates
    with ONLINE empirical sampling.
    """
    def rollout_all(
        x0: Array,
        switching_offsets: Array,
        rng_matrix: Array,     # [M, N, 2] PRNG keys
        env_params_batch: Array, # [M, N, theta_dim]
        compute_grad: bool = False,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[Array, Optional[Array], Array]:
        
        # 1. Autoregressive Sampling Pass
        # We run the rollout and collect the realized noise trajectory [M, N, T, 3]
        def _single_sample(m, key, th):
            return rollout_single_empirical(
                x0, m, key, th, uncertainty_data, gate_id,
                dynamics_fn, nominal_policy_fn, backup_policy_fn,
                safety_fn, pcis_fn, T,
                empirical_feature_fn, empirical_sampler_fn,
                nominal_trajectory=nominal_trajectory
            )

        v_sample = jax.vmap(_single_sample, in_axes=(None, 0, 0))
        all_MN_sample = jax.vmap(v_sample, in_axes=(0, 0, 0))
        
        # [M, N], [M, N, T, 3] and [M, N, T+1, 2]
        H_matrix, sampled_noise_batch, X_all = all_MN_sample(switching_offsets, rng_matrix, env_params_batch)

        if not compute_grad:
            return H_matrix, None, X_all

        # 2. Gradient Pass (Conditional)
        # Use the realized noise as if it were fixed input for the differentiable core
        def _single_grad(m, w, th):
            return rollout_with_grad(
                x0, m, w, th, dynamics_fn, nominal_policy_fn,
                backup_policy_fn, safety_fn, pcis_fn, T,
                nominal_trajectory=nominal_trajectory
            )

        v_grad = jax.vmap(_single_grad, in_axes=(None, 0, 0))
        all_MN_grad = jax.vmap(v_grad, in_axes=(0, 0, 0))
        
        # We re-run to get derivatives w.r.t. the fixed noise sequence
        _, Grad_matrix = all_MN_grad(switching_offsets, sampled_noise_batch, env_params_batch)
        return H_matrix, Grad_matrix, X_all

    return rollout_all



# ---------------------------------------------------------------------------
# Clopper-Pearson upper confidence bound
# ---------------------------------------------------------------------------


def _betaincinv_scipy(a: float, b: float, y: float) -> float:
    """
    Scalar fallback: inverse of the regularised incomplete beta function
    using scipy.special.betaincinv.

    Returns the value x in [0, 1] such that betainc(a, b, x) = y.
    """
    # All-failed edge case: betaincinv undefined when b ≤ 0
    if b <= 0.0:
        return 1.0
    return float(sc_special.betaincinv(float(a), float(b), float(y)))


def _betaincinv_callback(a: Array, b: Array, y: Array) -> Array:
    """
    JAX pure_callback wrapper around scipy.special.betaincinv.

    This allows betaincinv to be used inside jax.jit and jax.vmap even
    when the installed JAX version (e.g. 0.9.x) does not expose
    jax.scipy.special.betaincinv.

    ``vmap_method='sequential'`` ensures it can be batched inside jax.vmap
    by running the callback once per element sequentially.
    """
    def _impl(a_np, b_np, y_np):
        a_s, b_s, y_s = float(a_np), float(b_np), float(y_np)
        if b_s <= 0.0:
            return np.float32(1.0)
        return np.float32(sc_special.betaincinv(a_s, b_s, y_s))

    return jax.pure_callback(
        _impl,
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        y.astype(jnp.float32),
        vmap_method="sequential",
    )



def compute_q_bar(
    k: int,
    N: int,
    rho: float,
) -> float:
    """
    Compute the Clopper-Pearson upper bound on the failure probability.

    This implements Theorem 4 of Vincent et al. (2024)::

        q̄ = max{ q' ∈ [0,1] | Bin(k; N, q') ≥ ρ }
           = Beta.ppf(1 - ρ, k+1, N-k)

    Using the identity::
        1 - CDF_Bin(k; N, p) = CDF_Beta(p; k+1, N-k)

    we have::
        q̄ = betaincinv(k+1, N-k, 1-ρ)

    Implemented via scipy.special.betaincinv (pure Python / NumPy).
    This function is NOT JAX-traceable — it is called on concrete Python
    scalars after the JAX rollout stage returns k_vec to the host.

    Edge cases:
    - If k == N (all failures), q̄ = 1.
    - If k == 0, q̄ = betaincinv(1, N, 1-ρ) = (1-ρ)^{1/N}.

    Parameters
    ----------
    k : int
        Number of observed failures out of N samples (concrete int).
    N : int
        Total number of samples.
    rho : float
        Per-candidate incorrectness level.

    Returns
    -------
    q_bar : float
        Upper confidence bound on the true failure probability.
    """
    k = int(k)
    b = N - k
    if b <= 0:
        return 1.0
    a = k + 1.0
    return float(sc_special.betaincinv(float(a), float(b), float(1.0 - rho)))


def compute_q_bar_vec(k_vec: np.ndarray, N: int, rho: float) -> np.ndarray:
    """
    Vectorised wrapper: apply compute_q_bar to each element of k_vec.

    Parameters
    ----------
    k_vec : np.ndarray [M] int
        Observed failure counts per candidate switching time.
    N : int
        Number of samples per candidate.
    rho : float
        Per-candidate incorrectness level.

    Returns
    -------
    q_bars : np.ndarray [M] float32
    """
    return np.array(
        [compute_q_bar(int(k), N, rho) for k in k_vec],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Core algorithm — JIT-able
# ---------------------------------------------------------------------------


def _compute_h_and_failures(
    x0: Array,
    switching_offsets: Array,
    noise_batch: Array,
    env_params_batch: Array,
    dr_buffer: float,
    rollout_all_fn: Callable,
    compute_grad: bool = False,
) -> Tuple[Array, Array, Optional[Array]]:
    """
    JAX-JITable kernel: compute H_matrix and k_vec.

    Returns k_vec [M] int32, H_matrix [M, N], Grad_matrix [M, N, T, d]
    """
    H_matrix, Grad_matrix = rollout_all_fn(
        x0, switching_offsets, noise_batch, env_params_batch,
        compute_grad=compute_grad
    )
    failure_mask = H_matrix <= jnp.float32(dr_buffer)
    k_vec = jnp.sum(failure_mask, axis=1).astype(jnp.int32)
    return k_vec, H_matrix, Grad_matrix


def run_gatekeeper(
    x0: Array,                  # [STATE_DIM] current flat state
    s_prev: int,                # previous switching time (absolute, Python int)
    t: int,                     # current absolute timestep (Python int)
    switching_offsets: Array,   # [M] int32 — candidate relative offsets {1,...,M}
    noise_batch: Optional[Array], # [M, N, T, noise_dim] (None if empirical)
    env_params_batch: Array,    # [M, N, T, theta_dim]  or  [M, N, theta_dim]
    params: GatekeeperParams,
    rollout_all_fn: Callable,   # standard offline rollout
    uncertainty_data: Optional[JAXEmpiricalData] = None,
    rollout_all_empirical_fn: Optional[Callable] = None,
    rng_key: Optional[Array] = None,
    nominal_trajectory: Optional[Array] = None,
) -> Tuple[int, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int, float, dict]:
    """
    Execute one iteration of DRS-GK (Algorithm 2).

    Design note
    -----------
    The rollout and failure-counting stages are JAX-JITable.  The
    Clopper-Pearson q̄ computation is performed on the CPU with
    scipy.special.betaincinv (which is not JAX-traceable) after the
    JAX kernel returns concrete k_vec values.  The selection step then
    uses pure Python/NumPy.

    Parameters
    ----------
    x0 : Array [STATE_DIM]
    s_prev : int
        Switching time committed in the previous iteration.
    t : int
        Current absolute timestep.
    switching_offsets : Array [M] int32
        Candidate relative offsets {1, ..., M}.
    noise_batch : Array [M, N, T, noise_dim]
    env_params_batch : Array [M, N, T, theta_dim]
    params : GatekeeperParams
    rollout_all_fn : callable built by build_rollout_all

    Returns
    -------
    s_t : int
        New switching time (absolute timestep).
    I_valid : np.ndarray [M] bool
        Feasible switching times.
    q_bars : np.ndarray [M] float32
        Confidence bounds on failure probability for each candidate.
    L_H : float
        The Lipschitz constant used (either fixed or estimated).
    """
    # --- Step 1+3: JAX stage — rollouts + failure counting ---
    timings = {
        "jax_rollout": 0.0,
        "lipschitz_failure_counting": 0.0,
        "cpu_selection": 0.0,
    }
    compute_grad = (params.lipschitz_mode == "autodiff")
    
    L_H_fixed = float(params.lipschitz_constant)
    dr_buffer_init = L_H_fixed * (float(params.beta) ** (1.0 / float(params.p)))

    # t0 = time.perf_counter()
    if uncertainty_data is not None and rollout_all_empirical_fn is not None and rng_key is not None:
        # Step 1: Rollout with ONLINE empirical sampling
        # Split key for vmap: [M, N, 2]
        rng_matrix = jax.random.split(rng_key, params.M * params.N).reshape(params.M, params.N, 2)
        H_matrix, Grad_matrix, X_all = rollout_all_empirical_fn(
            x0, switching_offsets, rng_matrix, env_params_batch,
            compute_grad, nominal_trajectory
        )
    else:
        # Step 1: Rollout with pre-sampled noise
        H_matrix, Grad_matrix, X_all = rollout_all_fn(
            x0, switching_offsets, noise_batch, env_params_batch,
            compute_grad, nominal_trajectory
        )
    
    # Ensure JAX operations are finished for accurate timing
    # H_matrix.block_until_ready()
    # if Grad_matrix is not None:
    #     Grad_matrix.block_until_ready()
    # X_all.block_until_ready()
    # timings["jax_rollout"] = time.perf_counter() - t0

    # failure counting
    # t1 = time.perf_counter()
    failure_mask_init = H_matrix <= jnp.float32(dr_buffer_init)
    k_vec_jax = jnp.sum(failure_mask_init, axis=1).astype(jnp.int32)
    
    if compute_grad and Grad_matrix is not None:
        # L_H = max_{m, i} ||grad_w H(w_i, m)||_2
        # Grad_matrix shape: [M, N, T, noise_dim]
        # Reshape to [M, N, T*noise_dim] to compute trajectory-wise norm
        grad_norms = jnp.linalg.norm(Grad_matrix.reshape(params.M, params.N, -1), axis=-1)
        L_H = float(jnp.max(grad_norms))
        # print(f"Estimated L_H: {L_H:.4f}")
        
        # Re-compute failures with the true dr_buffer
        dr_buffer = L_H * (float(params.beta) ** (1.0 / float(params.p)))
        failure_mask = H_matrix <= jnp.float32(dr_buffer)
        k_vec_jax = jnp.sum(failure_mask, axis=1).astype(jnp.int32)
    else:
        L_H = L_H_fixed

    k_vec = np.array(k_vec_jax)  # to CPU: [M] int32
    # timings["lipschitz_failure_counting"] = time.perf_counter() - t1

    # --- Step 4: CPU stage — Clopper-Pearson bounds ---
    # t2 = time.perf_counter()
    rho = 1.0 - (1.0 - params.delta) ** (1.0 / params.M)
    q_bars = compute_q_bar_vec(k_vec, params.N, rho)

    # --- Step 5: Determine feasible set ---
    feasibility_threshold = (params.epsilon - params.alpha) / (1.0 - params.alpha)
    I_valid = q_bars <= float(feasibility_threshold)

    # --- Step 6: Select maximum feasible switching time ---
    abs_times = t + np.asarray(switching_offsets, dtype=int)
    if np.any(I_valid):
        m_idx_star = int(np.where(I_valid)[0][-1])
        s_t = int(abs_times[m_idx_star])
    else:
        m_idx_star = 0 # Default to earliest
        s_t = int(s_prev)

    # ExtractTrajectories for visualization
    # X_all shape [M, N, T+1, 2]
    trajectories = np.asarray(X_all[m_idx_star], dtype=np.float32)
    # failure_mask_init is [M, N] boolean
    mask = np.asarray(failure_mask_init[m_idx_star], dtype=bool)
    m_star = int(switching_offsets[m_idx_star])
    q_bar_star = float(q_bars[m_idx_star])

    # timings["cpu_selection"] = time.perf_counter() - t2

    return s_t, I_valid, q_bars, L_H, trajectories, mask, m_star, q_bar_star, timings


# ---------------------------------------------------------------------------
# Python-level stateful wrapper
# ---------------------------------------------------------------------------


class DRSGatekeeper:
    """
    Stateful wrapper for the Distributionally Robust Stochastic Gatekeeper.

    This class:
    - Holds the JAX-JIT-compiled ``run_gatekeeper`` call.
    - Tracks the mutable runtime state (``GatekeeperState``).
    - Pre-samples noise batches (currently a TODO stub).
    - Dispatches to the nominal or backup policy based on ``s_t``.

    Parameters
    ----------
    params : GatekeeperParams
        Algorithm hyperparameters.
    dynamics_fn : callable
        JAX-compatible stepped dynamics.
        Signature: (state [STATE_DIM], action [ACTION_DIM], noise [noise_dim])
                   -> next_state [STATE_DIM].
    nominal_policy_fn : callable
        Signature: (state [STATE_DIM],) -> action [ACTION_DIM].
    backup_policy_fn : callable
        Signature: (state [STATE_DIM],) -> action [ACTION_DIM].
    safety_fn : callable
        Unsafe-set distance h(state, env_param) > 0 ↔ state is safe.
        Signature: (state [STATE_DIM], env_param [theta_dim]) -> scalar.
    pcis_fn : callable
        PCIS membership h_c(state) > 0 ↔ state is in C.
        Signature: (state [STATE_DIM],) -> scalar.
    noise_dim : int
        Dimension of process noise w^f at each step.
    theta_dim : int
        Dimension of environment-parameter samples w^θ.
    noise_sampler : callable or None
        Function that returns a noise batch of shape
        [M, N, T, noise_dim + theta_dim].
        If None, Gaussian noise is used as a placeholder (TODO).
    seed : int
        JAX random seed used internally if noise_sampler is None.

    Example
    -------
    >>> gk = DRSGatekeeper(params, dyn, pi_N, pi_B, h, h_c, noise_dim=3, theta_dim=0)
    >>> gk.reset(initial_state_flat, t=0)
    >>> action = gk.step(current_state_flat)  # returns π_N or π_B action
    """

    def __init__(
        self,
        params: GatekeeperParams,
        dynamics_fn: Callable,
        nominal_policy_fn: Callable,
        backup_policy_fn: Callable,
        safety_fn: Callable,
        pcis_fn: Callable,
        noise_dim: int,
        theta_dim: int = 2,
        noise_sampler: Optional[Callable] = None,
        uncertainty_model: Optional[EmpiricalUncertaintyModel] = None,
        empirical_feature_fn: Optional[Callable] = None,
        empirical_sampler_fn: Optional[Callable] = None,
        calibration_model: Optional[Any] = None,
        track: Optional[Any] = None,
        vehicle_config: Optional[Any] = None,
        initial_track_bounds: Optional["TrackBoundsEstimate"] = None,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        params : GatekeeperParams
            Algorithm hyperparameters.
        dynamics_fn : callable
            (state, action, noise) -> next_state  (JAX-compatible).
        nominal_policy_fn : callable
            (state,) -> action  (JAX-compatible).
        backup_policy_fn : callable
            (state,) -> action  (JAX-compatible).
        safety_fn : callable
            (state, env_param) -> scalar  JAX-compatible.
        pcis_fn : callable
            (state,) -> scalar  JAX-compatible.
        noise_dim : int
            Dimension of process noise w^f per step.
        theta_dim : int
            Dimension of the environment-parameter vector θ.
        noise_sampler : callable or None
            If provided, overrides internal sampling logic.
        uncertainty_model : EmpiricalUncertaintyModel or None
            Model for empirical state-dependent sampling.
        calibration_model : Any or None
            (Optional) Calibration model for mean residuals.
        track : TrackModel or None
            Reference to the track for context rollouts.
        vehicle_config : VehicleConfig or None
            Reference to vehicle parameters for feature building.
        initial_track_bounds : TrackBoundsEstimate or None
            Initial perception estimate.
        seed : int
            NumPy/JAX random seed.
        """
        self.params = params
        self.dynamics_fn = dynamics_fn # JAX function
        self.nominal_policy_fn = nominal_policy_fn
        self.backup_policy_fn = backup_policy_fn
        self.noise_sampler = noise_sampler
        self.uncertainty_model = uncertainty_model
        self.empirical_feature_fn = empirical_feature_fn
        self.empirical_sampler_fn = empirical_sampler_fn if empirical_sampler_fn is not None else sample_empirical_jax
        self.calibration_model = calibration_model
        self.track = track
        self.vehicle_config = vehicle_config
        self.noise_dim = noise_dim
        self.theta_dim = theta_dim
        self._rng = np.random.default_rng(seed)
        self._key = jax.random.PRNGKey(seed)
        self.last_L_H = float(params.lipschitz_constant)

        # Default track bounds
        self._track_bounds: TrackBoundsEstimate = (
            initial_track_bounds
            if initial_track_bounds is not None
            else TrackBoundsEstimate.unknown()
        )

        # Preferred: Device-resident uncertainty data
        self.uncertainty_data: Optional[JAXEmpiricalData] = None
        if uncertainty_model is not None and hasattr(uncertainty_model, "to_jax"):
            self.uncertainty_data = uncertainty_model.to_jax()

        # Build the vmapped rollout
        # We define two versions: one for pre-sampled noise, one for empirical JAX noise.
        # Note: rollout_all has compute_grad as the 5th argument (index 4).
        self._rollout_all = jax.jit(
            build_rollout_all(
                dynamics_fn=dynamics_fn,
                nominal_policy_fn=nominal_policy_fn,
                backup_policy_fn=backup_policy_fn,
                safety_fn=safety_fn,
                pcis_fn=pcis_fn,
                T=params.T,
            ),
            static_argnums=4,
        )
        
        if self.uncertainty_data is not None:
            feature_fn = self.empirical_feature_fn
            if feature_fn is None:
                def feature_fn(state, action, prev_action, step_idx):
                    del action, prev_action, step_idx
                    return jax_feature_extractor(state, 0.0)
            # We'll assume a default gate_id of 0 for now or manage it per step
            self._rollout_all_empirical = jax.jit(
                build_rollout_all_empirical(
                    dynamics_fn=dynamics_fn,
                    nominal_policy_fn=nominal_policy_fn,
                    backup_policy_fn=backup_policy_fn,
                    safety_fn=safety_fn,
                    pcis_fn=pcis_fn,
                    T=params.T,
                    uncertainty_data=self.uncertainty_data,
                    gate_id=0, # Default, can be overridden if needed
                    empirical_feature_fn=feature_fn,
                    empirical_sampler_fn=self.empirical_sampler_fn,
                ),
                static_argnums=4,
            )

        # Candidate switching time offsets {1, ..., M}
        self._switching_offsets = jnp.arange(0, params.M, dtype=jnp.int32)

        # Runtime state
        self.state: Optional[GatekeeperState] = None
        self.last_timings: dict = {}

    # ------------------------------------------------------------------
    # Noise sampling
    # ------------------------------------------------------------------

    def _sample_noise(
        self,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
        x_flat: Optional[Array] = None,
        nominal_trajectory: Optional[Array] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw process-noise and environment-parameter batches for one algorithm
        iteration.

        Process noise (w^f)
        ~~~~~~~~~~~~~~~~~~~
        Shape: [M, N, T, noise_dim].  Represents stochastic model error on the
        vehicle dynamics (e.g. delta_vx / delta_vy / delta_yaw_rate).

        Currently uses i.i.d. Gaussian as a placeholder.
        TODO: Replace with EmpiricalUncertaintyModel.sample() calls.  The
              empirical model is NumPy-based and context-dependent; the caller
              should pre-sample a batch and supply noise_sampler= instead.

        Environment parameters (w^θ)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Shape: [M, N, T, theta_dim].  Represents the uncertain track-boundary
        geometry drawn from P̂^θ_t = P(θ | x_t, z_t).

        If a TrackBoundsEstimate is provided (either here or via the stored
        self._track_bounds), samples are drawn from its Gaussian model.  Each
        (m, i) pair gets an independent θ sample, broadcast across all T steps
        (static parameter assumption).

        Parameters
        ----------
        track_bounds : TrackBoundsEstimate or None
            Perception estimate to use.  If None, uses self._track_bounds.

        Returns
        -------
        noise_batch : np.ndarray [M, N, T, noise_dim]  float32
        env_batch   : np.ndarray [M, N, T, theta_dim]  float32
        """
        p = self.params
        M, N, T = p.M, p.N, p.T
        bounds = track_bounds if track_bounds is not None else self._track_bounds

        if self.noise_sampler is not None:
            # Fully user-supplied sampler overrides everything.
            try:
                return self.noise_sampler(
                    M,
                    N,
                    T,
                    self._rng,
                    x_flat=x_flat,
                    nominal_trajectory=nominal_trajectory,
                    track_bounds=bounds,
                )
            except TypeError:
                return self.noise_sampler(M, N, T, self._rng)

        # --- Process noise (w^f): i.i.d. Gaussian placeholder ---
        # TODO: Replace with EmpiricalUncertaintyModel.sample() calls.
        self._key, subkey_f = jax.random.split(self._key)
        noise_batch = np.asarray(
            jax.random.normal(
                subkey_f,
                shape=(M, N, T, self.noise_dim),
                dtype=jnp.float32,
            )
        )

        # --- Environment parameters (w^θ): sample from TrackBoundsEstimate ---
        env_batch = bounds.sample_env_params(M, N, T, self._rng)  # [M,N,T,2]

        # If theta_dim != 2, pad or truncate to match
        if self.theta_dim != 2:
            if self.theta_dim == 0:
                env_batch = np.zeros((M, N, T, 0), dtype=np.float32)
            elif self.theta_dim > 2:
                pad = np.zeros((M, N, T, self.theta_dim - 2), dtype=np.float32)
                env_batch = np.concatenate([env_batch, pad], axis=-1)
            else:
                env_batch = env_batch[..., :self.theta_dim]

        return noise_batch, env_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, x0_flat: Array, t: int = 0) -> None:
        """
        Reset the gatekeeper at the start of an episode.

        Parameters
        ----------
        x0_flat : Array [STATE_DIM]
            Initial flat state.
        t : int
            Starting absolute timestep (typically 0).
        """
        # Initialise with s_prev = t (switch immediately = use backup from start)
        self.state = GatekeeperState(t=t, s_prev=t, using_backup=False, plan_start_t=t)

    def update(
        self,
        x_flat: Array,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
        nominal_trajectory: Optional[Array] = None,
        max_steps: Optional[int] = None,
    ) -> GatekeeperState:
        """
        Run one iteration of DRS-GK to update the switching time.

        Parameters
        ----------
        x_flat : Array [STATE_DIM]
            Flat state vector.
        track_bounds : TrackBoundsEstimate or None
            New perception estimate.
        nominal_trajectory : Array [T, ACTION_DIM] or None
            Optional open-loop future actions for the nominal policy.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before update().")

        if track_bounds is not None:
            self._track_bounds = track_bounds

        # 1. Sample environment parameters (and process noise if not in empirical mode)
        # We always call this to get the latest env_batch from self._track_bounds
        t_sample_start = time.perf_counter()
        noise_batch, env_batch = self._sample_noise(
            self._track_bounds,
            x_flat=x_flat,
            nominal_trajectory=nominal_trajectory,
        )
        t_sample = time.perf_counter() - t_sample_start

        # Prepare sampling arguments for JAX
        noise_batch_jax = None
        env_batch_jax = jnp.asarray(env_batch)
        rng_key_jax = None
        rollout_emp_fn = None
        
        if self.uncertainty_data is not None:
            # Online sampling in JAX — skip the offline noise_batch
            self._key, rng_key_jax = jax.random.split(self._key)
            rollout_emp_fn = self._rollout_all_empirical
        else:
            # Standard offline mode
            noise_batch_jax = jnp.asarray(noise_batch)

        s_t, I_valid, q_bars, L_H, trajectories, mask, m_star, q_bar_star, timings = run_gatekeeper(
            x0=x_flat,
            s_prev=self.state.s_prev,
            t=self.state.t,
            switching_offsets=self._switching_offsets,
            noise_batch=noise_batch_jax,
            env_params_batch=env_batch_jax,
            params=self.params,
            rollout_all_fn=self._rollout_all,
            uncertainty_data=self.uncertainty_data,
            rollout_all_empirical_fn=rollout_emp_fn,
            rng_key=rng_key_jax,
            nominal_trajectory=nominal_trajectory,
        )

        timings["sampling"] = t_sample
        total_time = sum(timings.values())

        if self.params.debug_timing:
            progress_str = f"Step {self.state.t}"
            if max_steps is not None:
                progress_str += f"/{max_steps}"
            print(f"\n--- DRS-GK Timing [{progress_str}] ---")
            print(f"  Sampling:          {timings['sampling']*1000:7.2f} ms")
            print(f"  JAX Rollout:       {timings['jax_rollout']*1000:7.2f} ms")
            print(f"  Lipschitz/Fail:    {timings['lipschitz_failure_counting']*1000:7.2f} ms")
            print(f"  CPU Selection:     {timings['cpu_selection']*1000:7.2f} ms")
            print(f"  --------------------------")
            print(f"  Total Update:      {total_time*1000:7.2f} ms")

        self.last_L_H = L_H
        self.last_timings = timings
        
        # Determine if we found a new feasible switching time
        found_new = np.any(I_valid)
        using_backup = self.state.t >= s_t

        # --- Pinning Logic ---
        # We only consider an update a "New Commitment" if s_t has specifically changed.
        # If s_t == s_prev, we are just adhering to a previously committed plan,
        # so we should keep the trajectories "pinned" to where they were first calculated.
        is_new_commitment = found_new and (s_t != self.state.s_prev)

        if is_new_commitment:
            # Fresh plan commitment: record trajectories starting from current state
            self.state = GatekeeperState(
                t=self.state.t,
                s_prev=s_t,
                using_backup=using_backup,
                predicted_trajectories=trajectories,
                failure_mask=mask,
                m_star=m_star,
                q_bar_star=q_bar_star, # Bound from when the switch was chosen
                is_reverting=False,
                plan_start_t=self.state.t, # Mark the birth of this plan
            )
        else:
            # Reversion or Adherence: sticking to the previously chosen s_prev.
            # We retain the trajectories and mask from whoever s_prev was originaly chosen.
            current_m_star = self.state.s_prev - self.state.t
            
            retained_trajectores = self.state.predicted_trajectories
            retained_mask = self.state.failure_mask
            retained_q_bar = self.state.q_bar_star
            retained_start_t = self.state.plan_start_t
            
            # Initialization fallback: if this is step 0 and we failed immediately,
            # the "chosen" switching time is s_prev=t (backup immediately).
            if retained_trajectores is None:
                retained_trajectores = trajectories
                retained_mask = mask
                retained_q_bar = q_bar_star
                retained_start_t = self.state.t

            self.state = GatekeeperState(
                t=self.state.t,
                s_prev=self.state.s_prev,
                using_backup=using_backup,
                predicted_trajectories=retained_trajectores,
                failure_mask=retained_mask,
                m_star=current_m_star,
                q_bar_star=retained_q_bar,
                is_reverting=(not found_new),
                plan_start_t=retained_start_t,
            )
        return self.state

    def act(self, x_flat: Array) -> Array:
        """
        Return the control action for the current state.

        Executes the nominal policy if ``t < s_t``, otherwise the backup.

        Parameters
        ----------
        x_flat : Array [STATE_DIM]
            Current flat state vector.

        Returns
        -------
        action : Array [ACTION_DIM]
        """
        if self.state is None:
            raise RuntimeError("Call reset() and update() before act().")

        if self.state.using_backup:
            return self.backup_policy_fn(x_flat)
        else:
            return self.nominal_policy_fn(x_flat)

    def tick(self) -> None:
        """Advance the internal timestep by one step."""
        if self.state is None:
            raise RuntimeError("Call reset() before tick().")
        t_new = self.state.t + 1
        using_backup = t_new >= self.state.s_prev
        self.state = GatekeeperState(
            t=t_new,
            s_prev=self.state.s_prev,
            using_backup=using_backup,
            predicted_trajectories=self.state.predicted_trajectories,
            failure_mask=self.state.failure_mask,
            m_star=self.state.m_star,
            q_bar_star=self.state.q_bar_star,
            is_reverting=self.state.is_reverting,
            plan_start_t=self.state.plan_start_t,
        )

    def step(
        self,
        x_flat: Array,
        track_bounds: Optional["TrackBoundsEstimate"] = None,
        nominal_trajectory: Optional[Array] = None,
    ) -> Array:
        """
        Convenience method combining update(), act(), and tick().

        Typical usage inside a control loop::

            action = gk.step(x_flat, track_bounds=perception_estimate)
            env.apply(action)

        Parameters
        ----------
        x_flat : Array [STATE_DIM]
            Current flat state vector.
        track_bounds : TrackBoundsEstimate or None
            Current perception estimate of the track boundaries.
            Pass None to reuse the last supplied estimate.
        nominal_trajectory : Array [T, ACTION_DIM] or None
            Optional open-loop future actions for the nominal policy.

        Returns
        -------
        action : Array [ACTION_DIM]
        """
        self.update(x_flat, track_bounds=track_bounds, nominal_trajectory=nominal_trajectory)
        action = self.act(x_flat)
        self.tick()
        return action

    # ------------------------------------------------------------------
    # Utility: convert JaxRacecarState to flat array
    # ------------------------------------------------------------------

    @staticmethod
    def state_from_flat(s_flat: Array) -> VehicleState:
        """Convert flat array back to VehicleState dataclass."""
        s = np.asarray(s_flat)
        return VehicleState(
            x=float(s[0]), y=float(s[1]), yaw=float(s[2]),
            vx=float(s[3]), vy=float(s[4]), yaw_rate=float(s[5]),
            steer=float(s[6]), throttle=float(s[7]), brake=float(s[8]),
            progress=float(s[9]), lateral_error=float(s[10]), heading_error=float(s[11]),
            wheel_rotation=0.0
        )

    @staticmethod
    def state_to_flat(s: VehicleState) -> Array:
        """Convert VehicleState dataclass to flat array."""
        return jnp.array([
            s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate,
            s.steer, s.throttle, s.brake,
            s.progress, s.lateral_error, s.heading_error
        ], dtype=jnp.float32)

    @staticmethod
    def state_from_jax_racecar(jax_state) -> Array:
        """
        Convert a ``JaxRacecarState`` NamedTuple into the flat STATE_DIM vector
        used internally by the gatekeeper.

        Parameters
        ----------
        jax_state : JaxRacecarState

        Returns
        -------
        flat : Array [STATE_DIM]
        """
        return jnp.array(
            [
                jax_state.x,
                jax_state.y,
                jax_state.yaw,
                jax_state.vx,
                jax_state.vy,
                jax_state.yaw_rate,
                jax_state.steer,
                jax_state.throttle,
                jax_state.brake,
                jax_state.progress,
                jax_state.lateral_error,
                jax_state.heading_error,
            ],
            dtype=jnp.float32,
        )

    @staticmethod
    def state_from_vehicle_state(vehicle_state) -> Array:
        """
        Convert a ``VehicleState`` dataclass into the flat STATE_DIM vector.

        Parameters
        ----------
        vehicle_state : VehicleState (from uncertain_racecar_gym.dynamics)

        Returns
        -------
        flat : Array [STATE_DIM]
        """
        return jnp.array(
            [
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw,
                vehicle_state.vx,
                vehicle_state.vy,
                vehicle_state.yaw_rate,
                vehicle_state.steer,
                vehicle_state.throttle,
                vehicle_state.brake,
                vehicle_state.progress,
                vehicle_state.lateral_error,
                vehicle_state.heading_error,
            ],
            dtype=jnp.float32,
        )


# ---------------------------------------------------------------------------
# Default safety / PCIS functions for the racecar scenario
# ---------------------------------------------------------------------------


def make_track_safety_fn(road_half_width: float) -> Callable:
    """
    Create a safety function h(state, env_param) for track-boundary safety.

    In the racecar setting the "unsafe set" is outside the track boundaries.
    The signed distance to the boundary is::

        h(x) = road_half_width - |lateral_error|

    Positive ↔ inside the track (safe).

    The environment-parameter ``env_param`` is unused here (the track geometry
    is known analytically).  When the track boundaries are uncertain, replace
    this with a sampled version.

    Parameters
    ----------
    road_half_width : float
        Half-width of the driveable track in metres.

    Returns
    -------
    safety_fn : callable  (state [STATE_DIM], env_param [*]) -> scalar float32
    """
    half_w = jnp.float32(road_half_width)

    def safety_fn(state: Array, env_param: Array) -> Array:
        lateral_error = state[10]
        return half_w - jnp.abs(lateral_error)

    return safety_fn


def make_track_pcis_fn(pcis_half_width: float) -> Callable:
    """
    Create a PCIS membership function h_c(state).

    The PCIS is the set of states with |lateral_error| < pcis_half_width,
    which the backup (centerline-following) policy renders probabilistically
    controlled-invariant.

    Parameters
    ----------
    pcis_half_width : float
        Half-width of the PCIS region (should be ≤ road_half_width).

    Returns
    -------
    pcis_fn : callable  (state [STATE_DIM],) -> scalar float32
    """
    half_w = jnp.float32(pcis_half_width)

    def pcis_fn(state: Array) -> Array:
        lateral_error = state[10]
        return half_w - jnp.abs(lateral_error)

    return pcis_fn
