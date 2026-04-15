# F-16 Data-Driven Uncertainty Modeling

This document formally outlines the data-centric system identification and empirical uncertainty calibration pipeline deployed inside the `jsbsim-gym` environment. Inspired by analogous hybrid architectures, this framework employs a fully deterministic baseline rigid body kinematic mapping and extracts the non-linear unmodeled aerodynamics directly from data via polynomial Ridge Regression. High-dimensional stochasticity is isolated applying a context-conditioned nearest-neighbor retrieval.

## Data Collection Pipeline

The data generation sequence constructs massive, offline sets describing temporal step tuples $\mathcal{D} = \{ x_t, z_t, u_t, x_{t+1}^{true} \}$ under the presence of spatially driven unobserved disturbances. 

### 1. Procedural Canyon Geometry
To introduce diverse, state-dependent domain friction, we propagate the F-16 through a continuously evolving geometric canyon. The absolute spatial width depends fundamentally upon the aircraft's Cartesian translation, namely the North displacement $p_N$.

$$ W_c(p_N) = W_{base} + \sum_i A_i \sin(\omega_i p_N + \phi_i) $$

The analytical gradients $\nabla W_c(p_N) = \frac{\partial{W_c}}{\partial{p_N}}$ represent local topographical contraction and expansion phenomena (e.g. Venturi effects compress local airflow boundaries).

### 2. State-Dependent Wind Shear 
The true dynamics are subject to unobserved continuous variations driven by a correlated Ornstein-Uhlenbeck (OU) formulation representing localized atmospheric airframe shear. These processes are inherently heteroscedastic, scaling their standard deviation inversely proportion to the lateral canyon clearance. 

$$ dW_{\text{wind}} = -\theta_{\text{OU}} (\mu - W_{\text{wind}})dt + \frac{\sigma_{\text{base}}}{W_c + \epsilon_W} d\mathcal{B}_t $$

Where $d\mathcal{B}_t$ exhibits classical Gaussian Brownian mechanics.

### 3. Persistent Excitation
System identification routines require dynamically rich control manifolds to bypass multicollinearity defects in the parameter estimations.
A tracking PID algorithm preserves general bounding, while independent, functionally orthogonal multisine perturbations inject excitation directly across the non-linear aerodynamic control surfaces.

$$ \delta_{cmd}^j(t) = \delta_{\text{nominal}}^j(t) + \sum_{k=1}^N \mathcal{P}_k^j \sin(2\pi f_k t + \theta_k) $$

The sequences are serialized out of the loop at discrete times producing an extremely clean target file: `f16_dataset.parquet`.

---

## System Identification & Uncertainty Calibration

Operating exclusively via offline dataset inspection, the process attempts to estimate a robust, completely discretized polynomial map for $x_{t+1} \approx f_{\text{nom}}(x_t, u_t)$.

### 1. Kinematic Target Extraction
Unlike typical deep-learning formulations, we utilize an explicit rigid-body decomposition that separates Coriolis and gravitational coupling from aerodynamic specific-force channels while preserving full rotational inertia coupling.

$$ \hat{X}_{t} = \frac{u_{t+1} - u_t}{\Delta t} - (r_t v_t - q_t w_t - g \sin\theta_t) $$
$$ \hat{Y}_{t} = \frac{v_{t+1} - v_t}{\Delta t} - (p_t w_t - r_t u_t + g \sin\phi_t \cos\theta_t) $$
$$ \hat{Z}_{t} = \frac{w_{t+1} - w_t}{\Delta t} - (q_t u_t - p_t v_t + g \cos\phi_t \cos\theta_t) $$

Angular accelerations are first recovered from the dataset:

$$ \dot p = \frac{p_{t+1} - p_t}{\Delta t},\quad \dot q = \frac{q_{t+1} - q_t}{\Delta t},\quad \dot r = \frac{r_{t+1} - r_t}{\Delta t}. $$

Then rotational channels are mapped to body moments via the full Euler rigid-body relation:

$$ \begin{bmatrix} \hat{L} \\ \hat{M} \\ \hat{N} \end{bmatrix} = I\,\dot\omega + \omega \times (I\omega),\quad \omega = [p, q, r]^T, $$

using the F-16 inertia tensor (including non-zero $I_{xz}$).

### 2. Analytical Ridge Regression (Nominal Base)
Rather than regressing dimensional force/moment channels directly, we first convert them to aerodynamic coefficients and fit those coefficients with a single global polynomial map.

Using $S$ (wing area), $b$ (wingspan), and $\bar c$ (mean aerodynamic chord), the coefficient targets are

$$ C_{X,t} = \frac{\hat{X}_t}{q_{\text{bar},t}S/m_t},\quad C_{Y,t} = \frac{\hat{Y}_t}{q_{\text{bar},t}S/m_t},\quad C_{Z,t} = \frac{\hat{Z}_t}{q_{\text{bar},t}S/m_t}, $$
$$ C_{L,t} = \frac{\hat{L}_t}{q_{\text{bar},t}Sb},\quad C_{M,t} = \frac{\hat{M}_t}{q_{\text{bar},t}S\bar c},\quad C_{N,t} = \frac{\hat{N}_t}{q_{\text{bar},t}Sb}. $$

The polynomial feature vector is

$$\upsilon = [\alpha,\beta,\mathrm{Mach},p,q,r,\delta_t,\delta_e,\delta_a,\delta_r]^T,$$

and we use degree-$3$ expansion $\phi_3(\upsilon_t)$ with ridge regularization for each coefficient channel:

$$ \min_{\Theta_j} \sum_{t=1}^{N} \left\lVert C_{j,t} - \Theta_j\phi_3(\upsilon_t) \right\rVert_2^2 + \lambda \lVert \Theta_j \rVert_2^2,\quad j \in \{X,Y,Z,L,M,N\}. $$

At rollout time, predicted coefficients are mapped back to dimensional channels by the same scaling:

$$ \hat{X}=\frac{q_{\text{bar}}S}{m}\hat{C}_X,\;\hat{Y}=\frac{q_{\text{bar}}S}{m}\hat{C}_Y,\;\hat{Z}=\frac{q_{\text{bar}}S}{m}\hat{C}_Z,\;\hat{L}=q_{\text{bar}}Sb\hat{C}_L,\;\hat{M}=q_{\text{bar}}S\bar c\hat{C}_M,\;\hat{N}=q_{\text{bar}}Sb\hat{C}_N. $$

### 3. Residual Extraction
Projecting the resulting trained analytical weights back through rigid mechanics produces the *Nominal Kinematic Prediction* string $x_{t+1}^{nom}$. The difference to the environment observation defines the unfiltered baseline variations holding all stochastic structure.

$$ \tilde{w}_t = x_{t+1}^{true} - x_{t+1}^{nom} $$

### 4. Context-Conditioned Local Retrievals
Because errors strongly manifest under specific geometric and spatial regimes, a bias component $c(z_t)$ operates.

$$ z_t = [ \alpha, \beta, p \dots u_{t-1}, q_{dyn}, \dot{\alpha}, W_c, \nabla W_c ]^T $$

Applying standard vector regularization and constructing a $k-d$ tree, we isolate the precise $K$-neighborhood queries within the ambient vector state.

$$ c_{kNN}(z_t) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(z_t)} \tilde{w}^{(i)} $$

### 5. Ground Truth Sampling
Subtracting $c_{kNN}$ natively centers the data perfectly relative to zero. $\mu \rightarrow 0$. These normalized records represent the absolute intrinsic limits defining the dynamic noise capability of the underlying engine model.

$$ w_t = \tilde{w}_t - c_{kNN}(z_t) $$

At inference, the `RuntimeUncertaintySampler` invokes these histories matching the specific spatial constraints (i.e. bounding relative distances matching $|W_c - W_{c_{query}}| < \epsilon_W$) to output empirical noise trajectories bridging reality to the Gym framework accurately.

---

## Nominal Model Used by MPPI (and How It Differs from the True F-16)

The uncertainty pipeline above produces a *nominal* one-step predictor $f_{\text{nom}}(x_t, u_t)$ used in two places:

1. **Offline calibration / residual construction** (Sections 1–5 above).
2. **Runtime planning inside MPPI**, where this nominal predictor is rolled out many times per control update.

In this repo, MPPI uses a JAX-compiled surrogate dynamics model implemented in `jsbsim_gym/mppi_jax.py` (`f16_kinematics_step`). It is a hybrid model:

### 1) аэрodynamics / moments surrogate (polynomial ridge map)

MPPI does *not* call JSBSim during its internal rollouts. Instead, it predicts the six specific-force / moment channels

$$[X, Y, Z, L, M, N]^T$$

via a **degree-2 polynomial feature expansion** of a compact feature vector

$$\upsilon = [\alpha,\beta,p,q,r,\delta_e,\delta_a,\delta_r,\delta_t]^T,$$

where $(\alpha,\beta)$ are derived from body velocities and $(p,q,r)$ are body rates, and $(\delta_\cdot)$ are actuator commands (including throttle).

Let $\phi(\upsilon)$ denote the same ordering as `sklearn.PolynomialFeatures(degree=2, include_bias=True)`:

$$\phi(\upsilon) = [1,\upsilon,\upsilon_1^2,\upsilon_1\upsilon_2,\ldots,\upsilon_n^2]^T.$$

Then the nominal aerodynamic/moment prediction used by MPPI is an affine map

$$[X,Y,Z,L,M,N] = \phi(\upsilon)^T W + B,$$

with $(W,B)$ loaded from `jsbsim_gym/mppi_nominal_weights.npz` (generated by `extract_nominal_weights.py`).

### 2) rigid-body propagation (deterministic kinematics + Euler integration)

Given $(X,Y,Z,L,M,N)$, MPPI advances a reduced 12D rigid-body state

$$x = [p_N, p_E, h, u, v, w, p, q, r, \phi, \theta, \psi]^T$$

using the standard Newton–Euler structure (Coriolis + gravity terms explicitly included) and a fixed-step Euler integrator with

$$\Delta t = 1/30\ \text{s}.$$

This is the fast nominal rollout model that MPPI evaluates inside `rollout_trajectory(...)` / `single_rollout_cost(...)`.

### 3) Full discrete-time nominal step equations $f_{\text{nom}}$

MPPI’s nominal dynamics operate as a discrete-time map

$$x_{t+1}^{\text{nom}} = f_{\text{nom}}(x_t, u_t),$$

with the 12D state and 4D control

$$x = [p_N, p_E, h, u, v, w, p, q, r, \phi, \theta, \psi]^T,\qquad u_c = [\delta_a,\delta_e,\delta_r,\delta_t]^T.$$

Given $x_t$ and $u_{c,t}$, MPPI computes:

**(a) aerodynamic angles / speed (from body velocities)**

$$\alpha = \operatorname{atan2}\left(w,\max(u,1)\right),$$

$$V = \sqrt{\max(u^2+v^2+w^2,1)},$$

$$\beta = \arcsin\left(\operatorname{clip}\left(\frac{v}{V},-1,1\right)\right).$$

**(b) feature vector and quadratic polynomial basis**

Let

$$\upsilon = [\alpha,\beta,p,q,r,\delta_e,\delta_a,\delta_r,\delta_t]^T \in \mathbb{R}^9.$$

Define the degree-2 feature vector (same ordering as `sklearn.PolynomialFeatures(degree=2, include_bias=True)` used by this repo):

$$\phi(\upsilon) = \Big[1,\; \upsilon_1,\ldots,\upsilon_9,\; \upsilon_1\upsilon_1,\upsilon_1\upsilon_2,\ldots,\upsilon_9\upsilon_9\Big]^T \in \mathbb{R}^{55},$$

where the quadratic block enumerates products for all index pairs $(i,j)$ with $1\le i\le j\le 9$.

**(c) nominal specific forces / moments**

With weights loaded from `jsbsim_gym/mppi_nominal_weights.npz`:

$$W\in\mathbb{R}^{55\times 6},\quad B\in\mathbb{R}^{6},$$

the prediction is

$$[X,Y,Z,L,M,N]^T = \phi(\upsilon)^T W + B.$$

**(d) continuous-time rigid-body dynamics (Newton–Euler structure)**

Let $g = 32.174$ (ft/s\textsuperscript{2}). Then

$$\dot u = X + rv - qw - g\sin\theta,$$
$$\dot v = Y + pw - ru + g\sin\phi\cos\theta,$$
$$\dot w = Z + qu - pv + g\cos\phi\cos\theta,$$

and MPPI uses the full coupled angular-rate model

$$\dot\omega = I^{-1}\!\left(\begin{bmatrix}L\\M\\N\end{bmatrix} - \omega \times (I\omega)\right),\qquad \omega = [p, q, r]^T.$$

Euler angle rates:

$$\dot\phi = p + q\sin\phi\tan\theta + r\cos\phi\tan\theta,$$
$$\dot\theta = q\cos\phi - r\sin\phi,$$
$$\dot\psi = \frac{q\sin\phi + r\cos\phi}{\cos\theta}.$$

North/East/Up kinematics (body-to-inertial velocity projection):

$$\dot p_N = u(c_\theta c_\psi) + v(s_\phi s_\theta c_\psi - c_\phi s_\psi) + w(c_\phi s_\theta c_\psi + s_\phi s_\psi),$$

$$\dot p_E = u(c_\theta s_\psi) + v(s_\phi s_\theta s_\psi + c_\phi c_\psi) + w(c_\phi s_\theta s_\psi - s_\phi c_\psi),$$

$$\dot h = u s_\theta - v(s_\phi c_\theta) - w(c_\phi c_\theta),$$

where $c_{(\cdot)}=\cos(\cdot)$ and $s_{(\cdot)}=\sin(\cdot)$.

**(e) discretization (explicit Euler, fixed step)**

With

$$\Delta t = 1/30\ \text{s},$$

the nominal step is

$$x_{t+1}^{\text{nom}} = x_t + \Delta t\,\dot x_t,$$

where $\dot x_t$ is assembled from the derivatives above.

> Note: the implementation additionally clips several intermediate derivatives to keep rollouts numerically stable during optimization; the expressions above describe the underlying nominal model structure.

---

## What the “True” Dynamics Are in the Environment

The environment state transitions $x_{t+1}^{\text{true}}$ come from **JSBSim’s full nonlinear F-16 model** (see `jsbsim_gym/env.py` and the aircraft definition under `aircraft/f16/`). This includes many effects intentionally *not* represented in the MPPI nominal model, for example:

- **Nonlinear aerodynamic coefficient lookup tables** across wide envelopes (not a single global quadratic map).
- **Coupled actuator / FCS / engine dynamics** (MPPI treats inputs as direct surface/throttle commands).
- **Configuration-dependent effects** (e.g., detailed lift/drag moment coupling, damping derivatives, saturations).
- **Higher-fidelity integration / internal solver behavior** (JSBSim advances an internal continuous-time model; the gym wrapper may downsample multiple JSBSim substeps per environment step).
- **Exogenous disturbances and environment-driven stochasticity** used in data collection (e.g., canyon-conditioned wind-shear processes described earlier).

As a result, we treat the mismatch with an **exact additive decomposition** defined by the residual:

$$x_{t+1}^{\text{true}} \;=\; f_{\text{JSBSim}}(x_t, u_t, d_t),$$

$$\tilde{w}_t \;\triangleq\; x_{t+1}^{\text{true}} - f_{\text{nom}}(x_t, u_t),$$

$$c(z_t) \;\triangleq\; \frac{1}{K}\sum_{i\in \mathcal{N}_K(z_t)} \tilde{w}^{(i)},$$

$$w_t \;\triangleq\; \tilde{w}_t - c(z_t),$$

so that, by construction,

$$x_{t+1}^{\text{true}} \;=\; f_{\text{nom}}(x_t, u_t) + c(z_t) + w_t.$$

Here $f_{\text{nom}}$ is the fast MPPI surrogate rollout model, $c(z_t)$ is the context-conditioned bias estimated from stored residuals, and $w_t$ is the centered empirical residual sampled at runtime.
