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
Unlike typical deep-learning formulations, we utilize explicit rigid body kinematic simplifications separating inherent non-linear Coriolis and Gravitational effects from the purely interactive specific aerodynamic forces.

$$ \hat{X}_{t} = \frac{u_{t+1} - u_t}{\Delta t} - (r_t v_t - q_t w_t - g \sin\theta_t) $$
$$ \hat{Y}_{t} = \frac{v_{t+1} - v_t}{\Delta t} - (p_t w_t - r_t u_t + g \sin\phi_t \cos\theta_t) $$
$$ \hat{Z}_{t} = \frac{w_{t+1} - w_t}{\Delta t} - (q_t u_t - p_t v_t + g \cos\phi_t \cos\theta_t) $$

By assuming arbitrary constant inertial decoupling, the specific rolling elements $\hat{L}, \hat{M}, \hat{N}$ strictly evaluate temporal displacement scaling.

### 2. Analytical Ridge Regression (Nominal Base)
Rather than parameterizing strict F-16 wind tunnel coefficient graphs ($C_{L_q}$, $C_{m_{\delta_e}}$, etc.), we subsume the target dimensional forces across a general $L_2$-regularized multivariate polynomial.

Extracting a structured space vector: $\upsilon = [\alpha, \beta, p, q, r, \delta_e, \delta_a, \delta_r]^T$. 

We construct the expanded $2^{\text{nd}}$ degree combinations $\mathcal{X}_{poly} = \text{Poly}^2(\upsilon_t)$. 

$$ \min_{\Theta} \sum_{t=1}^{N} || \hat{X}_{t} - \Theta \mathcal{X}_{poly}^{(t)} ||^2_2 + \lambda ||\Theta||^2_2 $$

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
