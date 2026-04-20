from typing import NamedTuple

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import pandas as pd


class JAXEmpiricalData(NamedTuple):
    features_scaled: jax.Array
    residuals: jax.Array
    feature_mean: jax.Array
    feature_std: jax.Array
    alpha_values: jax.Array
    canyon_width_values: jax.Array
    sorted_features_scaled: jax.Array
    sorted_residuals: jax.Array
    sorted_alpha_values: jax.Array
    sorted_canyon_width_values: jax.Array
    alpha_feature_index: int
    canyon_width_feature_index: int
    neighbor_count: int
    max_pool_size: int
    epsilon_w: float
    alpha_threshold: float


def sample_empirical_jax(
    feature_vector: jax.Array,
    gate_id: jax.Array,
    rng_key: jax.Array,
    data: JAXEmpiricalData,
) -> jax.Array:
    del gate_id

    normalized_q = (feature_vector - data.feature_mean) / data.feature_std
    query_width = feature_vector[data.canyon_width_feature_index]

    total_samples = data.sorted_canyon_width_values.shape[0]
    lower = query_width - jnp.float32(data.epsilon_w)
    upper = query_width + jnp.float32(data.epsilon_w)

    lo = jnp.searchsorted(data.sorted_canyon_width_values, lower, side="left")
    hi = jnp.searchsorted(data.sorted_canyon_width_values, upper, side="right")
    width_count = hi - lo

    # If width-gated window is empty, fallback to a centered local pool.
    center = jnp.searchsorted(data.sorted_canyon_width_values, query_width, side="left")
    half_pool = jnp.int32(data.max_pool_size // 2)
    fallback_lo = jnp.clip(center - half_pool, 0, jnp.maximum(total_samples - data.max_pool_size, 0))
    fallback_count = jnp.minimum(jnp.int32(data.max_pool_size), total_samples - fallback_lo)

    pool_start = jnp.where(width_count > 0, lo, fallback_lo)
    pool_count = jnp.where(width_count > 0, width_count, fallback_count)
    pool_count = jnp.minimum(pool_count, jnp.int32(data.max_pool_size))

    offsets = jnp.arange(data.max_pool_size, dtype=jnp.int32)
    pool_indices = pool_start + offsets
    pool_indices_clipped = jnp.clip(pool_indices, 0, total_samples - 1)
    valid_mask = offsets < pool_count

    pool_features = data.sorted_features_scaled[pool_indices_clipped]
    pool_alpha = data.sorted_alpha_values[pool_indices_clipped]
    diffs = pool_features - normalized_q
    dists_sq = jnp.sum(jnp.square(diffs), axis=-1)

    regime_mask = jnp.abs(pool_alpha) < jnp.float32(data.alpha_threshold)
    valid_regime_mask = jnp.logical_and(valid_mask, regime_mask)
    has_valid_regime = jnp.any(valid_regime_mask)
    active_mask = jnp.where(has_valid_regime, valid_regime_mask, valid_mask)

    penalized_dists = jnp.where(active_mask, dists_sq, 1.0e12)
    vals, local_indices = jax.lax.top_k(-penalized_dists, data.neighbor_count)
    indices = pool_indices_clipped[local_indices]
    distances = -vals

    scale = jnp.mean(distances) + 1.0e-6
    logits = -distances / scale
    weights = jax.nn.softmax(logits)
    choice_idx = jax.random.choice(rng_key, indices, p=weights)
    return data.residuals[choice_idx]

class RuntimeUncertaintySampler:
    def __init__(self, artifact_path='f16_uncertainty_model.pkl'):
        """
        Loads the compiled empirical uncertainty model artifact.
        """
        artifact = joblib.load(artifact_path)
        
        # Unpack serialized components
        self.kdtree = artifact['kdtree']
        self.knn_scaler = artifact['knn_scaler']
        self.aug_features = artifact['aug_features']
        self.dataset = artifact['historical_dataset']
        
        # We assume dataset index corresponds to the row indices returned by KDTree.
        # But wait, KDTree query returns positional indices (0...N). We need to ensure
        # we can iloc appropriately.
        self.dataset = self.dataset.reset_index(drop=True)
        
        artifact_residual_columns = artifact.get('residual_columns')
        if artifact_residual_columns is None:
            raise KeyError(
                "Uncertainty artifact is missing required 'residual_columns'. "
                "Regenerate with python -m jsbsim_gym.calibration."
            )
        self.residual_columns = [str(column) for column in artifact_residual_columns]

        missing_columns = [column for column in self.residual_columns if column not in self.dataset.columns]
        if missing_columns:
            raise KeyError(
                "Uncertainty artifact is missing expected coefficient residual columns: "
                f"{missing_columns}. Regenerate with python -m jsbsim_gym.calibration."
            )

    def to_jax(
        self,
        neighbor_count: int = 20,
        max_pool_size: int = 512,
        epsilon_w: float = 200.0,
        alpha_threshold: float = 0.8,
    ) -> JAXEmpiricalData:
        feature_values = self.dataset[self.aug_features].to_numpy(dtype=np.float32, copy=True)
        features_scaled = self.knn_scaler.transform(feature_values).astype(np.float32, copy=False)
        residual_values = self.dataset[self.residual_columns].to_numpy(dtype=np.float32, copy=True)

        feature_mean = np.asarray(getattr(self.knn_scaler, "mean_", np.zeros(feature_values.shape[1])), dtype=np.float32)
        feature_std = np.asarray(getattr(self.knn_scaler, "scale_", np.ones(feature_values.shape[1])), dtype=np.float32)
        feature_std[np.abs(feature_std) < 1.0e-6] = 1.0

        alpha_feature_index = int(self.aug_features.index("alpha"))
        canyon_width_feature_index = int(self.aug_features.index("canyon_width"))
        alpha_values = self.dataset["alpha"].to_numpy(dtype=np.float32, copy=True)
        canyon_width_values = self.dataset["canyon_width"].to_numpy(dtype=np.float32, copy=True)

        order = np.argsort(canyon_width_values, kind="stable")
        sorted_features_scaled = features_scaled[order]
        sorted_residual_values = residual_values[order]
        sorted_alpha_values = alpha_values[order]
        sorted_canyon_width_values = canyon_width_values[order]

        total_samples = int(features_scaled.shape[0])
        max_neighbors = max(1, min(int(neighbor_count), total_samples))
        max_pool = max(max_neighbors, min(int(max_pool_size), total_samples))
        return JAXEmpiricalData(
            features_scaled=jnp.asarray(features_scaled, dtype=jnp.float32),
            residuals=jnp.asarray(residual_values, dtype=jnp.float32),
            feature_mean=jnp.asarray(feature_mean, dtype=jnp.float32),
            feature_std=jnp.asarray(feature_std, dtype=jnp.float32),
            alpha_values=jnp.asarray(alpha_values, dtype=jnp.float32),
            canyon_width_values=jnp.asarray(canyon_width_values, dtype=jnp.float32),
            sorted_features_scaled=jnp.asarray(sorted_features_scaled, dtype=jnp.float32),
            sorted_residuals=jnp.asarray(sorted_residual_values, dtype=jnp.float32),
            sorted_alpha_values=jnp.asarray(sorted_alpha_values, dtype=jnp.float32),
            sorted_canyon_width_values=jnp.asarray(sorted_canyon_width_values, dtype=jnp.float32),
            alpha_feature_index=alpha_feature_index,
            canyon_width_feature_index=canyon_width_feature_index,
            neighbor_count=max_neighbors,
            max_pool_size=max_pool,
            epsilon_w=float(epsilon_w),
            alpha_threshold=float(alpha_threshold),
        )
        
    def sample(self, z_q, W_c_q, config=None):
        """
        Retrieves a temporal block sequence of sample residuals matching the query context.
        
        Args:
            z_q (dict): Dictionary mapping augmented feature names to the runtime query values.
            W_c_q (float): Current canyon width of the query scenario.
            config (dict, optional): Customizations mapping parameters like epsilon_W, N_neighbors, N_block, etc.
            
        Returns:
            list or generator returning dicts of residuals for each step of the block.
        """
        if config is None:
            config = {
                'epsilon_W': 200.0,
                'N_neighbors': 20,
                'N_block': 10,
                'alpha_threshold': 0.8
            }
            
        # 1. Width Gating (Pre-filtering)
        # Note: KDTree holds all points! If we just want to query KDTree, we query nearest 
        # and THEN filter. Or rebuild KDTree dynamically? No, that's slow. 
        # Better strategy: Query an excess number of neighbors, then filter them by Width and Regime.
        
        # Extract features ensuring ordering
        vector_q = np.array([z_q.get(k, 0.0) for k in self.aug_features]).reshape(1, -1)
        z_q_scaled = self.knn_scaler.transform(vector_q)
        
        query_k = max(200, config['N_neighbors'] * 5)
        dists, idxs = self.kdtree.query(z_q_scaled, k=query_k)
        candidate_indices = idxs[0]
        
        candidates = self.dataset.iloc[candidate_indices]
        
        # Filter condition 1: canyon width epsilon_W
        mask_W = np.abs(candidates['canyon_width'] - W_c_q) <= config['epsilon_W']
        
        # Filter condition 2: regime constraints (e.g. valid alpha range)
        mask_regime = np.abs(candidates['alpha']) < config['alpha_threshold']
        
        valid_candidates = candidates[mask_W & mask_regime]
        
        # If filtering knocked out everything, just fallback to closest available point
        if valid_candidates.empty:
            valid_candidates = candidates.iloc[:config['N_neighbors']]
        else:
            valid_candidates = valid_candidates.iloc[:config['N_neighbors']]
            
        # Sample one sequence starting index
        selected_row = valid_candidates.sample(1).iloc[0]
        start_index = selected_row.name # index relative to self.dataset 
        
        # Block trailing samples
        end_index = min(len(self.dataset), start_index + config['N_block'])
        
        sampled_block = self.dataset.iloc[start_index:end_index]
        return sampled_block[self.residual_columns].to_dict('records')
