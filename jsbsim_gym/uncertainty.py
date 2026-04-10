import numpy as np
import pandas as pd
import joblib

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
        
        # State variables containing centered residuals
        self.residual_columns = ['w_u', 'w_v', 'w_w', 'w_p', 'w_q', 'w_r']
        
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
