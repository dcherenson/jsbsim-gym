import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from scipy.stats import wasserstein_distance
import joblib
import os

DT = 1.0 / 30.0
G = 32.174

def rigid_body_kinematics(df):
    targets = pd.DataFrame(index=df.index)
    targets['X'] = (df['next_u'] - df['u']) / DT - (df['r']*df['v'] - df['q']*df['w'] - G*np.sin(df['theta']))
    targets['Y'] = (df['next_v'] - df['v']) / DT - (df['p']*df['w'] - df['r']*df['u'] + G*np.sin(df['phi'])*np.cos(df['theta']))
    targets['Z'] = (df['next_w'] - df['w']) / DT - (df['q']*df['u'] - df['p']*df['v'] + G*np.cos(df['phi'])*np.cos(df['theta']))
    targets['L'] = (df['next_p'] - df['p']) / DT
    targets['M'] = (df['next_q'] - df['q']) / DT
    targets['N'] = (df['next_r'] - df['r']) / DT
    return targets

class NominalModel:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.ridge_models = {}
        self.features = ['alpha', 'beta', 'p', 'q', 'r', 'delta_e', 'delta_a', 'delta_r']
        self.targets = ['X', 'Y', 'Z', 'L', 'M', 'N']
        
    def fit(self, df, targets):
        X_poly = self.poly.fit_transform(df[self.features].values)
        for t in self.targets:
            m = Ridge(alpha=1.0)
            m.fit(X_poly, targets[t].values)
            self.ridge_models[t] = m
            
    def predict(self, df):
        X_poly = self.poly.transform(df[self.features].values)
        preds = pd.DataFrame(index=df.index)
        for t in self.targets:
            preds[t] = self.ridge_models[t].predict(X_poly)
        return preds

def add_prev_actions(df):
    df = df.copy()
    for c in ['delta_t', 'delta_e', 'delta_a', 'delta_r']:
        df[f'prev_{c}'] = df[c].shift(1).fillna(df[c].iloc[0])
    return df

def _save_raw_vs_centered_histograms(raw_res, centered_res, state_names, plot_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    colors = {"raw": "#94a3b8", "centered": "#2563eb"}
    axes = axes.flatten()
    for i, channel in enumerate(state_names):
        axes[i].hist(raw_res[channel], bins=60, color=colors["raw"], alpha=0.85, label="Raw")
        axes[i].hist(centered_res[channel], bins=60, color=colors["centered"], alpha=0.85, label="Centered")
        axes[i].set_title(f"{channel} kinematics residual")
        axes[i].set_xlabel("Residual")
        axes[i].set_ylabel("Count")
        axes[i].legend()
        
        p1 = np.percentile(raw_res[channel].dropna(), 1)
        p99 = np.percentile(raw_res[channel].dropna(), 99)
        axes[i].set_xlim(p1, p99)
        
    fig.tight_layout()
    path = os.path.join(plot_dir, "raw_vs_centered_histograms.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_canyon_bias_reduction(raw_res, centered_res, df, state_names, plot_dir):
    raw = raw_res.copy()
    centered = centered_res.copy()
    
    bin_edges = np.linspace(df['canyon_width'].min(), df['canyon_width'].max(), 31)
    
    raw['canyon_bin'] = pd.cut(df['canyon_width'], bins=bin_edges, include_lowest=True, right=True)
    centered['canyon_bin'] = pd.cut(df['canyon_width'], bins=bin_edges, include_lowest=True, right=True)
    
    fig, axes = plt.subplots(len(state_names)//2, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    
    for i, channel in enumerate(state_names):
        rg = raw.groupby('canyon_bin', observed=False)[channel].agg(["mean", "std"]).reset_index()
        cg = centered.groupby('canyon_bin', observed=False)[channel].agg(["mean", "std"]).reset_index()
        
        # Calculate centers
        mids = [(v.left + v.right)/2.0 for v in rg['canyon_bin']]
        
        axes[i].plot(mids, rg["mean"], color="#94a3b8", label="Raw mean")
        axes[i].plot(mids, cg["mean"], color="#2563eb", label="Centered mean")
        axes[i].fill_between(mids, rg["mean"] - rg["std"].fillna(0.0), rg["mean"] + rg["std"].fillna(0.0), color="#cbd5e1", alpha=0.35)
        axes[i].fill_between(mids, cg["mean"] - cg["std"].fillna(0.0), cg["mean"] + cg["std"].fillna(0.0), color="#93c5fd", alpha=0.30)
        axes[i].axhline(0.0, color="#111827", linewidth=0.8, alpha=0.4)
        axes[i].set_ylabel(channel)
        axes[i].legend(loc="upper right")
        axes[i].set_title(f"Bias along $W_c$: {channel}")
        
    axes[-1].set_xlabel("Canyon Width (ft)")
    axes[-2].set_xlabel("Canyon Width (ft)")
    fig.tight_layout()
    path = os.path.join(plot_dir, "canyon_width_bias_reduction.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_calibration_scatter(calib_pred, raw_res, state_names, plot_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, channel in enumerate(state_names):
        axes[i].hexbin(calib_pred[channel], raw_res[channel], gridsize=32, cmap="magma", mincnt=1)
        low = min(calib_pred[channel].min(), raw_res[channel].min())
        high = max(calib_pred[channel].max(), raw_res[channel].max())
        axes[i].plot([low, high], [low, high], linestyle="--", color="#e5e7eb", linewidth=1.0)
        axes[i].set_title(f"Predicted calib vs raw {channel}")
        axes[i].set_xlabel("Predicted deterministic+kNN mean")
        axes[i].set_ylabel("Raw residual")
    fig.tight_layout()
    path = os.path.join(plot_dir, "calibration_scatter.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_telemetry_context_plots(raw_res, centered_res, df, state_names, plot_dir):
    telemetry_channels = ['alpha', 'beta', 'p', 'q', 'r', 'qbar']
    fig, axes = plt.subplots(len(state_names), len(telemetry_channels), figsize=(24, 18))
    
    for i, s in enumerate(state_names):
        for j, tc in enumerate(telemetry_channels):
            ax = axes[i, j]
            ax.hexbin(df[tc], centered_res[s], gridsize=30, cmap="magma", mincnt=1)
            
            if i == 0:
                ax.set_title(tc)
            if j == 0:
                ax.set_ylabel(f"{s} Centered Res.")
            if i == len(state_names) - 1:
                ax.set_xlabel(tc)
                
    fig.tight_layout()
    path = os.path.join(plot_dir, "telemetry_context_plots.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_wasserstein_comparison(raw_res, sampled_res, state_names, plot_dir):
    wass_baseline = []
    wass_model = []
    
    for s in state_names:
        actual = raw_res[s].dropna().to_numpy()
        sampled = sampled_res[s].dropna().to_numpy()
        wass_baseline.append(float(wasserstein_distance(actual, np.zeros(len(actual), dtype=float))))
        wass_model.append(float(wasserstein_distance(actual, sampled)))
        
    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(state_names))
    width = 0.32
    axis.bar(
        x - width / 2,
        wass_baseline,
        width=width,
        label="zero baseline",
        color="#94a3b8",
    )
    axis.bar(
        x + width / 2,
        wass_model,
        width=width,
        label="sampled model",
        color="#2563eb",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(state_names)
    axis.set_ylabel("Wasserstein distance")
    axis.set_title("Distributional error on held-out residuals")
    axis.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "wasserstein_comparison.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_sampled_distribution_overlay(raw_res, sampled_res, state_names, plot_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {
        "actual": "#111827",
        "sampled": "#2563eb",
        "baseline": "#dc2626",
    }
    axes = axes.flatten()
    for axis, channel in zip(axes, state_names):
        actual = raw_res[channel].dropna().to_numpy()
        sampled_values = sampled_res[channel].dropna().to_numpy()
        zero_baseline = np.zeros(len(actual), dtype=float)
        
        # Calculate percentiles dynamically to drop extreme outliers
        p1 = min(np.percentile(actual, 1), np.percentile(sampled_values, 1), 0.0)
        p99 = max(np.percentile(actual, 99), np.percentile(sampled_values, 99), 0.0)
        
        bins = np.linspace(p1, p99, 36)
        axis.hist(actual, bins=bins, density=True, alpha=0.35, color=colors["actual"], label="actual")
        axis.hist(sampled_values, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["sampled"], label="model sampled")
        axis.hist(zero_baseline, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["baseline"], label="zero baseline")
        axis.set_title(f"Distribution match: {channel}")
        axis.set_xlabel("Residual value")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(plot_dir, "sampled_distribution_overlay.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_multimodal_slices(df, centered_res, state_names, kdtree, scaler, aug_features, plot_dir):
    # Find 3 extremely tight canyon scenarios
    tight_indices = df.nsmallest(3, 'canyon_width').index
    
    fig, axes = plt.subplots(3, len(state_names), figsize=(18, 10))
    
    Z_scaled = scaler.transform(df.loc[tight_indices, aug_features].values)
    K = 15
    dists, inds = kdtree.query(Z_scaled, k=K)
    
    for row_idx, (real_idx, k_inds) in enumerate(zip(tight_indices, inds)):
        canyon_w = df.loc[real_idx, 'canyon_width']
        for col_idx, s in enumerate(state_names):
            ax = axes[row_idx, col_idx]
            slice_vals = centered_res[s].values[k_inds]
            
            p1 = np.percentile(centered_res[s].dropna(), 1)
            p99 = np.percentile(centered_res[s].dropna(), 99)
            bins = np.linspace(p1, p99, 15)
            
            ax.hist(slice_vals, bins=bins, color="#8b5cf6", alpha=0.8, density=True)
            ax.set_xlim(p1, p99)
            
            if row_idx == 0:
                ax.set_title(f"{s}")
            if col_idx == 0:
                ax.set_ylabel(f"Slice {row_idx}\n($W_c$={canyon_w:.1f})\ndensity")
                
    fig.suptitle("Multimodal kNN Neighborhood Slices (K=15) in Tight Canyons", fontsize=16)
    fig.tight_layout()
    path = os.path.join(plot_dir, "multimodal_slices.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def generate_nominal_calibration_package():
    print("Loading Parquet dataset...")
    df = pd.read_parquet('f16_dataset.parquet')
    df = add_prev_actions(df)
    
    print("Module 1: Nominal System Identification")
    targets = rigid_body_kinematics(df)
    nom_model = NominalModel()
    nom_model.fit(df, targets)
    
    nom_preds = nom_model.predict(df)
    
    nom_next = pd.DataFrame(index=df.index)
    nom_next['u'] = df['u'] + DT * (nom_preds['X'] + df['r']*df['v'] - df['q']*df['w'] - G*np.sin(df['theta']))
    nom_next['v'] = df['v'] + DT * (nom_preds['Y'] + df['p']*df['w'] - df['r']*df['u'] + G*np.sin(df['phi'])*np.cos(df['theta']))
    nom_next['w'] = df['w'] + DT * (nom_preds['Z'] + df['q']*df['u'] - df['p']*df['v'] + G*np.cos(df['phi'])*np.cos(df['theta']))
    nom_next['p'] = df['p'] + DT * nom_preds['L']
    nom_next['q'] = df['q'] + DT * nom_preds['M']
    nom_next['r'] = df['r'] + DT * nom_preds['N']
    
    state_names = ['u', 'v', 'w', 'p', 'q', 'r']
    raw_res = pd.DataFrame(index=df.index)
    for s in state_names:
        raw_res[s] = df[f'next_{s}'] - nom_next[s]
        
    print("Module 2: Parametric Deterministic Calibration")
    param_features = ['canyon_width', 'canyon_width_grad', 'qbar']
    P_poly = PolynomialFeatures(degree=2, include_bias=True)
    P_X = P_poly.fit_transform(df[param_features].values)
    
    c_param_models = {}
    c_param_preds = pd.DataFrame(index=df.index)
    for s in state_names:
        m = Ridge(alpha=10.0)
        m.fit(P_X, raw_res[s].values)
        c_param_models[s] = m
        c_param_preds[s] = m.predict(P_X)
        
    inter_res = raw_res - c_param_preds
    
    print("Module 3: Context-Conditioned kNN Calibration")
    aug_features = ['alpha', 'beta', 'p', 'q', 'r', 'delta_e', 'delta_a', 'delta_r', 
                    'prev_delta_e', 'prev_delta_a', 'prev_delta_r', 
                    'qbar', 'alpha_dot', 'wind_u', 'wind_v', 'wind_w', 
                    'canyon_width', 'canyon_width_grad']
    
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(df[aug_features].values)
    
    kdtree = KDTree(Z_scaled)
    K = 15
    dists, inds = kdtree.query(Z_scaled, k=K+1)
    inds = inds[:, 1:] # Drop self
    
    c_knn_preds = pd.DataFrame(index=df.index)
    for s in state_names:
        nearest_vals = inter_res[s].values[inds]
        c_knn_preds[s] = nearest_vals.mean(axis=1)
        
    final_res = inter_res - c_knn_preds
    
    # Pack the dataset with actual tracked residuals
    df_resid = df.copy()
    for s in state_names:
        df_resid[f'w_{s}'] = final_res[s]
        
    print("Plotting Calibration...")
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    _save_raw_vs_centered_histograms(raw_res, final_res, state_names, plot_dir)
    _save_canyon_bias_reduction(raw_res, final_res, df, state_names, plot_dir)
    
    # Total calibration mean
    total_calib_mean = c_param_preds + c_knn_preds
    _save_calibration_scatter(total_calib_mean, raw_res, state_names, plot_dir)
    
    # Generate globally sampled distributions to measure kNN subset sampler effectiveness
    rand_choice = np.random.randint(0, K, size=len(df))
    sampled_idx = inds[np.arange(len(df)), rand_choice]
    sampled_res = pd.DataFrame(index=df.index)
    for s in state_names:
        sampled_res[s] = final_res[s].values[sampled_idx]
    
    # New parity plots
    _save_telemetry_context_plots(raw_res, final_res, df, state_names, plot_dir)
    _save_wasserstein_comparison(raw_res, sampled_res, state_names, plot_dir)
    _save_sampled_distribution_overlay(raw_res, sampled_res, state_names, plot_dir)
    _save_multimodal_slices(df, final_res, state_names, kdtree, scaler, aug_features, plot_dir)
    
    print("Module 5: Artifact Serialization")
    artifact = {
        'nominal_model': nom_model,
        'param_poly': P_poly,
        'c_param_models': c_param_models,
        'aug_features': aug_features,
        'knn_scaler': scaler,
        'kdtree': kdtree,
        'historical_dataset': df_resid
    }
    
    joblib.dump(artifact, 'f16_uncertainty_model.pkl')
    print("Artifact successfully exported to f16_uncertainty_model.pkl")

if __name__ == "__main__":
    generate_nominal_calibration_package()
