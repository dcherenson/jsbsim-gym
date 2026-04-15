import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import joblib
import os

DT = 1.0 / 30.0
G = 32.174
IXX = 9496.0
IYY = 55814.0
IZZ = 63100.0
IXZ = -982.0
INERTIA_DET = IXX * IZZ - IXZ * IXZ
WING_AREA_FT2 = 300.0
WING_SPAN_FT = 30.0
MEAN_AERODYNAMIC_CHORD_FT = 11.32
DEFAULT_MASS_LBS = 17400.0 + 230.0 + 2000.0
DEFAULT_MASS_SLUGS = DEFAULT_MASS_LBS / G
MIN_QBAR_PSF = 1.0
STANDARD_SPEED_OF_SOUND_FPS = 1116.45
NOMINAL_COEFF_WEIGHTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nominal_coeff_weights.npz")


def angular_rate_derivatives_to_moments(p, q, r, p_dot, q_dot, r_dot):
    h_x = IXX * p + IXZ * r
    h_y = IYY * q
    h_z = IXZ * p + IZZ * r

    cross_x = q * h_z - r * h_y
    cross_y = r * h_x - p * h_z
    cross_z = p * h_y - q * h_x

    L = IXX * p_dot + IXZ * r_dot + cross_x
    M = IYY * q_dot + cross_y
    N = IXZ * p_dot + IZZ * r_dot + cross_z
    return L, M, N


def moments_to_angular_rate_derivatives(p, q, r, L, M, N):
    h_x = IXX * p + IXZ * r
    h_y = IYY * q
    h_z = IXZ * p + IZZ * r

    cross_x = q * h_z - r * h_y
    cross_y = r * h_x - p * h_z
    cross_z = p * h_y - q * h_x

    rhs_x = L - cross_x
    rhs_y = M - cross_y
    rhs_z = N - cross_z

    p_dot = (IZZ * rhs_x - IXZ * rhs_z) / INERTIA_DET
    q_dot = rhs_y / IYY
    r_dot = (IXX * rhs_z - IXZ * rhs_x) / INERTIA_DET
    return p_dot, q_dot, r_dot

def rigid_body_kinematics(df):
    targets = pd.DataFrame(index=df.index)
    targets['X'] = (df['next_u'] - df['u']) / DT - (df['r']*df['v'] - df['q']*df['w'] - G*np.sin(df['theta']))
    targets['Y'] = (df['next_v'] - df['v']) / DT - (df['p']*df['w'] - df['r']*df['u'] + G*np.sin(df['phi'])*np.cos(df['theta']))
    targets['Z'] = (df['next_w'] - df['w']) / DT - (df['q']*df['u'] - df['p']*df['v'] + G*np.cos(df['phi'])*np.cos(df['theta']))
    p_dot = (df['next_p'] - df['p']) / DT
    q_dot = (df['next_q'] - df['q']) / DT
    r_dot = (df['next_r'] - df['r']) / DT
    targets['L'], targets['M'], targets['N'] = angular_rate_derivatives_to_moments(
        df['p'],
        df['q'],
        df['r'],
        p_dot,
        q_dot,
        r_dot,
    )
    return targets


def _mach_from_dataframe(df):
    if 'mach' in df.columns:
        mach = df['mach'].to_numpy(dtype=float)
        if np.all(np.isfinite(mach)):
            return mach

    if 'V' in df.columns:
        speed_fps = df['V'].to_numpy(dtype=float)
    else:
        u = df['u'].to_numpy(dtype=float)
        v = df['v'].to_numpy(dtype=float)
        w = df['w'].to_numpy(dtype=float)
        speed_fps = np.sqrt(np.maximum(u*u + v*v + w*w, 1e-6))
    return speed_fps / STANDARD_SPEED_OF_SOUND_FPS


def _effective_mass_slugs(df):
    if 'mass_slugs' not in df.columns:
        return np.full(len(df), DEFAULT_MASS_SLUGS, dtype=float)

    mass_slugs = df['mass_slugs'].to_numpy(dtype=float)
    valid = np.isfinite(mass_slugs) & (mass_slugs > 1e-3)
    return np.where(valid, mass_slugs, DEFAULT_MASS_SLUGS)


def _aero_scales(df):
    qbar_psf = np.clip(df['qbar'].to_numpy(dtype=float), MIN_QBAR_PSF, None)
    mass_slugs = np.clip(_effective_mass_slugs(df), 1e-3, None)

    force_scale = qbar_psf * WING_AREA_FT2 / mass_slugs
    roll_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT
    pitch_moment_scale = qbar_psf * WING_AREA_FT2 * MEAN_AERODYNAMIC_CHORD_FT
    yaw_moment_scale = qbar_psf * WING_AREA_FT2 * WING_SPAN_FT

    return {
        'force_scale': force_scale,
        'roll_moment_scale': roll_moment_scale,
        'pitch_moment_scale': pitch_moment_scale,
        'yaw_moment_scale': yaw_moment_scale,
    }


def _collect_nominal_coefficient_weights(nom_model):
    weights = []
    intercepts = []
    for target in nom_model.coeff_targets:
        model = nom_model.coeff_models[target]
        weights.append(model.coef_)
        intercepts.append(model.intercept_)

    W = np.stack(weights, axis=-1).astype(np.float32)
    B = np.stack(intercepts, axis=0).astype(np.float32)
    return W, B


def export_nominal_coefficient_weights(
    nom_model,
    output_path=NOMINAL_COEFF_WEIGHTS_OUTPUT_PATH,
    source_dataset='f16_dataset.parquet',
):
    output_path = str(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    W, B = _collect_nominal_coefficient_weights(nom_model)
    np.savez(
        output_path,
        W=W,
        B=B,
        feature_names=np.asarray(nom_model.features),
        target_names=np.asarray(nom_model.coeff_targets),
        poly_degree=np.asarray([int(nom_model.poly.degree)], dtype=np.int32),
        include_bias=np.asarray([int(bool(nom_model.poly.include_bias))], dtype=np.int32),
        source_dataset=np.asarray([str(source_dataset)]),
        model_space=np.asarray(["aerodynamic_coefficients"]),
        wing_area_ft2=np.asarray([WING_AREA_FT2], dtype=np.float32),
        wing_span_ft=np.asarray([WING_SPAN_FT], dtype=np.float32),
        mean_aerodynamic_chord_ft=np.asarray([MEAN_AERODYNAMIC_CHORD_FT], dtype=np.float32),
    )
    return output_path

class NominalModel:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=3, include_bias=True)
        self.coeff_models = {}
        self.features = ['alpha', 'beta', 'mach', 'p', 'q', 'r', 'delta_t', 'delta_e', 'delta_a', 'delta_r']
        self.coeff_targets = ['C_X', 'C_Y', 'C_Z', 'C_L', 'C_M', 'C_N']
        self.targets = ['X', 'Y', 'Z', 'L', 'M', 'N']

    def _build_feature_frame(self, df):
        feature_df = pd.DataFrame(index=df.index)
        feature_df['alpha'] = df['alpha'].to_numpy(dtype=float)
        feature_df['beta'] = df['beta'].to_numpy(dtype=float)
        feature_df['mach'] = _mach_from_dataframe(df)
        for channel in ['p', 'q', 'r', 'delta_t', 'delta_e', 'delta_a', 'delta_r']:
            feature_df[channel] = df[channel].to_numpy(dtype=float)
        return feature_df

    def _targets_to_coefficients(self, df, targets):
        scales = _aero_scales(df)
        coeffs = pd.DataFrame(index=df.index)
        coeffs['C_X'] = targets['X'].to_numpy(dtype=float) / scales['force_scale']
        coeffs['C_Y'] = targets['Y'].to_numpy(dtype=float) / scales['force_scale']
        coeffs['C_Z'] = targets['Z'].to_numpy(dtype=float) / scales['force_scale']
        coeffs['C_L'] = targets['L'].to_numpy(dtype=float) / scales['roll_moment_scale']
        coeffs['C_M'] = targets['M'].to_numpy(dtype=float) / scales['pitch_moment_scale']
        coeffs['C_N'] = targets['N'].to_numpy(dtype=float) / scales['yaw_moment_scale']
        return coeffs

    def _coefficients_to_targets(self, df, coeff_preds):
        scales = _aero_scales(df)
        targets = pd.DataFrame(index=df.index)
        targets['X'] = coeff_preds['C_X'].to_numpy(dtype=float) * scales['force_scale']
        targets['Y'] = coeff_preds['C_Y'].to_numpy(dtype=float) * scales['force_scale']
        targets['Z'] = coeff_preds['C_Z'].to_numpy(dtype=float) * scales['force_scale']
        targets['L'] = coeff_preds['C_L'].to_numpy(dtype=float) * scales['roll_moment_scale']
        targets['M'] = coeff_preds['C_M'].to_numpy(dtype=float) * scales['pitch_moment_scale']
        targets['N'] = coeff_preds['C_N'].to_numpy(dtype=float) * scales['yaw_moment_scale']
        return targets

    def fit(self, df, targets):
        feature_df = self._build_feature_frame(df)
        coeff_targets = self._targets_to_coefficients(df, targets)
        X_poly = self.poly.fit_transform(feature_df[self.features].values)
        for t in self.coeff_targets:
            m = Ridge(alpha=1.0)
            m.fit(X_poly, coeff_targets[t].values)
            self.coeff_models[t] = m

    def predict_coefficients(self, df):
        feature_df = self._build_feature_frame(df)
        X_poly = self.poly.transform(feature_df[self.features].values)
        preds = pd.DataFrame(index=df.index)
        for t in self.coeff_targets:
            preds[t] = self.coeff_models[t].predict(X_poly)
        return preds

    def predict(self, df):
        coeff_preds = self.predict_coefficients(df)
        return self._coefficients_to_targets(df, coeff_preds)

def add_prev_actions(df):
    df = df.copy()
    for c in ['delta_t', 'delta_e', 'delta_a', 'delta_r']:
        df[f'prev_{c}'] = df[c].shift(1).fillna(df[c].iloc[0])
    return df

def _save_raw_vs_centered_histograms(raw_res, centered_res, state_names, plot_dir):
    fig, axes = plt.subplots(len(state_names), 2, figsize=(14, max(10, 2.4 * len(state_names))))
    colors = {"raw": "#94a3b8", "centered": "#2563eb"}
    for row_index, channel in enumerate(state_names):
        axes[row_index, 0].hist(raw_res[channel], bins=60, color=colors["raw"], alpha=0.85)
        axes[row_index, 0].set_title(f"Raw {channel}")
        axes[row_index, 0].set_xlabel("residual")
        axes[row_index, 0].set_ylabel("count")

        axes[row_index, 1].hist(centered_res[channel], bins=60, color=colors["centered"], alpha=0.85)
        axes[row_index, 1].set_title(f"Centered {channel}")
        axes[row_index, 1].set_xlabel("residual")
        axes[row_index, 1].set_ylabel("count")
        
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

    fig, axes = plt.subplots(len(state_names), 1, figsize=(12, max(10, 2.2 * len(state_names))), sharex=True)
    if len(state_names) == 1:
        axes = np.asarray([axes])

    for axis, channel in zip(axes, state_names):
        raw_group = raw.groupby('canyon_bin', observed=False)[channel].agg(["mean", "std"]).reset_index(drop=True)
        centered_group = centered.groupby('canyon_bin', observed=False)[channel].agg(["mean", "std"]).reset_index(drop=True)
        axis.plot(raw_group.index, raw_group["mean"], color="#94a3b8", label="raw mean")
        axis.plot(centered_group.index, centered_group["mean"], color="#2563eb", label="centered mean")
        axis.fill_between(
            raw_group.index,
            raw_group["mean"] - raw_group["std"].fillna(0.0),
            raw_group["mean"] + raw_group["std"].fillna(0.0),
            color="#cbd5e1",
            alpha=0.35,
        )
        axis.fill_between(
            centered_group.index,
            centered_group["mean"] - centered_group["std"].fillna(0.0),
            centered_group["mean"] + centered_group["std"].fillna(0.0),
            color="#93c5fd",
            alpha=0.30,
        )
        axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.4)
        axis.set_ylabel(channel)
        axis.legend(loc="upper right")

    axes[0].set_title("Canyon-width residual mean and spread before/after nominal calibration")
    axes[-1].set_xlabel("canyon width bin")
    fig.tight_layout()
    path = os.path.join(plot_dir, "canyon_width_bias_reduction.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_calibration_scatter(calib_pred, raw_res, state_names, plot_dir):
    n_channels = len(state_names)
    n_cols = 3
    n_rows = int(np.ceil(n_channels / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for axis, channel in zip(axes, state_names):
        axis.hexbin(calib_pred[channel], raw_res[channel], gridsize=32, cmap="magma", mincnt=1)
        low = min(calib_pred[channel].min(), raw_res[channel].min())
        high = max(calib_pred[channel].max(), raw_res[channel].max())
        axis.plot([low, high], [low, high], linestyle="--", color="#e5e7eb", linewidth=1.0)
        axis.set_title(f"Predicted mean vs raw {channel}")
        axis.set_xlabel("predicted mean correction")
        axis.set_ylabel("raw residual")
    for axis in axes[n_channels:]:
        axis.axis("off")
    fig.tight_layout()
    path = os.path.join(plot_dir, "calibration_scatter.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _save_telemetry_context_plots(raw_res, centered_res, df, state_names, plot_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    axes[0].hist(df['alpha'], bins=50, color="#2563eb", alpha=0.85)
    axes[0].set_title("Angle-of-attack distribution")
    axes[0].set_xlabel("alpha [rad]")
    axes[0].set_ylabel("count")

    axes[1].hist(df['beta'], bins=50, color="#dc2626", alpha=0.82)
    axes[1].set_title("Sideslip distribution")
    axes[1].set_xlabel("beta [rad]")
    axes[1].set_ylabel("count")

    axes[2].hist(df['qbar'], bins=50, color="#0f766e", alpha=0.82)
    axes[2].set_title("Dynamic pressure distribution")
    axes[2].set_xlabel("qbar [psf]")
    axes[2].set_ylabel("count")

    if all(channel in centered_res.columns for channel in ['C_Z', 'C_M', 'C_N']):
        z_channel, m_channel, n_channel = 'C_Z', 'C_M', 'C_N'
        z_title = "alpha vs centered C_Z residual"
        m_title = "q rate vs centered C_M residual"
        n_title = "r rate vs centered C_N residual"
        z_ylabel = "C_Z residual"
        m_ylabel = "C_M residual"
        n_ylabel = "C_N residual"
    elif all(channel in centered_res.columns for channel in ['w', 'q', 'r']):
        z_channel, m_channel, n_channel = 'w', 'q', 'r'
        z_title = "alpha vs centered w residual"
        m_title = "q rate vs centered q residual"
        n_title = "r rate vs centered r residual"
        z_ylabel = "w residual"
        m_ylabel = "q residual"
        n_ylabel = "r residual"
    else:
        fallback_channels = list(centered_res.columns[:3])
        while len(fallback_channels) < 3:
            fallback_channels.append(fallback_channels[-1] if fallback_channels else state_names[0])
        z_channel, m_channel, n_channel = fallback_channels
        z_title = f"alpha vs centered {z_channel} residual"
        m_title = f"q rate vs centered {m_channel} residual"
        n_title = f"r rate vs centered {n_channel} residual"
        z_ylabel = f"{z_channel} residual"
        m_ylabel = f"{m_channel} residual"
        n_ylabel = f"{n_channel} residual"

    image = axes[3].hexbin(
        df['alpha'],
        centered_res[z_channel],
        C=df['qbar'],
        reduce_C_function=np.mean,
        gridsize=30,
        cmap="viridis",
        mincnt=1,
    )
    axes[3].set_title(z_title)
    axes[3].set_xlabel("alpha [rad]")
    axes[3].set_ylabel(z_ylabel)
    fig.colorbar(image, ax=axes[3], fraction=0.046, pad=0.04, label="mean qbar")

    axes[4].hexbin(df['q'], centered_res[m_channel], gridsize=30, cmap="magma", mincnt=1)
    axes[4].set_title(m_title)
    axes[4].set_xlabel("q [rad/s]")
    axes[4].set_ylabel(m_ylabel)

    axes[5].hexbin(df['r'], centered_res[n_channel], gridsize=30, cmap="magma", mincnt=1)
    axes[5].set_title(n_title)
    axes[5].set_xlabel("r [rad/s]")
    axes[5].set_ylabel(n_ylabel)

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

        low = min(float(actual.min()), float(sampled_values.min()), 0.0)
        high = max(float(actual.max()), float(sampled_values.max()), 0.0)
        bins = np.linspace(low, high, 36)
        axis.hist(actual, bins=bins, density=True, alpha=0.35, color=colors["actual"], label="actual")
        axis.hist(sampled_values, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["sampled"], label="model sampled")
        axis.hist(zero_baseline, bins=bins, density=True, histtype="step", linewidth=2.0, color=colors["baseline"], label="zero baseline")
        axis.set_title(f"Distribution match: {channel}")
        axis.set_xlabel("Residual value")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    for axis in axes[len(state_names):]:
        axis.axis("off")
    fig.tight_layout()
    path = os.path.join(plot_dir, "sampled_distribution_overlay.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _multimodal_histogram(values):
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values) + 1e-6)
    hist, edges = np.histogram(values, bins=80, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.2)
    peaks, _ = find_peaks(smooth, prominence=max(float(smooth.max()) * 0.045, 1e-6), distance=4)
    return centers, smooth, peaks


def _format_multimodal_group(example):
    parts = [f"{name}={value}" for name, value in zip(example['group_cols'], example['group_key'])]
    return ", ".join(parts)


def _collect_multimodal_examples(df, centered_res, state_names):
    frame = df.copy().reset_index(drop=True)
    for channel in state_names:
        frame[channel] = centered_res[channel].to_numpy(dtype=float)

    frame['canyon_width_slice'] = pd.qcut(frame['canyon_width'], q=4, duplicates='drop')
    frame['qbar_slice'] = pd.qcut(frame['qbar'], q=4, duplicates='drop')
    frame['alpha_slice'] = pd.cut(frame['alpha'], bins=12, include_lowest=True)
    frame['beta_slice'] = pd.cut(frame['beta'], bins=12, include_lowest=True)
    frame['p_slice'] = pd.qcut(frame['p'], q=4, duplicates='drop')
    frame['q_slice'] = pd.qcut(frame['q'], q=4, duplicates='drop')
    frame['r_slice'] = pd.qcut(frame['r'], q=4, duplicates='drop')
    frame['roll_mode'] = pd.cut(frame['delta_a'], bins=[-np.inf, -0.10, 0.10, np.inf], labels=['left', 'neutral', 'right'])
    frame['pitch_mode'] = pd.cut(frame['delta_e'], bins=[-np.inf, -0.10, 0.10, np.inf], labels=['up', 'neutral', 'down'])
    frame['yaw_mode'] = pd.cut(frame['delta_r'], bins=[-np.inf, -0.10, 0.10, np.inf], labels=['left', 'neutral', 'right'])

    groupings = [
        ('canyon_width_slice', 'alpha_slice'),
        ('canyon_width_slice', 'qbar_slice'),
        ('roll_mode', 'alpha_slice'),
        ('pitch_mode', 'qbar_slice'),
        ('yaw_mode', 'beta_slice'),
        ('p_slice', 'q_slice'),
        ('q_slice', 'r_slice'),
        ('canyon_width_slice', 'roll_mode'),
    ]

    examples = []
    min_group_count = 200
    for channel in state_names:
        multimodal_candidates = []
        fallback_candidates = []
        for group_cols in groupings:
            grouped = frame.groupby(list(group_cols), observed=True)
            for key, group in grouped:
                values = group[channel].to_numpy(dtype=float)
                if len(values) < min_group_count:
                    continue
                centers, smooth, peaks = _multimodal_histogram(values)
                candidate = {
                    'channel': channel,
                    'group_cols': group_cols,
                    'group_key': key if isinstance(key, tuple) else (key,),
                    'count': int(len(values)),
                    'peak_count': int(len(peaks)),
                    'peak_score': float(np.sum(smooth[peaks])) if len(peaks) > 0 else 0.0,
                    'values': values,
                    'centers': centers,
                    'smooth': smooth,
                    'peaks': peaks,
                }
                if len(peaks) >= 2:
                    multimodal_candidates.append(candidate)
                else:
                    fallback_candidates.append(candidate)

        if multimodal_candidates:
            multimodal_candidates.sort(key=lambda item: (-item['peak_count'], -item['peak_score'], -item['count']))
            chosen = dict(multimodal_candidates[0])
            chosen['selection_mode'] = 'multimodal'
            examples.append(chosen)
            continue

        if fallback_candidates:
            fallback_candidates.sort(key=lambda item: (-item['peak_score'], -item['count']))
            chosen = dict(fallback_candidates[0])
            chosen['selection_mode'] = 'slice_fallback'
            examples.append(chosen)
            continue

        # Last-resort fallback: use the full-channel residual distribution.
        values = centered_res[channel].to_numpy(dtype=float)
        centers, smooth, peaks = _multimodal_histogram(values)
        examples.append(
            {
                'channel': channel,
                'group_cols': ('scope',),
                'group_key': ('all',),
                'count': int(len(values)),
                'peak_count': int(len(peaks)),
                'peak_score': float(np.sum(smooth[peaks])) if len(peaks) > 0 else 0.0,
                'values': values,
                'centers': centers,
                'smooth': smooth,
                'peaks': peaks,
                'selection_mode': 'global_fallback',
            }
        )
    return examples


def _save_multimodal_slices(df, centered_res, state_names, plot_dir):
    examples = _collect_multimodal_examples(df, centered_res, state_names)
    n_examples = max(len(examples), 1)
    n_cols = 2
    n_rows = int(np.ceil(n_examples / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    selection_label = {
        'multimodal': 'multimodal',
        'slice_fallback': 'slice fallback',
        'global_fallback': 'global fallback',
    }

    for axis, example in zip(axes, examples):
        axis.hist(example['values'], bins=50, density=True, color="#bfdbfe", alpha=0.6)
        axis.plot(example['centers'], example['smooth'], color="#1d4ed8", linewidth=2.0)
        axis.scatter(
            example['centers'][example['peaks']],
            example['smooth'][example['peaks']],
            color="#dc2626",
            s=24,
            zorder=3,
        )
        axis.set_title(
            f"{example['channel']} | peaks={example['peak_count']} | n={example['count']} | {selection_label.get(example.get('selection_mode'), 'multimodal')}\n{_format_multimodal_group(example)}",
            fontsize=9,
        )
        axis.set_xlabel("residual")
        axis.set_ylabel("density")

    for axis in axes[len(examples):]:
        axis.axis('off')

    fig.tight_layout()
    path = os.path.join(plot_dir, "multimodal_slices.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

def generate_nominal_calibration_package():
    print("Loading Parquet dataset...")
    df = pd.read_parquet('f16_dataset.parquet')
    df = add_prev_actions(df)

    required_columns = ['mach', 'mass_slugs']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Dataset missing required columns "
            f"{missing_columns}. Regenerate the dataset with current collection pipeline."
        )
    
    print("Module 1: Nominal Coefficient Model Identification (degree=3)")
    targets = rigid_body_kinematics(df)
    nom_model = NominalModel()
    nom_model.fit(df, targets)

    nominal_coeff_preds = nom_model.predict_coefficients(df)
    true_coeffs = nom_model._targets_to_coefficients(df, targets)

    state_names = list(nom_model.coeff_targets)
    raw_res = pd.DataFrame(index=df.index)
    for s in state_names:
        raw_res[s] = true_coeffs[s] - nominal_coeff_preds[s]

    nominal_weights_path = export_nominal_coefficient_weights(
        nom_model,
        output_path=NOMINAL_COEFF_WEIGHTS_OUTPUT_PATH,
        source_dataset='f16_dataset.parquet',
    )
    print(f"Exported nominal coefficient weights to {nominal_weights_path}")
        
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
    aug_features = ['alpha', 'beta', 'mach', 'p', 'q', 'r', 'delta_t', 'delta_e', 'delta_a', 'delta_r', 
                    'prev_delta_t', 'prev_delta_e', 'prev_delta_a', 'prev_delta_r', 
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
    
    # Pack the dataset with centered coefficient residuals for runtime sampling.
    df_resid = df.copy()
    residual_columns = []
    for s in state_names:
        column_name = f'w_{s}'
        df_resid[column_name] = final_res[s]
        residual_columns.append(column_name)
        
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
    _save_multimodal_slices(df, final_res, state_names, plot_dir)
    
    print("Module 5: Artifact Serialization")
    artifact = {
        'source_dataset': 'f16_dataset.parquet',
        'nominal_model': nom_model,
        'nominal_weights_path': nominal_weights_path,
        'nominal_weight_feature_names': list(nom_model.features),
        'nominal_weight_targets': list(nom_model.coeff_targets),
        'nominal_weight_poly_degree': int(nom_model.poly.degree),
        'residual_space': 'aerodynamic_coefficients',
        'residual_columns': residual_columns,
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
    # Dispatch through the package import path so pickled artifacts keep stable
    # module-qualified class references.
    import sys

    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from jsbsim_gym.calibration import generate_nominal_calibration_package as _run_calibration

    _run_calibration()
