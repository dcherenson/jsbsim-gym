import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


DT = 1.0 / 30.0
G = 32.174
DEFAULT_FEATURES = (
    "alpha",
    "beta",
    "p",
    "q",
    "r",
    "delta_e",
    "delta_a",
    "delta_r",
    "delta_t",
)
TARGETS = ("X", "Y", "Z", "L", "M", "N")


def rigid_body_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DataFrame(index=df.index)
    targets["X"] = (df["next_u"] - df["u"]) / DT - (
        df["r"] * df["v"] - df["q"] * df["w"] - G * np.sin(df["theta"])
    )
    targets["Y"] = (df["next_v"] - df["v"]) / DT - (
        df["p"] * df["w"] - df["r"] * df["u"] + G * np.sin(df["phi"]) * np.cos(df["theta"])
    )
    targets["Z"] = (df["next_w"] - df["w"]) / DT - (
        df["q"] * df["u"] - df["p"] * df["v"] + G * np.cos(df["phi"]) * np.cos(df["theta"])
    )
    targets["L"] = (df["next_p"] - df["p"]) / DT
    targets["M"] = (df["next_q"] - df["q"]) / DT
    targets["N"] = (df["next_r"] - df["r"]) / DT
    return targets


def compute_fit_metrics(poly: PolynomialFeatures, ridge_models: dict[str, Ridge], X: np.ndarray, Y: pd.DataFrame) -> dict[str, float]:
    X_poly = poly.transform(X)
    metrics: dict[str, float] = {}
    for target in TARGETS:
        pred = ridge_models[target].predict(X_poly)
        residual = pred - Y[target].to_numpy(dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        metrics[f"{target}_rmse"] = rmse
        metrics[f"{target}_mae"] = mae
    return metrics


def fit_nominal_mppi_weights(
    dataset_path: Path,
    output_path: Path,
    ridge_alpha: float,
    degree: int,
) -> None:
    if degree != 2:
        raise ValueError("Only polynomial degree=2 is currently supported for MPPI export.")

    df = pd.read_parquet(dataset_path)
    missing = [column for column in (*DEFAULT_FEATURES, "u", "v", "w", "p", "q", "r", "phi", "theta", "next_u", "next_v", "next_w", "next_p", "next_q", "next_r") if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df.loc[:, list(DEFAULT_FEATURES)].to_numpy(dtype=np.float64)
    Y = rigid_body_kinematics(df)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    ridge_models: dict[str, Ridge] = {}
    weights = []
    intercepts = []
    for target in TARGETS:
        model = Ridge(alpha=float(ridge_alpha))
        model.fit(X_poly, Y[target].to_numpy(dtype=np.float64))
        ridge_models[target] = model
        weights.append(model.coef_)
        intercepts.append(model.intercept_)

    W = np.stack(weights, axis=-1).astype(np.float32)
    B = np.stack(intercepts, axis=0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        W=W,
        B=B,
        feature_names=np.asarray(DEFAULT_FEATURES),
        target_names=np.asarray(TARGETS),
        poly_degree=np.asarray([degree], dtype=np.int32),
        include_bias=np.asarray([1], dtype=np.int32),
        ridge_alpha=np.asarray([ridge_alpha], dtype=np.float32),
        source_dataset=np.asarray([str(dataset_path)]),
    )

    metrics = compute_fit_metrics(poly, ridge_models, X, Y)
    print(f"Saved throttle-inclusive MPPI weights to: {output_path}")
    print(f"W shape: {W.shape}, B shape: {B.shape}")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MPPI nominal force/moment weights from a canonical F-16 dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("f16_dataset.parquet"),
        help="Input canonical dataset parquet path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jsbsim_gym/mppi_nominal_weights.npz"),
        help="Output NPZ path for MPPI nominal weights.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization coefficient.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree (currently degree=2 only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_nominal_mppi_weights(
        dataset_path=args.dataset,
        output_path=args.output,
        ridge_alpha=args.ridge_alpha,
        degree=args.degree,
    )


if __name__ == "__main__":
    main()
