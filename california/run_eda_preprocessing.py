from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = {
    0: "Grapes",
    1: "Rice",
    2: "Alfalfa",
    3: "Almonds",
    4: "Pistachios",
    5: "Others",
}

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
TIMESTEPS = list(range(36))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="California EDA + preprocessing for MCTNet-like training."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/California_MCTNet_S2_360f_Mask36_10000_opt.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("prepared"),
        help="Output directory for figures and npz.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split ratio (global ratio, not relative).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Create an interpolated version (optional branch for report).",
    )
    return parser.parse_args()


def sort_feature_columns(cols: List[str]) -> List[str]:
    def key_fn(col: str) -> Tuple[str, int]:
        band, t = col.split("_t")
        return band, int(t)

    return sorted(cols, key=key_fn)


def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feat_re = re.compile(r"^B\d{1,2}A?_t\d+$")
    mask_re = re.compile(r"^m_t\d+$")
    feature_cols = [c for c in df.columns if feat_re.match(c)]
    mask_cols = [c for c in df.columns if mask_re.match(c)]
    feature_cols = sort_feature_columns(feature_cols)
    mask_cols = sorted(mask_cols, key=lambda x: int(x.split("_t")[-1]))
    return feature_cols, mask_cols


def validate_schema(feature_cols: List[str], mask_cols: List[str], df: pd.DataFrame) -> None:
    if "crop_label" not in df.columns:
        raise ValueError("Column 'crop_label' is missing.")
    if len(feature_cols) != 360:
        raise ValueError(f"Expected 360 feature columns, got {len(feature_cols)}.")
    if len(mask_cols) != 36:
        raise ValueError(f"Expected 36 mask columns, got {len(mask_cols)}.")

    expected_feat = [f"{b}_t{t}" for b in BANDS for t in TIMESTEPS]
    missing_feat = [c for c in expected_feat if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing expected feature columns, first 10: {missing_feat[:10]}")

    expected_masks = [f"m_t{t}" for t in TIMESTEPS]
    missing_masks = [c for c in expected_masks if c not in df.columns]
    if missing_masks:
        raise ValueError(f"Missing expected mask columns: {missing_masks}")


def build_tensors(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack(
        [df[[f"{b}_t{t}" for b in BANDS]].to_numpy(dtype=np.float32) for t in TIMESTEPS],
        axis=1,
    )
    m = df[[f"m_t{t}" for t in TIMESTEPS]].to_numpy(dtype=np.float32)
    y = df["crop_label"].to_numpy(dtype=np.int64)
    return x, m, y


def save_class_distribution(y: np.ndarray, outdir: Path) -> Dict[int, int]:
    counts = pd.Series(y).value_counts().sort_index()
    dct = {int(k): int(v) for k, v in counts.to_dict().items()}

    plt.figure(figsize=(8, 4))
    plt.bar([CLASS_NAMES.get(i, str(i)) for i in counts.index], counts.values)
    plt.title("California - class distribution")
    plt.ylabel("Samples")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(outdir / "eda_class_distribution.png", dpi=150)
    plt.close()
    return dct


def save_missing_analysis(m: np.ndarray, outdir: Path) -> Dict[str, float]:
    missing_rate_sample = 1.0 - m.mean(axis=1)
    missing_rate_timestep = 1.0 - m.mean(axis=0)

    plt.figure(figsize=(7, 4))
    plt.hist(missing_rate_sample, bins=30)
    plt.title("Missing rate per sample")
    plt.xlabel("Missing rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "eda_missing_rate_sample_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(TIMESTEPS, missing_rate_timestep, marker="o")
    plt.title("Missing rate per timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Missing rate")
    plt.tight_layout()
    plt.savefig(outdir / "eda_missing_rate_per_timestep.png", dpi=150)
    plt.close()

    return {
        "missing_rate_sample_mean": float(missing_rate_sample.mean()),
        "missing_rate_sample_min": float(missing_rate_sample.min()),
        "missing_rate_sample_max": float(missing_rate_sample.max()),
        "missing_rate_timestep_mean": float(missing_rate_timestep.mean()),
    }


def save_ndvi_curves(x: np.ndarray, y: np.ndarray, outdir: Path) -> None:
    idx_b4 = BANDS.index("B4")
    idx_b8 = BANDS.index("B8")
    ndvi = (x[:, :, idx_b8] - x[:, :, idx_b4]) / (x[:, :, idx_b8] + x[:, :, idx_b4] + 1e-6)

    plt.figure(figsize=(10, 5))
    for c in sorted(np.unique(y)):
        c_mask = y == c
        if c_mask.sum() == 0:
            continue
        mean_curve = ndvi[c_mask].mean(axis=0)
        std_curve = ndvi[c_mask].std(axis=0)
        name = CLASS_NAMES.get(int(c), f"class_{c}")
        plt.plot(TIMESTEPS, mean_curve, label=f"{name} (n={c_mask.sum()})")
        plt.fill_between(TIMESTEPS, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

    plt.title("California NDVI time-series per class")
    plt.xlabel("10-day timestep")
    plt.ylabel("NDVI")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "eda_ndvi_curves_by_class.png", dpi=150)
    plt.close()


def split_data(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    holdout_size = val_size + test_size
    if holdout_size <= 0.0 or holdout_size >= 1.0:
        raise ValueError("val_size + test_size must be in (0, 1).")

    x_train, x_hold, m_train, m_hold, y_train, y_hold = train_test_split(
        x, m, y, test_size=holdout_size, random_state=seed, stratify=y
    )

    rel_test = test_size / holdout_size
    x_val, x_test, m_val, m_test, y_val, y_test = train_test_split(
        x_hold, m_hold, y_hold, test_size=rel_test, random_state=seed, stratify=y_hold
    )

    return x_train, x_val, x_test, m_train, m_val, m_test, y_train, y_val, y_test


def normalize_by_band(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    sigma = x_train.reshape(-1, x_train.shape[-1]).std(axis=0) + 1e-6

    def norm(x_in: np.ndarray) -> np.ndarray:
        return ((x_in - mu[None, None, :]) / sigma[None, None, :]).astype(np.float32)

    return norm(x_train), norm(x_val), norm(x_test), mu.astype(np.float32), sigma.astype(np.float32)


def interpolate_timewise(x_ntc: np.ndarray, m_nt: np.ndarray) -> np.ndarray:
    x = x_ntc.copy()
    n, t, c = x.shape
    tgrid = np.arange(t)
    for i in range(n):
        valid = np.where(m_nt[i] > 0.5)[0]
        if len(valid) == 0:
            continue
        for ch in range(c):
            x[i, :, ch] = np.interp(tgrid, valid, x[i, valid, ch])
    return x


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    plt.style.use("seaborn-v0_8")

    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    for col in ("system:index", ".geo"):
        if col in df.columns:
            df = df.drop(columns=col)

    feature_cols, mask_cols = infer_columns(df)
    validate_schema(feature_cols, mask_cols, df)

    x, m, y = build_tensors(df)
    class_dist = save_class_distribution(y, args.outdir)
    missing_stats = save_missing_analysis(m, args.outdir)
    save_ndvi_curves(x, y, args.outdir)

    (
        x_train,
        x_val,
        x_test,
        m_train,
        m_val,
        m_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(x, m, y, args.val_size, args.test_size, args.seed)

    x_train_n, x_val_n, x_test_n, mu, sigma = normalize_by_band(x_train, x_val, x_test)
    class_weight = compute_class_weights(y_train)

    np.savez_compressed(
        args.outdir / "california_mctnet_ready.npz",
        X_train=x_train_n,
        X_val=x_val_n,
        X_test=x_test_n,
        M_train=m_train.astype(np.float32),
        M_val=m_val.astype(np.float32),
        M_test=m_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        mu=mu,
        sigma=sigma,
    )

    if args.interpolate:
        x_train_i = interpolate_timewise(x_train, m_train)
        x_val_i = interpolate_timewise(x_val, m_val)
        x_test_i = interpolate_timewise(x_test, m_test)
        x_train_i_n, x_val_i_n, x_test_i_n, mu_i, sigma_i = normalize_by_band(
            x_train_i, x_val_i, x_test_i
        )
        np.savez_compressed(
            args.outdir / "california_mctnet_ready_interpolated.npz",
            X_train=x_train_i_n,
            X_val=x_val_i_n,
            X_test=x_test_i_n,
            M_train=m_train.astype(np.float32),
            M_val=m_val.astype(np.float32),
            M_test=m_test.astype(np.float32),
            y_train=y_train.astype(np.int64),
            y_val=y_val.astype(np.int64),
            y_test=y_test.astype(np.int64),
            mu=mu_i,
            sigma=sigma_i,
        )

    report = {
        "input_csv": str(args.csv.resolve()),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "class_distribution": class_dist,
        "missing_stats": missing_stats,
        "split_shapes": {
            "train": list(x_train_n.shape),
            "val": list(x_val_n.shape),
            "test": list(x_test_n.shape),
        },
        "class_weight": class_weight,
        "interpolated_saved": bool(args.interpolate),
    }

    with open(args.outdir / "eda_preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"- Ready arrays: {args.outdir / 'california_mctnet_ready.npz'}")
    print(f"- Figures + report: {args.outdir}")
    if args.interpolate:
        print(f"- Interpolated arrays: {args.outdir / 'california_mctnet_ready_interpolated.npz'}")


if __name__ == "__main__":
    main()

