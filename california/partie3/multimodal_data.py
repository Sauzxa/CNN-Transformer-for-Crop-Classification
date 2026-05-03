from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def s1_monthly_feature_columns(n_months: int = 12) -> list[str]:
    cols: list[str] = []
    for m in range(1, n_months + 1):
        mm = f"{m:02d}"
        cols.extend([f"ratio_{mm}", f"vh_{mm}", f"vv_{mm}"])
    return cols


def s1_dataframe_to_tensor(df: pd.DataFrame, *, n_months: int = 12) -> np.ndarray:
    cols = s1_monthly_feature_columns(n_months)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing S1 columns: {missing[:8]}")
    arr = df[cols].to_numpy(dtype=np.float32)
    return arr.reshape(len(df), n_months, 3)


def _sanitize_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _concat_splits(processed_dir: Path, stem: str) -> np.ndarray:
    parts = []
    for split in ("train", "val", "test"):
        p = processed_dir / f"{stem}_{split}.npy"
        if p.exists():
            parts.append(np.load(p))
    if not parts:
        raise FileNotFoundError(f"No split files found for stem '{stem}' in {processed_dir}")
    return np.concatenate(parts, axis=0)


def _extract_or_create_id(df: pd.DataFrame, id_col: str) -> pd.Series:
    if id_col in df.columns:
        return pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    if "system:index" in df.columns:
        return pd.to_numeric(df["system:index"], errors="coerce").astype("Int64")
    return pd.Series(np.arange(len(df), dtype=np.int64), index=df.index, dtype="Int64")


def load_aligned_multimodal(
    project_root: Path | str,
    *,
    n_months_s1: int = 12,
    id_col: str = "id",
    n_classes: int = 6,
    s2_temporal_resample: int | None = None,
    s2_append_obs_mask_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load aligned California multimodal arrays:
    - X_s2: (N, T, C)
    - M_s2: (N, T)
    - X_s1: (N, 12, 3)
    - X_static: (N, F)
    - y: (N,)
    """
    root = Path(project_root)
    processed_dir = root / "processed_data"
    s1_path = root / "partie3" / "s1_monthly_timeseries.csv"
    cov_path = root / "dataset" / "covariables_environnementales.csv"

    X_s2 = _concat_splits(processed_dir, "X")
    M_s2 = _concat_splits(processed_dir, "m")
    y = _concat_splits(processed_dir, "y").astype(np.int32).reshape(-1)
    y = np.clip(y, 0, max(0, int(n_classes) - 1))

    if not cov_path.exists():
        raise FileNotFoundError(f"Missing file: {cov_path}")

    df_cov = pd.read_csv(cov_path)
    if s1_path.exists():
        df_s1 = pd.read_csv(s1_path)
    else:
        # Fallback: allow the pipeline to run without GEE S1 extraction.
        # The S1 branch will receive zeros with proper (N, 12, 3) shape.
        print(f"[multimodal_data] Warning: missing {s1_path}. Using zero S1 features fallback.")
        df_tmp = df_cov.copy()
        tmp_id = _extract_or_create_id(df_tmp, id_col)
        df_tmp[id_col] = tmp_id
        df_tmp = df_tmp.dropna(subset=[id_col]).astype({id_col: int})
        s1_zero_cols = s1_monthly_feature_columns(n_months_s1)
        df_s1 = pd.DataFrame({id_col: df_tmp[id_col].to_numpy(dtype=np.int64)})
        for c in s1_zero_cols:
            df_s1[c] = 0.0

    df_s1 = df_s1.copy()
    df_cov = df_cov.copy()
    df_s1[id_col] = _extract_or_create_id(df_s1, id_col)
    df_cov[id_col] = _extract_or_create_id(df_cov, id_col)
    df_s1 = df_s1.dropna(subset=[id_col]).astype({id_col: int})
    df_cov = df_cov.dropna(subset=[id_col]).astype({id_col: int})

    merged = df_s1.merge(df_cov, on=id_col, how="inner", suffixes=("", "_covdup"))
    merged = merged.loc[:, ~merged.columns.str.endswith("_covdup")]
    merged = merged.sort_values(id_col).reset_index(drop=True)

    static_exclude = {id_col, "lon", "lat", "system:index"}
    static_cols = [c for c in df_cov.columns if c not in static_exclude]
    if not static_cols:
        raise ValueError("No static covariables found in covariables CSV.")

    ids = merged[id_col].to_numpy(dtype=np.int64)
    valid_ids = ids[(ids >= 0) & (ids < len(X_s2))]
    if merged.empty or valid_ids.size == 0:
        # Robust fallback when id semantics differ between files (e.g., S1 id=-1 from system:index parse).
        print(
            "[multimodal_data] Warning: empty/invalid id merge between S1 and covariables. "
            "Falling back to row-order alignment."
        )
        n = min(len(X_s2), len(df_s1), len(df_cov))
        X_s2 = X_s2[:n].astype(np.float32, copy=False)
        M_s2 = M_s2[:n].astype(np.float32, copy=False)
        y = y[:n]
        X_s1 = s1_dataframe_to_tensor(df_s1.iloc[:n], n_months=n_months_s1)
        X_static = df_cov.iloc[:n][static_cols].to_numpy(dtype=np.float32)
        meta = pd.DataFrame({id_col: np.arange(n, dtype=np.int64)})
        X_s2 = _sanitize_array(X_s2)
        M_s2 = _sanitize_array(M_s2)
        X_s1 = _sanitize_array(X_s1)
        X_static = _sanitize_array(X_static)
        return X_s2, M_s2, X_s1, X_static, y, meta

    if np.any(ids < 0) or np.any(ids >= len(X_s2)):
        bad = ids[(ids < 0) | (ids >= len(X_s2))]
        raise IndexError(f"Invalid ids after merge for S2 arrays: {bad[:8]}")

    X_s2 = X_s2[ids].astype(np.float32, copy=False)
    M_s2 = M_s2[ids].astype(np.float32, copy=False)
    y = y[ids]
    X_s1 = s1_dataframe_to_tensor(merged, n_months=n_months_s1)
    X_static = merged[static_cols].to_numpy(dtype=np.float32)

    if s2_temporal_resample is not None:
        from partie3.temporal_regularization import (
            append_observation_mask_channel,
            build_valid_timestep_mask,
            resample_temporal_linear,
            resample_timestep_mask_nearest,
        )

        vm = build_valid_timestep_mask(X_s2)
        X_s2 = resample_temporal_linear(X_s2, int(s2_temporal_resample))
        M_s2 = resample_timestep_mask_nearest(vm, int(s2_temporal_resample))
        if s2_append_obs_mask_channel:
            X_s2 = append_observation_mask_channel(X_s2, M_s2)

    X_s2 = _sanitize_array(X_s2)
    M_s2 = _sanitize_array(M_s2)
    X_s1 = _sanitize_array(X_s1)
    X_static = _sanitize_array(X_static)
    return X_s2, M_s2, X_s1, X_static, y, merged[[id_col]].copy()


def stratified_train_val_test_indices(
    y: np.ndarray,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    y = np.asarray(y).reshape(-1)
    idx = np.arange(len(y), dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        stratify=y[idx_trainval],
        random_state=random_state,
    )
    return idx_train.astype(np.int64), idx_val.astype(np.int64), idx_test.astype(np.int64)


def standardize_modalities_fit_transform(
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    idx_train: np.ndarray,
):
    from sklearn.preprocessing import StandardScaler

    sc_s2 = StandardScaler()
    sc_s1 = StandardScaler()
    sc_st = StandardScaler()

    tr_s2 = X_s2[idx_train].reshape(len(idx_train), -1)
    tr_s1 = X_s1[idx_train].reshape(len(idx_train), -1)
    tr_st = X_static[idx_train]
    sc_s2.fit(tr_s2)
    sc_s1.fit(tr_s1)
    sc_st.fit(tr_st)

    X_s2_n = sc_s2.transform(X_s2.reshape(len(X_s2), -1)).reshape(X_s2.shape).astype(np.float32)
    X_s1_n = sc_s1.transform(X_s1.reshape(len(X_s1), -1)).reshape(X_s1.shape).astype(np.float32)
    X_st_n = sc_st.transform(X_static).astype(np.float32)
    scalers = {"s2": sc_s2, "s1": sc_s1, "static": sc_st}
    return (X_s2_n, X_s1_n, X_st_n), scalers


__all__ = [
    "load_aligned_multimodal",
    "s1_dataframe_to_tensor",
    "s1_monthly_feature_columns",
    "stratified_train_val_test_indices",
    "standardize_modalities_fit_transform",
]

