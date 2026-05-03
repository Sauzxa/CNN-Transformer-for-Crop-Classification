from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np


def parse_safe_acquisition_date(safe_path: Path) -> date:
    name = safe_path.name
    m = re.search(r"_(\d{8})T\d{6}_", name)
    if m:
        s = m.group(1)
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    m2 = re.search(r"(\d{8})", name)
    if m2:
        s = m2.group(1)
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"Date not found in SAFE name: {safe_path}")


def decadal_median_composite(
    X: np.ndarray,
    dates: list[date] | list[datetime],
    *,
    year: int,
    n_bins: int = 36,
    bin_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must be (N, T, C)")
    n, t_obs, c = X.shape
    if len(dates) != t_obs:
        raise ValueError("len(dates) must match T")

    bin_to_cols: dict[int, list[int]] = {}
    for j, d in enumerate(dates):
        if isinstance(d, datetime):
            d = d.date()
        if d.year != year:
            continue
        doy = int(d.timetuple().tm_yday)
        bi = min((doy - 1) // bin_days, n_bins - 1)
        bin_to_cols.setdefault(bi, []).append(j)

    X_out = np.zeros((n, n_bins, c), dtype=np.float32)
    mask_obs = np.zeros((n, n_bins), dtype=bool)
    for bi in range(n_bins):
        cols = bin_to_cols.get(bi, [])
        if not cols:
            continue
        sub = X[:, cols, :]
        med = np.nanmedian(sub, axis=1)
        X_out[:, bi, :] = np.nan_to_num(med, nan=0.0).astype(np.float32)
        mask_obs[:, bi] = np.any(np.isfinite(sub), axis=(1, 2))
    return X_out, mask_obs

