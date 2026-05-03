"""
Composites temporels type article : fenêtres de `bin_days` jours, médiane par fenêtre, manquants → 0.

Référence : médiane après masquage nuage sur intervalles de 10 jours, 36 pas pour une saison ~annuelle.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np

__all__ = [
    "parse_safe_acquisition_date",
    "decadal_median_composite",
    "year_window_bins",
]


def parse_safe_acquisition_date(safe_path: Path) -> date:
    """
    Date d'acquisition depuis le nom du dossier .SAFE (Sentinel-2).
    Ex. S2A_MSIL2A_20250601T162901_...
    """
    name = safe_path.name
    m = re.search(r"_(\d{8})T\d{6}_", name)
    if m:
        s = m.group(1)
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    m2 = re.search(r"(\d{8})", name)
    if m2:
        s = m2.group(1)
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"Date introuvable dans le nom SAFE : {safe_path}")


def year_window_bins(
    year: int,
    *,
    n_bins: int = 36,
    bin_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne les bornes [start, end) en jour de l'année pour chaque bin (0-indexé).
    `n_bins` * `bin_days` couvre typiquement 360 jours (comme beaucoup d'articles).
    """
    starts = np.arange(n_bins, dtype=np.int32) * bin_days + 1
    ends = starts + bin_days
    return starts, ends


def _doy(d: date) -> int:
    return int(d.timetuple().tm_yday)


def decadal_median_composite(
    X: np.ndarray,
    dates: list[date] | list[datetime],
    *,
    year: int,
    n_bins: int = 36,
    bin_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Agrège (N, T_obs, C) en (N, n_bins, C) par médiane dans chaque fenêtre décadaire.

    Chaque observation est assignée à un bin par jour de l'année : ``bi = min((doy-1)//bin_days, n_bins-1)``.
    Les jours 361–365 tombent ainsi dans le dernier bin (comme une fin de saison regroupée).

    Les dates hors ``year`` sont ignorées.

    Returns
    -------
    X_out : (N, n_bins, C) float32 — médiane puis 0 si tout NaN dans le bin
    mask_obs : (N, n_bins) bool — True si au moins une observation valide dans le bin
    """
    from collections import defaultdict

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X doit être (N, T, C)")
    n, t_obs, c = X.shape
    if len(dates) != t_obs:
        raise ValueError(f"len(dates)={len(dates)} != T={t_obs}")

    bin_to_cols: dict[int, list[int]] = defaultdict(list)
    for j, d in enumerate(dates):
        if isinstance(d, datetime):
            d = d.date()
        if d.year != year:
            continue
        doy = _doy(d)
        bi = (doy - 1) // bin_days
        if bi >= n_bins:
            bi = n_bins - 1
        bin_to_cols[bi].append(j)

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
