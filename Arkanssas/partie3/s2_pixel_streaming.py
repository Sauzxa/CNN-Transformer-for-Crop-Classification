"""
Lecture Sentinel-2 L2A (.SAFE) en streaming sur des pixels échantillonnés — même logique que le notebook Partie 1.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install rasterio") from e

# Bandes « article » (10 bandes, sans B1/B9/B10 — 60 m)
S2_BANDS_PAPER: Tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
)


def find_safe_granule_dir(safe_root: Path) -> Path:
    granule = safe_root / "GRANULE"
    if not granule.is_dir():
        raise FileNotFoundError(f"Pas de dossier GRANULE dans {safe_root}")
    for child in sorted(granule.iterdir()):
        if child.is_dir() and child.name.startswith("L2A_"):
            return child
    raise FileNotFoundError(f"Aucun sous-dossier L2A_* dans {granule}")


def list_band_paths(granule_dir: Path, resolution: str = "10m") -> Dict[str, Path]:
    img_data = granule_dir / "IMG_DATA"
    if not img_data.is_dir():
        raise FileNotFoundError(f"IMG_DATA manquant dans {granule_dir}")
    bands: Dict[str, Path] = {}
    for sub_name in (f"R{resolution}", "R10m", "R20m"):
        sub = img_data / sub_name
        if not sub.is_dir():
            continue
        for p in sub.glob("*.jp2"):
            m = re.search(r"_B([0-9A]+)_", p.name, re.I)
            if m:
                key = f"B{m.group(1).upper()}"
                if key not in bands:
                    bands[key] = p
    return bands


def select_pixel_indices_stratified_with_y(
    labels_hw: np.ndarray,
    max_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = labels_hw.shape
    flat_y = labels_hw.reshape(-1)
    valid = flat_y >= 0
    classes = np.unique(flat_y[valid])
    per_class = max(1, max_samples // int(classes.size))
    parts = []
    y_parts = []
    for cl in classes:
        pos = np.flatnonzero(valid & (flat_y == cl))
        if pos.size > 0:
            n_take = min(per_class, pos.size)
            chosen = rng.choice(pos, size=n_take, replace=False)
            parts.append(chosen)
            y_parts.append(np.full(n_take, cl, dtype=np.int32))
    if not parts:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    idx = np.concatenate(parts)
    y_concat = np.concatenate(y_parts)
    if idx.size > max_samples:
        sel = rng.choice(idx.size, size=max_samples, replace=False)
        idx = idx[sel]
        y_concat = y_concat[sel]
    rows = idx // w
    cols = idx % w
    return rows.astype(np.int32), cols.astype(np.int32), y_concat.astype(np.int32)


def scale_to_reflectance(x: np.ndarray) -> np.ndarray:
    """L2A DN souvent 0–10000 ; ramène vers ~0–1 pour indices spectraux."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x > 1.5, x / 10000.0, np.clip(x, 0.0, 1.0))


def build_pixel_dataset_streaming(
    safe_paths: List[Path],
    labels_hw: np.ndarray,
    max_samples: int,
    seed: int = 42,
    band_names: Tuple[str, ...] | None = None,
    resolution: str = "10m",
    stratified: bool = True,
    *,
    add_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (N, T, C) avec NaN pour nodata ; mask_valid_time: (N, T).

    Si ``add_indices`` : ajoute NDVI et NDWI (C = 10 + 2) après réflectance.
    """
    if band_names is None:
        band_names = S2_BANDS_PAPER
    band_names = tuple(band_names)
    safe_paths = [Path(p) for p in safe_paths]

    if stratified:
        rows, cols, y = select_pixel_indices_stratified_with_y(labels_hw, max_samples=max_samples, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        h, w = labels_hw.shape
        flat_y = labels_hw.reshape(-1)
        valid = flat_y >= 0
        idx = np.flatnonzero(valid)
        if idx.size > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        rows = (idx // w).astype(np.int32)
        cols = (idx % w).astype(np.int32)
        y = flat_y[idx].astype(np.int32)

    n = rows.size
    t = len(safe_paths)
    c0 = len(band_names)

    granule0 = find_safe_granule_dir(safe_paths[0])
    paths0 = list_band_paths(granule0, resolution=resolution)
    first_path = paths0[band_names[0]]
    with rasterio.open(first_path) as ref:
        transform = ref.transform
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        points = list(zip(xs, ys))

    X = np.empty((n, t, c0), dtype=np.float32)
    X[:] = np.nan

    for ti, safe in enumerate(safe_paths):
        gran = find_safe_granule_dir(safe)
        band_paths = list_band_paths(gran, resolution=resolution)
        missing = [b for b in band_names if b not in band_paths]
        if missing:
            raise KeyError(f"Bandes manquantes dans {safe}: {missing}")

        for bi, b in enumerate(band_names):
            with rasterio.open(band_paths[b]) as src:
                nodata = src.nodata
                vals = np.array([v[0] for v in src.sample(points)], dtype=np.float32)
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)
                X[:, ti, bi] = vals

    mask_valid_time = np.any(~np.isnan(X), axis=-1)

    if add_indices:
        idx = {n: i for i, n in enumerate(band_names)}
        eps = 1e-6
        X_ref = scale_to_reflectance(X)
        B03 = X_ref[:, :, idx["B03"]]
        B04 = X_ref[:, :, idx["B04"]]
        B08 = X_ref[:, :, idx["B08"]]
        ndvi = (B08 - B04) / (B08 + B04 + eps)
        ndwi = (B03 - B08) / (B03 + B08 + eps)
        X = np.concatenate([X_ref, ndvi[..., None], ndwi[..., None]], axis=-1).astype(np.float32)
        mask_valid_time = np.any(~np.isnan(X), axis=-1)

    return X, y, mask_valid_time


__all__ = [
    "S2_BANDS_PAPER",
    "build_pixel_dataset_streaming",
    "find_safe_granule_dir",
    "list_band_paths",
    "scale_to_reflectance",
    "select_pixel_indices_stratified_with_y",
]
