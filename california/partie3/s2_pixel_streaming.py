from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install rasterio") from e

S2_BANDS_PAPER: Tuple[str, ...] = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")


def find_safe_granule_dir(safe_root: Path) -> Path:
    granule = safe_root / "GRANULE"
    if not granule.is_dir():
        raise FileNotFoundError(f"No GRANULE in {safe_root}")
    for child in sorted(granule.iterdir()):
        if child.is_dir() and child.name.startswith("L2A_"):
            return child
    raise FileNotFoundError(f"No L2A_* in {granule}")


def list_band_paths(granule_dir: Path, resolution: str = "10m") -> Dict[str, Path]:
    img_data = granule_dir / "IMG_DATA"
    if not img_data.is_dir():
        raise FileNotFoundError(f"IMG_DATA missing in {granule_dir}")
    bands: Dict[str, Path] = {}
    for sub_name in (f"R{resolution}", "R10m", "R20m"):
        sub = img_data / sub_name
        if not sub.is_dir():
            continue
        for p in sub.glob("*.jp2"):
            m = re.search(r"_B([0-9A]+)_", p.name, re.I)
            if m:
                key = f"B{m.group(1).upper()}"
                bands.setdefault(key, p)
    return bands


def build_pixel_dataset_streaming(
    safe_paths: List[Path],
    labels_hw: np.ndarray,
    max_samples: int,
    seed: int = 42,
    band_names: Tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if band_names is None:
        band_names = S2_BANDS_PAPER
    band_names = tuple(band_names)
    safe_paths = [Path(p) for p in safe_paths]

    rng = np.random.default_rng(seed)
    h, w = labels_hw.shape
    flat = labels_hw.reshape(-1)
    valid = np.flatnonzero(flat >= 0)
    idx = valid if valid.size <= max_samples else rng.choice(valid, size=max_samples, replace=False)
    rows = (idx // w).astype(np.int32)
    cols = (idx % w).astype(np.int32)
    y = flat[idx].astype(np.int32)

    granule0 = find_safe_granule_dir(safe_paths[0])
    paths0 = list_band_paths(granule0)
    with rasterio.open(paths0[band_names[0]]) as ref:
        xs, ys = rasterio.transform.xy(ref.transform, rows, cols, offset="center")
        points = list(zip(xs, ys))

    n, t, c = rows.size, len(safe_paths), len(band_names)
    X = np.full((n, t, c), np.nan, dtype=np.float32)
    for ti, safe in enumerate(safe_paths):
        gran = find_safe_granule_dir(safe)
        bp = list_band_paths(gran)
        for bi, b in enumerate(band_names):
            with rasterio.open(bp[b]) as src:
                vals = np.array([v[0] for v in src.sample(points)], dtype=np.float32)
                if src.nodata is not None:
                    vals = np.where(vals == src.nodata, np.nan, vals)
                X[:, ti, bi] = vals
    mask_valid_time = np.any(~np.isnan(X), axis=-1)
    return X, y, mask_valid_time

