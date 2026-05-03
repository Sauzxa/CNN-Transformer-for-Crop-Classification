"""
Prétraitement « type article » (CDL + WorldCover + confiance), année configurable (défaut 2025).

- ESA WorldCover : classe 40 = Cultures / Cropland (masque le non-agricole).
- CDL USDA : couche de confiance (1–100), seuil typique ≥ 95 % comme dans les papiers.

Chemins par défaut relatifs à `dataset/` à la racine du projet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install rasterio") from e


# Année de campagne (S2 + étiquettes) — projet 2025
DATA_YEAR = 2025

# ESA WorldCover v200 : https://esa-worldcover.org/en/data-access
WORLDCOVER_CROPLAND_CLASS = 40

DEFAULT_CDL_PATH = "dataset/cdl_2025_clip.tif"
DEFAULT_CDL_CONF_PATH = "dataset/cdl_2025_confidence_clip.tif"
DEFAULT_WORLDCOVER_PATH = "dataset/worldcover_2021_esa.tif"


def load_cdl_labels(path: Path) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        y = src.read(1).astype(np.int32)
        profile = src.profile
    return y, profile


def remap_labels_cdl(raw: np.ndarray, ignore: tuple[int, ...] = (0,)) -> tuple[np.ndarray, dict[int, int]]:
    """Indices contigus pour stratification ; `rev` : index -> code CDL USDA."""
    ignore_set = set(ignore)
    codes = sorted(int(c) for c in np.unique(raw) if c not in ignore_set and c >= 0)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    out = np.full_like(raw, -1, dtype=np.int32)
    for c, i in code_to_idx.items():
        out[raw == c] = i
    rev = {i: c for c, i in code_to_idx.items()}
    return out, rev


def reproject_band_to_reference(
    src_path: Path,
    ref_path: Path,
    *,
    resampling: Resampling = Resampling.nearest,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Rééchantillonne la bande 1 de `src_path` sur la grille (H,W) et le CRS de `ref_path`.
    """
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        height, width = ref.height, ref.width

    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(dtype, copy=False)
        dst = np.zeros((height, width), dtype=dtype)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
        )
    return dst


def build_eligible_pixel_mask(
    cdl_raw: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    worldcover: np.ndarray | None = None,
    conf_min: float = 95.0,
    require_cropland_worldcover: bool = True,
) -> np.ndarray:
    """
    Masque booléen (H, W) : pixels utilisables pour tirer des échantillons.

    - Exclut fond CDL 0 (no data).
    - Si `confidence` : garde les pixels avec confiance >= conf_min.
    - Si `worldcover` : garde les pixels classés cultures (classe 40).
    """
    eligible = cdl_raw.astype(np.int32) != 0

    if confidence is not None:
        eligible &= confidence.astype(np.float32) >= float(conf_min)

    if worldcover is not None and require_cropland_worldcover:
        wc_i = np.rint(worldcover).astype(np.int32)
        eligible &= wc_i == WORLDCOVER_CROPLAND_CLASS

    return eligible


def apply_mask_to_label_raster(labels_hw: np.ndarray, eligible_hw: np.ndarray) -> np.ndarray:
    """Met les classes à -1 hors masque (pour que l'échantillonnage stratifié les ignore)."""
    out = labels_hw.copy()
    out[~eligible_hw] = -1
    return out


def load_optional_worldcover_aligned(
    cdl_path: Path,
    worldcover_path: Path | None,
) -> np.ndarray | None:
    if worldcover_path is None or not Path(worldcover_path).is_file():
        return None
    return reproject_band_to_reference(Path(worldcover_path), Path(cdl_path), dtype="float32")


def load_optional_cdl_confidence_aligned(
    cdl_path: Path,
    conf_path: Path | None,
) -> np.ndarray | None:
    if conf_path is None or not Path(conf_path).is_file():
        return None
    return reproject_band_to_reference(Path(conf_path), Path(cdl_path), dtype="float32")


__all__ = [
    "DATA_YEAR",
    "WORLDCOVER_CROPLAND_CLASS",
    "DEFAULT_CDL_PATH",
    "DEFAULT_CDL_CONF_PATH",
    "DEFAULT_WORLDCOVER_PATH",
    "apply_mask_to_label_raster",
    "build_eligible_pixel_mask",
    "load_cdl_labels",
    "load_optional_cdl_confidence_aligned",
    "load_optional_worldcover_aligned",
    "remap_labels_cdl",
    "reproject_band_to_reference",
]
