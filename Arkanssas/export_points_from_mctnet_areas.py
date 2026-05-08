"""
Génère `dataset/points_agricoles.csv` depuis :
- la grille (36 tifs) `dataset/MCTNet_Area{1,2}/*.tif`
- le raster de labels `dataset/cdl_area{1,2}.tif`

But : exporter des points (lon/lat) **alignés avec les pixels** échantillonnés qui servent à construire X/y
dans la partie 1 (mêmes classes et mêmes règles de sampling).

Usage (depuis la racine du projet) :
  python Arkanssas/export_points_from_mctnet_areas.py --area 1 --max-samples 10000 --seed 42
  python Arkanssas/export_points_from_mctnet_areas.py --area 2 --max-samples 10000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.transform import xy as transform_xy


def load_tif_list(area_dir: Path) -> list[Path]:
    tifs = sorted(area_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"Aucun .tif dans {area_dir}")
    return tifs


def reproject_cdl_to_tif(cdl_path: Path, ref_tif_path: Path) -> np.ndarray:
    """
    Reprojette/re-échantillonne le CDL sur la grille du premier tif (nearest pour labels entiers).
    Retourne un tableau (H, W) int32.
    """
    with rasterio.open(ref_tif_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height = ref.height
        ref_width = ref.width

    with rasterio.open(cdl_path) as src:
        dst_arr = np.empty((ref_height, ref_width), dtype=np.int32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=WarpResampling.nearest,
        )
    return dst_arr


def map_cdl_to_5classes(cdl_arr: np.ndarray) -> np.ndarray:
    """
    Même mapping que dans le notebook (Partie 1) pour Arkansas / CDL.
    """
    out = np.full_like(cdl_arr, -1, dtype=np.int32)
    out[cdl_arr == 1] = 0  # Corn
    out[cdl_arr == 2] = 1  # Cotton
    out[cdl_arr == 5] = 2  # Soybean
    out[cdl_arr == 3] = 3  # Rice

    # Tous les autres codes "crop" -> Others (classe 4), en excluant les codes déjà mappés.
    crop_codes = set(range(1, 61))
    main_codes = {1, 2, 3, 5}
    for code in crop_codes - main_codes:
        out[cdl_arr == code] = 4
    return out


def select_pixel_indices_stratified(
    labels_hw: np.ndarray,
    max_samples: int,
    min_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replique la logique du notebook : échantillonnage stratifié sur y (labels remappés 5 classes).
    Retourne (rows, cols) dans l'ordre de concat des classes (classes triées).
    """
    rng = np.random.default_rng(seed)
    h, w = labels_hw.shape
    flat_y = labels_hw.reshape(-1)
    valid = flat_y >= 0
    classes = np.unique(flat_y[valid])

    parts: list[np.ndarray] = []
    for cl in classes:
        cl = int(cl)
        pos = np.flatnonzero(valid & (flat_y == cl))
        if pos.size == 0:
            continue
        n_take = min(min_per_class, pos.size)
        # replace=True pour rester robuste si pos.size < min_per_class (cas rare).
        parts.append(rng.choice(pos, size=n_take, replace=True))

    if not parts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    idx = np.concatenate(parts, axis=0)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)

    rows = (idx // w).astype(np.int32)
    cols = (idx % w).astype(np.int32)
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=int, choices=[1, 2], required=True)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--min-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir or (repo_root / "dataset")

    area_dir = dataset_dir / f"MCTNet_Area{args.area}"
    cdl_path = dataset_dir / f"cdl_area{args.area}.tif"
    if not area_dir.is_dir():
        raise FileNotFoundError(f"Zone tifs introuvable : {area_dir}")
    if not cdl_path.is_file():
        raise FileNotFoundError(f"CDL introuvable : {cdl_path}")

    tifs = load_tif_list(area_dir)
    ref_tif = tifs[0]

    # 1) Labels remappés sur la grille Area{area}
    y_raw_reproj = reproject_cdl_to_tif(cdl_path, ref_tif)
    y_5 = map_cdl_to_5classes(y_raw_reproj)

    # 2) Sampling stratifié (mêmes classes/seed que partie 1)
    rows, cols = select_pixel_indices_stratified(
        y_5,
        max_samples=args.max_samples,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )
    if rows.size == 0:
        raise RuntimeError("Aucun point sélectionné (vérifier mapping CDL et masques).")

    # 3) Conversion pixels -> coordonnées (lon/lat EPSG:4326)
    with rasterio.open(ref_tif) as src:
        if src.crs is None:
            raise RuntimeError(f"CRS manquant dans {ref_tif}")
        xs, ys = transform_xy(src.transform, rows, cols, offset="center")

        # rasterio.warp.transform fournit lon/lat depuis src.crs
        from rasterio.warp import transform as warp_transform

        lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)

    df = pd.DataFrame(
        {
            "id": np.arange(len(lons), dtype=np.int64),
            "lon": lons,
            "lat": lats,
        }
    )

    out_path = args.out or (dataset_dir / "points_agricoles.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(
        f"OK: {len(df)} points exportés pour Area{args.area} -> {out_path} "
        f"(seed={args.seed}, max_samples={args.max_samples})"
    )


if __name__ == "__main__":
    main()

