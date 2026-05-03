"""
Exporte dataset/points_agricoles.csv pour GEE / extraction S1 / covariables.

Améliorations « type article » (année 2025) :
- Masque ESA WorldCover : ne garder que les pixels classés cultures (classe 40),
  si `dataset/worldcover_2021_esa.tif` (ou `--worldcover`) est fourni ; rééchantillonnage sur la grille du CDL.
- Couche de confiance CDL USDA : seuil 95 % par défaut,
  si `dataset/cdl_2025_confidence_clip.tif` est fourni (même emprise que le CDL).

Téléchargements typiques :
- CDL : https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php
- Confiance CDL : produit associé *Confidence Layer* (même année / emprise).
- WorldCover 2021 : https://esa-worldcover.org/en/data-access (rééchantillonnage nearest sur la grille CDL).

Fichiers optionnels : si absents, l’étape correspondante est ignorée (avertissement).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from partie3.paper_dataset import (  # noqa: E402
    DATA_YEAR,
    DEFAULT_CDL_CONF_PATH,
    DEFAULT_CDL_PATH,
    DEFAULT_WORLDCOVER_PATH,
    apply_mask_to_label_raster,
    build_eligible_pixel_mask,
    load_cdl_labels,
    load_optional_cdl_confidence_aligned,
    load_optional_worldcover_aligned,
    remap_labels_cdl,
)


def select_pixel_indices_stratified(labels_hw: np.ndarray, max_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    h, w = labels_hw.shape
    flat_y = labels_hw.reshape(-1)
    valid = flat_y >= 0
    classes = np.unique(flat_y[valid])
    if classes.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    per_class = max(1, max_samples // int(classes.size))
    parts = []
    for cl in classes:
        pos = np.flatnonzero(valid & (flat_y == cl))
        if pos.size > 0:
            n_take = min(per_class, pos.size)
            parts.append(rng.choice(pos, size=n_take, replace=False))
    idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    rows = idx // w
    cols = idx % w
    return rows.astype(np.int32), cols.astype(np.int32)


def export_coordinates(
    *,
    dataset_dir: Path,
    cdl_path: Path,
    max_samples: int,
    seed: int,
    conf_path: Path | None,
    worldcover_path: Path | None,
    conf_min: float,
    skip_worldcover: bool,
    skip_confidence: bool,
) -> None:
    print(f"Année de référence projet : {DATA_YEAR}")
    print("Chargement du CDL...")
    y_raw, _ = load_cdl_labels(cdl_path)

    conf_arr = None
    if not skip_confidence:
        conf_arr = load_optional_cdl_confidence_aligned(cdl_path, conf_path)
        if conf_arr is None:
            print(f"[info] Pas de couche de confiance ({conf_path or 'défaut'}) — filtre confiance désactivé.")
    wc_arr = None
    if not skip_worldcover:
        wc_arr = load_optional_worldcover_aligned(cdl_path, worldcover_path)
        if wc_arr is None:
            print(f"[info] Pas de WorldCover ({worldcover_path or 'défaut'}) — masque cultures désactivé.")

    eligible = build_eligible_pixel_mask(
        y_raw,
        confidence=conf_arr,
        worldcover=wc_arr,
        conf_min=conf_min,
        require_cropland_worldcover=wc_arr is not None,
    )

    labels_hw, _ = remap_labels_cdl(y_raw, ignore=(0,))
    labels_hw = apply_mask_to_label_raster(labels_hw, eligible)

    n_elig = int(np.sum(eligible))
    n_lbl = int(np.sum(labels_hw >= 0))
    print(f"Pixels éligibles (masques) : {n_elig:,} | avec label CDL valide : {n_lbl:,}")
    if n_lbl == 0:
        raise RuntimeError(
            "Aucun pixel échantillonnable : vérifiez les chemins WorldCover / confiance / CDL."
        )

    print("Échantillonnage stratifié sur les classes CDL...")
    rows, cols = select_pixel_indices_stratified(labels_hw, max_samples=max_samples, seed=seed)

    with rasterio.open(cdl_path) as src:
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        print(f"Conversion {src.crs} → EPSG:4326...")
        lons, lats = transform(src.crs, "EPSG:4326", xs, ys)

    df = pd.DataFrame({"id": np.arange(len(lons), dtype=np.int64), "lon": lons, "lat": lats})
    out_csv = dataset_dir / "points_agricoles.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"OK — {len(df):,} points → {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export points_agricoles.csv (Little Rock / Arkansas, etc.)")
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Défaut : <repo>/dataset")
    ap.add_argument("--cdl", type=Path, default=None)
    ap.add_argument("--confidence", type=Path, default=None, help="Raster confiance CDL (optionnel)")
    ap.add_argument("--worldcover", type=Path, default=None, help="Raster WorldCover (optionnel)")
    ap.add_argument("--max-samples", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf-min", type=float, default=95.0)
    ap.add_argument("--skip-worldcover", action="store_true")
    ap.add_argument("--skip-confidence", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    ds = args.dataset_dir or (root / "dataset")
    cdl = args.cdl or (ds / Path(DEFAULT_CDL_PATH).name)
    conf = args.confidence or (ds / Path(DEFAULT_CDL_CONF_PATH).name)
    wc = args.worldcover or (ds / Path(DEFAULT_WORLDCOVER_PATH).name)

    export_coordinates(
        dataset_dir=ds,
        cdl_path=cdl,
        max_samples=args.max_samples,
        seed=args.seed,
        conf_path=conf,
        worldcover_path=wc,
        conf_min=args.conf_min,
        skip_worldcover=args.skip_worldcover,
        skip_confidence=args.skip_confidence,
    )


if __name__ == "__main__":
    main()
