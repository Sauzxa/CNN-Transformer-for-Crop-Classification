"""
Construit dataset/X_train.npy et dataset/y_train.npy à partir de dataset/safes.txt.

- Bandes « article » (10 canaux Sentinel-2 L2A).
- Option composites décadaires : (N, 36, 10) + masque temporel, année DATA_YEAR (2025).

Prérequis : CDL clip + optionnel confiance + WorldCover (voir export_coords.py).
Exécution depuis la racine du projet :

  python build_s2_training_arrays.py --project-root .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Racine projet sur sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from partie3.paper_dataset import (  # noqa: E402
    DATA_YEAR,
    load_cdl_labels,
    remap_labels_cdl,
)
from partie3.s2_decadal_composites import (  # noqa: E402
    decadal_median_composite,
    parse_safe_acquisition_date,
)
from partie3.s2_pixel_streaming import S2_BANDS_PAPER, build_pixel_dataset_streaming  # noqa: E402


def _load_safe_paths(safes_file: Path) -> list[Path]:
    lines = [
        ln.strip()
        for ln in safes_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    paths = [Path(p) for p in lines]
    paths.sort(key=parse_safe_acquisition_date)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Construit X_train.npy / y_train.npy (S2 L2A, style article).")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--safes-file", type=Path, default=None, help="Défaut: <root>/dataset/safes.txt")
    ap.add_argument("--cdl", type=Path, default=None, help="Défaut: <root>/dataset/cdl_2025_clip.tif")
    ap.add_argument("--max-samples", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-decadal",
        action="store_true",
        help="Ne pas agréger en 36 pas (garde T = nombre de SAFE).",
    )
    ap.add_argument("--n-bins", type=int, default=36)
    ap.add_argument("--bin-days", type=int, default=10)
    args = ap.parse_args()

    root = args.project_root.resolve()
    safes_file = args.safes_file or (root / "dataset" / "safes.txt")
    cdl_path = args.cdl or (root / "dataset" / "cdl_2025_clip.tif")

    if not safes_file.is_file():
        raise FileNotFoundError(f"safes.txt introuvable : {safes_file}")
    if not cdl_path.is_file():
        raise FileNotFoundError(f"CDL introuvable : {cdl_path}")

    y_raw, _ = load_cdl_labels(cdl_path)
    labels_hw, rev = remap_labels_cdl(y_raw, ignore=(0,))

    safe_paths = _load_safe_paths(safes_file)
    if not safe_paths:
        raise ValueError("Aucun chemin .SAFE dans safes.txt")
    dates = [parse_safe_acquisition_date(p) for p in safe_paths]

    print(f"SAFE : {len(safe_paths)} acquisitions, année cible {DATA_YEAR}")
    print(f"Première / dernière date : {dates[0]} … {dates[-1]}")

    X, y_idx, mask_t = build_pixel_dataset_streaming(
        safe_paths,
        labels_hw,
        max_samples=args.max_samples,
        seed=args.seed,
        band_names=S2_BANDS_PAPER,
        resolution="10m",
        stratified=True,
    )
    y_cdl = np.array([rev[int(i)] for i in y_idx], dtype=np.int32)

    if not args.no_decadal:
        X, mask_bins = decadal_median_composite(
            X,
            dates,
            year=DATA_YEAR,
            n_bins=args.n_bins,
            bin_days=args.bin_days,
        )
        print(f"Composites décadaires : X {X.shape}, masque bins {mask_bins.shape}")
    else:
        print(f"Série brute : X {X.shape}")

    out_x = root / "dataset" / "X_train.npy"
    out_y = root / "dataset" / "y_train.npy"
    out_x.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_x, X.astype(np.float32))
    np.save(out_y, y_cdl.astype(np.int32))
    print(f"Enregistré : {out_x}")
    print(f"Enregistré : {out_y}")


if __name__ == "__main__":
    main()
