from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample CDL raster on CSV points and create labels."
    )
    parser.add_argument("--csv", type=Path, default=BASE_DIR / "dataset/California_MCTNet_S2_360f_Mask36_10000_opt.csv")
    parser.add_argument("--cdl", type=Path, default=BASE_DIR / "dataset/2021_30m_cdls.zip")
    parser.add_argument("--out-csv", type=Path, default=BASE_DIR / "dataset/California_MCTNet_S2_360f_Mask36_10000_with_cdl.csv")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional {cdl_code: class_id}.")
    parser.add_argument("--unknown-label", type=int, default=-1)
    parser.add_argument("--drop-unknown", action="store_true")
    return parser.parse_args()


def _read_geo_point(raw: str) -> Tuple[float, float]:
    obj = json.loads(raw)
    if obj.get("type") != "Point":
        raise ValueError(f"Expected Point geometry, got {obj.get('type')}")
    coords = obj.get("coordinates", [])
    if len(coords) < 2:
        raise ValueError("Invalid .geo coordinates")
    return float(coords[0]), float(coords[1])  # lon, lat


def _iter_points(geo_col: Iterable[str]) -> List[Tuple[float, float]]:
    return [_read_geo_point(v) for v in geo_col]


def _resolve_cdl_tif_candidates(cdl_path: Path) -> List[str]:
    if not cdl_path.exists():
        raise FileNotFoundError(f"CDL path not found: {cdl_path}")
    if cdl_path.suffix.lower() in {".tif", ".tiff"}:
        return [str(cdl_path)]
    if cdl_path.suffix.lower() == ".zip":
        tif_name = None
        try:
            with zipfile.ZipFile(cdl_path) as zf:
                tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
                if tif_names:
                    tif_name = tif_names[0]
        except zipfile.BadZipFile:
            # Some archives are partially readable by GDAL (/vsizip) but rejected by
            # Python's zipfile central-directory checks.
            pass

        if tif_name is None:
            # Best-effort fallback: common naming convention <zip_stem>.tif
            tif_name = f"{cdl_path.stem}.tif"
        zip_posix = cdl_path.as_posix()
        zip_abs = str(cdl_path.resolve()).replace("\\", "/")
        return [
            f"zip://{zip_posix}!{tif_name}",
            f"zip://{zip_abs}!{tif_name}",
            f"/vsizip/{zip_posix}/{tif_name}",
            f"/vsizip//{zip_abs}/{tif_name}",
        ]
    raise ValueError(f"Unsupported CDL format: {cdl_path.suffix}")


def _load_mapping(path: Path | None) -> Dict[int, int] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in raw.items()}


def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    # First honor user's current working directory semantics.
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # Fallback to script-local paths (california/...)
    return (BASE_DIR / p).resolve()


def main() -> None:
    args = parse_args()
    args.csv = _resolve_path(args.csv)
    args.cdl = _resolve_path(args.cdl)
    args.out_csv = _resolve_path(args.out_csv)
    if args.mapping_json is not None:
        args.mapping_json = _resolve_path(args.mapping_json)

    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    try:
        import rasterio
        from rasterio.warp import transform as rio_transform
    except Exception as exc:
        raise RuntimeError(
            "rasterio is required. Install with: pip install rasterio"
        ) from exc

    df = pd.read_csv(args.csv)
    if ".geo" not in df.columns:
        raise ValueError("Input CSV must contain '.geo' column with GeoJSON Point.")

    points_wgs84 = _iter_points(df[".geo"].astype(str))
    xs = np.array([p[0] for p in points_wgs84], dtype=np.float64)
    ys = np.array([p[1] for p in points_wgs84], dtype=np.float64)

    cdl_candidates = _resolve_cdl_tif_candidates(args.cdl)
    last_err: Exception | None = None
    src = None
    cdl_uri = None
    for cand in cdl_candidates:
        try:
            src = rasterio.open(cand)
            cdl_uri = cand
            break
        except Exception as exc:
            last_err = exc
            continue

    if src is None or cdl_uri is None:
        tried = "\n".join(f"- {c}" for c in cdl_candidates)
        raise RuntimeError(
            "Unable to open CDL raster from provided source.\n"
            f"Tried:\n{tried}\n"
            "The archive may be corrupted. Re-download or extract the .tif manually and pass --cdl path/to/file.tif"
        ) from last_err

    with src:
        raster_crs = src.crs
        if raster_crs is None:
            raise ValueError("CDL raster has no CRS.")

        tx, ty = rio_transform("EPSG:4326", raster_crs, xs.tolist(), ys.tolist())
        sampled = list(src.sample(zip(tx, ty)))
        cdl_codes = np.array([int(v[0]) if len(v) else 0 for v in sampled], dtype=np.int64)

        nodata = src.nodata
        if nodata is not None:
            cdl_codes[cdl_codes == int(nodata)] = 0

    df_out = df.copy()
    df_out["cdl_code"] = cdl_codes

    mapping = _load_mapping(args.mapping_json)
    if mapping is None:
        # Default behavior: use raw CDL code as label.
        df_out["crop_label_cdl"] = df_out["cdl_code"].astype(np.int64)
    else:
        df_out["crop_label_cdl"] = df_out["cdl_code"].map(mapping).fillna(args.unknown_label).astype(np.int64)
        if args.drop_unknown:
            df_out = df_out[df_out["crop_label_cdl"] != args.unknown_label].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    report = {
        "input_csv": str(args.csv.resolve()),
        "input_cdl": str(args.cdl.resolve()),
        "output_csv": str(args.out_csv.resolve()),
        "n_rows_input": int(df.shape[0]),
        "n_rows_output": int(df_out.shape[0]),
        "n_unique_cdl_codes": int(df_out["cdl_code"].nunique()),
        "top_cdl_codes": {
            str(k): int(v)
            for k, v in df_out["cdl_code"].value_counts().head(15).to_dict().items()
        },
    }
    if mapping is not None:
        report["n_unique_labels"] = int(df_out["crop_label_cdl"].nunique())
        report["top_labels"] = {
            str(k): int(v)
            for k, v in df_out["crop_label_cdl"].value_counts().head(15).to_dict().items()
        }

    report_path = args.out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Done. Saved: {args.out_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
