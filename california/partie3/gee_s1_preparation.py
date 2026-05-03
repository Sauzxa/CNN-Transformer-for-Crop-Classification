from __future__ import annotations

import glob
from pathlib import Path

import ee
import numpy as np
import pandas as pd


def init_gee() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize()


def load_points(project_root: Path) -> pd.DataFrame:
    points_path = project_root / "dataset" / "points_agricoles.csv"
    if points_path.exists():
        df = pd.read_csv(points_path)
    else:
        pattern = str(project_root / "dataset" / "California_MCTNet_PAPERSTYLE_2021_chunk*.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError("No points source found (points_agricoles.csv or chunk CSVs).")
        dfs = [pd.read_csv(p) for p in paths]
        cols = [c for c in ["system:index", "lon", "lat"] if c in dfs[0].columns]
        df = pd.concat([d[cols] for d in dfs], axis=0).drop_duplicates(subset=["system:index"])
    if "id" not in df.columns:
        if "system:index" in df.columns:
            parsed = pd.to_numeric(df["system:index"], errors="coerce")
            # If system:index is non-numeric (common for GEE exports), use row index ids.
            if parsed.notna().sum() < max(1, int(0.5 * len(parsed))):
                df["id"] = np.arange(len(df), dtype=np.int64)
            else:
                df["id"] = parsed
        else:
            df["id"] = np.arange(len(df), dtype=np.int64)
    df = df.dropna(subset=["lon", "lat"]).copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if df["id"].isna().any():
        missing = df["id"].isna()
        df.loc[missing, "id"] = np.arange(len(df), dtype=np.int64)[missing.to_numpy()]
    df["id"] = df["id"].astype(np.int64)
    return df


def extract_s1_monthly_averages(points_df: pd.DataFrame, *, year: int = 2021, orbit: str | None = None) -> pd.DataFrame:
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterMetadata("instrumentMode", "equals", "IW")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )
    if orbit in ("ASCENDING", "DESCENDING"):
        s1_col = s1_col.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    if s1_col.size().getInfo() == 0:
        raise RuntimeError(f"No Sentinel-1 images found for year {year}.")

    monthly_images = []
    for m in range(1, 13):
        m_col = s1_col.filter(ee.Filter.calendarRange(m, m, "month"))
        m_img = ee.Image(ee.Algorithms.If(m_col.size().gt(0), m_col.mean(), s1_col.mean()))
        vv = m_img.select("VV").rename(f"vv_{m:02d}")
        vh = m_img.select("VH").rename(f"vh_{m:02d}")
        ratio = vh.subtract(vv).rename(f"ratio_{m:02d}")
        monthly_images.append(ee.Image.cat([vv, vh, ratio]))
    all_months_img = ee.Image.cat(monthly_images)

    features = [
        ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]), {"id": int(row["id"])})
        for _, row in points_df.iterrows()
    ]

    data: list[dict] = []
    chunk_size = 1500
    for i in range(0, len(features), chunk_size):
        chunk = features[i : i + chunk_size]
        sampled_fc = all_months_img.reduceRegions(
            collection=ee.FeatureCollection(chunk),
            reducer=ee.Reducer.first(),
            scale=10,
        )
        rows = sampled_fc.getInfo().get("features", [])
        for f in rows:
            data.append(f.get("properties", {}))
    out = pd.DataFrame(data).fillna(0)
    return out.sort_values("id").reset_index(drop=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    init_gee()
    df_points = load_points(root)
    s1_data = extract_s1_monthly_averages(df_points, year=2021)
    out_path = root / "partie3" / "s1_monthly_timeseries.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s1_data.to_csv(out_path, index=False)
    print(f"S1 monthly time-series saved to {out_path}")

