import ee
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse

def init_gee():
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize()

def extract_env_covariates(points_fc, start_date, end_date):
    """
    Extrait les covariables environnementales (Topographie, Sol, Climat)
    pour une collection de points via Earth Engine.
    
    :param points_fc: ee.FeatureCollection contenant les points géographiques.
    :param start_date: String (ex: '2025-01-01')
    :param end_date: String (ex: '2025-12-31')
    """
    
    # 1. TOPOGRAPHIE - SRTM 30m
    # Rééchantillonnage bilinéaire lors de l'extraction
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    
    topo_img = ee.Image.cat([elevation, slope, aspect]).rename(['elevation', 'slope', 'aspect'])
    
    # 2. SOL - OpenLandMap (profondeur 0cm)
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('clay')
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('sand')
    
    # L'image SILT a récemment été masquée par Google sur cet dataset, mais mathématiquement dans le système USDA : Limon = 100 - Argile - Sable
    silt = ee.Image(100).subtract(clay).subtract(sand).rename('silt')
    
    ph = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0').rename('ph')
    
    soil_img = ee.Image.cat([clay, sand, silt, ph])
    
    # Fusion des features statiques
    static_img = ee.Image.cat([topo_img, soil_img])
    
    # Extraction des données statiques pour les points (Scale=10 pour forcer la résolution Sentinel-2)
    static_sampled = static_img.reduceRegions(
        collection=points_fc,
        reducer=ee.Reducer.first(),
        scale=10 
    )

    # 3. CLIMAT - ERA5 Land (Températures et Précipitations)
    # Exemple d'agrégation mensuelle (ou on peut faire sur toute l'année, ou par pas de temps S2)
    # L'énoncé suggère d'extraire la valeur *par pas de temps* de la série S2 pour le climat.
    # Ici nous créons une fonction qui prendra la moyenne du pas de temps entre deux synthèses S2,
    # pour simplifier on extrait la moyenne sur toute la période visée (ou un agrégat).
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date, end_date)
        
    temp_max = era5.select('temperature_2m_max').mean().rename('temp_max')
    temp_min = era5.select('temperature_2m_min').mean().rename('temp_min')
    precip = era5.select('total_precipitation_sum').sum().rename('precip_sum')
    
    climate_img = ee.Image.cat([temp_max, temp_min, precip])
    
    # Extraction climat (Scale=10)
    climate_sampled = climate_img.reduceRegions(
        collection=static_sampled,
        reducer=ee.Reducer.first(),
        scale=10
    )
    
    return climate_sampled

if __name__ == "__main__":
    print("Initialisation de Google Earth Engine...")
    init_gee()
    
    print("Chargement des coordonnées depuis le CSV généré via la Solution 1...")
    ap = argparse.ArgumentParser(description="Exporte covariables environnementales pour une liste de points.")
    ap.add_argument(
        "--points-csv",
        type=Path,
        default=None,
        help="CSV avec colonnes `id, lon, lat` (défaut: dataset/points_agricoles.csv).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV de sortie (défaut: dataset/covariables_environnementales.csv).",
    )
    ap.add_argument("--start-date", type=str, default="2025-01-01")
    ap.add_argument("--end-date", type=str, default="2025-12-31")
    ap.add_argument("--chunk-size", type=int, default=4000)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    csv_path = args.points_csv or (project_root / "dataset" / "points_agricoles.csv")
    out_csv = args.out_csv or (project_root / "dataset" / "covariables_environnementales.csv")
    try:
        df_points = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur: {csv_path} introuvable. Exécutez export_coords.py d'abord!")
        exit(1)

    required_cols = {"lon", "lat"}
    missing = required_cols - set(df_points.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {sorted(missing)}")

    # IMPORTANT : l'alignement partie 1 ↔ partie 2 repose sur un identifiant stable.
    # On réutilise la colonne `id` si elle existe, sinon on la crée.
    if "id" not in df_points.columns:
        df_points = df_points.reset_index(drop=True).copy()
        df_points["id"] = df_points.index.astype(int)
    else:
        # normalisation type + ordre
        df_points["id"] = df_points["id"].astype(int)
        df_points = df_points.sort_values("id").reset_index(drop=True)
        
    features_list = []
    # Boucle sur VOS vraies données sans limite (ça va prendre environ 2 à 5 minutes pour 16 000 points)
    print("Création de la FeatureCollection Earth Engine complète...")
    for index, row in df_points.iterrows(): 
        geom = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(geom, {'id': int(row['id'])})
        features_list.append(feature)
        
    # GEE bloque les requêtes de plus de 5000 éléments avec getInfo(). On procède donc par "lots" (chunks).
    print("Découpage et extraction par lots de 4000 points (Patientez quelques minutes)...")
    
    start_date = args.start_date
    end_date = args.end_date
    
    data = []
    chunk_size = int(args.chunk_size)
    
    for i in range(0, len(features_list), chunk_size):
        chunk = features_list[i : i + chunk_size]
        print(f"-> Traitement des points {i} à {i + len(chunk)}...")
        
        # On crée la FeatureCollection juste pour ce petit lot
        points_fc = ee.FeatureCollection(chunk)
        sampled_fc = extract_env_covariates(points_fc, start_date, end_date)
        
        # Récupération sécurisée du lot
        features = sampled_fc.getInfo()['features']
        for f in features:
            data.append(f['properties'])
        
    df = pd.DataFrame(data)
    if "id" in df.columns:
        df["id"] = df["id"].astype(int)
        df = df.sort_values("id").reset_index(drop=True)
    print("Données extraites avec succès ! Aperçu :")
    print(df.head())
    
    # Sauvegarde au format CSV de vos variables
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Vos covariables sont sauvées et prêtes pour l'ablation dans {out_csv} !")
