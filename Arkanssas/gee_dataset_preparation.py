import ee
import pandas as pd
from datetime import datetime, timedelta

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
    csv_path = "dataset/points_agricoles.csv"
    try:
        df_points = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur: {csv_path} introuvable. Exécutez export_coords.py d'abord!")
        exit(1)
        
    features_list = []
    # Boucle sur VOS vraies données sans limite (ça va prendre environ 2 à 5 minutes pour 16 000 points)
    print("Création de la FeatureCollection Earth Engine complète...")
    for index, row in df_points.iterrows(): 
        geom = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(geom, {'id': index})
        features_list.append(feature)
        
    # GEE bloque les requêtes de plus de 5000 éléments avec getInfo(). On procède donc par "lots" (chunks).
    print("Découpage et extraction par lots de 4000 points (Patientez quelques minutes)...")
    
    start_date = '2025-01-01'
    end_date = '2025-12-31'
    
    data = []
    chunk_size = 4000
    
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
    print("Données extraites avec succès ! Aperçu :")
    print(df.head())
    
    # Sauvegarde au format CSV de vos variables
    out_csv = "dataset/covariables_environnementales.csv"
    df.to_csv(out_csv, index=False)
    print(f"Vos covariables sont sauvées et prêtes pour l'ablation dans {out_csv} !")
