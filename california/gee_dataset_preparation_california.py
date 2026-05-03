import ee
import pandas as pd
import glob
import os

def init_gee():
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize()

def extract_env_covariates(points_fc, start_date, end_date):
    # 1. TOPOGRAPHIE - SRTM 30m
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    topo_img = ee.Image.cat([elevation, slope, aspect]).rename(['elevation', 'slope', 'aspect'])
    
    # 2. SOL - OpenLandMap
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('clay')
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('sand')
    silt = ee.Image(100).subtract(clay).subtract(sand).rename('silt')
    ph = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0').rename('ph')
    soil_img = ee.Image.cat([clay, sand, silt, ph])
    
    static_img = ee.Image.cat([topo_img, soil_img])
    static_sampled = static_img.reduceRegions(
        collection=points_fc,
        reducer=ee.Reducer.first(),
        scale=10 
    )

    # 3. CLIMAT - ERA5 Land
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date, end_date)
        
    temp_max = era5.select('temperature_2m_max').mean().rename('temp_max')
    temp_min = era5.select('temperature_2m_min').mean().rename('temp_min')
    precip = era5.select('total_precipitation_sum').sum().rename('precip_sum')
    climate_img = ee.Image.cat([temp_max, temp_min, precip])
    
    climate_sampled = climate_img.reduceRegions(
        collection=static_sampled,
        reducer=ee.Reducer.first(),
        scale=10
    )
    
    return climate_sampled

if __name__ == "__main__":
    print("Initialisation de Google Earth Engine...")
    init_gee()
    
    print("Chargement des coordonnées depuis les CSV S2 existants...")
    # On recupere les points depuis les chunks de donnees Sentinel-2 deja exportes
    csv_pattern = os.path.join("dataset", "California_MCTNet_PAPERSTYLE_2021_chunk*.csv")
    paths = sorted(glob.glob(csv_pattern))
    if not paths:
        print(f"Erreur: Aucun fichier trouvé pour {csv_pattern}")
        exit(1)
        
    dfs = [pd.read_csv(p) for p in paths]
    df_merged = pd.concat([df[['system:index', 'lat', 'lon']] for df in dfs if 'lat' in df.columns and 'lon' in df.columns]).drop_duplicates(subset=['system:index'])
    
    print(f"Nombre de points uniques trouvés : {len(df_merged)}")
        
    features_list = []
    print("Création de la FeatureCollection Earth Engine complète...")
    for _, row in df_merged.iterrows(): 
        geom = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(geom, {'system:index': row['system:index']})
        features_list.append(feature)
        
    print("Découpage et extraction par lots de 4000 points (Patientez quelques minutes)...")
    
    start_date = '2021-01-01' # Année 2021 pour la Californie
    end_date = '2021-12-31'
    
    data = []
    chunk_size = 4000
    
    for i in range(0, len(features_list), chunk_size):
        chunk = features_list[i : i + chunk_size]
        print(f"-> Traitement des points {i} à {i + len(chunk)}...")
        
        points_fc = ee.FeatureCollection(chunk)
        sampled_fc = extract_env_covariates(points_fc, start_date, end_date)
        
        features = sampled_fc.getInfo()['features']
        for f in features:
            props = f['properties']
            # Conserver l'ID d'origine (system:index de GEE) pour pouvoir fusionner plus tard
            if 'system:index' in f['id']:
                props['system:index'] = f['id']
            data.append(props)
        
    df = pd.DataFrame(data)
    print("Données extraites avec succès ! Aperçu :")
    print(df.head())
    
    out_csv = "dataset/covariables_environnementales.csv"
    df.to_csv(out_csv, index=False)
    print(f"Vos covariables sont sauvées et prêtes pour l'ablation dans {out_csv} !")
