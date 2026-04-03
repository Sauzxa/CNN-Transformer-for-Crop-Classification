import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from mctnet_model import build_mctnet
from training_utils import sparse_categorical_focal_loss

import pandas as pd

def load_real_covariates(n_samples, n_time):
    """
    Charge le fichier CSV généré par l'Extraction GEE pour remplacer les dummies.
    """
    print("Chargement des covariables depuis Google Earth Engine...")
    try:
        df = pd.read_csv("dataset/covariables_environnementales.csv")
        # On s'assure de l'ordre exact correspondant à X_train.npy
        df = df.sort_values(by="id").reset_index(drop=True)
        
        # Statiques (7 variables): Topo (elevation, slope, aspect) + SOIL (clay, sand, silt, ph)
        static_cols = ['elevation', 'slope', 'aspect', 'clay', 'sand', 'silt', 'ph']
        X_static = df[static_cols].values # Forme (N, 7)
        
        # Temporel (Le climat ERA5, 3 variables)
        clim_cols = ['temp_max', 'temp_min', 'precip_sum']
        X_clim_static = df[clim_cols].values # Forme (N, 3)
        
        # Astuce Deep Learning : le climat évolue dans le temps normal, on duplique donc le vecteur sur les "n_time" pas de temps de l'image Sentinel 2 pour qu'il soit compatible avec l'architecture du Transformer
        X_clim = np.repeat(X_clim_static[:, np.newaxis, :], n_time, axis=1) # Forme (N, Time, 3)
        
        # On coupe s'il y a plus de covariables que d'images (cas des données factices)
        X_clim = X_clim[:n_samples]
        X_static = X_static[:n_samples]
        
        return X_clim, X_static
        
    except FileNotFoundError:
        print("Fichier covariables_environnementales.csv introuvable, veuillez lancer le script GEE.")
        exit(1)

def run_ablation_experiment():
    print("Chargement des données S2 originales...")
    try:
        X_train_s2 = np.load("dataset/X_train.npy")
        y_train = np.load("dataset/y_train.npy")
        # Limitation pour l'exemple
        X_train_s2 = X_train_s2[:500]
        y_train = y_train[:500]
    except FileNotFoundError:
        print("X_train.npy introuvable, création de variables dummy de test.")
        X_train_s2 = np.random.normal(size=(100, 12, 10))
        y_train = np.random.randint(0, 5, size=(100,))
        
    n_time = X_train_s2.shape[1]
    X_clim, X_static = load_real_covariates(len(X_train_s2), n_time)
    n_classes = len(np.unique(y_train))
    
    # 5 configurations prévues par l'étude d'ablation de l'énoncé
    configs = [
        {"name": "1. S2 original", "use_clim": False, "use_static": False},
        {"name": "2. S2 + Climat", "use_clim": True, "use_static": False},
        {"name": "3. S2 + Sol", "use_clim": False, "use_static": "soil"}, # Soil: colonnes 3 à 6 (4 vars)
        {"name": "4. S2 + Topographie", "use_clim": False, "use_static": "topo"}, # Topo: colonnes 0 à 2 (3 vars)
        {"name": "5. S2 + Fusion Totale", "use_clim": True, "use_static": "all"}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Lancement configuration: {config['name']} ---")
        
        n_channels = 10 # 10 Bandes S2 classiques
        X_temporal_input = X_train_s2.copy()
        
        # A. Injection du Climat en dimension Temporelle / Channels
        if config["use_clim"]:
            n_channels += X_clim.shape[-1]
            X_temporal_input = np.concatenate([X_temporal_input, X_clim], axis=-1)
            
        n_static_features = None
        X_static_input = None
        
        # B. Injection Sol/Topo en dimension Statique (secondaire)
        if config["use_static"] == "all":
            n_static_features = X_static.shape[-1]
            X_static_input = X_static
        elif config["use_static"] == "soil":
            n_static_features = 4
            X_static_input = X_static[:, 3:7] 
        elif config["use_static"] == "topo":
            n_static_features = 3
            X_static_input = X_static[:, 0:3]
            
        # Création du modèle adapté via le code modifié de MCTNet
        model = build_mctnet(
            n_timesteps=n_time,
            n_channels=n_channels,
            n_classes=n_classes,
            n_static_features=n_static_features,
            add_ndvi_if_missing=False
        )
        
        model.compile(
            optimizer="adam",
            loss=sparse_categorical_focal_loss(),
            metrics=["accuracy"]
        )
        
        # Adaptation des inputs selon le format de la configuration (1 ou 2 tenseurs)
        if n_static_features is not None:
            train_inputs = [X_temporal_input, X_static_input]
        else:
            train_inputs = [X_temporal_input]
            
        print(f"Shape Temporel: {X_temporal_input.shape}", end="")
        if n_static_features: print(f" / Shape Statique: {X_static_input.shape}")
        else: print(" / Shape Statique: N/A")
        
        print("Modèle compilé avec succès. (Remplacer ce commentaire par model.fit(...))")
        # En situation réelle:
        # history = model.fit(train_inputs, y_train, epochs=20, validation_split=0.2, verbose=0)
        # y_pred = np.argmax(model.predict(test_inputs), axis=-1)
        # oa = accuracy_score(y_test, y_pred)
        # kappa = cohen_kappa_score(y_test, y_pred)
        
        # Valeurs mock pour la démo
        results[config['name']] = {
            "OA": 0.85 + np.random.uniform(0, 0.05),
            "Kappa": 0.82 + np.random.uniform(0, 0.05),
            "F1": 0.83 + np.random.uniform(0, 0.05)
        }

    import pandas as pd
    res_df = pd.DataFrame(results).T
    print("\n=== RAPPEL DES RÉSULTATS D'ABLATION ===")
    print(res_df.round(4))

if __name__ == "__main__":
    run_ablation_experiment()
