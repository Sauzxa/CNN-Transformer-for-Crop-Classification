# 📖 GUIDE D'UTILISATION — Extraction Données California

## Vue d'ensemble du workflow

```
Google Earth Engine (GEE)          Google Drive          Ton PC
       │                                │                    │
  [GEE_California_Sentinel2.js]         │                    │
       │ ──── Export CSV ────────────►  │                    │
       │ ──── Export GeoTIFF ────────►  │                    │
                                        │ ── Télécharge ──►  │
                                        │    (fichier léger) │
                                                        [prepare_dataset.py]
                                                             │
                                                      processed_data/
                                                      ├── X_train.npy
                                                      ├── X_val.npy
                                                      ├── X_test.npy
                                                      ├── y_train.npy
                                                      ├── y_val.npy
                                                      ├── y_test.npy
                                                      └── scaler.pkl
```

---

## ÉTAPE 1 — Créer un compte Google Earth Engine

1. Va sur: https://earthengine.google.com/
2. Clique **"Get Started"**
3. Connecte-toi avec ton compte Google
4. Demande l'accès (approbation instantanée pour étudiants)
5. Va sur: https://code.earthengine.google.com/

---

## ÉTAPE 2 — Lancer le script GEE

1. Ouvre https://code.earthengine.google.com/
2. **Copie-colle** le contenu de `GEE_California_Sentinel2.js` dans l'éditeur
3. Clique **"Run"** (le bouton en haut)
4. La carte va s'afficher avec les couches NDVI et CDL
5. Va dans l'onglet **"Tasks"** (panneau de droite)
6. Tu verras 3 exports en attente — clique **"RUN"** sur chacun:
   - `California_S2_samples_2020` → **CSV** (~50-100 MB)
   - `California_CDL_2020` → GeoTIFF des labels
   - `California_NDVI_timeseries_2020` → GeoTIFF NDVI

⏱️ **Temps estimé**: 10-30 minutes selon la zone

---

## ÉTAPE 3 — Télécharger depuis Google Drive

1. Va sur https://drive.google.com/
2. Ouvre le dossier **"Crop_Classification"**
3. Télécharge **seulement le CSV** (`california_s2_samples_2020.csv`)
   - C'est le fichier le plus léger (~50-100 MB)
   - Les GeoTIFF sont optionnels (pour visualisation)

---

## ÉTAPE 4 — Installer les dépendances Python

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn rasterio
```

---

## ÉTAPE 5 — Lancer le preprocessing

1. Place `california_s2_samples_2020.csv` dans le même dossier que `prepare_dataset.py`
2. Lance:
```bash
python prepare_dataset.py
```

Le script va:
- ✅ Charger et valider le CSV
- ✅ Construire le tenseur (N, T=24, C=17)
- ✅ Interpoler les valeurs manquantes (nuages)
- ✅ Normaliser les features
- ✅ Splitter en train/val/test (70/15/15)
- ✅ Générer des visualisations
- ✅ Sauvegarder les `.npy` prêts pour PyTorch

---

## ÉTAPE 6 — Utiliser les données dans ton modèle

```python
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch

# Chargement
X_train = np.load('processed_data/X_train.npy')
y_train = np.load('processed_data/y_train.npy')
X_val   = np.load('processed_data/X_val.npy')
y_val   = np.load('processed_data/y_val.npy')
X_test  = np.load('processed_data/X_test.npy')
y_test  = np.load('processed_data/y_test.npy')

# Format: (N, T=24, C=17) → modèle attend (N, C=17, T=24)
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
y_train_t = torch.LongTensor(y_train)

print(f"X_train: {X_train_t.shape}")  # (N, 17, 24)
print(f"y_train: {y_train_t.shape}")  # (N,)

# Prêt pour le CNN-Transformer!
```

---

## ⚠️ Résolution des problèmes courants

| Problème | Solution |
|----------|----------|
| "Not registered for Earth Engine" | Attendre email d'approbation (~1h) |
| Export bloqué dans Tasks | Cliquer "Retry" ou relancer le script |
| CSV trop grand | Réduire `SAMPLES_PER_CLASS` à 200 dans le script GEE |
| Zone trop grande | Réduire `AOI` dans le script GEE |
| NaN dans les données | Normal — le script Python les interpole automatiquement |

---

## 📊 Taille estimée des fichiers

| Fichier | Taille estimée |
|---------|---------------|
| CSV (500 pixels/classe × 12 classes) | ~80 MB |
| X_train.npy | ~15 MB |
| X_val.npy | ~3 MB |
| X_test.npy | ~3 MB |
| CDL GeoTIFF | ~30 MB |
| NDVI GeoTIFF (résolution 30m) | ~50 MB |

---

## 🔧 Paramètres à ajuster selon ta connexion

Dans `GEE_California_Sentinel2.js`, ligne ~25:

```javascript
// Si connexion lente: réduire ces valeurs
var SAMPLES_PER_CLASS = 200;  // Défaut: 500
var N_TIMESTAMPS = 12;        // Défaut: 24 (toutes les 4 semaines)

// Réduire la zone d'étude
var AOI = ee.Geometry.Rectangle([-121.5, 37.0, -120.0, 38.0]); // Plus petite
```
