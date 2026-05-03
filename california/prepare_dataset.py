"""
=============================================================================
POST-TRAITEMENT DES DONNÉES GEE — California Crop Classification
=============================================================================
Ce script prend les fichiers CSV exportés depuis Google Earth Engine
et les transforme en tenseurs PyTorch prêts pour l'entraînement.

PRÉREQUIS:
    pip install pandas numpy torch scikit-learn matplotlib seaborn rasterio
    
UTILISATION:
    1. Télécharge le fichier CSV depuis Google Drive
    2. Place-le dans le même dossier que ce script
    3. Lance: python prepare_dataset.py
=============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH      = os.path.join("dataset", "California_MCTNet_PAPERSTYLE_2021_chunk*_t*.csv")  # 4 chunks exportés depuis GEE
OUTPUT_DIR    = "./processed_data"
N_TIMESTAMPS  = 36  # article: 36 pas (10 jours)
SPECTRAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']  # article: 10 bandes
ALL_BANDS      = SPECTRAL_BANDS
N_CHANNELS     = len(ALL_BANDS)  # 10

# Labels California (paperstyle export in this repo).
# On garde les 6 classes, y compris Rice.
CROP_NAMES = {
    0: 'Grapes',
    1: 'Rice',
    2: 'Alfalfa',
    3: 'Almonds',
    4: 'Pistachios',
    5: 'Others',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

def load_gee_csv(csv_path: str) -> pd.DataFrame:
    """
    Charge un CSV GEE.
    Supporte un glob (ex: "California_MCTNet_PAPERSTYLE_2021*.csv") pour fusionner
    les exports chunkés sur la clé 'system:index'.
    """
    paths = sorted(glob.glob(csv_path))
    if not paths and os.path.exists(csv_path):
        paths = [csv_path]
    if not paths:
        raise FileNotFoundError(f"No CSV found for: {csv_path}")

    print(f"Chargement CSV: {len(paths)} fichier(s)")
    for p in paths:
        print(f"   - {p}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)

    # Extrait toutes les entités uniques et leurs propriétés depuis n'importe quel chunk
    all_props = []
    for df in dfs:
        # Extraire les colonnes de base s'elles existent
        keep = [c for c in ['system:index', 'crop_label', 'cdl_code', 'lon', 'lat', 'longitude', 'latitude', '.geo'] if c in df.columns]
        all_props.append(df[keep].copy())
    
    props_df = pd.concat(all_props).drop_duplicates(subset=['system:index'])
    
    # Outer Merge de tous les chunks depuis la base universelle
    df_merged = props_df
    for df in dfs:
        if 'system:index' not in df.columns:
            continue
        feat_cols = ['system:index'] + [c for c in df.columns if '_t' in c]
        df_merged = df_merged.merge(df[feat_cols], on='system:index', how='left')

    # Important: pour 'm_tX' (masque de nuages), si NaN (chunk manquant), on met 0 (donnée manquante)
    mt_cols = [c for c in df_merged.columns if c.startswith('m_t')]
    if mt_cols:
        df_merged[mt_cols] = df_merged[mt_cols].fillna(0)

    print(f"   Forme merged: {df_merged.shape}")
    print(f"   Colonnes trouvées: {len(df_merged.columns)}")

    # Suppression des colonnes GEE inutiles
    cols_to_drop = [c for c in df_merged.columns if c in ['.geo', 'system:time_start']]
    df_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Suppression des lignes avec LABEL manquant uniquement (on garde les NaN spectraux pour l'interpolation ensuite !)
    n_before = len(df_merged)
    if 'crop_label' in df_merged.columns:
        df_merged.dropna(subset=['crop_label'], inplace=True)
    n_after = len(df_merged)
    print(f"   Lignes supprimées (Label manquant): {n_before - n_after}")

    return df_merged


def extract_feature_matrix(df: pd.DataFrame):
    """
    Extrait le tenseur de features de forme (N, T, C).
    N = nombre de pixels
    T = timestamps (24)
    C = canaux spectraux (17)
    """
    N = len(df)
    X = np.zeros((N, N_TIMESTAMPS, N_CHANNELS), dtype=np.float32)
    
    print(f"\nConstruction du tenseur (N={N}, T={N_TIMESTAMPS}, C={N_CHANNELS})...")
    
    for t in range(N_TIMESTAMPS):
        for c_idx, band in enumerate(ALL_BANDS):
            col_name = f"{band}_t{t}"
            if col_name in df.columns:
                X[:, t, c_idx] = df[col_name].values.astype(np.float32)
            else:
                print(f"   [WARN] Colonne manquante: {col_name} — remplie avec 0")

    # Target column (paper-style export uses crop_label)
    if 'crop_label' in df.columns:
        y = df['crop_label'].values.astype(np.int64)
    elif 'label' in df.columns:
        y = df['label'].values.astype(np.int64)
    else:
        raise KeyError("Target column not found. Expected 'crop_label' (paper-style) or 'label'.")

    # Missing mask (m_t0..m_t35) if present
    mask_missing = None
    mt_cols = [f"m_t{t}" for t in range(N_TIMESTAMPS)]
    if all(c in df.columns for c in mt_cols):
        mask_missing = df[mt_cols].fillna(0).values.astype(np.int8)  # (N, T)
    
    # Coordonnées GPS si disponibles
    coords = None
    if '.geo' in df.columns or ('longitude' in df.columns and 'latitude' in df.columns):
        if 'longitude' in df.columns:
            coords = df[['longitude', 'latitude']].values

    # Often exported as lon/lat
    if coords is None and ('lon' in df.columns and 'lat' in df.columns):
        coords = df[['lon', 'lat']].values
    
    print(f"   OK Tenseur créé: X={X.shape}, y={y.shape}")
    print(f"   Classes présentes: {np.unique(y)}")
    
    return X, y, coords, mask_missing


def paper_split_indices(y: np.ndarray, seed: int = 42):
    """
    Split like Table 2: per class (California) → train=240, val=60, test=rest.
    Assumes total N ~ 10k and classes 0..5.
    """
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = min(240, len(idx))
        n_val = min(60, max(0, len(idx) - n_train))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


# =============================================================================
# 2. EXPLORATION DES DONNÉES
# =============================================================================

def explore_data(X: np.ndarray, y: np.ndarray, save_dir: str):
    """Génère les visualisations d'exploration."""
    print("\nExploration des données...")
    
    # --- 2.1 Distribution des classes ---
    fig, ax = plt.subplots(figsize=(12, 5))
    unique, counts = np.unique(y, return_counts=True)
    crop_labels = [CROP_NAMES.get(int(i), f'Class {i}') for i in unique]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    
    bars = ax.bar(crop_labels, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Type de culture', fontsize=12)
    ax.set_ylabel('Nombre de pixels', fontsize=12)
    ax.set_title('Distribution des classes de culture — California 2021', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   OK class_distribution.png sauvegardé")
    
    # --- 2.2 Séries temporelles NDVI par classe (si NDVI exporté) ---
    if 'NDVI' in ALL_BANDS:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        ndvi_idx = ALL_BANDS.index('NDVI')

        time_axis = np.arange(N_TIMESTAMPS) * (365 / N_TIMESTAMPS)  # Jours

        for i, (cls, name) in enumerate(CROP_NAMES.items()):
            if i >= len(axes):
                break
            mask = (y == cls)
            if mask.sum() == 0:
                axes[i].set_title(f'{name}\n(pas de données)')
                continue

            ndvi_cls = X[mask, :, ndvi_idx]  # (N_cls, T)
            mean_ndvi = ndvi_cls.mean(axis=0)
            std_ndvi  = ndvi_cls.std(axis=0)

            axes[i].plot(time_axis, mean_ndvi, color='green', linewidth=2, label='Moyenne')
            axes[i].fill_between(
                time_axis,
                mean_ndvi - std_ndvi,
                mean_ndvi + std_ndvi,
                alpha=0.3,
                color='green',
                label='±1 std',
            )
            axes[i].set_title(f'{name} (n={mask.sum()})', fontsize=10, fontweight='bold')
            axes[i].set_ylim(-0.1, 1.0)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Jour de l\'année')
            axes[i].set_ylabel('NDVI')

        plt.suptitle(
            'Séries temporelles NDVI par type de culture — California 2021',
            fontsize=14,
            fontweight='bold',
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ndvi_timeseries_by_class.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   OK ndvi_timeseries_by_class.png sauvegardé")
    else:
        print("   INFO NDVI non présent dans les features exportées -> plot NDVI ignoré")
    
    # --- 2.3 Matrice de corrélation des features (au timestamp 12 = été) ---
    T_summer = 12
    X_summer = X[:, T_summer, :]  # (N, C)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = pd.DataFrame(X_summer, columns=ALL_BANDS).corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Corrélation entre features spectrales (été, t=12)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   OK feature_correlation.png sauvegardé")
    
    # --- 2.4 Statistiques résumé ---
    stats = {
        'N_pixels_total': len(y),
        'N_timestamps': N_TIMESTAMPS,
        'N_channels': N_CHANNELS,
        'N_classes': len(np.unique(y)),
        'Classes': {CROP_NAMES.get(int(c), str(c)): int((y==c).sum()) for c in np.unique(y)},
        **(
            {
                'NDVI_mean': float(X[:, :, ALL_BANDS.index('NDVI')].mean()),
                'NDVI_std':  float(X[:, :, ALL_BANDS.index('NDVI')].std()),
            }
            if 'NDVI' in ALL_BANDS
            else {}
        ),
        'NaN_count': int(np.isnan(X).sum()),
    }
    
    print("\nStatistiques du dataset:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    return stats


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def normalize_features(X: np.ndarray, fit: bool = True, scaler=None):
    """
    Normalisation par canal (StandardScaler sur dimension temporelle).
    X shape: (N, T, C)
    """
    N, T, C = X.shape
    X_flat = X.reshape(-1, C)  # (N*T, C)
    
    if fit or scaler is None:
        scaler = StandardScaler()
        X_flat_norm = scaler.fit_transform(X_flat)
    else:
        X_flat_norm = scaler.transform(X_flat)
    
    X_norm = X_flat_norm.reshape(N, T, C)
    return X_norm.astype(np.float32), scaler


def handle_missing_values(X: np.ndarray) -> np.ndarray:
    """
    Interpolation linéaire sur l'axe temporel pour combler les NaN
    (pixels nuageux non masqués).
    """
    print("\nTraitement des valeurs manquantes...")
    nan_count = np.isnan(X).sum()
    
    if nan_count == 0:
        print("   OK Aucune valeur manquante")
        return X
    
    print(f"   {nan_count} NaN trouvés, interpolation en cours...")
    X_fixed = X.copy()
    
    N, T, C = X.shape
    for n in range(N):
        for c in range(C):
            series = X[n, :, c]
            nan_mask = np.isnan(series)
            if nan_mask.any() and not nan_mask.all():
                valid_idx = np.where(~nan_mask)[0]
                nan_idx   = np.where(nan_mask)[0]
                # Interpolation linéaire
                X_fixed[n, nan_idx, c] = np.interp(
                    nan_idx, valid_idx, series[valid_idx]
                )
            elif nan_mask.all():
                X_fixed[n, :, c] = 0.0  # Pixel entièrement nuageux → 0
    
    print(f"   OK NaN restants après interpolation: {np.isnan(X_fixed).sum()}")
    return X_fixed


def split_dataset(X: np.ndarray, y: np.ndarray, 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  random_state=42):
    """Split stratifié train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio),
        stratify=y, random_state=random_state
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size),
        stratify=y_temp, random_state=random_state
    )
    
    print(f"\nSplit du dataset:")
    print(f"   Train: {X_train.shape[0]} pixels")
    print(f"   Val:   {X_val.shape[0]} pixels")
    print(f"   Test:  {X_test.shape[0]} pixels")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 4. DATASET PYTORCH
# =============================================================================

class CropDataset(Dataset):
    """Dataset PyTorch pour la classification de cultures."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features de forme (N, T, C)
            y: Labels de forme (N,)
        """
        # Le modèle CNN-Transformer attend (N, C, T) — on transpose
        self.X = torch.from_numpy(X).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    @property
    def n_classes(self):
        return len(torch.unique(self.y))
    
    @property
    def n_channels(self):
        return self.X.shape[1]
    
    @property
    def seq_len(self):
        return self.X.shape[2]


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=64):
    """Crée les DataLoaders PyTorch."""
    
    train_ds = CropDataset(X_train, y_train)
    val_ds   = CropDataset(X_val,   y_val)
    test_ds  = CropDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\nDataLoaders créés:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    print(f"   Format tenseur: (batch, C={train_ds.n_channels}, T={train_ds.seq_len})")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# 5. PIPELINE PRINCIPALE
# =============================================================================

def main():
    print("=" * 60)
    print("  PIPELINE DE PRÉPARATION — California Crop Dataset")
    print("=" * 60)
    
    # --- Étape 1: Chargement ---
    paths = sorted(glob.glob(CSV_PATH))
    if not paths:
        print(f"\n[ERREUR] Aucun CSV trouvé pour: {CSV_PATH}")
        print("        Vérifie que les 4 chunks sont bien dans le dossier `dataset/`.")
        return
    
    df = load_gee_csv(CSV_PATH)
    
    # --- Étape 2: Extraction features ---
    X, y, coords, mask_missing = extract_feature_matrix(df)
    
    # --- Étape 3: Traitement NaN ---
    X = handle_missing_values(X)
    
    # --- Étape 4: Exploration ---
    explore_data(X, y, OUTPUT_DIR)
    
    # --- Étape 5: Normalisation ---
    X_norm, scaler = normalize_features(X, fit=True)
    
    # --- Étape 6: Split (article) ---
    train_idx, val_idx, test_idx = paper_split_indices(y, seed=42)
    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_val, y_val = X_norm[val_idx], y[val_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]

    print(f"\nSplit paper-style (Table 2):")
    print(f"   Train: {X_train.shape[0]} pixels")
    print(f"   Val:   {X_val.shape[0]} pixels")
    print(f"   Test:  {X_test.shape[0]} pixels")
    
    # --- Étape 7: Sauvegarde ---
    print("\nSauvegarde des données traitées...")
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),   X_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   y_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)

    # Sauvegarde du masque de valeurs manquantes si disponible (N,T)
    if mask_missing is not None:
        np.save(os.path.join(OUTPUT_DIR, 'm_train.npy'), mask_missing[train_idx])
        np.save(os.path.join(OUTPUT_DIR, 'm_val.npy'),   mask_missing[val_idx])
        np.save(os.path.join(OUTPUT_DIR, 'm_test.npy'),  mask_missing[test_idx])
    
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # --- Étape 8: Création DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64
    )
    
    # Vérification d'un batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nVérification batch:")
    print(f"   X_batch: {X_batch.shape}  (batch, channels, time)")
    print(f"   y_batch: {y_batch.shape}  (batch,)")
    print(f"   dtype X: {X_batch.dtype}")
    print(f"   dtype y: {y_batch.dtype}")
    
    print("\n" + "=" * 60)
    print("  OK PIPELINE TERMINÉ AVEC SUCCÈS!")
    print(f"  Données sauvegardées dans: {OUTPUT_DIR}/")
    print("  Visualisations générées")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    loaders = main()
