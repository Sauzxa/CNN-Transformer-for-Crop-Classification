import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# --- CONFIGURATION ---
CSV_PATH      = os.path.join("dataset", "California_MCTNet_PAPERSTYLE_2021_chunk*_t*.csv")
ENV_CSV_PATH  = os.path.join("dataset", "covariables_environnementales.csv")
OUTPUT_DIR    = "./processed_data"
N_TIMESTAMPS  = 36
SPECTRAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
N_CHANNELS     = len(SPECTRAL_BANDS)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    # 1. Spectral Data
    paths = sorted(glob.glob(CSV_PATH))
    dfs = [pd.read_csv(p) for p in paths]
    
    all_props = []
    for df in dfs:
        keep = [c for c in ['system:index', 'crop_label', 'cdl_code', 'lon', 'lat'] if c in df.columns]
        all_props.append(df[keep].copy())
    
    df_merged = pd.concat(all_props).drop_duplicates(subset=['system:index'])
    
    for df in dfs:
        feat_cols = ['system:index'] + [c for c in df.columns if '_t' in c]
        df_merged = df_merged.merge(df[feat_cols], on='system:index', how='left')

    # 2. Environmental Data
    if os.path.exists(ENV_CSV_PATH):
        print(f"Loading environmental data from {ENV_CSV_PATH}")
        df_env = pd.read_csv(ENV_CSV_PATH)
        # On suppose l'ordre des lignes identique après le merge de df_merged
        # Pour être sûr, on concatène les colonnes de df_env à df_merged
        # Note: df_merged a la même taille que df_env (9996)
        if len(df_env) == len(df_merged):
            df_merged = pd.concat([df_merged.reset_index(drop=True), df_env.reset_index(drop=True)], axis=1)
        else:
            print(f"Warning: Environmental data size ({len(df_env)}) mismatch with S2 data ({len(df_merged)})")
    
    return df_merged

def extract_tensors(df):
    N = len(df)
    X = np.zeros((N, N_TIMESTAMPS, N_CHANNELS), dtype=np.float32)
    for t in range(N_TIMESTAMPS):
        for c, band in enumerate(SPECTRAL_BANDS):
            col = f"{band}_t{t}"
            if col in df.columns:
                X[:, t, c] = df[col].values
    
    y = df['crop_label'].values.astype(np.int32)
    
    # Static features
    static_cols = ['aspect', 'clay', 'elevation', 'ph', 'precip_sum', 'sand', 'silt', 'slope', 'temp_max', 'temp_min']
    X_static = df[static_cols].values.astype(np.float32)
    
    # Mask
    mt_cols = [f"m_t{t}" for t in range(N_TIMESTAMPS)]
    M = df[mt_cols].fillna(0).values.astype(np.float32)
    M = (M > 0.5).astype(np.float32)
    
    return X, X_static, M, y

def paper_split_indices(y, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = min(240, len(idx))
        n_val = min(60, max(0, len(idx) - n_train))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)

def main():
    df = load_data()
    X, X_static, M, y = extract_tensors(df)
    
    # Handle NaNs in X
    for n in range(X.shape[0]):
        for c in range(X.shape[2]):
            s = X[n, :, c]
            nan_mask = np.isnan(s)
            if nan_mask.any() and not nan_mask.all():
                v = np.where(~nan_mask)[0]
                m = np.where(nan_mask)[0]
                X[n, m, c] = np.interp(m, v, s[v])
            elif nan_mask.all():
                X[n, :, c] = 0
    X = np.nan_to_num(X)

    # Normalization
    # S2
    scaler_s2 = StandardScaler()
    X_shape = X.shape
    X = scaler_s2.fit_transform(X.reshape(-1, N_CHANNELS)).reshape(X_shape)
    # Static
    scaler_static = StandardScaler()
    X_static = scaler_static.fit_transform(X_static)
    
    # Split
    train_idx, val_idx, test_idx = paper_split_indices(y, seed=42)
    
    # Save
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        np.save(os.path.join(OUTPUT_DIR, f'X_{name}.npy'), X[idx])
        np.save(os.path.join(OUTPUT_DIR, f'X_static_{name}.npy'), X_static[idx])
        np.save(os.path.join(OUTPUT_DIR, f'm_{name}.npy'), M[idx])
        np.save(os.path.join(OUTPUT_DIR, f'y_{name}.npy'), y[idx])
    
    with open(os.path.join(OUTPUT_DIR, 'scaler_s2.pkl'), 'wb') as f:
        pickle.dump(scaler_s2, f)
    with open(os.path.join(OUTPUT_DIR, 'scaler_static.pkl'), 'wb') as f:
        pickle.dump(scaler_static, f)
        
    print("Multi-modal data preparation complete.")

if __name__ == "__main__":
    main()
