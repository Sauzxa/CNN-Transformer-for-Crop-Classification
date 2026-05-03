# California Partie 3

Cette version reproduit la logique de `Arkanssas/partie3` pour California.

## Fichiers principaux

- `multimodal_data.py`: chargement/alignement S2, S1, covariables statiques.
- `mctnet_v2_model.py`: architecture multimodale (branche S2, branche S1, fusion, statiques).
- `train_ablation.py`: ablations `S2_only`, `S2_static`, `S2_S1`, `S2_S1_static`.
- `gee_s1_preparation.py`: extraction GEE S1 mensuelle (VV/VH/ratio) avec fallback local.
- `dataset_audit.py`: audit dimensions + distribution des classes.

## Convention des données

- S2: `processed_data/X_train.npy`, `X_val.npy`, `X_test.npy`
- Masques: `processed_data/m_train.npy`, `m_val.npy`, `m_test.npy`
- Labels: `processed_data/y_train.npy`, `y_val.npy`, `y_test.npy`
- Covariables: `dataset/covariables_environnementales.csv`
- S1 mensuel: `partie3/s1_monthly_timeseries.csv`

## Exécution rapide

1. Générer/placer `s1_monthly_timeseries.csv` (via `gee_s1_preparation.py`).
2. Charger et aligner avec `load_aligned_multimodal(...)`.
3. Lancer les splits stratifiés et la normalisation.
4. Exécuter `run_ablation_suite(...)`.

