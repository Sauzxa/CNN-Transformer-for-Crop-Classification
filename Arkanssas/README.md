# MCTNet — Crop Classification from Sentinel-2 Time Series

Ce projet réalise la classification de cultures sur l'Arkansas à partir de séries temporelles Sentinel-2, avec des labels CDL et une architecture MCTNet (hybride CNN-Transformer).

## Dataset

- Zone: Arkansas (Area1 + Area2)
- Sources:
  - Sentinel-2 (séries temporelles multi-bandes)
  - CDL (Cropland Data Layer) pour les labels
- Classes finales (5):
  - Corn
  - Cotton
  - Soybean
  - Rice
  - Others

## Modèle

- Architecture: `MCTNet` (CNN-Transformer hybride)
- Implémentation principale: `mctnet_model.py`
- Entraînement et utilitaires: `training_utils.py`
- Notebook principal: `notebooks/Classification de cultures (1) (1).ipynb`

## Résultats

| Expérience | OA (Accuracy) | Kappa | F1 Macro |
|---|---:|---:|---:|
| Part 1 baseline (MCTNet) | 91.82% | 0.8977 | 91.83% |
| Part 2 ablation — S2 | 91.0% | - | - |
| Part 2 ablation — S2 + Climat | 90.2% | - | - |
| Part 2 ablation — S2 + Sol | ?% | - | - |
| Part 2 ablation — S2 + Topographie | 90.3% | - | - |
| Part 2 ablation — S2 + Fusion Totale | ?% | - | - |

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

```bash
jupyter notebook
```

Puis ouvrir le notebook `notebooks/Classification de cultures (1) (1).ipynb`.
