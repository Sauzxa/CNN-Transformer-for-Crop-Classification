import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def eda_covariates(df, target_col: str | None = None, out_dir: str | Path = "."):
    """
    Analyse Exploratoire des variables environnementales
    1. Matrice de corrélation
    2. Feature Importance avec Random Forest
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Matrice de corrélation de Spearman (robuste aux non-linéarités)
    # Séparez les covariables de la variable cible (si fournie)
    if target_col is not None and target_col in df.columns:
        features = df.drop(columns=[target_col])
    else:
        features = df.copy()
    
    plt.figure(figsize=(10, 8))
    corr_matrix = features.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matrice de Corrélation de Spearman - Covariables Environnementales")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png")
    plt.show()

    # 2. Random Forest Feature Importance
    if target_col is not None and target_col in df.columns:
        print("Entraînement d'un Random Forest Classifier pour l'importance des variables...")
        X = features.fillna(0)  # Gérer les NaN éventuels
        y = df[target_col]
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Importance des Covariables (Random Forest — prédiction de la classe)")
        sns.barplot(x=importances[indices], y=features.columns[indices], palette="viridis")
        plt.xlabel("Importance relative")
        plt.ylabel("Covariables")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance_rf.png")
        plt.show()
    else:
        print(
            "RF importance ignorée : aucune colonne cible réelle fournie. "
            "Pour une importance pertinente, joindre les vraies classes (CDL) alignées par `id` "
            "et appeler eda_covariates(..., target_col='class_id')."
        )

if __name__ == "__main__":
    print("Chargement des véritables données extraites par GEE...")
    try:
        project_root = Path(__file__).resolve().parent
        df = pd.read_csv(project_root / "dataset" / "covariables_environnementales.csv")
    except FileNotFoundError:
        print("Erreur: Le fichier 'dataset/covariables_environnementales.csv' n'a pas été trouvé. Lancez gee_dataset_preparation.py d'abord.")
        exit(1)
        
    # EDA corrélation toujours valide sur covariables réelles.
    # Pour l'importance RF, fournir une cible réelle alignée (ex: class_id) ; sinon on la saute.
    eda_covariates(df.drop(columns=["id"]) if "id" in df.columns else df, target_col="class_id" if "class_id" in df.columns else None, out_dir=project_root)
    print("Analyse exploratoire terminée. Les graphiques ont été sauvegardés.")
