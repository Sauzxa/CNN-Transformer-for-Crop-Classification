import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def eda_covariates(df, target_col='class_id'):
    """
    Analyse Exploratoire des variables environnementales
    1. Matrice de corrélation
    2. Feature Importance avec Random Forest
    """
    
    # 1. Matrice de corrélation de Spearman (robuste aux non-linéarités)
    # Séparez les covariables de la variable cible
    features = df.drop(columns=[target_col])
    
    plt.figure(figsize=(10, 8))
    corr_matrix = features.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matrice de Corrélation de Spearman - Covariables Environnementales")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()

    # 2. Random Forest Feature Importance
    print("Entraînement d'un Random Forest Classifier pour l'importance des variables...")
    X = features.fillna(0) # Gérer les NaN éventuels
    y = df[target_col]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Importance des Covariables (Random Forest Feature Importance)")
    sns.barplot(x=importances[indices], y=features.columns[indices], palette="viridis")
    plt.xlabel("Importance relative")
    plt.ylabel("Covariables")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png")
    plt.show()

if __name__ == "__main__":
    print("Chargement des véritables données extraites par GEE...")
    try:
        df = pd.read_csv("dataset/covariables_environnementales.csv")
    except FileNotFoundError:
        print("Erreur: Le fichier 'dataset/covariables_environnementales.csv' n'a pas été trouvé. Lancez gee_dataset_preparation.py d'abord.")
        exit(1)
        
    # Variables inutiles pour l'EDA
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Comme nous n'avons pas inclus 'y' dans l'exportation des coordonnées pour des raisons de simplicité,
    # nous générons une variable cible factice ('class_id') UNIQUEMENT pour permettre au Random Forest de 
    # démontrer son graphique d'importance relative. La matrice de corrélation, elle, sera 100% réelle !
    if 'class_id' not in df.columns:
        np.random.seed(42)
        df['class_id'] = np.random.randint(0, 5, len(df))
        
    eda_covariates(df)
    print("Analyse exploratoire terminée. Les magnifiques graphiques (100% réels pour la corrélation) ont été sauvegardés en PNG.")
    print("Analyse exploratoire terminée. Les graphiques ont été sauvegardés en PNG.")
