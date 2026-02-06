import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('clean')

def plot_distribution(df, column, output_path):
    """
    Crée un graphique de distribution pour identifier les anomalies
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution de {column}')
    plt.savefig(output_path)
    plt.close()

def detect_outliers(df, column, method='zscore', threshold=3):
    """
    Détecte les valeurs aberrantes dans une colonne.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        column (str): Nom de la colonne à vérifier
        method (str): Méthode de détection ('zscore' ou 'iqr')
        threshold (float): Seuil de détection (pour z-score uniquement)
        
    Returns:
        Series: Masque booléen indiquant les valeurs aberrantes
    """
    if method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    elif method == 'iqr':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'zscore' ou 'iqr'.")

def clean_data(input_dir='extracted_data', output_dir='cleaned_data'):
    """
    Nettoie les données extraites et les sauvegarde dans un nouveau format.
    
    Args:
        input_dir (str): Répertoire contenant les données extraites
        output_dir (str): Répertoire pour les données nettoyées
        
    Returns:
        tuple: (DataFrame capteurs nettoyé, DataFrame défaillances nettoyé)
    """
    try:
        # Création du répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Création d'un sous-répertoire pour les visualisations

        
        # Chargement des données extraites

        
        logger.info(f"Chargement des données capteurs depuis {sensor_data_path}")
        sensor_df = pd.read_parquet(sensor_data_path)
        
        logger.info(f"Chargement des données de défaillance depuis {failure_data_path}")
        failure_df = pd.read_parquet(failure_data_path)
        
        # --- Nettoyage des données capteurs ---
        
        # 1. Vérification des valeurs manquantes
        missing_values_sensor = sensor_df.isnull().sum()
        logger.info(f"Valeurs manquantes dans les données capteurs:\n{missing_values_sensor}")

        # Remplacer les valeurs infinies et NaN par 0

        
        # 2. Suppression des lignes avec valeurs manquantes (ou imputation selon la stratégie)

        logger.info(f"Lignes supprimées pour valeurs manquantes: {original_len - len(sensor_df)}")
        
        # 3. Vérification des doublons

        logger.info(f"Nombre de doublons dans les données capteurs: {duplicates}")
        sensor_df = sensor_df.drop_duplicates()
        
        # 4. Détection et traitement des valeurs aberrantes
        numeric_columns = ['temperature', 'vibration', 'pressure', 'current']
        
        for column in numeric_columns:
            # Visualiser la distribution
            plot_path = os.path.join(viz_dir, f'{column}_distribution.png')
            plot_distribution(sensor_df, column, plot_path)
            
            # Détecter les valeurs aberrantes
            outliers_mask = detect_outliers(sensor_df, column, method='iqr')
            outliers_count = outliers_mask.sum()
            logger.info(f"Valeurs aberrantes détectées dans {column}: {outliers_count}")
            
            # Pour les valeurs aberrantes, les remplacer par des NaN puis imputer
            if outliers_count > 0:
                # Option 1: Conserver les valeurs aberrantes avec un indicateur
                sensor_df[f'{column}_outlier'] = outliers_mask
                
                # Option 2: Remplacer par la médiane (à décommenter si préféré)
                # median_value = sensor_df[~outliers_mask][column].median()
                # sensor_df.loc[outliers_mask, column] = median_value
        
        # 5. Vérification de la cohérence des timestamps
        sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp'])
        
        # --- Nettoyage des données de défaillance ---
        
        # 1. Vérification des valeurs manquantes

        logger.info(f"Valeurs manquantes dans les données de défaillance:\n{missing_values_failure}")
        
        # 2. Imputation des valeurs manquantes (si applicable)
        # Pour repair_duration et repair_cost, utiliser la médiane par type de défaillance
        for column in ['repair_duration', 'repair_cost']:
            if failure_df[column].isnull().sum() > 0:
                # Calculer les médianes par failure_type
                medians = failure_df.groupby('failure_type')[column].median()
                
                # Appliquer les médianes correspondantes
                for failure_type in failure_df['failure_type'].unique():
                    mask = (failure_df['failure_type'] == failure_type) & (failure_df[column].isnull())
                    failure_df.loc[mask, column] = medians[failure_type]
                
                # Pour les types de défaillance sans valeur, utiliser la médiane globale
                failure_df[column] = failure_df[column].fillna(failure_df[column].median())
        
        # 3. Vérification des doublons
        
        logger.info(f"Nombre de doublons dans les données de défaillance: {duplicates}")
        
        # 4. Vérification de la cohérence temporelle
        # S'assurer que les défaillances sont pour des équipements existants
        valid_equipment_ids = sensor_df['equipment_id'].unique()
        invalid_ids = failure_df[~failure_df['equipment_id'].isin(valid_equipment_ids)]
        
        if len(invalid_ids) > 0:
            logger.warning(f"Défaillances pour des équipements inexistants: {len(invalid_ids)}")
            failure_df = failure_df[failure_df['equipment_id'].isin(valid_equipment_ids)]
        
        # Sauvegarde des données nettoyées

        
        # Également sauvegarder en CSV pour faciliter l'inspection
        sensor_df.to_csv(os.path.join(output_dir, 'clean_sensor_data.csv'), index=False)
        failure_df.to_csv(os.path.join(output_dir, 'clean_failure_data.csv'), index=False)
        
        logger.info(f"Données nettoyées sauvegardées dans {output_dir}")
        
        # Créer un rapport de nettoyage
        cleaning_report = {
            "date_nettoyage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_enregistrements_capteurs_initial": original_len,
            "nb_enregistrements_capteurs_final": len(sensor_df),
            "nb_enregistrements_defaillances_initial": len(failure_df) + len(invalid_ids),
            "nb_enregistrements_defaillances_final": len(failure_df),
            "nb_valeurs_aberrantes_detectees": {col: detect_outliers(sensor_df, col).sum() for col in numeric_columns}
        }
        
        pd.DataFrame([cleaning_report]).to_csv(os.path.join(output_dir, 'cleaning_report.csv'), index=False)
        
        return sensor_df, failure_df
    
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        raise

if __name__ == "__main__":
    # Exécution du nettoyage des données
    clean_sensor_df, clean_failure_df = clean_data()
    
    # Affichage des informations de base sur les données nettoyées
    print("\nRésumé des données capteurs nettoyées:")
    print(clean_sensor_df.describe())
    
    print("\nRésumé des données de défaillance nettoyées:")
    print(clean_failure_df.describe())