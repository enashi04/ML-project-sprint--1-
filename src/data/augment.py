from narwhals import exclude
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("augment_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('augment')

def create_time_features(df):
    """
    Crée des caractéristiques temporelles à partir de la colonne timestamp.
    
    Args:
        df (DataFrame): DataFrame avec une colonne timestamp
        
    Returns:
        DataFrame: DataFrame avec des caractéristiques temporelles ajoutées
    """
    df = df.copy()
    
    # Extraire les composantes temporelles
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    
    # Créer des indicateurs pour jour/nuit et semaine/weekend
    df['is_night'] = (df['hour'] < 6) | (df['hour'] > 18)
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    return df

def create_rolling_features(df, window_sizes=[5, 10, 30], group_by='equipment_id'):
    """
    Crée des caractéristiques basées sur des fenêtres glissantes.
    
    Args:
        df (DataFrame): DataFrame avec des données de capteurs
        window_sizes (list): Liste des tailles de fenêtre à utiliser
        group_by (str): Colonne à utiliser pour le regroupement
        
    Returns:
        DataFrame: DataFrame avec des caractéristiques de fenêtre glissante ajoutées
    """
    df = df.copy().sort_values(by=[group_by, 'timestamp'])
    
    # Colonnes numériques pour les calculs de fenêtre glissante
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        return df
    
    # Pour chaque taille de fenêtre
    for window in window_sizes:
        # Pour chaque colonne numérique
        for col in numeric_cols:
            # Grouper par equipment_id et calculer les statistiques sur la fenêtre glissante
            df[f'{col}_rolling_mean_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_rolling_std_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).std())
            df[f'{col}_rolling_min_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'{col}_rolling_max_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).max())
    
    return df

def create_lag_features(df, lag_periods=[1, 3, 5, 10], group_by='equipment_id'):
    """
    Crée des caractéristiques de lag pour les variables numériques.
    
    Args:
        df (DataFrame): DataFrame avec des données de capteurs
        lag_periods (list): Liste des périodes de lag à utiliser
        group_by (str): Colonne à utiliser pour le regroupement
        
    Returns:
        DataFrame: DataFrame avec des caractéristiques de lag ajoutées
    """
    df = df.copy().sort_values(by=[group_by, 'timestamp'])
    
    # Colonnes numériques pour les calculs de lag
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        return df

    
    # Pour chaque période de lag
    for lag in lag_periods:
        # Pour chaque colonne numérique
        for col in numeric_cols:
            # Créer une colonne de lag
            df[f'{col}_lag_{lag}'] = df.groupby(group_by)[col].shift(lag)
            
            # Créer une colonne de différence avec le lag (taux de changement)
            df[f'{col}_change_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            df[f'{col}_pct_change_{lag}'] = df.groupby(group_by)[col].pct_change(periods=lag)
    
    return df

def add_failure_indicators(sensor_df, failure_df, time_window=24):
    """
    Ajoute des indicateurs de défaillance prochaine aux données de capteurs.
    
    Args:
        sensor_df (DataFrame): DataFrame avec des données de capteurs
        failure_df (DataFrame): DataFrame avec des données de défaillance
        time_window (int): Fenêtre de temps (en heures) avant une défaillance pour l'étiquetage
        
    Returns:
        DataFrame: DataFrame capteurs avec des indicateurs de défaillance ajoutés
    """
    sensor_df = sensor_df.copy()
    failure_df=failure_df.copy()
    
    # Initialiser les colonnes d'indicateur de défaillance
    sensor_df['failure_soon'] = 0
    sensor_df['time_to_failure'] = pd.NA
    sensor_df['next_failure_type'] = pd.NA
    
    # Pour chaque défaillance enregistrée
    for _, failure in failure_df.iterrows():
        equipment_id = failure['equipment_id']
        failure_time = failure['failure_timestamp']
        failure_type = failure['failure_type']
        
        # Trouver les enregistrements pour cet équipement dans la fenêtre de temps avant la défaillance
        window_mask = (
            (sensor_df['equipment_id'] == equipment_id) & 
            (sensor_df['timestamp'] <= failure_time) & 
            (sensor_df['timestamp'] >= failure_time - pd.Timedelta(hours=time_window))
        )
        
        if window_mask.any():
            #temps jusqu'à la défaillance
            ttf_hours = (
                (failure_time - sensor_df.loc[window_mask, 'timestamp'])
                .dt.total_seconds() / 3600
            )
            #maj si la ligne pas encore labellisé ou si la panne est plus proche
            current_ttf = sensor_df.loc[window_mask, 'time_to_failure']
            current_ttf_num = pd.to_numeric(current_ttf, errors='coerce')
            update_mask = current_ttf_num.isna() | (ttf_hours < current_ttf_num)

            if update_mask.any():
                idx_to_update=sensor_df.loc[window_mask].index[update_mask]

                # Marquer ces enregistrements comme précédant une défaillance
                sensor_df.loc[idx_to_update, 'failure_soon'] = 1

                  # Mettre le temps jusqu'à la défaillance
                sensor_df.loc[idx_to_update, 'time_to_failure'] = ttf_hours.loc[idx_to_update]

                # Mettre le type de défaillance suivant (le plus proche)
                sensor_df.loc[idx_to_update, 'next_failure_type'] = failure_type
    
    return sensor_df

def create_component_health_features(sensor_df, failure_df):
    """
    Crée des caractéristiques estimées de santé des composants en fonction de l'historique des défaillances.
    
    Args:
        sensor_df (DataFrame): DataFrame avec des données de capteurs
        failure_df (DataFrame): DataFrame avec des données de défaillance
        
    Returns:
        DataFrame: DataFrame avec des indicateurs de santé des composants ajoutés
    """
    sensor_df = sensor_df.copy()
    failure_df=failure_df.copy()
    
    # Initialiser colonnes de santé
    sensor_df['days_since_last_failure'] = np.inf
    sensor_df['failures_count_last_30days'] = 0
    
    # Pour chaque équipement unique
   # Pour chaque équipement
    for equipment_id in sensor_df['equipment_id'].unique():
        # Sous-ensembles capteurs / pannes pour cet équipement
        eq_mask = sensor_df['equipment_id'] == equipment_id
        eq_sensor = sensor_df.loc[eq_mask].sort_values('timestamp')

        eq_failures = failure_df[failure_df['equipment_id'] == equipment_id].sort_values('failure_timestamp')

        # Si pas de pannes, on met days_since_last_failure à inf (ou NA si tu préfères)
        if eq_failures.empty:
            sensor_df.loc[eq_sensor.index, 'days_since_last_failure'] = np.inf
            sensor_df.loc[eq_sensor.index, 'failures_count_last_30days'] = 0
            continue

        # --- 1) Days since last failure ---
        # Idée : pour chaque ligne capteur, trouver la dernière panne STRICTEMENT avant timestamp
        last_failure_times = []

        failure_times = eq_failures['failure_timestamp'].values

        for ts in eq_sensor['timestamp'].values:
            # pannes avant ts
            past_failures = failure_times[failure_times < ts]
            if len(past_failures) == 0:
                last_failure_times.append(pd.NaT)
            else:
                last_failure_times.append(past_failures[-1])

        last_failure_times = pd.to_datetime(last_failure_times)

        days_since = (pd.to_datetime(eq_sensor['timestamp']).reset_index(drop=True) - last_failure_times).dt.total_seconds() / 86400.0
        # Si aucune panne passée -> inf (ou NA)
        days_since = days_since.fillna(np.inf)

        sensor_df.loc[eq_sensor.index, 'days_since_last_failure'] = days_since.values

        # --- 2) Count of failures in last 30 days ---
        # Fenêtre [t - window_days, t) => pas le futur
        window = pd.Timedelta(days=30)

        counts = []
        for ts in eq_sensor['timestamp'].values:
            start = ts - window
            # compter les pannes entre start (inclus) et ts (exclus)
            c = ((eq_failures['failure_timestamp'] >= start) & (eq_failures['failure_timestamp'] < ts)).sum()
            counts.append(int(c))

        sensor_df.loc[eq_sensor.index, 'failures_count_last_30days'] = counts

    return sensor_df

def feature_scaling(df, method='standard', exclude_cols=None):
    """
    Applique une mise à l'échelle aux caractéristiques numériques.
    
    Args:
        df (DataFrame): DataFrame à mettre à l'échelle
        method (str): Méthode de mise à l'échelle ('standard' ou 'minmax')
        exclude_cols (list): Colonnes à exclure de la mise à l'échelle
        
    Returns:
        DataFrame: DataFrame avec les caractéristiques mises à l'échelle
    """
    df = df.copy()
    
    # Identifier les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    #Exclure les colonnes identifiées.
    exclude = {"equipment_id", "timestamp", "failure_soon", "time_to_failure", "next_failure_type"}
    if exclude_cols:
        exclude |= set(exclude_cols)

    numeric_cols = [c for c in numeric_cols if c not in exclude]
    logger.info(f"Colonnes numériques identifiées pour la mise à l'échelle (hors labels): {numeric_cols}")

    if not numeric_cols:
            return df
    
    # Appliquer la mise à l'échelle
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'standard' ou 'minmax'.")
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].replace((np.inf, -np.inf, np.nan), 0))
    
    return df

def create_interaction_features(df):
    """
    Crée des caractéristiques d'interaction entre les variables numériques.
    
    Args:
        df (DataFrame): DataFrame avec des variables numériques
        
    Returns:
        DataFrame: DataFrame avec des caractéristiques d'interaction ajoutées
    """
    df = df.copy()
    
    # Colonnes de base pour les interactions
    base_cols = ['temperature', 'vibration', 'pressure', 'current']
    
    # Créer les interactions multiplicatives
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # Créer les ratios (éviter les divisions par zéro)
    eps = 1e-6
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            denom_2 = df[col2].where(df[col2].abs() > eps, np.nan)
            denom_1 = df[col1].where(df[col1].abs() > eps, np.nan)

            df[f'{col1}_div_{col2}'] = df[col1] / denom_2
            df[f'{col2}_div_{col1}'] = df[col2] / denom_1

    return df

def plot_feature_importances(df, target_col='failure_soon', output_path=None):
    """
    Trace et sauvegarde les corrélations entre les caractéristiques et la cible.
    """
    # Travailler sur une copie
    df = df.copy()

    # Vérifier que la colonne cible existe
    if target_col not in df.columns:
        logger.warning(f"La colonne cible '{target_col}' est absente du DataFrame. Aucun plot généré.")
        return

    # Si la cible n'est pas numérique, essayer de la convertir
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isna().all():
            logger.warning(f"La colonne cible '{target_col}' n'a pas pu être convertie en numérique. Aucun plot généré.")
            return

    # Sélectionner uniquement les colonnes numériques pour le calcul de corrélation
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # S'assurer que la colonne cible figure bien dans numeric_df
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]

    # Calculer les corrélations avec la cible
    corr_with_target = numeric_df.corr()[target_col].sort_values(ascending=False)

    # Filtrer pour n'inclure que les plus importantes (top 20)
    top_corr = corr_with_target.drop(labels=[target_col], errors='ignore')[0:20]

    if top_corr.empty:
        logger.warning("Aucune corrélation numérique disponible à tracer.")
        return

    # Créer le graphique
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'Top 20 des caractéristiques corrélées avec {target_col}')
    plt.tight_layout()

    # Sauvegarder ou afficher
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def augment_data(input_dir='cleaned_data', output_dir='augmented_data'):
    """
    Enrichit les données nettoyées avec des caractéristiques supplémentaires.
    
    Args:
        input_dir (str): Répertoire contenant les données nettoyées
        output_dir (str): Répertoire pour les données enrichies
        
    Returns:
        DataFrame: DataFrame enrichi pour l'entraînement du modèle
    """
    try:
        # Création du répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Création d'un sous-répertoire pour les visualisations
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Chargement des données nettoyées
        sensor_data_path = os.path.join(input_dir, 'clean_sensor_data.parquet')
        failure_data_path = os.path.join(input_dir, 'clean_failure_data.parquet')
        
        logger.info(f"Chargement des données capteurs depuis {sensor_data_path}")
        sensor_df = pd.read_parquet(sensor_data_path)
        
        logger.info(f"Chargement des données de défaillance depuis {failure_data_path}")
        failure_df = pd.read_parquet(failure_data_path)
        
        # --- Augmentation des données ---
        
        # 1. Créer des caractéristiques temporelles
        logger.info("Création des caractéristiques temporelles")
        sensor_df = create_time_features(sensor_df)
        
        # 2. Créer des caractéristiques de fenêtre glissante
        logger.info("Création des caractéristiques de fenêtre glissante")
        sensor_df = create_rolling_features(sensor_df, window_sizes=[5, 10, 30])
        
        # 3. Créer des caractéristiques de lag
        logger.info("Création des caractéristiques de lag")
        sensor_df = create_lag_features(sensor_df, lag_periods=[1, 3, 5, 10])
        
        # 4. Ajouter des indicateurs de défaillance prochaine
        logger.info("Ajout des indicateurs de défaillance")
        sensor_df = add_failure_indicators(sensor_df, failure_df, time_window=24)
        
        # 5. Créer des caractéristiques de santé des composants
        logger.info("Création des caractéristiques de santé des composants")
        sensor_df = create_component_health_features(sensor_df, failure_df)
        
        # 6. Créer des caractères d'interaction
        logger.info("Création des caractéristiques d'interaction")
        sensor_df = create_interaction_features(sensor_df)

        # 7. Mise à l'échelle des caractéristiques
        logger.info("Application de la mise à l'échelle des caractéristiques")
        sensor_df = feature_scaling(sensor_df, method='standard', exclude_cols=['timestamp', 'equipment_id'])
        
        # 8. Tracer les importances des caractéristiques
        logger.info("Traçage des importances des caractéristiques")
        plot_feature_importances(sensor_df, target_col='failure_soon', output_path=os.path.join(viz_dir, 'feature_importances.png'))
        
        # Sauvegarde des données augmentées
        augmented_data_path = os.path.join(output_dir, 'augmented_sensor_data.parquet')
        logger.info(f"Sauvegarde des données enrichies sous {augmented_data_path}")
        sensor_df.to_parquet(augmented_data_path, index=False)
        
        logger.info("Augmentation des données terminée avec succès.")
        return sensor_df

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'augmentation des données: {e}")
        raise
        
if __name__ == "__main__":
    # Exécution de l'augmentation des données
    augmented_data = augment_data()

    # Affichage des informations de base sur les données nettoyées
    print("\nRésumé des données capteurs et de défaillance nettoyées:")
    print(augmented_data.describe())
    
    # print("\nRésumé des données de défaillance nettoyées:")
    # print(augmented_data.describe())