import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("build_features_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('build_features')

def create_polynomial_features(df, degree=2):
    """
    Crée des caractéristiques polynomiales pour capturer les relations non linéaires.
    
    Args:
        df (DataFrame): DataFrame avec les données de capteurs
        degree (int): Degré du polynôme à générer
        
    Returns:
        DataFrame: DataFrame avec les caractéristiques polynomiales ajoutées
    """
    df = df.copy()
    
    # Colonnes numériques de base pour les polynômes
    base_cols = ['temperature', 'vibration', 'pressure', 'current']
    
    # Pour chaque colonne, créer les puissances jusqu'au degré spécifié
    for col in base_cols:

    
    logger.info(f"Caractéristiques polynomiales de degré {degree} créées")
    return df

def create_cycle_features(df, equipment_ids=None):
    """
    Crée des caractéristiques basées sur les cycles d'opération des équipements.
    
    Args:
        df (DataFrame): DataFrame avec les données de capteurs
        equipment_ids (list): Liste des IDs d'équipement à traiter (None = tous)
        
    Returns:
        DataFrame: DataFrame avec les caractéristiques de cycle ajoutées
    """
    df = df.copy()
    
    # Si aucun equipment_id n'est spécifié, utiliser tous les équipements
    if equipment_ids is None:
        equipment_ids = df['equipment_id'].unique()
    
    # Initialiser les colonnes de cycle
    df['cycle_id'] = np.nan
    df['cycle_phase'] = np.nan
    df['time_in_cycle'] = np.nan
    df['cycle_duration'] = np.nan
    
    # Pour chaque équipement
    for equip_id in equipment_ids:
        # Filtrer les données pour cet équipement
        
        if len(equip_data) == 0:
            continue
        
        # Utiliser le courant pour identifier les cycles (supposant que le courant indique l'activité)
        # Détecter les démarrages lorsque le courant passe au-dessus d'un seuil

        
        # Détecter les changements d'état (démarrage/arrêt)
        
        # Identifier les cycles (un cycle commence quand la machine démarre)
        
        current_cycle = 1
        for i in range(len(cycle_starts)):
            start_idx = cycle_starts[i]
            
            # Validate start_idx
            if start_idx < 0 or start_idx >= len(equip_data):
                print(f"Invalid start index at cycle {current_cycle}: {start_idx}")
                continue  # Skip to the next iteration if the index is invalid

            # Determine the end of the cycle (either the next start, or the end of the data)
            if i < len(cycle_starts) - 1:  # There is a next cycle
                end_idx = cycle_starts[i + 1]
            else:  # This is the last cycle
                end_idx = len(equip_data)  # Use the length of the DataFrame directly

            # Validate end_idx
            if end_idx < 0 or end_idx > len(equip_data):
                print(f"Invalid end index at cycle {current_cycle}: {end_idx}")
                continue  # Skip to the next iteration if the index is invalid
            
            # Ensure end_idx is not less than start_idx
            if end_idx <= start_idx:
                print(f"End index {end_idx} is not greater than start index {start_idx} at cycle {current_cycle}")
                continue  # Skip to the next iteration if the indices are invalid
            
            # Assigner l'ID du cycle
            df.loc[equip_data.iloc[start_idx:end_idx].index, 'cycle_id'] = current_cycle
            
            # Calculer la durée du cycle
            #print(f"Start Index: {start_idx}, End Index: {end_idx}, DataFrame Length: {len(equip_data)}")
            if start_idx < 0 or start_idx >= len(equip_data):
                print(f"Invalid start index: {start_idx}")
            if end_idx < 0 or end_idx >= len(equip_data):
                print(f"Invalid end index: {end_idx}")
            if equip_data.empty:
                print("The equipped data is empty.")
                return  # Or handle as needed

            cycle_duration =  # en minutes
            df.loc[equip_data.iloc[start_idx:end_idx].index, 'cycle_duration'] = cycle_duration
            
            # Calculer le temps écoulé dans le cycle et la phase du cycle (0-1)
            for j in range(start_idx, end_idx):
            
            current_cycle += 1
    
    logger.info(f"Caractéristiques de cycle créées pour {len(equipment_ids)} équipements")
    return df

def encode_categorical_features(df, method='onehot'):
    """
    Encode les variables catégorielles.
    
    Args:
        df (DataFrame): DataFrame avec les données
        method (str): Méthode d'encodage ('onehot' ou 'label')
        
    Returns:
        DataFrame: DataFrame avec les variables catégorielles encodées
    """
    df = df.copy()
    
    # Variables catégorielles à encoder
    cat_columns = ['equipment_type']
    
    if 'next_failure_type' in df.columns:
        cat_columns.append('next_failure_type')
    
    if 'component_affected' in df.columns:
        cat_columns.append('component_affected')
    
    # Filtrer pour inclure uniquement les colonnes présentes dans le DataFrame
    cat_columns = [col for col in cat_columns if col in df.columns]
    
    encoders = {}
    
    if method == 'onehot':
        for col in cat_columns:
            # Appliquer l'encodage one-hot
            encoded = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, encoded], axis=1)
            
            # Enregistrer le mapping pour référence future
            unique_values = df[col].unique().tolist()
            encoders[col] = unique_values
            
            # Supprimer la colonne originale
            df = df.drop(col, axis=1)
    
    elif method == 'label':
        for col in cat_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            
            # Enregistrer l'encodeur pour référence future
            encoders[col] = le
            
            # Conserver la colonne originale pour référence
    
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'onehot' ou 'label'.")
    
    # Sauvegarder les encodeurs pour une utilisation future
    logger.info(f"Variables catégorielles encodées avec la méthode '{method}'")
    return df, encoders

def create_frequency_domain_features(df, columns=['vibration'], fs=1.0, group_by='equipment_id'):
    """
    Crée des caractéristiques dans le domaine fréquentiel à partir des signaux temporels.
    
    Args:
        df (DataFrame): DataFrame avec les données de capteurs
        columns (list): Liste des colonnes à analyser
        fs (float): Fréquence d'échantillonnage (Hz)
        group_by (str): Colonne à utiliser pour le regroupement
        
    Returns:
        DataFrame: DataFrame avec les caractéristiques fréquentielles ajoutées
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Colonne {col} non trouvée, ignorée pour l'analyse fréquentielle")
            continue
            
        # Pour chaque équipement
        for equip_id in df[group_by].unique():
            # Filtrer et trier les données pour cet équipement
            equip_data = df[df[group_by] == equip_id].sort_values('timestamp')
            
            if len(equip_data) < 10:  # Besoin d'un minimum de points
                continue
                
            # Calculer la FFT
            signal = equip_data[col].values
            fft_result = np.fft.rfft(signal)
            fft_freq = np.fft.rfftfreq(len(signal), d=1/fs)
            fft_magnitude = np.abs(fft_result)
            
            # Extraire des caractéristiques fréquentielles
            dominant_freq_idx = np.argmax(fft_magnitude)
            dominant_freq = fft_freq[dominant_freq_idx] if dominant_freq_idx < len(fft_freq) else 0
            
            # Calculer des statistiques spectrales
            spectral_mean = np.mean(fft_magnitude)
            spectral_std = np.std(fft_magnitude)
            spectral_kurtosis = stats.kurtosis(fft_magnitude) if len(fft_magnitude) > 3 else 0
            spectral_skewness = stats.skew(fft_magnitude) if len(fft_magnitude) > 2 else 0
            
            # Assigner les caractéristiques à chaque ligne de cet équipement

    
    logger.info(f"Caractéristiques fréquentielles créées pour {len(columns)} colonnes")
    return df

def reduce_dimensionality(df, n_components=None, method='pca', exclude_cols=None):
    """
    Réduit la dimensionnalité des caractéristiques numériques.
    
    Args:
        df (DataFrame): DataFrame avec les caractéristiques
        n_components (int): Nombre de composantes à garder (None = automatique)
        method (str): Méthode de réduction ('pca' uniquement pour l'instant)
        exclude_cols (list): Liste des colonnes à exclure de la réduction
        
    Returns:
        tuple: (DataFrame avec dimensions réduites, transformateur utilisé)
    """
    df = df.copy()
    
    # Identifier les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclure les colonnes spécifiées
    if exclude_cols:
        for col in exclude_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
    
    # Si trop peu de colonnes, retourner tel quel
    if len(numeric_cols) <= 2:
        logger.warning("Trop peu de colonnes numériques pour la réduction de dimensions")
        return df, None
    
    # Créer une copie des données pour la réduction
    X = df[numeric_cols].fillna(0).copy()
    
    if method == 'pca':
        # Déterminer le nombre de composantes automatiquement si non spécifié
        if n_components is None:
            n_components = min(len(numeric_cols) // 2, len(X) // 10)
            n_components = max(2, n_components)  # Au moins 2 composantes
        
        # Appliquer PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)
        
        # Ajouter les composantes au DataFrame
        for i in range(n_components):
            df[f'pca_component_{i+1}'] = transformed[:, i]
        
        # Calculer et afficher la variance expliquée
        explained_variance = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA: {n_components} composantes expliquent {explained_variance:.2%} de la variance")
        
        return df, pca
    
    else:
        raise ValueError(f"Méthode de réduction '{method}' non supportée")

def create_anomaly_scores(df, columns=None, window_size=20, method='zscore'):
    """
    Calcule des scores d'anomalie pour les variables sélectionnées.
    
    Args:
        df (DataFrame): DataFrame avec les données de capteurs
        columns (list): Liste des colonnes à analyser (None = toutes les numériques)
        window_size (int): Taille de la fenêtre glissante pour la détection contextuelle
        method (str): Méthode de calcul du score ('zscore' ou 'mahalanobis')
        
    Returns:
        DataFrame: DataFrame avec les scores d'anomalie ajoutés
    """
    df = df.copy()
    
    # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclure les colonnes qui ne sont pas des mesures de capteurs
        exclude_patterns = ['_id', 'timestamp', 'failure', 'encoded', 'component', 'pca_component']
        columns = [col for col in columns if not any(pattern in col for pattern in exclude_patterns)]
    
    # Initialiser une colonne pour le score d'anomalie global
    df['anomaly_score'] = 0
    
    if method == 'zscore':
        # Pour chaque équipement
        for equip_id in df['equipment_id'].unique():
            # Filtrer et trier les données pour cet équipement
            equip_data = df[df['equipment_id'] == equip_id].sort_values('timestamp')
            
            if len(equip_data) < window_size:
                continue
                
            # Pour chaque colonne d'intérêt
            for col in columns:
                # Calculer les z-scores dans une fenêtre glissante
                rolling_mean = equip_data[col].rolling(window=window_size).mean()
                rolling_std = equip_data[col].rolling(window=window_size).std()
                
                # Éviter la division par zéro
                rolling_std = rolling_std.replace(0, np.nan)
                
                # Calculer les z-scores (en évitant NaN au début)
                z_scores = np.abs((equip_data[col] - rolling_mean) / rolling_std)
                z_scores = z_scores.fillna(0)
                
                # Ajouter la colonne de score d'anomalie pour cette variable
                df.loc[equip_data.index, f'{col}_anomaly'] = z_scores
                
                # Contribuer au score global (moyenne des scores individuels)
                df.loc[equip_data.index, 'anomaly_score'] += z_scores / len(columns)
    
    elif method == 'mahalanobis':
        # Ce code est un exemple plus avancé qui nécessiterait des bibliothèques
        # comme scikit-learn pour calculer les distances de Mahalanobis
        logger.warning("Méthode 'mahalanobis' non implémentée dans cette version")
        pass
    
    else:
        raise ValueError(f"Méthode de score d'anomalie '{method}' non supportée")
    
    logger.info(f"Scores d'anomalie calculés pour {len(columns)} colonnes avec la méthode '{method}'")
    return df

def build_features(input_dir='augmented_data', output_dir='featured_data'):
    """
    Construit des caractéristiques avancées à partir des données augmentées.
    
    Args:
        input_dir (str): Répertoire contenant les données augmentées
        output_dir (str): Répertoire pour les données avec caractéristiques avancées
        
    Returns:
        DataFrame: DataFrame prêt pour l'entraînement du modèle
    """
    try:
        # Création du répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Création d'un sous-répertoire pour les artifacts
        artifacts_dir = os.path.join(output_dir, 'artifacts')
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        
        # Chargement des données augmentées
        input_data_path = os.path.join(input_dir, 'augmented_sensor_data.parquet')
        
        logger.info(f"Chargement des données augmentées depuis {input_data_path}")
        df = pd.read_parquet(input_data_path)
        
        # --- Construction des caractéristiques avancées ---
        
        # 1. Caractéristiques polynomiales
        logger.info("Création des caractéristiques polynomiales")
        
        # 2. Caractéristiques de cycle (si les données temporelles sont suffisantes)
        if len(df) > 1000:  # Seuil arbitraire pour éviter le traitement sur trop peu de données
            logger.info("Création des caractéristiques de cycle")
        
        # 3. Encoder les variables catégorielles
        logger.info("Encodage des variables catégorielles")
        
        # Sauvegarder les encodeurs pour une utilisation future
        dump(encoders, os.path.join(artifacts_dir, 'category_encoders.joblib'))
        
        # 4. Caractéristiques du domaine fréquentiel pour les capteurs de vibration
        if 'vibration' in df.columns:
            logger.info("Création des caractéristiques fréquentielles")
            df = create_frequency_domain_features(df, columns=['vibration'], fs=1.0)
        
        # 5. Scores d'anomalie
        logger.info("Calcul des scores d'anomalie")

        
        # 6. Réduction de dimensionnalité
        exclude_from_pca = ['equipment_id', 'timestamp', 'failure_soon', 'time_to_failure', 
                           'anomaly_score', 'days_since_last_failure']

        # !!! à implementer !!! Exclure les colonnes importantes pour la prédiction

        logger.info("Réduction de dimensionnalité")
        df, pca_transformer = reduce_dimensionality(df, method='pca', exclude_cols=exclude_from_pca)

        df = df.drop(columns=['timestamp', 'equipment_id'])

        df['failure_within_24h'] = np.where(df['time_to_failure'] > 0, 1, 0)
        df['time_to_failure'] = df['time_to_failure'].fillna(0)
        
        # Sauvegarder le transformateur PCA
        if pca_transformer:
            dump(pca_transformer, os.path.join(artifacts_dir, 'pca_transformer.joblib'))
        
        # Sélection des caractéristiques finales 
        # À ce stade, on pourrait appliquer une sélection de caractéristiques,
        # mais cela nécessiterait des tests complémentaires

        train, test = train_test_split(df, test_size=0.2)
        
        # Sauvegarde des données avec caractéristiques enrichies
        output_path = os.path.join(output_dir, 'featured_data.parquet')
        train.to_parquet(output_path)
        test_output_path = os.path.join(output_dir, 'featured_test_data.parquet')
        test.to_parquet(test_output_path)

        # Également sauvegarder en CSV pour inspection
        csv_output_path = os.path.join(output_dir, 'featured_data.csv')
        train.to_csv(csv_output_path, index=False)
        test_csv_output_path = os.path.join(output_dir, 'featured_test_data.csv')
        test.to_csv(test_csv_output_path, index=False)

        logger.info(f"Données avec caractéristiques avancées sauvegardées dans {output_dir}")
        logger.info(f"Nombre total de caractéristiques: {df.shape[1]}")
        
        # Créer un rapport sur les caractéristiques
        feature_report = {
            "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_lignes": len(df),
            "nb_caracteristiques": df.shape[1],
            "memoire_utilisation_mb": df.memory_usage().sum() / 1024 / 1024,
            "pct_valeurs_manquantes": df.isnull().mean().mean() * 100,
            "nb_caracteristiques_polynomiales": len([col for col in df.columns if 'power_' in col]),
            "nb_caracteristiques_cycle": len([col for col in df.columns if col in ['cycle_id', 'cycle_phase', 'time_in_cycle', 'cycle_duration']]),
            "nb_caracteristiques_frequentielles": len([col for col in df.columns if 'spectral_' in col or 'dominant_freq' in col]),
            "nb_composantes_pca": len([col for col in df.columns if 'pca_component_' in col])
        }
        
        pd.DataFrame([feature_report]).to_csv(os.path.join(output_dir, 'feature_report.csv'), index=False)
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors de la construction des caractéristiques: {str(e)}")
        raise

if __name__ == "__main__":
    # Exécution de la construction des caractéristiques
    featured_df = build_features()

    # Affichage des informations de base sur les données avec caractéristiques avancées
    print("\nRésumé des données avec caractéristiques avancées:")
    print(f"Dimensions: {featured_df.shape}")
    print("\nAperçu des colonnes:")
    print(featured_df.columns.tolist()[:10])  # Afficher seulement les 10 premières colonnes
