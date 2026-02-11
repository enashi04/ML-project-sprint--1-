#Importation des libraires
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


def _ensure_datetime(df, col='timestamp'):
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


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
    new_cols = {}
    for col in base_cols:
        if col in df.columns:
            for d in range(2, degree + 1):
                new_col_name = f'{col}_power_{d}'
                new_cols[new_col_name] = df[col].astype(float) ** d

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
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
    df = _ensure_datetime(df, 'timestamp')
        
    if equipment_ids is None:
        equipment_ids = df['equipment_id'].unique()
        
    # Initialiser les colonnes de cycle avec des valeurs par défaut
    # (On les crée en une fois pour éviter la fragmentation)
    cycle_init = pd.DataFrame(
        {
            'cycle_id': np.nan,
            'cycle_phase': np.nan,
            'time_in_cycle': np.nan,
            'cycle_duration': np.nan
        },
        index=df.index
    )
    df = pd.concat([df, cycle_init], axis=1)
        
    global_cycle_counter = 0

    # On prépare des Series de sortie qu'on remplira, puis on concat à la fin
    cycle_id_s = pd.Series(np.nan, index=df.index, dtype='float64')
    cycle_phase_s = pd.Series(np.nan, index=df.index, dtype='float64')
    time_in_cycle_s = pd.Series(np.nan, index=df.index, dtype='float64')
    cycle_duration_s = pd.Series(np.nan, index=df.index, dtype='float64')

    for equip_id in equipment_ids:
        # Filtrer les données pour cet équipement
        mask = df['equipment_id'] == equip_id
        equip_data = df.loc[mask].sort_values('timestamp')
        
        if len(equip_data) == 0:
            continue
            
        # Détection de seuil dynamique (ex: moyenne + marge, ou simple moyenne)
        if 'current' in equip_data.columns:
            cur = equip_data['current'].astype(float)
            base = cur.mean()
            threshold = base * 0.5
            
            # État binaire : 1 = ON, 0 = OFF
            is_running = (cur > threshold).astype(int)
            
            # Détection des changements d'état (1: démarrage, -1: arrêt)
            diffs = is_running.diff().fillna(0)
            starts = np.where(diffs.values == 1)[0]
            ends = np.where(diffs.values == -1)[0]
                
            # Nettoyage des indices pour avoir des paires start/end cohérentes
            if len(starts) > 0 and len(ends) > 0:
                if ends[0] < starts[0]:
                    ends = ends[1:]
                
                n_cycles = min(len(starts), len(ends))
                starts = starts[:n_cycles]
                ends = ends[:n_cycles]
                
                equip_index = equip_data.index.to_numpy()
                equip_ts = equip_data['timestamp'].to_numpy()

                for s_idx, e_idx in zip(starts, ends):
                    if e_idx <= s_idx:
                        continue

                    global_cycle_counter += 1
                    
                    start_time = equip_ts[s_idx]
                    end_time = equip_ts[e_idx]

                    if pd.isna(start_time) or pd.isna(end_time):
                        continue

                    duration_minutes = (end_time - start_time) / np.timedelta64(1, 'm')
                    if duration_minutes <= 0:
                        continue

                    cycle_indices = equip_index[s_idx:e_idx]
                    if len(cycle_indices) == 0:
                        continue
                    
                    current_times = df.loc[cycle_indices, 'timestamp']
                    tic = (current_times - start_time).dt.total_seconds() / 60.0

                    cycle_id_s.loc[cycle_indices] = float(global_cycle_counter)
                    cycle_duration_s.loc[cycle_indices] = float(duration_minutes)
                    time_in_cycle_s.loc[cycle_indices] = tic.values
                    cycle_phase_s.loc[cycle_indices] = (tic / duration_minutes).values

    df['cycle_id'] = cycle_id_s
    df['cycle_duration'] = cycle_duration_s
    df['time_in_cycle'] = time_in_cycle_s
    df['cycle_phase'] = cycle_phase_s

    logger.info(f"Caractéristiques de cycle créées. Cycles totaux détectés : {global_cycle_counter}")
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
    
    cat_columns = [col for col in cat_columns if col in df.columns]
    
    encoders = {}
    
    if method == 'onehot':
        dummies_list = []
        for col in cat_columns:
            encoded = pd.get_dummies(df[col], prefix=col, drop_first=False)
            dummies_list.append(encoded)
            unique_values = df[col].unique().tolist()
            encoders[col] = unique_values
        
        if dummies_list:
            df = pd.concat([df.drop(columns=cat_columns), *dummies_list], axis=1)
    
    elif method == 'label':
        for col in cat_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df = df.drop(col, axis=1)
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'onehot' ou 'label'.")
    
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
    df = _ensure_datetime(df, 'timestamp')

    feature_cols = []

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Colonne {col} non trouvée, ignorée pour l'analyse fréquentielle")
            continue

        per_equipment_features = {}

        for equip_id in df[group_by].unique():
            equip_data = df[df[group_by] == equip_id].sort_values('timestamp')
            if len(equip_data) < 10:
                continue

            signal = equip_data[col].astype(float).fillna(0.0).values

            fft_result = np.fft.rfft(signal)
            fft_freq = np.fft.rfftfreq(len(signal), d=1/fs)
            fft_magnitude = np.abs(fft_result)

            dominant_freq_idx = int(np.argmax(fft_magnitude)) if len(fft_magnitude) else 0
            dominant_freq = float(fft_freq[dominant_freq_idx]) if dominant_freq_idx < len(fft_freq) else 0.0

            spectral_mean = float(np.mean(fft_magnitude)) if len(fft_magnitude) else 0.0
            spectral_std = float(np.std(fft_magnitude)) if len(fft_magnitude) else 0.0
            spectral_kurtosis = float(stats.kurtosis(fft_magnitude)) if len(fft_magnitude) > 3 else 0.0
            spectral_skewness = float(stats.skew(fft_magnitude)) if len(fft_magnitude) > 2 else 0.0

            per_equipment_features[equip_id] = {
                f'{col}_dominant_freq': dominant_freq,
                f'{col}_spectral_mean': spectral_mean,
                f'{col}_spectral_std': spectral_std,
                f'{col}_spectral_kurtosis': spectral_kurtosis,
                f'{col}_spectral_skewness': spectral_skewness,
            }

        if per_equipment_features:
            tmp = pd.DataFrame.from_dict(per_equipment_features, orient='index')
            tmp.index.name = group_by
            feature_cols.append(tmp)

    if feature_cols:
        all_feat = pd.concat(feature_cols, axis=1)
        df = df.merge(all_feat, left_on=group_by, right_index=True, how='left')

    logger.info(f"Caractéristiques fréquentielles créées pour {len(columns)} colonnes")
    return df


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
    df = _ensure_datetime(df, 'timestamp')
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        exclude_patterns = ['_id', 'timestamp', 'failure', 'encoded', 'component', 'pca_component', '_anomaly']
        columns = [col for col in columns if not any(pattern in col for pattern in exclude_patterns)]
    
    if not columns:
        logger.warning("Aucune colonne valide trouvée pour le calcul des scores d'anomalie")
        df['anomaly_score'] = 0.0
        return df
    
    df['anomaly_score'] = 0.0
    
    if method == 'zscore':
        scores = pd.Series(0.0, index=df.index, dtype='float64')

        for equip_id in df['equipment_id'].unique():
            mask = df['equipment_id'] == equip_id
            equip = df.loc[mask].sort_values('timestamp')
            if len(equip) < window_size:
                continue

            local_sum = pd.Series(0.0, index=equip.index, dtype='float64')
            valid_cols = 0

            for col in columns:
                if col not in equip.columns:
                    continue
                x = equip[col].astype(float)
                if x.isna().all():
                    continue

                m = x.rolling(window=window_size, min_periods=1).mean()
                s = x.rolling(window=window_size, min_periods=1).std().replace(0, np.nan)
                z = ((x - m) / (s + 1e-8)).abs().fillna(0.0).replace([np.inf, -np.inf], 0.0)

                local_sum = local_sum.add(z, fill_value=0.0)
                valid_cols += 1

            if valid_cols > 0:
                scores.loc[equip.index] = (local_sum / valid_cols).values

        df['anomaly_score'] = scores
            
    elif method == 'mahalanobis':
        logger.warning("Méthode 'mahalanobis' non implémentée dans cette version")
        df['anomaly_score'] = 0.0
    
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
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Répertoire OK: {output_dir}")
        
        # Création d'un sous-répertoire pour les artifacts
        artifacts_dir = os.path.join(output_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Chargement des données augmentées
        input_data_path = os.path.join(input_dir, 'augmented_sensor_data.parquet')
        
        logger.info(f"Chargement des données augmentées depuis {input_data_path}")
        df = pd.read_parquet(input_data_path)

        # sécuriser timestamp
        df = _ensure_datetime(df, 'timestamp')
        
        # --- Construction des caractéristiques avancées ---
        
        # 1. Caractéristiques polynomiales
        logger.info("Création des caractéristiques polynomiales")
        df = create_polynomial_features(df, degree=2)
        
        # 2. Caractéristiques de cycle
        if len(df) > 1000:
            logger.info("Création des caractéristiques de cycle")
            df = create_cycle_features(df)
        
        # 3. Encoder les variables catégorielles
        logger.info("Encodage des variables catégorielles")
        df, encoders = encode_categorical_features(df, method='onehot')
        dump(encoders, os.path.join(artifacts_dir, 'category_encoders.joblib'))
        
        # 4. Caractéristiques fréquentielles
        if 'vibration' in df.columns:
            logger.info("Création des caractéristiques fréquentielles")
            df = create_frequency_domain_features(df, columns=['vibration'], fs=1.0)
        
        # 5. Scores d'anomalie
        logger.info("Calcul des scores d'anomalie")
        df = create_anomaly_scores(df, method='zscore')

        # ============================================================
        # MODIFICATION ANTI-LEAKAGE : SPLIT AVANT TARGET ET AVANT PCA
        # ============================================================
        if "equipment_id" not in df.columns:
            raise ValueError("equipment_id manquant : impossible de split correctement")

        equip_ids = df["equipment_id"].dropna().unique()
        train_ids, test_ids = train_test_split(equip_ids, test_size=0.2, random_state=42)

        train = df[df["equipment_id"].isin(train_ids)].copy()
        test = df[df["equipment_id"].isin(test_ids)].copy()

        # =========================
        # TARGET séparée sur train/test
        # =========================
        for _df in (train, test):
            if 'time_to_failure' in _df.columns:
                _df['failure_within_24h'] = ((_df['time_to_failure'] > 0) & (_df['time_to_failure'] <= 24)).astype(int)
                _df['time_to_failure'] = _df['time_to_failure'].fillna(0)
            else:
                _df['failure_within_24h'] = 0
                _df['time_to_failure'] = 0

            if 'failure_soon' in _df.columns:
                _df['failure_soon'] = (_df['failure_soon'] > 0).astype(int)

        # =========================
        # PCA : fit sur train puis transform sur test
        # =========================
        exclude_from_pca = [
            'equipment_id', 'timestamp',
            'failure_soon', 'time_to_failure', 'failure_within_24h',
            'anomaly_score', 'days_since_last_failure'
        ]

        numeric_cols = train.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in set(exclude_from_pca)]

        if len(numeric_cols) > 2:
            logger.info("Réduction de dimensionnalité avec PCA (fit train -> transform test)")
            X_train_pca = train[numeric_cols].fillna(0).values
            X_test_pca = test[numeric_cols].fillna(0).values

            pca_transformer = PCA(n_components=2, random_state=42)
            Z_train = pca_transformer.fit_transform(X_train_pca)
            Z_test = pca_transformer.transform(X_test_pca)

            train["pca_component_1"] = Z_train[:, 0]
            train["pca_component_2"] = Z_train[:, 1]
            test["pca_component_1"] = Z_test[:, 0]
            test["pca_component_2"] = Z_test[:, 1]

            dump(pca_transformer, os.path.join(artifacts_dir, 'pca_transformer.joblib'))
            logger.info(f"PCA: 2 composantes expliquent {float(np.sum(pca_transformer.explained_variance_ratio_)):.2%} de la variance")
        else:
            logger.info("PCA non appliqué (pas assez de colonnes numériques)")
            pca_transformer = None

        # Drop colonnes non désirées
        drop_cols = [c for c in ['timestamp', 'equipment_id'] if c in train.columns]
        if drop_cols:
            train = train.drop(columns=drop_cols)
            test = test.drop(columns=drop_cols)

        # Sauvegarde des données
        output_path = os.path.join(output_dir, 'featured_data.parquet')
        test_output_path = os.path.join(output_dir, 'featured_test_data.parquet')
        train.to_parquet(output_path, index=False)
        test.to_parquet(test_output_path, index=False)

        csv_output_path = os.path.join(output_dir, 'featured_data.csv')
        test_csv_output_path = os.path.join(output_dir, 'featured_test_data.csv')
        train.to_csv(csv_output_path, index=False)
        test.to_csv(test_csv_output_path, index=False)

        logger.info(f"Données avec caractéristiques avancées sauvegardées dans {output_dir}")
        logger.info(f"Train: {train.shape} | Test: {test.shape}")

        # Rapport features
        full_df = pd.concat([train, test], axis=0, ignore_index=True)

        feature_report = {
            "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_lignes": len(full_df),
            "nb_caracteristiques": full_df.shape[1],
            "memoire_utilisation_mb": full_df.memory_usage().sum() / 1024 / 1024,
            "pct_valeurs_manquantes": full_df.isnull().mean().mean() * 100,
            "nb_caracteristiques_polynomiales": len([col for col in full_df.columns if 'power_' in col]),
            "nb_caracteristiques_cycle": len([col for col in full_df.columns if col in ['cycle_id', 'cycle_phase', 'time_in_cycle', 'cycle_duration']]),
            "nb_caracteristiques_frequentielles": len([col for col in full_df.columns if 'spectral_' in col or 'dominant_freq' in col]),
            "nb_composantes_pca": len([col for col in full_df.columns if 'pca_component_' in col])
        }

        pd.DataFrame([feature_report]).to_csv(os.path.join(output_dir, 'feature_report.csv'), index=False)

        return full_df
    
    except Exception as e:
        logger.error(f"Erreur lors de la construction des caractéristiques: {str(e)}")
        raise


if __name__ == "__main__":
    featured_df = build_features()

    print("\nRésumé des données avec caractéristiques avancées:")
    print(f"Dimensions: {featured_df.shape}")
    print("\nAperçu des colonnes:")
    print(featured_df.columns.tolist()[:10])
