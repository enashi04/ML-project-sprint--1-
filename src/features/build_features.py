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
    """Assure que la colonne timestamp est au format datetime."""
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def create_polynomial_features(df, degree=2):
    """
    Crée des caractéristiques polynomiales pour capturer les relations non linéaires.
    VERSION OPTIMISÉE : utilise pd.concat pour éviter la fragmentation.
    """
    df = df.copy()
    
    # Colonnes numériques de base pour les polynômes
    base_cols = ['temperature', 'vibration', 'pressure', 'current']
    
    # 🔥 OPTIMISATION : créer toutes les colonnes d'un coup
    new_cols = {}
    for col in base_cols:
        if col in df.columns:
            for d in range(2, degree + 1):
                new_cols[f'{col}_power_{d}'] = df[col].astype(float) ** d
    
    # Concat une seule fois
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    logger.info(f"Caractéristiques polynomiales de degré {degree} créées")
    return df


def create_cycle_features(df, equipment_ids=None):
    """
    Crée des caractéristiques basées sur les cycles d'opération des équipements.
    VERSION OPTIMISÉE : utilise des Series temporaires puis concat final.
    """
    df = df.copy()
    df = _ensure_datetime(df, 'timestamp')
    
    if equipment_ids is None:
        equipment_ids = df['equipment_id'].unique()
    
    # 🔥 OPTIMISATION : créer des Series temporaires
    cycle_id_s = pd.Series(np.nan, index=df.index, dtype='float64')
    cycle_phase_s = pd.Series(np.nan, index=df.index, dtype='float64')
    time_in_cycle_s = pd.Series(np.nan, index=df.index, dtype='float64')
    cycle_duration_s = pd.Series(np.nan, index=df.index, dtype='float64')
    
    global_cycle_counter = 0
    
    for equip_id in equipment_ids:
        equip_data = df[df['equipment_id'] == equip_id].sort_values('timestamp')
        
        if len(equip_data) == 0:
            continue
        
        if 'current' in equip_data.columns:
            current_values = equip_data['current'].values
            threshold = np.mean(current_values) * 0.5
            
            is_running = (current_values > threshold).astype(int)
            state_changes = np.diff(is_running, prepend=0)
            cycle_starts = np.where(state_changes == 1)[0]
            cycle_ends = np.where(state_changes == -1)[0]
            
            if len(cycle_starts) > 0 and len(cycle_ends) > 0:
                if len(cycle_ends) > 0 and cycle_ends[0] < cycle_starts[0]:
                    cycle_ends = cycle_ends[1:]
                
                n_cycles = min(len(cycle_starts), len(cycle_ends))
                
                for i in range(n_cycles):
                    start_idx = cycle_starts[i]
                    end_idx = cycle_ends[i] if i < len(cycle_ends) else len(equip_data)
                    
                    if start_idx < 0 or start_idx >= len(equip_data):
                        continue
                    if end_idx < 0 or end_idx > len(equip_data):
                        continue
                    if end_idx <= start_idx:
                        continue
                    
                    global_cycle_counter += 1
                    
                    start_time = equip_data.iloc[start_idx]['timestamp']
                    end_time = equip_data.iloc[min(end_idx, len(equip_data)-1)]['timestamp']
                    
                    cycle_indices = equip_data.iloc[start_idx:end_idx].index
                    cycle_duration = (end_time - start_time).total_seconds() / 60.0
                    
                    # Remplir les Series
                    cycle_id_s.loc[cycle_indices] = float(global_cycle_counter)
                    cycle_duration_s.loc[cycle_indices] = float(cycle_duration)
                    
                    # Calculer time_in_cycle et phase pour chaque point
                    for j in range(start_idx, min(end_idx, len(equip_data))):
                        idx = equip_data.iloc[j].name
                        current_time = equip_data.iloc[j]['timestamp']
                        tic = (current_time - start_time).total_seconds() / 60.0
                        
                        time_in_cycle_s.loc[idx] = tic
                        if cycle_duration > 0:
                            cycle_phase_s.loc[idx] = tic / cycle_duration
    
    # 🔥 CONCAT une seule fois
    new_cols = {
        'cycle_id': cycle_id_s,
        'cycle_phase': cycle_phase_s,
        'time_in_cycle': time_in_cycle_s,
        'cycle_duration': cycle_duration_s
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    logger.info(f"Caractéristiques de cycle créées. Cycles totaux détectés : {global_cycle_counter}")
    return df


def encode_categorical_features(df, method='onehot'):
    """Encode les variables catégorielles."""
    df = df.copy()
    
    cat_columns = ['equipment_type']
    
    if 'next_failure_type' in df.columns:
        cat_columns.append('next_failure_type')
    if 'component_affected' in df.columns:
        cat_columns.append('component_affected')
    
    cat_columns = [col for col in cat_columns if col in df.columns]
    encoders = {}
    
    if method == 'onehot':
        # 🔥 OPTIMISATION : créer toutes les colonnes encodées puis concat
        encoded_dfs = []
        for col in cat_columns:
            encoded = pd.get_dummies(df[col], prefix=col, drop_first=False)
            encoded_dfs.append(encoded)
            encoders[col] = df[col].unique().tolist()
        
        # Concat toutes les colonnes encodées d'un coup
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)
            df = df.drop(columns=cat_columns)
    
    elif method == 'label':
        new_cols = {}
        for col in cat_columns:
            le = LabelEncoder()
            new_cols[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'onehot' ou 'label'.")
    
    logger.info(f"Variables catégorielles encodées avec la méthode '{method}'")
    return df, encoders


def create_frequency_domain_features(df, columns=['vibration'], fs=1.0, group_by='equipment_id'):
    """
    Crée des caractéristiques dans le domaine fréquentiel.
    VERSION OPTIMISÉE : utilise des Series temporaires puis concat final.
    """
    df = df.copy()
    
    # 🔥 OPTIMISATION : créer des Series temporaires pour chaque feature
    new_cols = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Colonne {col} non trouvée, ignorée pour l'analyse fréquentielle")
            continue
        
        # Initialiser les Series pour ce col
        for suffix in ['dominant_freq', 'spectral_mean', 'spectral_std', 'spectral_kurtosis', 'spectral_skewness']:
            new_cols[f'{col}_{suffix}'] = pd.Series(0.0, index=df.index, dtype='float64')
        
        # Pour chaque équipement
        for equip_id in df[group_by].unique():
            equip_data = df[df[group_by] == equip_id].sort_values('timestamp')
            
            if len(equip_data) < 10:
                continue
            
            signal = equip_data[col].values
            fft_result = np.fft.rfft(signal)
            fft_freq = np.fft.rfftfreq(len(signal), d=1/fs)
            fft_magnitude = np.abs(fft_result)
            
            dominant_freq_idx = np.argmax(fft_magnitude)
            dominant_freq = fft_freq[dominant_freq_idx] if dominant_freq_idx < len(fft_freq) else 0
            
            spectral_mean = np.mean(fft_magnitude)
            spectral_std = np.std(fft_magnitude)
            spectral_kurtosis = stats.kurtosis(fft_magnitude) if len(fft_magnitude) > 3 else 0
            spectral_skewness = stats.skew(fft_magnitude) if len(fft_magnitude) > 2 else 0
            
            # Remplir les Series
            equip_idx = equip_data.index
            new_cols[f'{col}_dominant_freq'].loc[equip_idx] = dominant_freq
            new_cols[f'{col}_spectral_mean'].loc[equip_idx] = spectral_mean
            new_cols[f'{col}_spectral_std'].loc[equip_idx] = spectral_std
            new_cols[f'{col}_spectral_kurtosis'].loc[equip_idx] = spectral_kurtosis
            new_cols[f'{col}_spectral_skewness'].loc[equip_idx] = spectral_skewness
    
    # 🔥 CONCAT une seule fois
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    logger.info(f"Caractéristiques fréquentielles créées pour {len(columns)} colonnes")
    return df


def reduce_dimensionality(df, n_components=None, method='pca', exclude_cols=None):
    """Réduit la dimensionnalité des caractéristiques numériques."""
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) <= 2:
        logger.warning("Trop peu de colonnes numériques pour la réduction de dimensions")
        return df, None
    
    X = df[numeric_cols].fillna(0).copy()
    
    if method == 'pca':
        if n_components is None:
            n_components = min(len(numeric_cols) // 2, len(X) // 10)
            n_components = max(2, n_components)
        
        n_components = min(n_components, len(numeric_cols), len(X))
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)
        
        # 🔥 OPTIMISATION : créer toutes les composantes puis concat
        pca_cols = {}
        for i in range(n_components):
            pca_cols[f'pca_component_{i+1}'] = transformed[:, i]
        
        df = pd.concat([df, pd.DataFrame(pca_cols, index=df.index)], axis=1)
        
        explained_variance = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA: {n_components} composantes expliquent {explained_variance:.2%} de la variance")
        
        return df, pca
    
    else:
        raise ValueError(f"Méthode de réduction '{method}' non supportée")


def create_anomaly_scores(df, columns=None, window_size=20, method='zscore'):
    """
    Calcule des scores d'anomalie pour les variables sélectionnées.
    VERSION OPTIMISÉE : utilise pd.concat pour éviter la fragmentation.
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        exclude_patterns = ['_id', 'timestamp', 'failure', 'encoded', 'component', 'pca_component']
        columns = [col for col in columns if not any(pattern in col for pattern in exclude_patterns)]
    
    if method == 'zscore':
        # 🔥 OPTIMISATION : créer toutes les colonnes d'anomalie d'un coup
        new_cols = {}
        global_anomaly_scores = pd.Series(0.0, index=df.index, dtype='float64')
        
        for equip_id in df['equipment_id'].unique():
            mask = df['equipment_id'] == equip_id
            equip_data = df.loc[mask].sort_values('timestamp')
            
            if len(equip_data) < window_size:
                continue
            
            equip_idx = equip_data.index
            anomaly_count = 0
            
            for col in columns:
                if col not in equip_data.columns:
                    continue
                
                rolling_mean = equip_data[col].rolling(window=window_size, min_periods=1).mean()
                rolling_std = equip_data[col].rolling(window=window_size, min_periods=1).std()
                rolling_std = rolling_std.replace(0, np.nan)
                
                z_scores = np.abs((equip_data[col] - rolling_mean) / (rolling_std + 1e-8))
                z_scores = z_scores.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Stocker dans le dictionnaire
                col_name = f'{col}_anomaly'
                if col_name not in new_cols:
                    new_cols[col_name] = pd.Series(0.0, index=df.index, dtype='float64')
                
                new_cols[col_name].loc[equip_idx] = z_scores.values
                global_anomaly_scores.loc[equip_idx] += z_scores.values
                anomaly_count += 1
            
            if anomaly_count > 0:
                global_anomaly_scores.loc[equip_idx] /= anomaly_count
        
        # 🔥 CONCAT UNE SEULE FOIS
        new_cols['anomaly_score'] = global_anomaly_scores
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    elif method == 'mahalanobis':
        logger.warning("Méthode 'mahalanobis' non implémentée dans cette version")
        df = pd.concat([df, pd.DataFrame({'anomaly_score': 0.0}, index=df.index)], axis=1)
    
    else:
        raise ValueError(f"Méthode de score d'anomalie '{method}' non supportée")
    
    logger.info(f"Scores d'anomalie calculés pour {len(columns)} colonnes avec la méthode '{method}'")
    return df


def build_features(input_dir='augmented_data', output_dir='featured_data'):
    """
    Construit des caractéristiques avancées à partir des données augmentées.
    🚨 VERSION CORRIGÉE : Split AVANT création de la target pour éviter le data leakage
    ⚡ VERSION OPTIMISÉE : Évite la fragmentation mémoire avec pd.concat
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        artifacts_dir = os.path.join(output_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        input_data_path = os.path.join(input_dir, 'augmented_sensor_data.parquet')
        logger.info(f"Chargement des données augmentées depuis {input_data_path}")
        df = pd.read_parquet(input_data_path)
        df = _ensure_datetime(df, 'timestamp')
        
        # --- Construction des caractéristiques avancées ---
        
        logger.info("Création des caractéristiques polynomiales")
        df = create_polynomial_features(df, degree=2)
        
        if len(df) > 1000:
            logger.info("Création des caractéristiques de cycle")
            df = create_cycle_features(df)
        
        logger.info("Encodage des variables catégorielles")
        df, encoders = encode_categorical_features(df, method='onehot')
        dump(encoders, os.path.join(artifacts_dir, 'category_encoders.joblib'))
        
        if 'vibration' in df.columns:
            logger.info("Création des caractéristiques fréquentielles")
            df = create_frequency_domain_features(df, columns=['vibration'], fs=1.0)
        
        logger.info("Calcul des scores d'anomalie")
        df = create_anomaly_scores(df, method='zscore', window_size=20)
        
        # Exclure les colonnes de leakage de la PCA
        exclude_from_pca = [
            'equipment_id', 'timestamp', 'failure_soon', 'time_to_failure', 
            'next_failure_type', 'anomaly_score', 'days_since_last_failure',
            'failures_count_last_30days'
        ]
        exclude_from_pca += [col for col in df.columns if col.startswith('next_failure_type_')]
        
        logger.info("Réduction de dimensionnalité avec PCA")
        df, pca_transformer = reduce_dimensionality(df, n_components=5, method='pca', exclude_cols=exclude_from_pca)
        
        if pca_transformer:
            dump(pca_transformer, os.path.join(artifacts_dir, 'pca_transformer.joblib'))
            logger.info(f"Variance expliquée par PCA : {sum(pca_transformer.explained_variance_ratio_):.2%}")
        
        # ============================================
        # 🎯 SPLIT TEMPOREL AVANT CRÉATION DE LA TARGET
        # ============================================
        logger.info("Split temporel des données (80% train, 20% test)")
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        split_date = train_df['timestamp'].max()
        logger.info(f"Split effectué à la date : {split_date}")
        logger.info(f"Train : {len(train_df)} lignes ({train_df['timestamp'].min()} à {train_df['timestamp'].max()})")
        logger.info(f"Test : {len(test_df)} lignes ({test_df['timestamp'].min()} à {test_df['timestamp'].max()})")
        
        # ============================================
        # 🎯 CRÉER LES TARGETS APRÈS LE SPLIT
        # ============================================
        
        # Pour TRAIN
        if 'time_to_failure' in train_df.columns:
            train_df['failure_within_24h'] = ((train_df['time_to_failure'] > 0) & 
                                             (train_df['time_to_failure'] <= 24)).astype(int)
            train_df['time_to_failure'] = train_df['time_to_failure'].fillna(0)
        else:
            train_df['failure_within_24h'] = 0
            train_df['time_to_failure'] = 0
        
        # Pour TEST
        if 'time_to_failure' in test_df.columns:
            test_df['failure_within_24h'] = ((test_df['time_to_failure'] > 0) & 
                                            (test_df['time_to_failure'] <= 24)).astype(int)
            test_df['time_to_failure'] = test_df['time_to_failure'].fillna(0)
        else:
            test_df['failure_within_24h'] = 0
            test_df['time_to_failure'] = 0
        
        if 'failure_soon' in train_df.columns:
            train_df['failure_soon'] = (train_df['failure_soon'] > 0).astype(int)
        if 'failure_soon' in test_df.columns:
            test_df['failure_soon'] = (test_df['failure_soon'] > 0).astype(int)
        
        logger.info(f"Train failures (failure_within_24h=1) : {train_df['failure_within_24h'].sum()} ({train_df['failure_within_24h'].mean():.2%})")
        logger.info(f"Test failures (failure_within_24h=1) : {test_df['failure_within_24h'].sum()} ({test_df['failure_within_24h'].mean():.2%})")
        
        # Drop colonnes non désirées
        drop_cols = ['timestamp', 'equipment_id']
        drop_cols = [col for col in drop_cols if col in train_df.columns]
        
        if drop_cols:
            logger.info(f"Suppression des colonnes : {drop_cols}")
            train_df = train_df.drop(columns=drop_cols)
            test_df = test_df.drop(columns=drop_cols)
        
        # Sauvegarde
        logger.info("Sauvegarde des données...")
        train_df.to_parquet(os.path.join(output_dir, 'featured_train_data.parquet'), index=False)
        test_df.to_parquet(os.path.join(output_dir, 'featured_test_data.parquet'), index=False)
        train_df.to_csv(os.path.join(output_dir, 'featured_train_data.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'featured_test_data.csv'), index=False)
        
        # Pour compatibilité
        train_df.to_csv(os.path.join(output_dir, 'featured_data.csv'), index=False)
        
        logger.info(f"✅ Données sauvegardées dans {output_dir}")
        logger.info(f"Nombre total de caractéristiques : {train_df.shape[1]}")
        
        # Rapport
        feature_report = {
            "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_lignes_train": len(train_df),
            "nb_lignes_test": len(test_df),
            "nb_caracteristiques": train_df.shape[1],
            "train_positive_rate": float(train_df['failure_within_24h'].mean()),
            "test_positive_rate": float(test_df['failure_within_24h'].mean()),
            "memoire_utilisation_train_mb": train_df.memory_usage().sum() / 1024 / 1024,
            "memoire_utilisation_test_mb": test_df.memory_usage().sum() / 1024 / 1024,
            "pct_valeurs_manquantes_train": train_df.isnull().mean().mean() * 100,
            "pct_valeurs_manquantes_test": test_df.isnull().mean().mean() * 100,
            "nb_caracteristiques_polynomiales": len([col for col in train_df.columns if 'power_' in col]),
            "nb_caracteristiques_cycle": len([col for col in train_df.columns if col in ['cycle_id', 'cycle_phase', 'time_in_cycle', 'cycle_duration']]),
            "nb_caracteristiques_frequentielles": len([col for col in train_df.columns if 'spectral_' in col or 'dominant_freq' in col]),
            "nb_composantes_pca": len([col for col in train_df.columns if 'pca_component_' in col])
        }
        
        pd.DataFrame([feature_report]).to_csv(os.path.join(output_dir, 'feature_report.csv'), index=False)
        
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ DU FEATURE ENGINEERING")
        logger.info("="*60)
        for key, value in feature_report.items():
            logger.info(f"{key}: {value}")
        logger.info("="*60 + "\n")
        
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Erreur lors de la construction des caractéristiques: {str(e)}")
        raise


if __name__ == "__main__":
    train_df, test_df = build_features()

    print("\n" + "="*60)
    print("RÉSUMÉ DES DONNÉES AVEC CARACTÉRISTIQUES AVANCÉES")
    print("="*60)
    print(f"\nTRAIN:")
    print(f"  Dimensions: {train_df.shape}")
    print(f"  Colonnes (10 premières): {train_df.columns.tolist()[:10]}")
    print(f"  Failures: {train_df['failure_within_24h'].sum()} ({train_df['failure_within_24h'].mean():.2%})")
    
    print(f"\nTEST:")
    print(f"  Dimensions: {test_df.shape}")
    print(f"  Failures: {test_df['failure_within_24h'].sum()} ({test_df['failure_within_24h'].mean():.2%})")
    print("="*60)