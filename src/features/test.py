import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
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
    base_cols = [c for c in base_cols if c in df.columns]

    # Pour chaque colonne, créer les puissances jusqu'au degré spécifié
    for col in base_cols:
        for p in range(2, degree + 1):
            df[f'{col}_power_{p}'] = df[col] ** p

    logger.info(f"Caractéristiques polynomiales de degré {degree} créées")
    return df


def encode_categorical_features_train_test(train_df, test_df, method='onehot'):
    """
    Encode les variables catégorielles (fit sur train, apply sur test).

    Args:
        train_df (DataFrame)
        test_df (DataFrame)
        method (str): 'onehot' ou 'label'

    Returns:
        tuple: (train_df_encoded, test_df_encoded, encoders)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Variables catégorielles à encoder
    cat_columns = ['equipment_type']
    if 'next_failure_type' in train_df.columns:
        cat_columns.append('next_failure_type')
    if 'component_affected' in train_df.columns:
        cat_columns.append('component_affected')

    cat_columns = [c for c in cat_columns if c in train_df.columns]

    encoders = {}

    if method == "onehot":
        # One-hot: on fige les colonnes vues dans train et on aligne test
        for col in cat_columns:
            train_dum = pd.get_dummies(train_df[col], prefix=col, drop_first=False)
            test_dum = pd.get_dummies(test_df[col], prefix=col, drop_first=False)

            # Aligner les colonnes test sur celles du train
            train_cols = train_dum.columns.tolist()
            test_dum = test_dum.reindex(columns=train_cols, fill_value=0)

            # Concat
            train_df = pd.concat([train_df.drop(columns=[col]), train_dum], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), test_dum], axis=1)

            encoders[col] = train_cols

    elif method == "label":
        # Label encoding (moins recommandé si catégories non ordonnées)
        for col in cat_columns:
            le = LabelEncoder()
            train_df[f"{col}_encoded"] = le.fit_transform(train_df[col].astype(str))
            test_df[f"{col}_encoded"] = le.transform(test_df[col].astype(str))

            encoders[col] = le

            # On drop la colonne originale pour éviter fuite / double info
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

    else:
        raise ValueError("Méthode non reconnue. Utilisez 'onehot' ou 'label'.")

    logger.info(f"Variables catégorielles encodées avec la méthode '{method}' (fit train, apply test)")
    return train_df, test_df, encoders


def reduce_dimensionality_train_test(train_df, test_df, n_components=None, method='pca', exclude_cols=None):
    """
    Réduit la dimensionnalité (fit sur train, transform train/test).

    Returns:
        tuple: (train_df, test_df, transformer)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()

    if exclude_cols:
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Si trop peu de colonnes, retourner tel quel
    if len(numeric_cols) <= 2:
        logger.warning("Trop peu de colonnes numériques pour la réduction de dimensions")
        return train_df, test_df, None

    X_train = train_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if method == "pca":
        if n_components is None:
            # règle simple : pas trop agressif
            n_components = min(25, max(2, len(numeric_cols) // 3))

        pca = PCA(n_components=n_components)
        Z_train = pca.fit_transform(X_train)
        Z_test = pca.transform(X_test)

        for i in range(n_components):
            train_df[f'pca_component_{i+1}'] = Z_train[:, i]
            test_df[f'pca_component_{i+1}'] = Z_test[:, i]

        explained_variance = float(np.sum(pca.explained_variance_ratio_))
        logger.info(f"PCA: {n_components} composantes expliquent {explained_variance:.2%} de la variance")

        return train_df, test_df, pca

    raise ValueError(f"Méthode de réduction '{method}' non supportée")


def create_anomaly_scores(df, columns=None, window_size=20, method='zscore'):
    """
    Calcule des scores d'anomalie pour les variables sélectionnées.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        exclude_patterns = ['_id', 'timestamp', 'failure', 'encoded', 'component', 'pca_component']
        columns = [col for col in columns if not any(p in col for p in exclude_patterns)]

    df['anomaly_score'] = 0.0

    if method == 'zscore':
        for equip_id in df['equipment_id'].dropna().unique():
            equip_data = df[df['equipment_id'] == equip_id].sort_values('timestamp')
            if len(equip_data) < window_size:
                continue

            for col in columns:
                if col not in df.columns:
                    continue

                rolling_mean = equip_data[col].rolling(window=window_size, min_periods=window_size).mean()
                rolling_std = equip_data[col].rolling(window=window_size, min_periods=window_size).std()
                rolling_std = rolling_std.replace(0, np.nan)

                z_scores = np.abs((equip_data[col] - rolling_mean) / rolling_std).fillna(0)

                df.loc[equip_data.index, f'{col}_anomaly'] = z_scores
                df.loc[equip_data.index, 'anomaly_score'] += z_scores / max(1, len(columns))

    else:
        raise ValueError(f"Méthode de score d'anomalie '{method}' non supportée")

    logger.info(f"Scores d'anomalie calculés pour {len(columns)} colonnes avec la méthode '{method}'")
    return df


def temporal_train_test_split(df, time_col="timestamp", test_size=0.2):
    """
    Split temporel simple : les derniers X% du temps en test.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    cutoff_idx = int(len(df) * (1 - test_size))
    cutoff_time = df.iloc[cutoff_idx][time_col]

    train = df[df[time_col] < cutoff_time].copy()
    test = df[df[time_col] >= cutoff_time].copy()

    logger.info(f"Split temporel: cutoff={cutoff_time} | train={len(train)} test={len(test)}")
    return train, test, cutoff_time


def build_features(input_dir='augmented_data', output_dir='featured_data'):
    """
    Construit des caractéristiques avancées à partir des données augmentées.

    Args:
        input_dir (str): Répertoire contenant les données augmentées
        output_dir (str): Répertoire pour les données avec caractéristiques avancées

    Returns:
        DataFrame: DataFrame prêt pour l'entraînement du modèle (train+test concat, pour inspection)
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

        # Sécuriser timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

        # --- Construction des caractéristiques avancées ---

        # 0. (Optionnel) Créer une cible "within 24h" si tu veux, sinon garde failure_soon
        if "time_to_failure" in df.columns:
            df["failure_within_24h"] = ((df["time_to_failure"].astype("float") > 0) &
                                        (df["time_to_failure"].astype("float") <= 24)).astype(int)
            df["time_to_failure"] = df["time_to_failure"].fillna(0)

        # 1. Caractéristiques polynomiales
        logger.info("Création des caractéristiques polynomiales")
        df = create_polynomial_features(df, degree=2)

        # 2. Scores d'anomalie
        logger.info("Calcul des scores d'anomalie")
        df = create_anomaly_scores(df,columns=['temperature', 'vibration', 'pressure', 'current'], window_size=20, method='zscore')
        df=df.copy() # pour éviter SettingWithCopyWarning

        # Split temporel AVANT PCA et AVANT tout fit de transformeurs
        train, test, cutoff_time = temporal_train_test_split(df, time_col="timestamp", test_size=0.2)

        # 3. Encoder les variables catégorielles
        logger.info("Encodage des variables catégorielles")
        train, test, encoders = encode_categorical_features_train_test(train, test, method='onehot')

        # Sauvegarder les encodeurs pour une utilisation future
        dump(encoders, os.path.join(artifacts_dir, 'category_encoders.joblib'))

        # 4. Réduction de dimensionnalité (PCA) — fit train, transform test
        exclude_from_pca = [
            'equipment_id', 'failure_soon', 'time_to_failure', 'failure_within_24h',
            'anomaly_score', 'days_since_last_failure', 'failures_count_last_30days'
        ]
        # NB: timestamp n'est pas numérique donc pas dans PCA de toute façon

        logger.info("Réduction de dimensionnalité")
        train, test, pca_transformer = reduce_dimensionality_train_test(
            train, test, method='pca', exclude_cols=exclude_from_pca
        )

        # Sauvegarder le transformateur PCA
        if pca_transformer:
            dump(pca_transformer, os.path.join(artifacts_dir, 'pca_transformer.joblib'))

        # 5. Construction des datasets finaux (X/y)
        # Garder le timestamp pour debug/split, mais on l'enlève des features modèle ensuite.
        target_col = 'failure_soon' if 'failure_soon' in train.columns else 'failure_within_24h'

        drop_cols = [c for c in ['timestamp'] if c in train.columns]
        # equipment_id: tu peux le garder pour analyse, mais en général on le drop pour éviter biais
        if 'equipment_id' in train.columns:
            drop_cols.append('equipment_id')

        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)

        X_train = train.drop(columns=[target_col] + drop_cols, errors='ignore')
        X_test = test.drop(columns=[target_col] + drop_cols, errors='ignore')

        # Sauvegarde (parquet + csv)
        train_out = pd.concat([X_train, y_train.rename(target_col)], axis=1)
        test_out = pd.concat([X_test, y_test.rename(target_col)], axis=1)

        output_path = os.path.join(output_dir, 'featured_data.parquet')
        test_output_path = os.path.join(output_dir, 'featured_test_data.parquet')
        train_out.to_parquet(output_path, index=False)
        test_out.to_parquet(test_output_path, index=False)

        csv_output_path = os.path.join(output_dir, 'featured_data.csv')
        test_csv_output_path = os.path.join(output_dir, 'featured_test_data.csv')
        train_out.to_csv(csv_output_path, index=False)
        test_out.to_csv(test_csv_output_path, index=False)

        logger.info(f"Données avec caractéristiques avancées sauvegardées dans {output_dir}")
        logger.info(f"Nombre total de caractéristiques (train): {X_train.shape[1]} | (test): {X_test.shape[1]}")

        # Créer un rapport sur les caractéristiques
        full_df = pd.concat([train_out.assign(split="train"), test_out.assign(split="test")], axis=0)

        feature_report = {
            "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cutoff_time": str(cutoff_time),
            "nb_lignes_total": int(len(full_df)),
            "nb_lignes_train": int(len(train_out)),
            "nb_lignes_test": int(len(test_out)),
            "nb_caracteristiques_train": int(train_out.shape[1]),
            "memoire_utilisation_mb": float(full_df.memory_usage().sum() / 1024 / 1024),
            "pct_valeurs_manquantes": float(full_df.isnull().mean().mean() * 100),
            "nb_caracteristiques_polynomiales": int(len([col for col in full_df.columns if '_power_' in col])),
            "nb_composantes_pca": int(len([col for col in full_df.columns if 'pca_component_' in col])),
            "target": target_col
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
