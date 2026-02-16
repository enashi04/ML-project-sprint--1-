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
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            logger.info(f"Répertoire de visualisation créé: {viz_dir}")

        # Chargement des données extraites
        sensor_data_path = os.path.join(input_dir, 'sensor_data.parquet')
        failure_data_path = os.path.join(input_dir, 'failure_data.parquet')

        logger.info(f"Chargement des données capteurs depuis {sensor_data_path}")
        sensor_df = pd.read_parquet(sensor_data_path)

        logger.info(f"Chargement des données de défaillance depuis {failure_data_path}")
        failure_df = pd.read_parquet(failure_data_path)

        # Sécuriser les timestamps (indispensable pour create_time_features / augment)
        if "timestamp" in sensor_df.columns:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
        if "failure_timestamp" in failure_df.columns:
            failure_df["failure_timestamp"] = pd.to_datetime(failure_df["failure_timestamp"], errors="coerce")

        # --- Nettoyage des données capteurs ---

        # modifier la colonne equipment_id, récupérer que le dernier caractère et convertir en int
        # (en pratique ici on récupère la suite de chiffres en fin de chaîne, pas juste 1 caractère)
        if "equipment_id" in sensor_df.columns:
            sensor_df["equipment_id"] = (
                sensor_df["equipment_id"]
                .astype(str)
                .str.extract(r"(\d+)$")[0]
                .astype("float")
                .astype("Int64")
            )

        # je fais un tri strict pour le calcul des fenetres temporelles dans create_time_features / augment
        if "equipment_id" in sensor_df.columns and "timestamp" in sensor_df.columns:
            sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)

        # 1. Vérification des valeurs manquantes
        missing_values_sensor = sensor_df.isnull().sum()
        logger.info(f"Valeurs manquantes dans les données capteurs:\n{missing_values_sensor}")

        # Remplacer les valeurs infinies et NaN par 0
        # Important: on ne remplace pas timestamp/equipment_id par 0 (sinon données invalides)
        sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # On drop les lignes avec timestamp/equipment_id manquant car critiques
        critical_cols = [c for c in ["timestamp", "equipment_id"] if c in sensor_df.columns]
        if critical_cols:
            sensor_df = sensor_df.dropna(subset=critical_cols)

        # Imputation uniquement sur colonnes numériques capteurs (si besoin)
        numeric_columns = ['temperature', 'vibration', 'pressure', 'current']
        numeric_columns = [c for c in numeric_columns if c in sensor_df.columns]

        for col in numeric_columns:
            sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

        # Imputer uniquement les colonnes numériques (médiane)
        # (évite de casser si une colonne manque et évite de drop trop de lignes)
        if numeric_columns:
            sensor_df[numeric_columns] = sensor_df[numeric_columns].fillna(
                sensor_df[numeric_columns].median(numeric_only=True)
            )

        # 2. Suppression des lignes avec valeurs manquantes (ou imputation selon la stratégie)
        # ⚠️ Ici on évite dropna() global (trop destructeur). On se limite aux colonnes critiques + capteurs.
        original_len = len(sensor_df)
        subset_for_na = critical_cols + numeric_columns
        if subset_for_na:
            sensor_df = sensor_df.dropna(subset=subset_for_na)

        logger.info(f"Lignes supprimées pour valeurs manquantes: {original_len - len(sensor_df)}")

        # 3. Vérification des doublons
        duplicates = sensor_df.duplicated().sum()
        logger.info(f"Nombre de doublons dans les données capteurs: {duplicates}")
        sensor_df = sensor_df.drop_duplicates()

        # 4. Détection et traitement des valeurs aberrantes
        # (stratégie recommandée time-series: outlier -> NaN -> ffill par équipement -> fallback médiane globale)
        outliers_report = {}

        # On s'assure que c'est bien trié avant ffill (important)
        if "equipment_id" in sensor_df.columns and "timestamp" in sensor_df.columns:
            sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)

        for column in numeric_columns:
            if column not in sensor_df.columns:
                continue

            # Visualiser la distribution
            plot_path = os.path.join(viz_dir, f'{column}_distribution.png')
            plot_distribution(sensor_df, column, plot_path)

            # Détecter les valeurs aberrantes
            outliers_mask = detect_outliers(sensor_df, column, method='iqr')
            outliers_count = int(outliers_mask.sum())
            outliers_report[column] = outliers_count
            logger.info(f"Valeurs aberrantes détectées dans {column}: {outliers_count}")

            # Pour les valeurs aberrantes, les remplacer par des NaN puis imputer
            if outliers_count > 0:
                # Option 1: Conserver les valeurs aberrantes avec un indicateur
                # sensor_df[f'{column}_outlier'] = outliers_mask

                # Option 2: Remplacer par une stratégie "passé" (anti-fuite):
                # outlier -> NaN -> forward fill par équipement -> fallback médiane globale
                sensor_df.loc[outliers_mask, column] = np.nan

                if "equipment_id" in sensor_df.columns:
                    sensor_df[column] = sensor_df.groupby("equipment_id")[column].ffill()
                else:
                    # fallback si pas d'equipment_id (rare)
                    sensor_df[column] = sensor_df[column].ffill()

                # Si outliers au début de série => encore NaN, on met la médiane globale
                median_value = sensor_df[column].median()
                sensor_df[column] = sensor_df[column].fillna(median_value)

        # 5. Vérification de la cohérence des timestamps
        if "equipment_id" in sensor_df.columns and "timestamp" in sensor_df.columns:
            sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)

        # --- Nettoyage des données de défaillance ---
        if "equipment_id" in failure_df.columns:
            failure_df["equipment_id"] = failure_df["equipment_id"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
            failure_df["equipment_id"] = failure_df["equipment_id"].astype("Int64")

        # Drop les lignes de panne sans timestamp (inutilisables)
        if "failure_timestamp" in failure_df.columns:
            failure_df = failure_df.dropna(subset=["failure_timestamp", "equipment_id"])

        # 1. Vérification des valeurs manquantes
        missing_values_failure = failure_df.isnull().sum()
        logger.info(f"Valeurs manquantes dans les données de défaillance:\n{missing_values_failure}")

        # 2. Imputation des valeurs manquantes (si applicable)
        # Pour repair_duration et repair_cost, utiliser la médiane par type de défaillance
        for column in ['repair_duration', 'repair_cost']:
            if column in failure_df.columns and failure_df[column].isnull().sum() > 0:
                # Calculer les médianes par failure_type
                medians = failure_df.groupby('failure_type')[column].median()

                # Appliquer les médianes correspondantes
                for failure_type in failure_df['failure_type'].dropna().unique():
                    mask = (failure_df['failure_type'] == failure_type) & (failure_df[column].isnull())
                    failure_df.loc[mask, column] = medians.get(failure_type, np.nan)

                # Pour les types de défaillance sans valeur, utiliser la médiane globale
                failure_df[column] = failure_df[column].fillna(failure_df[column].median())

        # 3. Vérification des doublons
        duplicates = failure_df.duplicated().sum()
        logger.info(f"Nombre de doublons dans les données de défaillance: {duplicates}")
        failure_df = failure_df.drop_duplicates()

        # 4. Vérification de la cohérence temporelle
        # S'assurer que les défaillances sont pour des équipements existants
        invalid_ids = pd.DataFrame()
        if "equipment_id" in sensor_df.columns and "equipment_id" in failure_df.columns:
            valid_equipment_ids = sensor_df['equipment_id'].dropna().unique()
            invalid_ids = failure_df[~failure_df['equipment_id'].isin(valid_equipment_ids)]

            if len(invalid_ids) > 0:
                logger.warning(f"Défaillances pour des équipements inexistants: {len(invalid_ids)}")
                failure_df = failure_df[failure_df['equipment_id'].isin(valid_equipment_ids)]

        # Sauvegarde des données nettoyées
        sensor_df.to_parquet(os.path.join(output_dir, 'clean_sensor_data.parquet'), index=False)
        failure_df.to_parquet(os.path.join(output_dir, 'clean_failure_data.parquet'), index=False)

        # Également sauvegarder en CSV pour faciliter l'inspection
        sensor_df.to_csv(os.path.join(output_dir, 'clean_sensor_data.csv'), index=False)
        failure_df.to_csv(os.path.join(output_dir, 'clean_failure_data.csv'), index=False)

        logger.info(f"Données nettoyées sauvegardées dans {output_dir}")

        # Créer un rapport de nettoyage
        cleaning_report = {
            "date_nettoyage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_enregistrements_capteurs_initial": int(original_len),
            "nb_enregistrements_capteurs_final": int(len(sensor_df)),
            "nb_enregistrements_defaillances_initial": int(len(failure_df) + (len(invalid_ids) if isinstance(invalid_ids, pd.DataFrame) else 0)),
            "nb_enregistrements_defaillances_final": int(len(failure_df)),
            "nb_valeurs_aberrantes_detectees": outliers_report
        }

        pd.DataFrame([cleaning_report]).to_csv(os.path.join(output_dir, 'cleaning_report.csv'), index=False)

        # Sauvegarde détaillée des outliers (plus lisible qu’un dict dans une cellule CSV)
        pd.DataFrame([outliers_report]).to_csv(os.path.join(output_dir, 'outliers_report.csv'), index=False)

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
