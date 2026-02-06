"""
Module de traitement de données pour la prédiction de risque de défaillance industrielle.

Ce module contient des outils pour extraire, nettoyer et augmenter des données
de maintenance prédictive issues de capteurs industriels.

Fonctions principales:
- process_data: Exécute le pipeline complet (extraction, nettoyage, augmentation)
- extract_data: Extrait les données des fichiers CSV sources
- clean_data: Nettoie les données extraites
- augment_data: Enrichit les données avec des caractéristiques additionnelles

Utilisation typique:
```python
from predictive_maintenance import process_data

# Exécuter le pipeline complet
processed_data = process_data(
    sensor_file_path="chemin/vers/predictive_maintenance_sensor_data.csv",
    failure_file_path="chemin/vers/predictive_maintanace_failure_log.csv",
    output_dir="données_traitées"
)
```
"""

import os
import logging
from datetime import datetime

# Import des fonctions principales de chaque module
from extract import extract_data
from clean import clean_data
from augment import augment_data

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictive_maintenance.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('predictive_maintenance')

__all__ = ['extract_data', 'clean_data', 'augment_data', 'process_data']

def process_data(sensor_file_path, failure_file_path, output_dir='processed_data', 
                 skip_existing=False, verbose=True):
    """
    Exécute le pipeline complet de traitement des données: extraction, nettoyage et augmentation.
    
    Args:
        sensor_file_path (str): Chemin vers le fichier de données capteurs
        failure_file_path (str): Chemin vers le fichier de journal des défaillances
        output_dir (str): Répertoire principal pour les sorties
        skip_existing (bool): Si True, ignore les étapes déjà complétées si les fichiers existent
        verbose (bool): Si True, affiche des informations supplémentaires pendant le traitement
    
    Returns:
        DataFrame: Les données finales augmentées prêtes pour l'entraînement du modèle
    """
    try:
        start_time = datetime.now()
        
        # Créer la structure de répertoires
        extracted_dir = os.path.join(output_dir, 'extracted_data')
        cleaned_dir = os.path.join(output_dir, 'cleaned_data')
        augmented_dir = os.path.join(output_dir, 'augmented_data')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire principal créé: {output_dir}")
        
        # Étape 1: Extraction
        extract_output_path = os.path.join(extracted_dir, 'sensor_data.parquet')
        if skip_existing and os.path.exists(extract_output_path):
            logger.info("Données extraites trouvées, chargement depuis les fichiers existants")
            sensor_df, failure_df = None, None  # Les données seront chargées dans l'étape suivante
        else:
            logger.info("Début de l'extraction des données")
            sensor_df, failure_df = extract_data(sensor_file_path, failure_file_path, output_dir=extracted_dir)
            logger.info("Extraction des données terminée")
        
        # Étape 2: Nettoyage
        clean_output_path = os.path.join(cleaned_dir, 'clean_sensor_data.parquet')
        if skip_existing and os.path.exists(clean_output_path):
            logger.info("Données nettoyées trouvées, chargement depuis les fichiers existants")
            sensor_clean_df, failure_clean_df = None, None  # Les données seront chargées dans l'étape suivante
        else:
            logger.info("Début du nettoyage des données")
            sensor_clean_df, failure_clean_df = clean_data(input_dir=extracted_dir, output_dir=cleaned_dir)
            logger.info("Nettoyage des données terminé")
        
        # Étape 3: Augmentation
        augmented_output_path = os.path.join(augmented_dir, 'augmented_data.parquet')
        if skip_existing and os.path.exists(augmented_output_path):
            logger.info("Données augmentées trouvées, chargement depuis les fichiers existants")
            augmented_df = pd.read_parquet(augmented_output_path)
        else:
            logger.info("Début de l'augmentation des données")
            augmented_df = augment_data(input_dir=cleaned_dir, output_dir=augmented_dir)
            logger.info("Augmentation des données terminée")
        
        # Calculer le temps total écoulé
        elapsed_time = datetime.now() - start_time
        logger.info(f"Traitement des données terminé en {elapsed_time}")
        
        if verbose:
            # Afficher un résumé
            print("\n=== Résumé du traitement des données ===")
            print(f"Nombre de capteurs originaux: {sensor_df.shape[0] if sensor_df is not None else 'N/A'}")
            print(f"Nombre de défaillances originales: {failure_df.shape[0] if failure_df is not None else 'N/A'}")
            print(f"Nombre de capteurs après nettoyage: {sensor_clean_df.shape[0] if sensor_clean_df is not None else 'N/A'}")
            print(f"Nombre de défaillances après nettoyage: {failure_clean_df.shape[0] if failure_clean_df is not None else 'N/A'}")
            print(f"Nombre de lignes dans le jeu de données final: {augmented_df.shape[0]}")
            print(f"Nombre de caractéristiques dans le jeu de données final: {augmented_df.shape[1]}")
            print(f"Temps de traitement total: {elapsed_time}")
            print("======================================\n")


            logger.info("\n=== Résumé du traitement des données ===")
            logger.info(f"Nombre de capteurs originaux: {sensor_df.shape[0] if sensor_df is not None else 'N/A'}")
            logger.info(f"Nombre de défaillances originales: {failure_df.shape[0] if failure_df is not None else 'N/A'}")
            logger.info(f"Nombre de capteurs après nettoyage: {sensor_clean_df.shape[0] if sensor_clean_df is not None else 'N/A'}")
            logger.info(f"Nombre de défaillances après nettoyage: {failure_clean_df.shape[0] if failure_clean_df is not None else 'N/A'}")
            logger.info(f"Nombre de lignes dans le jeu de données final: {augmented_df.shape[0]}")
            logger.info(f"Nombre de caractéristiques dans le jeu de données final: {augmented_df.shape[1]}")
            logger.info(f"Temps de traitement total: {elapsed_time}")
            logger.info("======================================\n")
        
        return augmented_df
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données: {str(e)}")
        raise

# Si besoin d'imports spécifiques pour le fonctionnement du __init__.py
import pandas as pd
import os

processed_data = process_data(
    sensor_file_path=os.path.abspath("../../data/raw/predictive_maintenance_sensor_data.csv"),
    failure_file_path=os.path.abspath("../../data/raw/predictive_maintenance_failure_logs.csv"),
    output_dir=os.path.abspath("../../data/processed/")
)

# Version du module
__version__ = '0.1.0'