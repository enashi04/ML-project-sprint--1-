"""
Package de Prédiction de Risque de Défaillance Industrielle

Ce package contient les modules nécessaires pour entraîner, évaluer et 
utiliser un modèle de prédiction des risques de défaillance industrielle.
"""

import os
import sys
from .train_model import train_and_evaluate
from .predict_model import PredictionEngine
from .evaluation import evaluate_model

# Ajouter le chemin du dossier 'features' au sys.path
features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "features"))
print(features_path)
if features_path not in sys.path:
    sys.path.append(features_path)

# Importer la fonction
from build_features import build_features


__all__ = [
    'train',
    'predict',
    'evaluate_model',
    'calculate_metrics'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_train_and_evaluate():
    data_path = os.path.join(os.path.join(BASE_DIR, "data", "processed"),"augmented_data","clean_sensor_failure_merged.csv")

    trained_models, evaluation_results, model_paths, best_model = train_and_evaluate(data_path, target_column='failure_within_24h', models_to_train=None, 
                      models_dir="models", test_size=0.2, random_state=42, cv=5)

featured_data = build_features(
    input_dir=os.path.abspath(os.path.join(BASE_DIR, "../../data/processed/augmented_data/")),
    output_dir=os.path.abspath(os.path.join(BASE_DIR, "../../data/processed/augmented_data/"))
)

# Version du package
__version__ = '0.1.0'

# Informations sur le projet
__project_name__ = 'Prédiction de Risque de Défaillance Industrielle'
__author__ = 'Classe de Machine Learning'
