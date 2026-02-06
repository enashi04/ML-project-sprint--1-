"""
Package de Prédiction de Risque de Défaillance Industrielle

Ce package contient les modules nécessaires pour entraîner, évaluer et 
utiliser un modèle de prédiction des risques de défaillance industrielle.
"""

from train_model import train_and_evaluate
from predict_model import predict
from evaluation import evaluate_model

__all__ = [
    'train',
    'predict',
    'evaluate_model',
    'calculate_metrics'
]

def model_train_and_evaluate():
    trained_models, evaluation_results, model_paths, best_model = train_and_evaluate(data_path, target_column='failure_within_24h', models_to_train=None, 
                      models_dir="models", test_size=0.2, random_state=42, cv=5)


featured_data = build_features(
    input_dir=os.path.abspath("../../data/processed/augmented_data/"),
    output_dir=os.path.abspath("../../data/processed/augmented_data/")
)

# Version du package
__version__ = '0.1.0'

# Informations sur le projet
__project_name__ = 'Prédiction de Risque de Défaillance Industrielle'
__author__ = 'Classe de Machine Learning'
