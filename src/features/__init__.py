"""
Module de construction de caractéristiques avancées pour la prédiction de risque de défaillance industrielle.

Ce module fournit des fonctions pour générer des caractéristiques avancées à partir des données 
augmentées, notamment des caractéristiques polynomiales, fréquentielles, de cycle, et la réduction
de dimensionnalité.

Fonctions principales:
- build_features: Exécute le processus complet de construction de caractéristiques
- create_polynomial_features: Crée des caractéristiques polynomiales
- create_cycle_features: Identifie et caractérise les cycles d'opération
- create_frequency_domain_features: Extrait des informations du domaine fréquentiel
- reduce_dimensionality: Applique des techniques de réduction de dimension (PCA)
- create_anomaly_scores: Calcule des scores d'anomalie pour la détection

Utilisation typique:
```python
from predictive_maintenance.build_features import build_features

# Construire les caractéristiques avancées
featured_data = build_features(
    input_dir="données_augmentées",
    output_dir="données_caractérisées"
)
```
"""
import os 

from build_features import (
    build_features,
    create_polynomial_features,
    create_cycle_features,
    encode_categorical_features,
    create_frequency_domain_features,
    reduce_dimensionality,
    create_anomaly_scores
)

__all__ = [
    'build_features',
    'create_polynomial_features',
    'create_cycle_features',
    'encode_categorical_features',
    'create_frequency_domain_features',
    'reduce_dimensionality',
    'create_anomaly_scores'
]


featured_data = build_features(
    input_dir=os.path.abspath("../../data/processed/augmented_data/"),
    output_dir=os.path.abspath("../../data/processed/")
)

# Version du module
__version__ = '0.1.0'
