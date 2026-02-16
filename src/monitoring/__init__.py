"""
Package de Prédiction de Risque de Défaillance Industrielle

Ce package contient les modules nécessaires pour entraîner, évaluer, 
utiliser et surveiller un modèle de prédiction des risques de défaillance industrielle.
"""

# Import des modules de monitoring
# from data_drift import check_drift  # À ajuster selon les fonctions réelles
from .data_drift import check_data_drift
from .performance_tracking import ModelPerformanceTracker #track_performance  # À ajuster selon les fonctions réelles

__all__ = [
    # Modules de monitoring
    # 'check_drift',  # À ajuster selon les fonctions réelles
    # 'detect_drift',
 # À ajuster selon les fonctions réelles"DataDriftMonitor",
    "compare_datasets",
    "generate_drift_report",
    "check_data_drift",
    "track_performance",
]

# Version du package
__version__ = '0.1.0'

# Informations sur le projet
__project_name__ = 'Prédiction de Risque de Défaillance Industrielle'
__author__ = 'Classe de Machine Learning'