"""
Module pour effectuer des prédictions à partir des modèles entraînés.
Ce script permet d'importer un modèle entraîné et de l'utiliser pour 
prédire les risques de défaillance sur de nouvelles données.
"""

import os
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PredictionEngine:
    """Classe pour effectuer des prédictions avec des modèles entraînés."""
    
    def __init__(self, model_path=None, models_dir="models"):
        """
        Initialise le moteur de prédiction.
        
        Args:
            model_path (str): Chemin vers un modèle spécifique
            models_dir (str): Répertoire contenant les modèles
        """
        self.model_path = model_path
        self.models_dir = models_dir
        self.model = None
        self.model_info = None
        self.features_info = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Charge un modèle entraîné à partir du chemin spécifié.
        
        Args:
            model_path (str): Chemin vers le modèle
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        logger.info(f"Chargement du modèle depuis {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_info = {
                'parameters': model_data.get('parameters', {}),
                'cv_score': model_data.get('cv_score', None),
                'evaluation': model_data.get('evaluation', {}),
                'timestamp': model_data.get('timestamp', 'Unknown')
            }
            self.features_info = model_data.get('features_info', {})
            
            logger.info(f"Modèle chargé avec succès. Timestamp: {self.model_info['timestamp']}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def find_latest_model(self, model_type=None):
        """
        Trouve le modèle le plus récent dans le répertoire des modèles.
        
        Args:
            model_type (str): Type de modèle à rechercher (None pour tous)
            
        Returns:
            str: Chemin vers le modèle le plus récent
        """
        logger.info(f"Recherche du modèle le plus récent dans {self.models_dir}")
        
        try:
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            
            if model_type:
                model_files = [f for f in model_files if f.startswith(f"{model_type}_")]
            
            if not model_files:
                logger.warning("Aucun modèle trouvé")
                return None
            
            # Trier par date de modification
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)), reverse=True)
            latest_model = os.path.join(self.models_dir, model_files[0])
            
            logger.info(f"Modèle le plus récent: {latest_model}")
            return latest_model
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du modèle le plus récent: {e}")
            return None
    
    def find_best_model(self, metric='auc'):
        """
        Trouve le meilleur modèle selon la métrique spécifiée.
        
        Args:
            metric (str): Métrique à utiliser ('auc' ou 'accuracy')
            
        Returns:
            str: Chemin vers le meilleur modèle
        """
        logger.info(f"Recherche du meilleur modèle selon {metric} dans {self.models_dir}")
        
        try:
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            
            if not model_files:
                logger.warning("Aucun modèle trouvé")
                return None
            
            # Évaluer chaque modèle
            best_score = -1
            best_model_path = None
            
            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Vérifier si les informations d'évaluation sont disponibles
                    if 'evaluation' in model_data and metric in model_data['evaluation']:
                        score = model_data['evaluation'][metric]
                        
                        if score > best_score:
                            best_score = score
                            best_model_path = model_path
                
                except Exception as e:
                    logger.warning(f"Impossible d'évaluer {model_file}: {e}")
            
            if best_model_path:
                logger.info(f"Meilleur modèle: {best_model_path} avec {metric}={best_score:.4f}")
                return best_model_path
            else:
                logger.warning(f"Aucun modèle avec métrique {metric} trouvé")
                return None
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du meilleur modèle: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        Prétraite les données pour la prédiction.
        
        Args:
            data (DataFrame): Données à prétraiter
            
        Returns:
            DataFrame: Données prétraitées
        """
        logger.info("Prétraitement des données pour la prédiction")
        
        # Vérifier que le modèle est chargé
        if not self.model or not self.features_info:
            logger.error("Modèle non chargé ou informations sur les caractéristiques manquantes")
            return None
        
        try:
            # Liste des caractéristiques attendues
            expected_features = self.features_info.get('feature_names', [])
            
            if not expected_features:
                logger.warning("Informations sur les caractéristiques manquantes")
                # Utiliser toutes les colonnes disponibles sauf l'identifiant

            
            # Vérifier les caractéristiques manquantes

            
            if missing_features:
                logger.warning(f"Caractéristiques manquantes: {missing_features}")
                # Ajouter les caractéristiques manquantes avec des valeurs par défaut (0)
                for feature in missing_features:
                    data[feature] = 0
            
            if extra_features:
                logger.warning(f"Caractéristiques supplémentaires ignorées: {extra_features}")
            
            # Sélectionner uniquement les caractéristiques nécessaires dans le bon ordre
            data_processed = data[expected_features].copy()
            
            logger.info(f"Données prétraitées: {data_processed.shape} échantillons, {data_processed.shape[1]} caractéristiques")
            return data_processed
        
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            return None
    
    def predict(self, data, return_probabilities=True, threshold=0.5):
        """
        Effectue des prédictions sur les données fournies.
        
        Args:
            data (DataFrame): Données pour la prédiction
            return_probabilities (bool): Renvoyer les probabilités de défaillance
            threshold (float): Seuil pour la classification binaire
            
        Returns:
            DataFrame: Résultats de prédiction
        """
        logger.info("Exécution des prédictions")
        
        # Vérifier que le modèle est chargé
        if not self.model:
            logger.error("Modèle non chargé")
            return None
        
        try:
            # Prétraiter les données
            X = self.preprocess_data(data)
            
            if X is None:
                return None
            
            # Sauvegarder l'ID de l'équipement si disponible
            has_equipment_id = 'equipment_id' in data.columns
            equipment_ids = data['equipment_id'].copy() if has_equipment_id else None
            
            # Effectuer les prédictions
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Créer le DataFrame des résultats
            results = pd.DataFrame({
                'failure_probability': y_pred_proba,
                'predicted_failure': y_pred
            })
            
            # Ajouter l'ID de l'équipement si disponible
            if has_equipment_id:
                results.insert(0, 'equipment_id', equipment_ids)
            
            # Ajouter un horodatage
            results['prediction_timestamp'] = datetime.now()
            
            logger.info(f"Prédictions terminées: {results.shape[0]} échantillons")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return None
    
    def calculate_risk_levels(self, probabilities, levels=5):
        """
        Convertit les probabilités en niveaux de risque.
        
        Args: