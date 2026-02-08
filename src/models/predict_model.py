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
                cols = [c for c in data.columns if c not in ['equipment_id', 'timestamp', 'prediction_timestamp']]
                expected_features = cols  

            
            # Vérifier les caractéristiques manquantes
            missing_features = [feature for feature in data.columns if feature not in data.columns]
            extra_features = [feature for feature in data.columns if feature not in expected_features and feature not in ['equipment_id','timestamp']]

            if missing_features:
                logger.warning(f"Caractéristiques manquantes: {missing_features}")
                # Ajouter les caractéristiques manquantes avec des valeurs par défaut (0)
                for feature in missing_features:
                    data[feature] = 0
            
            if extra_features:
                logger.warning(f"Caractéristiques supplémentaires ignorées: {extra_features}")
            
            # Sélectionner uniquement les caractéristiques nécessaires dans le bon ordre
            data_processed = data[expected_features].copy()

            data_processed = ( 
                data_processed
                .replace([np.inf, -np.inf],np.nan)
                .fillna(0)
            )
            
            #sécurité si y'a des booleans
            for col in data_processed.columns: 
                if data_processed[col].dtype=='bool':
                    data_processed[col]=data_processed[col].astype(int)
            
            #conversion non numériques à voir

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
        probabilities (array-like) : probabilités de défaillance
        levels (int): Nombre de niveaux

        Returns : 
        array : niveau de risque """
        try :
            proba= np.asarray(probabilities, dtype=float)
            proba=np.clip(proba, 0.0,1.0)

            #découpage en quantile
            quantiles = np.linspace(0,1,levels+1)
            bins = np.quantile(proba, quantiles)

            #si on a des proba constantesn on veut éviter les bins
            bins = np.nuique(bins)
            if len(bins)<=2:
                bins=np.linspace(0,1,levels+1)

            #risque
            risk = np.digitize(proba, bins[1:-1], right=True) +1
            risk=np.clip(risk, 1, levels)

            return risk

        except Exception as e:
            logger.error(f"Erreur lors du calcul des niveaux de risque: {e}")
            return None
        
    def predict_with_risk_levels(self, data, levels=5, threshold=0.5):
        """
        Prédit et ajoute un niveau de risque.
        """
        results = self.predict(data, return_probabilities=True, threshold=threshold)
        if results is None:
            return None
        results['risk_level'] = self.calculate_risk_levels(results['failure_probability'].values, levels=levels)
        return results

    def save_predictions(self, results, output_path=None):
        """
        Sauvegarde les prédictions en CSV (et JSON optionnel).
        
        Args:
            results (DataFrame): Résultats de prédiction
            output_path (str): Chemin de sortie (csv). Si None, généré automatiquement.
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if results is None or len(results) == 0:
            logger.warning("Aucun résultat à sauvegarder")
            return None

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.models_dir, f"predictions_{ts}.csv")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"Prédictions sauvegardées: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des prédictions: {e}")
            return None
        
def _read_input_data(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(input_path)
    if ext in [".csv"]:
        return pd.read_csv(input_path)
    if ext in [".json"]:
        with open(input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.DataFrame(obj)
    raise ValueError("Format non supporté. Utilisez .csv, .parquet ou .json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Effectuer des prédictions avec un modèle entraîné")
    parser.add_argument("--model_path", type=str, default=None, help="Chemin vers un modèle .pkl")
    parser.add_argument("--models_dir", type=str, default="models", help="Répertoire contenant les modèles")
    parser.add_argument("--input_path", type=str, required=True, help="Chemin vers les nouvelles données (csv/parquet/json)")
    parser.add_argument("--output_path", type=str, default=None, help="Chemin de sortie pour les prédictions (csv)")
    parser.add_argument("--best", action="store_true", help="Utiliser le meilleur modèle du dossier models_dir")
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "accuracy"], help="Métrique pour best model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil pour predicted_failure")
    parser.add_argument("--risk_levels", type=int, default=5, help="Nombre de niveaux de risque (ex: 5)")
    
    args = parser.parse_args()
    
    engine = PredictionEngine(model_path=args.model_path, models_dir=args.models_dir)

    # Sélection automatique du modèle
    if engine.model is None:
        if args.best:
            best_path = engine.find_best_model(metric=args.metric)
            if best_path is None:
                raise SystemExit("Aucun modèle trouvé pour best=True")
            if not engine.load_model(best_path):
                raise SystemExit("Impossible de charger le meilleur modèle")
        else:
            latest_path = engine.find_latest_model()
            if latest_path is None:
                raise SystemExit("Aucun modèle trouvé")
            if not engine.load_model(latest_path):
                raise SystemExit("Impossible de charger le modèle le plus récent")

    # Charger les nouvelles données
    data_in = _read_input_data(args.input_path)

    # Prédire + risk level
    results = engine.predict_with_risk_levels(
        data=data_in,
        levels=args.risk_levels,
        threshold=args.threshold
    )

    # Sauvegarder
    engine.save_predictions(results, output_path=args.output_path)