"""
Module pour l'entraînement des modèles de prédiction de défaillance industrielle.
Ce script prend en charge le chargement des données prétraitées, l'entraînement 
de différents modèles et leur sauvegarde.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import joblib
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour entraîner différents modèles de machine learning."""
    
    def __init__(self, data_path, models_dir="models", test_size=0.2, random_state=42, use_gpu=True):
        """
        Initialise le ModelTrainer.
        
        Args:
            data_path (str): Chemin vers les données prétraitées
            models_dir (str): Répertoire pour sauvegarder les modèles
            test_size (float): Proportion des données pour le test
            random_state (int): Graine aléatoire pour la reproductibilité
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # Création du répertoire pour les modèles s'il n'existe pas
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Option 1: avec RAPIDS cuML (si disponible)
        try:
            import cuml
            from cuml.ensemble import RandomForestClassifier as cuRF
            from cuml.linear_model import LogisticRegression as cuLR
            from cuml.svm import SVC as cuSVC
            GPU_AVAILABLE = True
        except ImportError:
            GPU_AVAILABLE = False
            print("RAPIDS cuML n'est pas disponible, utilisation des CPU fallbacks")

        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            self.models = {
                'random_forest': {
                    'model': cuRF(random_state=random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                # Autres modèles GPU...
                'xgboost': {
                    'model': xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMClassifier(device='gpu', random_state=random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100]
                    }
                }
            }

        else:
            self.models = {
                
            }
        
    def load_data(self):
        """Charge les données prétraitées depuis le chemin spécifié."""
        logger.info(f"Chargement des données depuis {self.data_path}")
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Données chargées avec succès: {data.shape} échantillons")
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
            
    def prepare_train_test_data(self, data, target_column='failure_within_24h'):
        """
        Prépare les ensembles d'entraînement et de test.
        
        Args:
            data (DataFrame): DataFrame contenant les données prétraitées
            target_column (str): Nom de la colonne cible
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Préparation des ensembles d'entraînement et de test")
        
        # Vérifier que la colonne cible existe
        if target_column not in data.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans les données")
        
        # Séparation des caractéristiques et de la cible
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Vérifier la distribution des classes
        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Distribution des classes: {class_distribution.to_dict()}")
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
        X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
        y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
        y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


        logger.info(f"Ensemble d'entraînement: {X_train.shape} échantillons")
        logger.info(f"Ensemble de test: {X_test.shape} échantillons")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, models_to_train=None, cv=5):
        """
        Entraîne les modèles spécifiés avec recherche d'hyperparamètres.
        
        Args:
            X_train (DataFrame): Caractéristiques d'entraînement
            y_train (Series): Cibles d'entraînement
            models_to_train (list): Liste des modèles à entraîner (None pour tous)
            cv (int): Nombre de plis pour la validation croisée
            
        Returns:
            dict: Dictionnaire des modèles entraînés
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        trained_models = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                logger.warning(f"Modèle '{model_name}' non reconnu. Ignoré.")
                continue
                
            logger.info(f"Entraînement du modèle: {model_name}")
            model_info = self.models[model_name]
            
            # Recherche des meilleurs hyperparamètres
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            try:
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                logger.info(f"Meilleurs paramètres pour {model_name}: {best_params}")
                logger.info(f"Meilleur score de validation croisée (AUC): {best_score:.4f}")
                
                trained_models[model_name] = {
                    'model': best_model,
                    'params': best_params,
                    'cv_score': best_score
                }
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement du modèle {model_name}: {e}")
        
        return trained_models
    
    def evaluate_models(self, trained_models, X_test, y_test):
        """
        Évalue les modèles entraînés sur l'ensemble de test.
        test_data.csv
        Args:
            trained_models (dict): Modèles entraînés
            X_test (DataFrame): Caractéristiques de test
            y_test (Series): Cibles de test
            
        Returns:
            dict: Résultats d'évaluation
        """
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            logger.info(f"Évaluation du modèle: {model_name}")
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques
            accuracy = 
            conf_matrix = 
            class_report = 
            auc_score = 
            
            logger.info(f"Précision sur l'ensemble de test: {accuracy:.4f}")
            logger.info(f"AUC sur l'ensemble de test: {auc_score:.4f}")
            logger.info(f"Matrice de confusion:\n{conf_matrix}")
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'auc': auc_score
            }
        
        return evaluation_results
    
    def save_models(self, trained_models, evaluation_results, features_info=None):
        """
        Sauvegarde les modèles entraînés et leurs résultats d'évaluation.
        
        Args:
            trained_models (dict): Modèles entraînés
            evaluation_results (dict): Résultats d'évaluation
            features_info (dict): Informations sur les caractéristiques utilisées
            
        Returns:
            dict: Chemins des modèles sauvegardés
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_paths = {}
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            # Créer un dictionnaire avec toutes les informations du modèle
            model_data = {
                'model': model,
                'parameters': model_info['params'],
                'cv_score': model_info['cv_score'],
                'evaluation': evaluation_results[model_name],
                'features_info': features_info,
                'timestamp': timestamp
            }
            
            # Créer le chemin de sauvegarde
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
            
            # Sauvegarder le modèle
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"Modèle {model_name} sauvegardé à {model_path}")
                model_paths[model_name] = model_path
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du modèle {model_name}: {e}")
        
        # Sauvegarder un fichier récapitulatif
        summary_path = os.path.join(self.models_dir, f"training_summary_{timestamp}.pkl")
        try:
            summary_data = {
                'models': list(trained_models.keys()),
                'evaluation_summary': {model: {'auc': eval_info['auc'], 'accuracy': eval_info['accuracy']} 
                                      for model, eval_info in evaluation_results.items()},
                'timestamp': timestamp,
                'features_info': features_info
            }
            with open(summary_path, 'wb') as f:
                pickle.dump(summary_data, f)
            logger.info(f"Résumé de l'entraînement sauvegardé à {summary_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du résumé d'entraînement: {e}")
        
        return model_paths
    
    def find_best_model(self, evaluation_results, metric='auc'):
        """
        Trouve le meilleur modèle selon la métrique spécifiée.
        
        Args:
            evaluation_results (dict): Résultats d'évaluation
            metric (str): Métrique à utiliser ('auc' ou 'accuracy')
            
        Returns:
            str: Nom du meilleur modèle
        """
        scores = 
        best_model = 
        logger.info(f"Meilleur modèle selon {metric}: {best_model} avec un score de {scores[best_model]:.4f}")
        return best_model
        
    def save_feature_importance(self, trained_models, feature_names, top_n=20):
        """
        Sauvegarde l'importance des caractéristiques pour les modèles qui le supportent.
        
        Args:
            trained_models (dict): Modèles entraînés
            feature_names (list): Noms des caractéristiques
            top_n (int): Nombre de caractéristiques importantes à sauvegarder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            # Vérifier si le modèle a un attribut feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Préparer un DataFrame des importances
                top_indices = indices[:top_n]
                importance_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_indices],
                    'Importance': importances[top_indices]
                })
                
                # Sauvegarder en CSV
                importance_path = os.path.join(
                    self.models_dir, 
                    f"{model_name}_feature_importance_{timestamp}.csv"
                )
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Importance des caractéristiques pour {model_name} sauvegardée à {importance_path}")
            
            elif hasattr(model, 'coef_'):
                # Pour les modèles linéaires
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                indices = np.argsort(np.abs(coefs))[::-1]
                
                # Préparer un DataFrame des coefficients
                top_indices = indices[:top_n]
                coef_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_indices],
                    'Coefficient': coefs[top_indices]
                })
                
                # Sauvegarder en CSV
                coef_path = os.path.join(
                    self.models_dir, 
                    f"{model_name}_coefficients_{timestamp}.csv"
                )
                coef_df.to_csv(coef_path, index=False)
                logger.info(f"Coefficients pour {model_name} sauvegardés à {coef_path}")

def train_and_evaluate(data_path, target_column='failure_within_24h', models_to_train=None, 
                      models_dir="models", test_size=0.2, random_state=42, cv=5):
    """
    Fonction principale pour entraîner et évaluer les modèles.
    
    Args:
        data_path (str): Chemin vers les données prétraitées
        target_column (str): Nom de la colonne cible
        models_to_train (list): Liste des modèles à entraîner
        models_dir (str): Répertoire pour sauvegarder les modèles
        test_size (float): Proportion des données pour le test
        random_state (int): Graine aléatoire pour la reproductibilité
        cv (int): Nombre de plis pour la validation croisée
        
    Returns:
        tuple: (trained_models, evaluation_results, model_paths)
    """
    # Initialiser le ModelTrainer
    trainer = ModelTrainer(
        data_path=data_path,
        models_dir=models_dir,
        test_size=test_size,
        random_state=random_state
    )
    
    # Charger les données
    data = trainer.load_data()
    
    # Préparer les ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = trainer.prepare_train_test_data(
        data=data,
        target_column=target_column
    )
    
    # Entraîner les modèles
    trained_models = trainer.train_models(
        X_train=X_train,
        y_train=y_train,
        models_to_train=models_to_train,
        cv=cv
    )
    
    # Évaluer les modèles
    evaluation_results = trainer.evaluate_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test
    )
    
    # Trouver le meilleur modèle
    best_model = trainer.find_best_model(evaluation_results)
    
    # Informations sur les caractéristiques
    features_info = {
        'feature_names': list(X_train.columns),
        'n_features': X_train.shape[1]
    }
    
    # Sauvegarder les modèles et les résultats
    model_paths = trainer.save_models(
        trained_models=trained_models,
        evaluation_results=evaluation_results,
        features_info=features_info
    )
    
    # Sauvegarder l'importance des caractéristiques
    trainer.save_feature_importance(
        trained_models=trained_models,
        feature_names=list(X_train.columns)
    )
    
    return trained_models, evaluation_results, model_paths, best_model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner des modèles de prédiction de défaillance industrielle")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers les données prétraitées")
    parser.add_argument("--target_column", type=str, default="failure_within_24h", help="Nom de la colonne cible")
    parser.add_argument("--models_dir", type=str, default="models", help="Répertoire pour sauvegarder les modèles")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion des données pour le test")
    parser.add_argument("--random_state", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis pour la validation croisée")
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["random_forest", "gradient_boosting", "logistic_regression", "svm"],
                        help="Modèles à entraîner (tous par défaut)")
    
    args = parser.parse_args()
    


    train_and_evaluate(
        data_path=args.data_path,
        target_column=args.target_column,
        models_to_train=args.models,
        models_dir=args.models_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv
    )
