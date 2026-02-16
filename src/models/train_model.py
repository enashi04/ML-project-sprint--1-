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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, average_precision_score
)
import logging
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
    
    def __init__(
        self,
        data_path,
        models_dir="models",
        test_size=0.2,
        random_state=42,
        use_gpu=False,
        n_jobs=1,
        safe_mode=True
    ):
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
        self.n_jobs = n_jobs
        self.safe_mode = safe_mode
        
        # Création du répertoire pour les modèles s'il n'existe pas
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Détection RAPIDS cuML
        try:
            import cuml  # noqa: F401
            from cuml.ensemble import RandomForestClassifier as cuRF
            GPU_AVAILABLE = True
        except ImportError:
            GPU_AVAILABLE = False
            cuRF = None
            logger.info("RAPIDS cuML n'est pas disponible, utilisation des CPU fallbacks")

        self.use_gpu = bool(use_gpu and GPU_AVAILABLE)

        # Grilles "safe" pour éviter de faire freezer la machine
        # -> tu peux élargir après validation
        if self.use_gpu:
            self.models = {
                'xgboost': {
                    'model': xgb.XGBClassifier(
                        tree_method='gpu_hist',
                        gpu_id=0,
                        random_state=random_state,
                        eval_metric='logloss'
                    ),
                    'params': {
                        'n_estimators': [200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMClassifier(device='gpu', random_state=random_state),
                    'params': {
                        'n_estimators': [300],
                        'learning_rate': [0.05, 0.1],
                        'num_leaves': [31, 63]
                    }
                },
            }

            # Optionnel: cuML RF si dispo (mais attention RAM GPU)
            if cuRF is not None:
                self.models['random_forest'] = {
                    'model': cuRF(random_state=random_state),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [10, 20],
                        # WARNING: cuML ne supporte pas tous les params sklearn
                        # 'min_samples_split': [2, 5]
                    }
                }

        else:
            self.models = {
                'logistic_regression': {
                    'model': LogisticRegression(
                        random_state=random_state,
                        max_iter=3000,
                        n_jobs=self.n_jobs,
                        solver='lbfgs',
                        class_weight='balanced'  # IMPORTANT: dataset déséquilibré
                    ),
                    'params': {
                        'C': [0.1, 1.0, 10.0]
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(
                        random_state=random_state,
                        n_jobs=self.n_jobs,
                        class_weight='balanced'  # IMPORTANT: dataset déséquilibré
                    ),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=random_state),
                    'params': {
                        'n_estimators': [200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5]
                    }
                },
                'svm': {
                    'model': SVC(
                        random_state=random_state,
                        probability=True,
                        class_weight='balanced'  # IMPORTANT: dataset déséquilibré
                    ),
                    'params': {
                        'C': [1.0, 2.0],
                        'kernel': ['rbf', 'linear']
                    }
                },
                'xgboost': {
                    'model': xgb.XGBClassifier(
                        tree_method='hist',
                        random_state=random_state,
                        eval_metric='logloss',
                        nthread=self.n_jobs  # <- évite d'utiliser tous les coeurs par surprise
                    ),
                    'params': {
                        'n_estimators': [300],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMClassifier(device='cpu', random_state=random_state),
                    'params': {
                        'n_estimators': [300],
                        'learning_rate': [0.05, 0.1],
                        'num_leaves': [31, 63]
                    }
                }
            }
        
    def load_data(self, sample_rows=None):
        """Charge les données prétraitées depuis le chemin spécifié."""
        logger.info(f"Chargement des données depuis {self.data_path}")
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Données chargées avec succès: {data.shape} échantillons")

            # Optionnel: tester sur un sous-ensemble pour éviter de faire planter le PC
            if sample_rows is not None and sample_rows > 0 and sample_rows < len(data):
                data = data.sample(n=sample_rows, random_state=self.random_state)
                logger.warning(f"Mode sample activé: utilisation de {sample_rows} lignes")

            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise

    def _compute_scale_pos_weight(self, y):
        """
        Calcule un poids de classe pour XGBoost/LightGBM: nb_negatifs / nb_positifs
        """
        y = pd.Series(y).astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def _drop_leakage_columns(self, X, target_column):
        """
        Supprime les colonnes qui fuient de l'information (futur / post-évènement).
        IMPORTANT: on ne doit jamais drop la cible elle-même (elle n'est pas dans X ici).
        """
        leak_cols = []

        # Cible "future" (dérivée de time_to_failure) -> enlever toutes colonnes qui donnent le futur
        if target_column == 'failure_within_24h':
            if 'time_to_failure' in X.columns:
                leak_cols.append('time_to_failure')
            if 'failure_soon' in X.columns:
                leak_cols.append('failure_soon')
            leak_cols += [c for c in X.columns if c.startswith("next_failure_type_")]
            leak_cols += [c for c in X.columns if c.startswith("component_affected_")]

        # Cible "failure_soon" -> enlever time_to_failure et les infos "panne suivante"
        if target_column == 'failure_soon':
            if 'time_to_failure' in X.columns:
                leak_cols.append('time_to_failure')
            leak_cols += [c for c in X.columns if c.startswith("next_failure_type_")]
            leak_cols += [c for c in X.columns if c.startswith("component_affected_")]

        if leak_cols:
            leak_cols = sorted(set(leak_cols))
            logger.warning(f"Colonnes supprimées pour éviter fuite de label: {leak_cols}")
            X = X.drop(columns=[c for c in leak_cols if c in X.columns])

        return X
    
    def _sanitize_X_y(self, X, y):
        """
        Nettoie X/y : supprime colonnes non numériques, remplace NaN/inf, cast y en int.
        """
        # Sécurité: retirer toute colonne non numérique si jamais il en reste
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            logger.warning(f"Colonnes non numériques supprimées (sécurité): {non_numeric}")
            X = X.drop(columns=non_numeric)

        X = X.replace((np.inf, -np.inf, np.nan), 0)
        y = pd.Series(y).replace((np.inf, -np.inf, np.nan), 0).astype(int)

        return X, y
            
    def prepare_train_test_data(self, data, target_column='failure_within_24h'):
        """
        Prépare les ensembles d'entraînement et de test.
        """
        logger.info("Préparation des ensembles d'entraînement et de test")
        
        # Vérifier que la colonne cible existe
        if target_column not in data.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans les données")
        
        # Séparation des caractéristiques et de la cible
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # IMPORTANT: éviter la fuite de label
        X = self._drop_leakage_columns(X, target_column)

        # Vérifier la distribution des classes
        X, y = self._sanitize_X_y(X, y)
        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Distribution des classes: {class_distribution.to_dict()}")
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)

        logger.info(f"Ensemble d'entraînement: {X_train.shape} échantillons")
        logger.info(f"Ensemble de test: {X_test.shape} échantillons")

        # Calcul du scale_pos_weight (utile pour XGB/LGBM)
        spw = self._compute_scale_pos_weight(y_train)
        logger.info(f"scale_pos_weight (neg/pos) calculé sur train: {spw:.2f}")
        
        return X_train, X_test, y_train, y_test

    def prepare_train_test_data_from_files(self, train_df, test_df, target_column='failure_within_24h'):
        """
        Prépare train/test quand on a déjà un split temporel (train + test séparés).
        IMPORTANT: appliquer le même nettoyage/leakage drop que prepare_train_test_data !
        """
        logger.info("Préparation des ensembles d'entraînement et de test (split temporel: fichiers séparés)")

        if target_column not in train_df.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le TRAIN")
        if target_column not in test_df.columns:
            raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le TEST")

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # IMPORTANT: éviter la fuite de label (sur TRAIN et TEST)
        X_train = self._drop_leakage_columns(X_train, target_column)
        X_test = self._drop_leakage_columns(X_test, target_column)

        # IMPORTANT: aligner les colonnes entre train et test (one-hot, etc.)
        X_train, y_train = self._sanitize_X_y(X_train, y_train)
        X_test, y_test = self._sanitize_X_y(X_test, y_test)

        # Si colonnes différentes -> on aligne (colonnes manquantes = 0)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # Sécurité: si test a des colonnes que train n'a pas, on les drop
        extra_in_test = [c for c in X_test.columns if c not in X_train.columns]
        if extra_in_test:
            logger.warning(f"Colonnes présentes uniquement dans le TEST supprimées: {extra_in_test}")
            X_test = X_test.drop(columns=extra_in_test)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = pd.Series(y_train).astype(int).reset_index(drop=True)
        y_test = pd.Series(y_test).astype(int).reset_index(drop=True)

        # Logs distribution classes
        class_distribution_train = y_train.value_counts(normalize=True)
        class_distribution_test = y_test.value_counts(normalize=True)
        logger.info(f"Distribution des classes (train): {class_distribution_train.to_dict()}")
        logger.info(f"Distribution des classes (test): {class_distribution_test.to_dict()}")

        logger.info(f"Ensemble d'entraînement (split temporel): {X_train.shape} échantillons")
        logger.info(f"Ensemble de test (split temporel): {X_test.shape} échantillons")

        spw = self._compute_scale_pos_weight(y_train)
        logger.info(f"scale_pos_weight (neg/pos) calculé sur train: {spw:.2f}")

        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, models_to_train=None, cv=3):
        """
        Entraîne les modèles spécifiés avec recherche d'hyperparamètres.
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        trained_models = {}

        # Poids classe pour XGB/LGBM
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        
        for model_name in models_to_train:
            if model_name not in self.models:
                logger.warning(f"Modèle '{model_name}' non reconnu. Ignoré.")
                continue
                
            logger.info(f"Entraînement du modèle: {model_name}")
            model_info = self.models[model_name]
            model = model_info['model']

            # Injecter le scale_pos_weight quand c'est pertinent
            if model_name in ("xgboost", "lightgbm"):
                try:
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    logger.info(f"{model_name}: scale_pos_weight={scale_pos_weight:.2f} appliqué")
                except Exception as e:
                    logger.warning(f"Impossible d'appliquer scale_pos_weight sur {model_name}: {e}")
            
            # Recherche des meilleurs hyperparamètres
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=model_info['params'],
                cv=cv,
                scoring='average_precision',  # mieux que roc_auc en dataset déséquilibré
                n_jobs=self.n_jobs,          # <- IMPORTANT: éviter n_jobs=-1 si ça freeze
                verbose=1
            )
            
            try:
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                logger.info(f"Meilleurs paramètres pour {model_name}: {best_params}")
                logger.info(f"Meilleur score de validation croisée (PR-AUC): {best_score:.4f}")
                
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
        """
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            logger.info(f"Évaluation du modèle: {model_name}")
            
            # Prédictions
            y_pred = model.predict(X_test)

            # Probabilités (si disponibles)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                y_pred_proba = scores
            else:
                y_pred_proba = y_pred.astype(float)
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            logger.info(f"Précision sur l'ensemble de test: {accuracy:.4f}")
            logger.info(f"AUC sur l'ensemble de test: {auc_score:.4f}")
            logger.info(f"PR-AUC sur l'ensemble de test: {pr_auc:.4f}")
            logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            logger.info(f"Matrice de confusion:\n{conf_matrix}")
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pr_auc': pr_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'auc': auc_score
            }
        
        return evaluation_results
    
    def save_models(self, trained_models, evaluation_results, features_info=None):
        """
        Sauvegarde les modèles entraînés et leurs résultats d'évaluation.
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
        
        return model_paths
    
    def find_best_model(self, evaluation_results, metric='auc'):
        """
        Trouve le meilleur modèle selon la métrique spécifiée.
        """
        if metric not in ("auc", "pr_auc", "f1", "recall", "accuracy"):
            logger.warning(f"Métrique '{metric}' non reconnue, fallback sur 'auc'")
            metric = "auc"

        scores = {m: evaluation_results[m].get(metric, float("-inf")) for m in evaluation_results.keys()}
        best_model = max(scores, key=scores.get)
        logger.info(f"Meilleur modèle selon {metric}: {best_model} avec un score de {scores[best_model]:.4f}")
        return best_model


def train_and_evaluate(data_path, target_column='failure_within_24h', models_to_train=None, 
                    models_dir="models", test_size=0.2, random_state=42, cv=5, use_gpu=False, n_jobs=1, sample_rows=None, test_path=None):
    """
    Fonction principale pour entraîner et évaluer les modèles.
    """
    trainer = ModelTrainer(
        data_path=data_path,
        models_dir=models_dir,
        test_size=test_size,
        random_state=random_state,
    )

    # Si test_path est fourni (ou détecté), on utilise split temporel
    # Sinon fallback: train_test_split classique
    if test_path is None:
        # Détection automatique: si un fichier *_test_data.csv existe à côté de data_path
        base, ext = os.path.splitext(data_path)
        candidate = f"{base.replace('featured_data', 'featured_data')}_test_data{ext}"
        if os.path.exists(candidate):
            test_path = candidate

    if test_path is not None and os.path.exists(test_path):
        logger.info(f"Split temporel détecté -> TRAIN={data_path} | TEST={test_path}")
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(test_path)

        X_train, X_test, y_train, y_test = trainer.prepare_train_test_data_from_files(
            train_df=train_df,
            test_df=test_df,
            target_column=target_column
        )
    else:
        #on commente pour l'instant
        data = trainer.load_data(sample_rows=sample_rows)
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_data(
            data=data,
            target_column=target_column
        )

    trained_models = trainer.train_models(
        X_train=X_train,
        y_train=y_train,
        models_to_train=models_to_train,
        cv=cv
    )
    
    evaluation_results = trainer.evaluate_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test
    )
    
    best_model = trainer.find_best_model(evaluation_results, metric="pr_auc")
    
    features_info = {
        'feature_names': list(X_train.columns),
        'n_features': X_train.shape[1]
    }
    
    model_paths = trainer.save_models(
        trained_models=trained_models,
        evaluation_results=evaluation_results,
        features_info=features_info
    )
    
    return trained_models, evaluation_results, model_paths, best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner des modèles de prédiction de défaillance industrielle")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers les données prétraitées")
    parser.add_argument("--test_path", type=str, default=None, help="Chemin vers le fichier de test (split temporel)")
    parser.add_argument("--target_column", type=str, default="failure_within_24h", help="Nom de la colonne cible")
    parser.add_argument("--models_dir", type=str, default="models", help="Répertoire pour sauvegarder les modèles")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion des données pour le test")
    parser.add_argument("--random_state", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    parser.add_argument("--cv", type=int, default=3, help="Nombre de plis pour la validation croisée")
    parser.add_argument("--n_jobs", type=int, default=1, help="Nombre de jobs parallèles (éviter -1 si freeze)")
    parser.add_argument("--use_gpu", action="store_true", help="Activer GPU si disponible")
    parser.add_argument("--sample_rows", type=int, default=None, help="Nombre de lignes à échantillonner pour test")

    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["random_forest", "gradient_boosting", "logistic_regression", "svm", "xgboost", "lightgbm"],
                        help="Modèles à entraîner (tous par défaut)")
    
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        test_path=args.test_path,
        target_column=args.target_column,
        models_to_train=args.models,
        models_dir=args.models_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
        sample_rows=args.sample_rows
    )
