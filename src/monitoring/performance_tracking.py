"""
Module de suivi des performances pour le projet de prédiction de risque de défaillance industrielle.

Ce module permet de surveiller l'évolution des performances des modèles au fil du temps,
de comparer différentes versions des modèles et de générer des alertes en cas de dégradation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from datetime import datetime
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


class ModelPerformanceTracker:
    """
    Classe pour le suivi des performances des modèles au fil du temps.
    """
    
    def __init__(self, model_name: str, model_version: str,
                 is_classification: bool = True,
                 baseline_metrics: Optional[Dict] = None,
                 output_dir: str = "performance_reports",
                 alert_threshold: float = 0.1):
        """
        Initialise le tracker de performance.
        
        Args:
            model_name: Nom du modèle suivi
            model_version: Version du modèle
            is_classification: True si c'est un modèle de classification, False pour régression
            baseline_metrics: Métriques de référence pour comparer (optionnel)
            output_dir: Répertoire où sauvegarder les rapports de performance
            alert_threshold: Seuil de dégradation des performances pour déclencher une alerte (en proportion)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.is_classification = is_classification
        self.baseline_metrics = baseline_metrics
        self.output_dir = output_dir
        self.alert_threshold = alert_threshold
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Historique des métriques
        self.metrics_history = []
        
        # Charger l'historique existant si disponible
        self._load_history()
    
    def _load_history(self):
        """
        Charge l'historique des métriques depuis un fichier JSON s'il existe.
        """
        history_file = f"{self.output_dir}/{self.model_name}_{self.model_version}_history.json"
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.metrics_history = json.load(f)
                print(f"Historique chargé avec {len(self.metrics_history)} entrées.")
            except Exception as e:
                print(f"Erreur lors du chargement de l'historique: {e}")
    
    def _save_history(self):
        """
        Sauvegarde l'historique des métriques dans un fichier JSON.
        """
        history_file = f"{self.output_dir}/{self.model_name}_{self.model_version}_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            print(f"Historique sauvegardé dans: {history_file}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'historique: {e}")
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calcule les métriques de performance pour un modèle de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions (classes)
            y_prob: Probabilités prédites (pour les métriques comme AUC-ROC)
            
        Returns:
            Dictionnaire des métriques calculées
        """
        metrics = {
            'accuracy': 
            'precision': 
            'recall': 
            'f1': 
        }
        
        # Classification binaire avec probabilités
        if y_prob is not None and len(np.unique(y_true)) == 2:
            # S'assurer que y_prob est un tableau de probabilités pour la classe positive
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                # Si probas multi-classes, prendre la colonne pour la classe positive (1)
                proba_positive = y_prob[:, 1]
            else:
                proba_positive = y_prob
                
            metrics['auc_roc'] = roc_auc_score(y_true, proba_positive)
        
        # Confusion matrix (convertir en liste pour la sérialisation JSON)
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcule les métriques de performance pour un modèle de régression.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            Dictionnaire des métriques calculées
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def track_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None,
                         dataset_name: str = "validation",
                         custom_metrics: Optional[Dict] = None) -> Dict:
        """
        Calcule et enregistre les métriques de performance actuelles.
        
        Args:
            y_true: Valeurs/classes réelles
            y_pred: Valeurs/classes prédites
            y_prob: Probabilités prédites (pour classification)
            dataset_name: Nom du dataset évalué (ex: "validation", "production")
            custom_metrics: Métriques personnalisées à ajouter au rapport
            
        Returns:
            Rapport de performance contenant les métriques calculées
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculer les métriques selon le type de modèle
        if self.is_classification:
            metrics = self._calculate_classification_metrics(y_true, y_pred, y_prob)
        else:
            metrics = self._calculate_regression_metrics(y_true, y_pred)
        
        # Ajouter les métriques personnalisées
        if custom_metrics:
            metrics.update(custom_metrics)
        
        # Créer le rapport de performance
        performance_report = {
            'timestamp': timestamp,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'dataset': dataset_name,
            'sample_size': len(y_true),
            'metrics': metrics
        }
        
        # Vérifier si les performances se dégradent par rapport à la baseline
        if self.baseline_metrics:
            degradation = self._check_degradation(metrics)
            performance_report['degradation'] = degradation
        
        # Ajouter à l'historique et sauvegarder
        self.metrics_history.append(performance_report)
        self._save_history()
        
        # Créer un rapport individuel pour cette évaluation
        report_file = f"{self.output_dir}/{self.model_name}_{self.model_version}_{dataset_name}_{timestamp.replace(' ', '_').replace(':', '-')}.json"
        with open(report_file, 'w') as f:
            json.dump(performance_report, f, indent=4)
        
        return performance_report
    
    def _check_degradation(self, current_metrics: Dict) -> Dict:
        """
        Vérifie si les métriques actuelles se sont dégradées par rapport à la baseline.
        
        Args:
            current_metrics: Métriques actuelles
            
        Returns:
            Rapport de dégradation
        """
        degradation_report = {'has_degradation': False}
        metrics_diff = {}
        
        # Pour chaque métrique présente dans les deux
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics and metric not in ['confusion_matrix']:
                current_value = current_metrics[metric]
                
                # Calculer la différence (positive = amélioration, négative = dégradation)
                # Pour MSE, RMSE, MAE une valeur plus basse est meilleure
                if metric in ['mse', 'rmse', 'mae']:
                    diff = baseline_value - current_value
                    rel_diff = diff / baseline_value if baseline_value else float('inf')
                    degraded = diff < 0 and abs(rel_diff) > self.alert_threshold
                else:
                    # Pour les autres métriques (accuracy, precision, recall, f1, r2, etc.)
                    diff = current_value - baseline_value
                    rel_diff = diff / baseline_value if baseline_value else float('inf')
                    degraded = diff < 0 and abs(rel_diff) > self.alert_threshold
                
                metrics_diff[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'difference': diff,
                    'relative_difference': rel_diff,
                    'degraded': degraded
                }
                
                if degraded:
                    degradation_report['has_degradation'] = True
        
        degradation_report['metrics_diff'] = metrics_diff
        
        return degradation_report
    
    def visualize_performance_trend(self, metric_name: str = 'f1') -> None:
        """
        Crée une visualisation de l'évolution d'une métrique au fil du temps.
        
        Args:
            metric_name: Nom de la métrique à visualiser
        """
        if not self.metrics_history:
            print("Pas d'historique de performances disponible.")
            return
        
        # Extraire les timestamps et les valeurs de la métrique
        timestamps = []
        values = []
        
        for entry in self.metrics_history:
            if metric_name in entry['metrics']:
                timestamps.append(datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S"))
                values.append(entry['metrics'][metric_name])
        
        if not timestamps:
            print(f"La métrique '{metric_name}' n'est pas disponible dans l'historique.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Ajouter une ligne horizontale pour la valeur de référence si disponible
        if self.baseline_metrics and metric_name in self.baseline_metrics:
            plt.axhline(y=self.baseline_metrics[metric_name], color='r', linestyle='--', 
                        label=f'Baseline ({self.baseline_metrics[metric_name]:.4f})')
        
        plt.title(f'Évolution de {metric_name} - {self.model_name} v{self.model_version}')
        plt.xlabel('Date')
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if self.baseline_metrics and metric_name in self.baseline_metrics:
            plt.legend()
        
        # Sauvegarder le graphique
        fig_path = f"{self.output_dir}/{self.model_name}_{self.model_version}_{metric_name}_trend.png"
        plt.savefig(fig_path)
        
        print(f"Visualisation sauvegardée dans: {fig_path}")
        plt.show()
    
    def compare_with_baseline(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Compare les performances actuelles avec la baseline et génère un rapport détaillé.
        
        Args:
            y_true: Valeurs/classes réelles
            y_pred: Valeurs/classes prédites
            y_prob: Probabilités prédites (pour classification)
            
        Returns:
            Rapport de comparaison
        """
        # Calculer les métriques actuelles
        if self.is_classification:
            current_metrics = 
        else:
            current_metrics = 
        
        # Si pas de baseline, on ne peut pas comparer
        if not self.baseline_metrics:
            return {'current_metrics': current_metrics, 'comparison': 'Pas de baseline disponible'}
        
        # Comparer avec la baseline
        
        # Créer le rapport de comparaison
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'sample_size': len(y_true),
            'current_metrics': current_metrics,
            'comparison': comparison
        }
        
        # Sauvegarder le rapport
        timestamp = report['timestamp'].replace(' ', '_').replace(':', '-')
        report_file = f"{self.output_dir}/{self.model_name}_{self.model_version}_comparison_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
    
    def set_baseline(self, metrics: Dict = None, y_true: np.ndarray = None, 
                     y_pred: np.ndarray = None, y_prob: np.ndarray = None) -> None:
        """
        Définit les métriques de référence (baseline).
        
        Args:
            metrics: Dictionnaire de métriques à utiliser comme référence
            y_true: Valeurs/classes réelles pour calculer de nouvelles métriques de référence
            y_pred: Valeurs/classes prédites pour calculer de nouvelles métriques de référence
            y_prob: Probabilités prédites pour calculer de nouvelles métriques de référence
        """
        if metrics:
            self.baseline_metrics = metrics
        elif y_true is not None and y_pred is not None:
            # Calculer les métriques selon le type de modèle
            if self.is_classification:
                self.baseline_metrics = self._calculate_classification_metrics(y_true, y_pred, y_prob