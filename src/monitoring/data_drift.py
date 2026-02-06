"""
Module de détection et d'analyse du data drift pour le projet de 
prédiction de risque de défaillance industrielle.

Ce module permet de surveiller les changements dans les distributions 
des données d'entrée au fil du temps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Union


class DataDriftMonitor:
    """
    Classe pour la détection et l'analyse du data drift.
    """
    
    def __init__(self, reference_data: pd.DataFrame, 
                 drift_threshold: float = 0.05,
                 output_dir: str = "drift_reports"):
        """
        Initialise le moniteur de data drift.
        
        Args:
            reference_data: Données de référence (généralement les données d'entraînement)
            drift_threshold: Seuil p-value pour considérer qu'il y a un drift (défaut: 0.05)
            output_dir: Répertoire où sauvegarder les rapports de drift
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.output_dir = output_dir
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculer et stocker les statistiques de référence
        self.reference_stats = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calcule les statistiques descriptives pour chaque feature.
        
        Args:
            data: DataFrame contenant les données à analyser
            
        Returns:
            Dictionnaire avec les statistiques par feature
        """
        stats = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats[col] = {
                    'mean': data[col].mean(),
                    'median': data[col].median(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'q1': data[col].quantile(0.25),
                    'q3': data[col].quantile(0.75),
                    'missing': data[col].isna().sum() / len(data)
                }
            else:
                # Pour les colonnes catégorielles
                value_counts = data[col].value_counts(normalize=True).to_dict()
                stats[col] = {
                    'value_counts': value_counts,
                    'missing': data[col].isna().sum() / len(data),
                    'n_categories': data[col].nunique()
                }
                
        return stats
        
    def detect_drift(self, new_data: pd.DataFrame) -> Dict:
        """
        Détecte le drift entre les données de référence et les nouvelles données.
        
        Args:
            new_data: Nouvelles données à comparer avec la référence
            
        Returns:
            Rapport de drift contenant les résultats des tests et métriques
        """
        drift_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_size': len(new_data),
            'features': {},
            'overall_drift_detected': False
        }
        
        # Calculer les statistiques des nouvelles données
        new_stats = self._calculate_statistics(new_data)
        
        # Comparer les caractéristiques une par une
        features_with_drift = 0
        
        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue
                
            feature_report = {'drift_detected': False}
            
            # Pour les features numériques, utiliser le test KS
            if pd.api.types.is_numeric_dtype(self.reference_data[col]) and pd.api.types.is_numeric_dtype(new_data[col]):
                # Suppression des valeurs manquantes pour le test KS
                ref_values = self.reference_data[col].dropna()
                new_values = new_data[col].dropna()
                
                if len(ref_values) > 0 and len(new_values) > 0:
                    ks_statistic, p_value = ks_2samp(ref_values, new_values)
                    
                    feature_report['ks_statistic'] = ks_statistic
                    feature_report['p_value'] = p_value
                    feature_report['drift_detected'] = p_value < self.drift_threshold
                    
                    # Calculer la différence relative des statistiques
                    rel_diff = {}
                    for stat in ['mean', 'median', 'std']:
                        if stat in self.reference_stats[col] and stat in new_stats[col]:
                            if self.reference_stats[col][stat] != 0:
                                rel_diff[stat] = (new_stats[col][stat] - self.reference_stats[col][stat]) / self.reference_stats[col][stat]
                            else:
                                rel_diff[stat] = float('inf') if new_stats[col][stat] != 0 else 0
                    
                    feature_report['relative_differences'] = rel_diff
            
            # Pour les features catégorielles, comparer les distributions
            elif col in self.reference_stats and col in new_stats:
                if 'value_counts' in self.reference_stats[col] and 'value_counts' in new_stats[col]:
                    # Calculer la distance euclidienne entre les distributions
                    ref_dist = self.reference_stats[col]['value_counts']
                    new_dist = new_stats[col]['value_counts']
                    
                    # Créer un ensemble de toutes les catégories
                    all_categories = set(ref_dist.keys()) | set(new_dist.keys())
                    
                    # Calculer la différence par catégorie
                    category_diffs = {}
                    total_diff = 0
                    
                    for category in all_categories:
                        ref_val = ref_dist.get(category, 0)
                        new_val = new_dist.get(category, 0)
                        diff = abs(ref_val - new_val)
                        category_diffs[category] = diff
                        total_diff += diff**2
                    
                    euclidean_dist = np.sqrt(total_diff)
                    feature_report['euclidean_distance'] = euclidean_dist
                    feature_report['category_differences'] = category_diffs
                    
                    # Considérer un drift si la distance est supérieure à un seuil
                    feature_report['drift_detected'] = euclidean_dist > 0.2  # Seuil arbitraire
            
            drift_report['features'][col] = feature_report
            
            if feature_report['drift_detected']:
                features_with_drift += 1
        
        # Déterminer s'il y a un drift global

        drift_report['overall_drift_detected'] = drift_report['drift_percentage'] > 0.1  # Seuil arbitraire
        
        # Sauvegarder le rapport
        self._save_report(drift_report)
        
        return drift_report
    
    def _save_report(self, report: Dict):
        """
        Sauvegarde le rapport de drift dans un fichier JSON.
        
        Args:
            report: Rapport de drift à sauvegarder
        """
        timestamp = report['timestamp'].replace(' ', '_').replace(':', '-')
        filename = f"{self.output_dir}/drift_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Rapport de drift sauvegardé dans: {filename}")
    
    def visualize_drift(self, new_data: pd.DataFrame, top_n: int = 5) -> None:
        """
        Crée des visualisations du data drift entre les données de référence et les nouvelles données.
        
        Args:
            new_data: Nouvelles données à comparer avec la référence
            top_n: Nombre de features à afficher (celles avec le plus de drift)
        """
        drift_report = self.detect_drift(new_data)
        
        # Trier les features par niveau de drift
        drift_features = []
        for feature, report in drift_report['features'].items():
            if report['drift_detected']:
                if 'p_value' in report:
                    drift_score = 1 - report['p_value']  # Plus le p est petit, plus le drift est grand
                elif 'euclidean_distance' in report:
                    drift_score = report['euclidean_distance']
                else:
                    drift_score = 0
                
                drift_features.append((feature, drift_score))
        
        # Trier par score de drift décroissant
        drift_features.sort(key=lambda x: x[1], reverse=True)
        
        # Limiter au top_n features
        top_features = [f[0] for f in drift_features[:top_n]]
        
        if not top_features:
            print("Aucun drift significatif détecté dans les données.")
            return
        
        # Créer des visualisations pour les top features
        n_plots = len(top_features)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            if pd.api.types.is_numeric_dtype(self.reference_data[feature]) and pd.api.types.is_numeric_dtype(new_data[feature]):
                # Histogramme pour les variables numériques
                ax.hist(self.reference_data[feature].dropna(), bins=30, alpha=0.5, label='Référence')
                ax.hist(new_data[feature].dropna(), bins=30, alpha=0.5, label='Nouvelles données')
                
                if 'p_value' in drift_report['features'][feature]:
                    p_value = drift_report['features'][feature]['p_value']
                    ax.set_title(f"{feature} (p-value: {p_value:.4f})")
                else:
                    ax.set_title(feature)
                    
            else:
                # Barplot pour les variables catégorielles
                ref_counts = self.reference_data[feature].value_counts(normalize=True)
                new_counts = new_data[feature].value_counts(normalize=True)
                
                # Limiter le nombre de catégories affichées si trop nombreuses
                categories = list(set(ref_counts.index) | set(new_counts.index))
                if len(categories) > 10:
                    categories = list(set(ref_counts.nlargest(10).index) | set(new_counts.nlargest(10).index))
                
                ref_values = [ref_counts.get(cat, 0) for cat in categories]
                new_values = [new_counts.get(cat, 0) for cat in categories]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, ref_values, width, label='Référence')
                ax.bar(x + width/2, new_values, width, label='Nouvelles données')
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha='right')
                
                if 'euclidean_distance' in drift_report['features'][feature]:
                    euc_dist = drift_report['features'][feature]['euclidean_distance']
                    ax.set_title(f"{feature} (dist. euclidienne: {euc_dist:.4f})")
                else:
                    ax.set_title(feature)
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt_path = f"{self.output_dir}/drift_visualization_{timestamp}.png"
        plt.savefig(plt_path)
        
        print(f"Visualisation du drift sauvegardée dans: {plt_path}")
        plt.show()


def compare_datasets(reference_data: pd.DataFrame, 
                     current_data: pd.DataFrame, 
                     threshold: float = 0.05,
                     output_dir: str = "drift_reports") -> Dict:
    """
    Fonction utilitaire pour comparer rapidement deux datasets et détecter le drift.
    
    Args:
        reference_data: Données de référence
        current_data: Données actuelles à comparer
        threshold: Seuil p-value pour considérer qu'il y a un drift
        output_dir: Répertoire où sauvegarder les rapports de drift
        
    Returns:
        Rapport de drift
    """
    monitor = DataDriftMonitor(reference_data, drift_threshold=threshold, output_dir=output_dir)
    return monitor.detect_drift(current_data)


def generate_drift_report(reference_data: pd.DataFrame, 
                          current_data: pd.DataFrame,
                          visualize: bool = True,
                          top_features: int = 5,
                          output_dir: str = "drift_reports") -> Dict:
    """
    Génère un rapport complet de data drift, incluant optionnellement des visualisations.
    
    Args:
        reference_data: Données de référence
        current_data: Données actuelles à comparer
        visualize: Si True, génère des visualisations
        top_features: Nombre de features à visualiser
        output_dir: Répertoire où sauvegarder les rapports
        
    Returns:
        Rapport de drift
    """
    monitor = DataDriftMonitor(reference_data, output_dir=output_dir)
    drift_report = monitor.detect_drift(current_data)
    
    if visualize:
        monitor.visualize_drift(current_data, top_n=top_features)
    
    return drift_report
