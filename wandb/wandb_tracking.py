import os
import sys
import wandb
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns

# Ajout du répertoire parent dans le PYTHONPATH pour les imports relatifs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports depuis les différents modules du projet
from src.data.extract import load_raw_data
from src.data.clean import clean_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluation import evaluate_model
from src.monitoring.data_drift import check_data_drift
from src.monitoring.performance_tracking import track_performance


class WandbExperimentTracker:
    """
    Classe pour suivre les expériences de machine learning avec Weights & Biases.
    """
    
    def __init__(self, project_name="industrial-failure-prediction", 
                 entity=None, config=None, tags=None, group=None, job_type=None):
        """
        Initialise un tracker d'expérience Weights & Biases.
        
        Args:
            project_name (str): Nom du projet sur Weights & Biases.
            entity (str): Nom de l'entité (utilisateur ou organisation) sur Weights & Biases.
            config (dict): Configuration de l'expérience.
            tags (list): Liste des tags pour l'expérience.
            group (str): Groupe d'expériences.
            job_type (str): Type de job (ex: 'training', 'evaluation', 'feature-engineering').
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.job_type = job_type
        self.artifacts = {}
        
    def start_run(self, run_name=None):
        """
        Démarre une nouvelle exécution (run) Weights & Biases.
        
        Args:
            run_name (str): Nom spécifique pour l'exécution.
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            name=run_name,
            reinit=True
        )
        print(f"Started wandb run: {self.run.name}")
        return self
    
    def end_run(self):
        """Termine l'exécution actuelle."""
        if self.run:
            self.run.finish()
            print(f"Finished wandb run: {self.run.name}")
    
    def log_config(self, config_dict):
        """
        Enregistre la configuration de l'expérience.
        
        Args:
            config_dict (dict): Dictionnaire de configuration à enregistrer.
        """
        if self.run:
            for key, value in config_dict.items():
                self.run.config[key] = value
    
    def log_metrics(self, metrics_dict, step=None):
        """
        Enregistre des métriques pendant l'expérience.
        
        Args:
            metrics_dict (dict): Dictionnaire de métriques à enregistrer.
            step (int, optional): Étape actuelle dans l'entraînement.
        """
        if self.run:
            self.run.log(metrics_dict, step=step)
    
    def log_artifact(self, artifact_name, artifact_type, description, path=None, data=None):
        """
        Enregistre un artefact (modèle, dataset, etc.).
        
        Args:
            artifact_name (str): Nom de l'artefact.
            artifact_type (str): Type d'artefact ('model', 'dataset', etc.).
            description (str): Description de l'artefact.
            path (str, optional): Chemin du fichier à enregistrer.
            data (any, optional): Données à enregistrer.
        """
        if self.run:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description
            )
            
            if path:
                artifact.add_file(path)
            
            if data is not None:
                # Enregistrement de données Python
                if isinstance(data, pd.DataFrame):
                    # Pour un DataFrame pandas
                    table = wandb.Table(dataframe=data)
                    artifact.add(table, "data_table")
                elif isinstance(data, dict):
                    # Pour un dictionnaire
                    with artifact.new_file("data.json", mode="w") as f:
                        json.dump(data, f)
                
            self.run.log_artifact(artifact)
            self.artifacts[artifact_name] = artifact
            print(f"Logged artifact: {artifact_name}")
    
    def log_model(self, model, model_name, description, model_path=None, metadata=None):
        """
        Enregistre un modèle ML comme artefact.
        
        Args:
            model: Modèle ML à enregistrer.
            model_name (str): Nom du modèle.
            description (str): Description du modèle.
            model_path (str, optional): Chemin où le modèle est sauvegardé.
            metadata (dict, optional): Métadonnées supplémentaires sur le modèle.
        """
        # Si le modèle n'est pas déjà sauvegardé, nous l'enregistrons temporairement
        if not model_path:
            import joblib
            model_path = f"{model_name}.joblib"
            joblib.dump(model, model_path)
        
        metadata = metadata or {}
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=description,
            metadata=metadata
        )
        
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
        self.artifacts[model_name] = artifact
        
        # Supprimer le fichier temporaire si nous l'avons créé
        if not model_path:
            os.remove(model_path)
        
        print(f"Logged model: {model_name}")
    
    def log_feature_importance(self, model, feature_names, model_name="model"):
        """
        Enregistre l'importance des caractéristiques pour les modèles qui le supportent.
        
        Args:
            model: Modèle ML avec feature_importances_ ou coef_.
            feature_names (list): Liste des noms des caractéristiques.
            model_name (str): Nom du modèle pour le logging.
        """
        if not self.run:
            return
            
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            else:
                print("Le modèle ne fournit pas d'importances de caractéristiques")
                return
                
            # Création d'une table wandb pour les importances
            feature_importance_data = {
                "feature": feature_names,
                "importance": importances
            }
            feature_importance_df = pd.DataFrame(feature_importance_data)
            feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)
            
            # Plot et log de l'importance des caractéristiques
            plt.figure(figsize=(10, 6))
            sns.barplot(x="importance", y="feature", data=feature_importance_df)
            plt.title(f"Feature Importance - {model_name}")
            plt.tight_layout()
            
            self.run.log({f"{model_name}_feature_importance": wandb.Image(plt)})
            plt.close()
            
            # Log des données sous forme de tableau
            self.run.log({f"{model_name}_feature_importance_table": wandb.Table(dataframe=feature_importance_df)})
            
        except Exception as e:
            print(f"Erreur lors du logging de l'importance des caractéristiques: {str(e)}")
    
    def log_confusion_matrix(self, y_true, y_pred, model_name="model"):
        """
        Enregistre une matrice de confusion.
        
        Args:
            y_true (array): Valeurs réelles.
            y_pred (array): Prédictions du modèle.
            model_name (str): Nom du modèle pour le logging.
        """
        if not self.run:
            return
            
        try:
            # Création de la matrice de confusion
            cm = confusion_matrix(y_true, y_pred)
            
            # Visualisation avec seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Non-défaillance", "Défaillance"],
                        yticklabels=["Non-défaillance", "Défaillance"])
            plt.xlabel("Prédiction")
            plt.ylabel("Réel")
            plt.title(f"Matrice de confusion - {model_name}")
            
            self.run.log({f"{model_name}_confusion_matrix": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Erreur lors du logging de la matrice de confusion: {str(e)}")
    
    def log_roc_curve(self, y_true, y_prob, model_name="model"):
        """
        Enregistre une courbe ROC.
        
        Args:
            y_true (array): Valeurs réelles.
            y_prob (array): Probabilités prédites pour la classe positive.
            model_name (str): Nom du modèle pour le logging.
        """
        if not self.run:
            return
            
        try:
            # Calcul de la courbe ROC
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            # Création d'un DataFrame pour le logging
            roc_df = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": [t if t != float("inf") else 0 for t in thresholds]
            })
            
            # Création du graphique
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{model_name}")
            plt.plot([0, 1], [0, 1], 'k--', label="Random")
            plt.xlabel("Taux de faux positifs")
            plt.ylabel("Taux de vrais positifs")
            plt.title(f"Courbe ROC - {model_name}")
            plt.legend()
            
            self.run.log({f"{model_name}_roc_curve": wandb.Image(plt)})
            plt.close()
            
            # Log des données sous forme de tableau
            self.run.log({f"{model_name}_roc_data": wandb.Table(dataframe=roc_df)})
            
        except Exception as e:
            print(f"Erreur lors du logging de la courbe ROC: {str(e)}")
    
    def log_precision_recall_curve(self, y_true, y_prob, model_name="model"):
        """
        Enregistre une courbe précision-rappel.
        
        Args:
            y_true (array): Valeurs réelles.
            y_prob (array): Probabilités prédites pour la classe positive.
            model_name (str): Nom du modèle pour le logging.
        """
        if not self.run:
            return
            
        try:
            # Calcul de la courbe précision-rappel
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            
            # Création d'un DataFrame pour le logging
            pr_df = pd.DataFrame({
                "precision": precision[:-1],  # dernier élément de precision n'a pas de seuil correspondant
                "recall": recall[:-1],
                "thresholds": thresholds
            })
            
            # Création du graphique
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f"{model_name}")
            plt.xlabel("Rappel")
            plt.ylabel("Précision")
            plt.title(f"Courbe Précision-Rappel - {model_name}")
            plt.legend()
            
            self.run.log({f"{model_name}_pr_curve": wandb.Image(plt)})
            plt.close()
            
            # Log des données sous forme de tableau
            self.run.log({f"{model_name}_pr_data": wandb.Table(dataframe=pr_df)})
            
        except Exception as e:
            print(f"Erreur lors du logging de la courbe précision-rappel: {str(e)}")
    
    def log_data_distribution(self, data, column_name, title=None):
        """
        Enregistre la distribution d'une colonne de données.
        
        Args:
            data (DataFrame): Les données à analyser.
            column_name (str): Nom de la colonne à visualiser.
            title (str, optional): Titre du graphique.
        """
        if not self.run:
            return
            
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column_name], kde=True)
            plt.title(title or f"Distribution de {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Fréquence")
            
            self.run.log({f"distribution_{column_name}": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Erreur lors du logging de la distribution: {str(e)}")
    
    def log_correlation_matrix(self, data, title="Matrice de corrélation"):
        """
        Enregistre une matrice de corrélation.
        
        Args:
            data (DataFrame): DataFrame contenant les variables numériques.
            title (str): Titre du graphique.
        """
        if not self.run:
            return
            
        try:
            # Calcul de la matrice de corrélation
            correlation_matrix = data.corr()
            
            # Visualisation
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", 
                       cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(title)
            plt.tight_layout()
            
            self.run.log({"correlation_matrix": wandb.Image(plt)})
            plt.close()
            
            # Log du tableau de données
            self.run.log({"correlation_data": wandb.Table(dataframe=correlation_matrix)})
            
        except Exception as e:
            print(f"Erreur lors du logging de la matrice de corrélation: {str(e)}")
    
    def log_learning_curves(self, train_scores, val_scores, epochs, metric_name="loss"):
        """
        Enregistre des courbes d'apprentissage.
        
        Args:
            train_scores (array): Scores sur l'ensemble d'entrainement.
            val_scores (array): Scores sur l'ensemble de validation.
            epochs (array): Numéros des époques/itérations.
            metric_name (str): Nom de la métrique (ex: 'loss', 'accuracy').
        """
        if not self.run:
            return
            
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_scores, label=f"Train {metric_name}")
            plt.plot(epochs, val_scores, label=f"Validation {metric_name}")
            plt.title(f"Courbe d'apprentissage - {metric_name}")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.legend()
            
            self.run.log({f"learning_curve_{metric_name}": wandb.Image(plt)})
            plt.close()
            
            # Log des données sous forme de tableau
            learning_df = pd.DataFrame({
                "epoch": epochs,
                f"train_{metric_name}": train_scores,
                f"val_{metric_name}": val_scores
            })
            self.run.log({f"learning_data_{metric_name}": wandb.Table(dataframe=learning_df)})
            
        except Exception as e:
            print(f"Erreur lors du logging des courbes d'apprentissage: {str(e)}")
    
    def log_drift_metrics(self, drift_metrics, title="Métriques de dérive des données"):
        """
        Enregistre les métriques de dérive des données.
        
        Args:
            drift_metrics (dict): Dictionnaire des métriques de dérive.
            title (str): Titre du tableau.
        """
        if not self.run:
            return
        
        try:
            # Conversion en DataFrame pour une meilleure visualisation
            drift_df = pd.DataFrame(list(drift_metrics.items()), columns=["Feature", "Drift Score"])
            
            # Log sous forme de tableau
            self.run.log({"drift_metrics": wandb.Table(dataframe=drift_df)})
            
            # Visualisation
            plt.figure(figsize=(12, 8))
            sns.barplot(x="Drift Score", y="Feature", data=drift_df.sort_values("Drift Score", ascending=False))
            plt.title(title)
            plt.tight_layout()
            
            self.run.log({"drift_chart": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Erreur lors du logging des métriques de dérive: {str(e)}")


def main():
    """
    Exemple d'utilisation du tracker pour un workflow complet
    """
    # Configuration de l'expérience
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "test_size": 0.2,
        "feature_engineering": {
            "use_rolling_stats": True,
            "window_size": 5,
            "create_lag_features": True,
            "lag_values": [1, 2, 3]
        }
    }
    
    # Initialisation du tracker
    experiment = WandbExperimentTracker(
        project_name="industrial-failure-prediction",
        entity="votre_entite",  # Remplacez par votre entité wandb
        config=config,
        tags=["défaillance", "industriel", "machine-learning"],
        group="expériences-initiales",
        job_type="training"
    )
    
    # Démarrage de l'expérience
    experiment.start_run(run_name="rf_model_v1")
    
    try:
        # Chargement des données
        print("Chargement des données...")
        raw_data = load_raw_data("path/to/data.csv")
        
        # Nettoyage des données
        print("Nettoyage des données...")
        clean_data_df = clean_data(raw_data)
        
        # Log des statistiques des données
        experiment.log_metrics({
            "data_rows": len(clean_data_df),
            "data_columns": len(clean_data_df.columns),
            "missing_values": clean_data_df.isna().sum().sum()
        })
        
        # Log de la distribution des caractéristiques principales
        for col in ["temperature", "pression", "vibration"]:
            if col in clean_data_df.columns:
                experiment.log_data_distribution(clean_data_df, col)
        
        # Log de la matrice de corrélation
        numerical_cols = clean_data_df.select_dtypes(include=['number']).columns
        experiment.log_correlation_matrix(clean_data_df[numerical_cols])
        
        # Construction des caractéristiques
        print("Construction des caractéristiques...")
        features_df = build_features(clean_data_df, config["feature_engineering"])
        
        # Division en ensembles d'entraînement et de test
        from sklearn.model_selection import train_test_split
        
        X = features_df.drop("defaillance", axis=1)
        y = features_df["defaillance"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["random_state"]
        )
        
        # Entraînement du modèle
        print("Entraînement du modèle...")
        model = train_model(X_train, y_train, 
                           model_type=config["model_type"],
                           params={
                               "n_estimators": config["n_estimators"],
                               "max_depth": config["max_depth"],
                               "random_state": config["random_state"]
                           })
        
        # Évaluation du modèle
        print("Évaluation du modèle...")
        eval_metrics = evaluate_model(model, X_test, y_test)
        
        # Log des métriques d'évaluation
        experiment.log_metrics(eval_metrics)
        
        # Log de l'importance des caractéristiques
        experiment.log_feature_importance(model, X.columns.tolist())
        
        # Log de la matrice de confusion
        y_pred = model.predict(X_test)
        experiment.log_confusion_matrix(y_test, y_pred)
        
        # Log de la courbe ROC et précision-rappel
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive
        experiment.log_roc_curve(y_test, y_prob)
        experiment.log_precision_recall_curve(y_test, y_prob)
        
        # Validation croisée
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        experiment.log_metrics({
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std()
        })
        
        # Vérification de la dérive des données (simulée)
        drift_metrics = check_data_drift(X_train, X_test)
        experiment.log_drift_metrics(drift_metrics)
        
        # Sauvegarde du modèle
        import joblib
        model_path = "models/rf_model_v1.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Log du modèle comme artefact
        experiment.log_model(
            model=model,
            model_name="rf_failure_prediction",
            description="Modèle de prédiction de défaillance industrielle",
            model_path=model_path,
            metadata={
                "accuracy": eval_metrics["accuracy"],
                "f1_score": eval_metrics["f1"],
                "features": list(X.columns)
            }
        )
        
        print("Expérience terminée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution de l'expérience: {str(e)}")
        # Log de l'erreur
        experiment.log_metrics({"error": True})
        experiment.run.log({"error_message": str(e)})
    
    finally:
        # Fin de l'expérience
        experiment.end_run()


if __name__ == "__main__":
    main()
