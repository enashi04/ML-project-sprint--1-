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
# NOTE: dans TON projet, on a extract_data / augment_data / build_features(input_dir, output_dir)
from src.data.extract import extract_data
from src.data.clean import clean_data
from src.data.augment import augment_data
from src.features.build_features import build_features
from src.models.train_model import train_and_evaluate
from src.models.evaluation import evaluate_model
from src.monitoring.data_drift import compare_datasets
from src.monitoring.performance_tracking import ModelPerformanceTracker


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
        # Login si une clé est disponible (sinon W&B tente un mode local/offline selon ta config)
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception as e:
                print(f"W&B login warning: {e}")

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

            if path and os.path.exists(path):
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
        temp_created = False
        if not model_path:
            import joblib
            model_path = f"{model_name}.joblib"
            joblib.dump(model, model_path)
            temp_created = True

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
        if temp_created and os.path.exists(model_path):
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
            sns.heatmap(correlation_matrix, mask=mask, annot=False,
                        cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(title)
            plt.tight_layout()

            self.run.log({"correlation_matrix": wandb.Image(plt)})
            plt.close()

            # Log du tableau de données (attention: peut être énorme)
            self.run.log({"correlation_data": wandb.Table(dataframe=correlation_matrix.reset_index())})

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


def _drift_report_to_scores(drift_report: dict) -> dict:
    """
    Convertit le rapport de drift (DataDriftMonitor) en {feature: score} pour le log W&B.
    """
    scores = {}
    if not isinstance(drift_report, dict):
        return scores
    features = drift_report.get("features", {}) or {}
    for feat, rep in features.items():
        if not isinstance(rep, dict):
            continue
        if "p_value" in rep and rep["p_value"] is not None:
            try:
                scores[feat] = float(1.0 - float(rep["p_value"]))
            except Exception:
                continue
        elif "euclidean_distance" in rep and rep["euclidean_distance"] is not None:
            try:
                scores[feat] = float(rep["euclidean_distance"])
            except Exception:
                continue
    return scores


def main():
    """
    Exemple d'utilisation du tracker pour un workflow complet
    """
    # Configuration de l'expérience
    config = {
        "sensor_path": "data/raw/predictive_maintenance_sensor_data.csv",
        "failure_path": "data/raw/predictive_maintenance_failure_logs.csv",

        "raw_dir": "raw_data",
        "cleaned_dir": "cleaned_data",
        "augmented_dir": "augmented_data",
        "featured_dir": "featured_data",
        "models_dir": "models",

        "model_type": "random_forest",
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
        entity=None,  # Remplacez par votre entité wandb si besoin
        config=config,
        tags=["défaillance", "industriel", "machine-learning"],
        group="expériences-initiales",
        job_type="training"
    )

    # Démarrage de l'expérience
    experiment.start_run(run_name=f"{config['model_type']}_run")

    try:
        # Chargement des données
        print("Chargement des données...")

        # 1) EXTRACT
        extract_data(
            sensor_file_path=config["sensor_path"],
            failure_file_path=config["failure_path"],
            output_dir=config["raw_dir"]
        )

        # 2) CLEAN
        clean_data(
            input_dir=config["raw_dir"],
            output_dir=config["cleaned_dir"]
        )

        # 3) AUGMENT
        augment_data(
            input_dir=config["cleaned_dir"],
            output_dir=config["augmented_dir"]
        )

        # 4) FEATURES
        build_features(
            input_dir=config["augmented_dir"],
            output_dir=config["featured_dir"]
        )

        featured_train_csv = os.path.join(config["featured_dir"], "featured_data.csv")
        featured_test_csv = os.path.join(config["featured_dir"], "featured_test_data.csv")

        # Log des datasets comme artefacts
        if os.path.exists(featured_train_csv):
            experiment.log_artifact(
                artifact_name="featured_train",
                artifact_type="dataset",
                description="Jeu d'entraînement feature-engineered",
                path=featured_train_csv
            )
        if os.path.exists(featured_test_csv):
            experiment.log_artifact(
                artifact_name="featured_test",
                artifact_type="dataset",
                description="Jeu de test feature-engineered",
                path=featured_test_csv
            )

        # Charger pour stats + drift + courbes
        train_df = pd.read_csv(featured_train_csv)
        test_df = pd.read_csv(featured_test_csv) if os.path.exists(featured_test_csv) else None

        experiment.log_metrics({
            "train_rows": int(len(train_df)),
            "n_features_plus_target": int(train_df.shape[1]),
            "missing_values_train": int(train_df.isna().sum().sum())
        })
        if test_df is not None:
            experiment.log_metrics({
                "test_rows": int(len(test_df)),
                "missing_values_test": int(test_df.isna().sum().sum())
            })

        # Matrice de corrélation sur sample (sinon trop gros)
        numerical_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
        if numerical_cols:
            sample_corr = train_df[numerical_cols].sample(
                n=min(3000, len(train_df)),
                random_state=config["random_state"]
            )
            experiment.log_correlation_matrix(sample_corr, title="Matrice de corrélation (sample)")

        # Entraînement du modèle via TON train_model.py
        print("Entraînement du modèle...")
        models_to_train = [config["model_type"]] if config.get("model_type") else None
        trained_models, evaluation_results, model_paths, best_model = train_and_evaluate(
            data_path=featured_train_csv,
            target_column="failure_within_24h",
            models_to_train=models_to_train,
            models_dir=config["models_dir"],
            test_size=config["test_size"],
            random_state=config["random_state"],
            cv=3
        )

        # Log des métriques du meilleur modèle (compat: ton evaluation_results contient auc/pr_auc/f1/etc selon ta version)
        best_metrics = evaluation_results.get(best_model, {})
        for k, v in best_metrics.items():
            if isinstance(v, (int, float, np.number)):
                experiment.log_metrics({f"{best_model}_{k}": float(v)})

        # Récupérer le modèle entraîné
        best_model_obj = trained_models[best_model]["model"]
        best_model_path = model_paths.get(best_model)

        # Évaluation sur test_df si présent (sinon on ne fait pas les courbes)
        if test_df is not None and len(test_df.columns) >= 2:
            X_test = test_df.drop(columns=["failure_within_24h"]) if "failure_within_24h" in test_df.columns else test_df.iloc[:, :-1]
            y_test = test_df["failure_within_24h"] if "failure_within_24h" in test_df.columns else test_df.iloc[:, -1]

            # Align colonnes comme l’entraînement
            X_train = train_df.drop(columns=["failure_within_24h"]) if "failure_within_24h" in train_df.columns else train_df.iloc[:, :-1]
            X_train = X_train.replace((np.inf, -np.inf, np.nan), 0)
            X_test = X_test.replace((np.inf, -np.inf, np.nan), 0)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            y_pred = best_model_obj.predict(X_test)
            experiment.log_confusion_matrix(y_test, y_pred, model_name=best_model)

            if hasattr(best_model_obj, "predict_proba"):
                y_prob = best_model_obj.predict_proba(X_test)[:, 1]
                experiment.log_roc_curve(y_test, y_prob, model_name=best_model)
                experiment.log_precision_recall_curve(y_test, y_prob, model_name=best_model)

            # Feature importance si possible
            experiment.log_feature_importance(best_model_obj, X_train.columns.tolist(), model_name=best_model)

            # Drift train vs test
            drift_report = compare_datasets(
                reference_data=X_train,
                current_data=X_test,
                threshold=0.05,
                output_dir="drift_reports"
            )
            drift_scores = _drift_report_to_scores(drift_report)
            if drift_scores:
                experiment.log_drift_metrics(drift_scores)

            # Tracking perf (ton module = classe)
            try:
                perf_tracker = ModelPerformanceTracker(
                    model_name=best_model,
                    model_version="v1",
                    is_classification=True
                )
                perf_tracker.track_performance(
                    y_true=np.array(y_test),
                    y_pred=np.array(y_pred),
                    y_prob=np.array(y_prob) if "y_prob" in locals() else None,
                    dataset_name="test"
                )
                perf_tracker.save_history()
            except Exception as e:
                print(f"Performance tracking warning: {e}")

        # Log du modèle comme artefact (on log le .pkl produit par save_models)
        if best_model_path and os.path.exists(best_model_path):
            experiment.log_artifact(
                artifact_name=f"{best_model}_model",
                artifact_type="model",
                description="Modèle entraîné (pickle dict)",
                path=best_model_path
            )

        print("Expérience terminée avec succès!")

    except Exception as e:
        print(f"Erreur lors de l'exécution de l'expérience: {str(e)}")
        # Log de l'erreur
        experiment.log_metrics({"error": True})
        if experiment.run:
            experiment.run.log({"error_message": str(e)})

    finally:
        # Fin de l'expérience
        experiment.end_run()


if __name__ == "__main__":
    main()
