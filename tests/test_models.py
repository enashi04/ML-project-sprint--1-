import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pickle
import os

# Supposons que vous avez un module models.py qui contient des fonctions 
# pour l'entraînement et l'évaluation des modèles
from models import (
    train_model, 
    evaluate_model, 
    save_model, 
    load_model, 
    cross_validate_model,
    predict_risk
)


class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        # Création de données de test
        np.random.seed(42)
        n_samples = 100
        
        # Caractéristiques
        self.X_train = pd.DataFrame({
            'temperature': np.random.normal(37, 2, n_samples),
            'pression': np.random.normal(103, 1, n_samples),
            'vibration': np.random.normal(0.4, 0.1, n_samples),
            'temp_rolling_mean': np.random.normal(37, 1, n_samples),
            'vibration_lag_1': np.random.normal(0.4, 0.05, n_samples)
        })
        
        # Cibles
        self.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Données de test
        self.X_test = pd.DataFrame({
            'temperature': np.random.normal(37, 2, 30),
            'pression': np.random.normal(103, 1, 30),
            'vibration': np.random.normal(0.4, 0.1, 30),
            'temp_rolling_mean': np.random.normal(37, 1, 30),
            'vibration_lag_1': np.random.normal(0.4, 0.05, 30)
        })
        
        self.y_test = pd.Series(np.random.randint(0, 2, 30))
        
        # Mock pour le modèle
        self.mock_model = MagicMock()
        self.mock_model.fit.return_value = self.mock_model
        self.mock_model.predict.return_value = np.array([0, 1, 0, 1] * 7 + [0, 1])
        self.mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]
        ] * 7 + [[0.7, 0.3], [0.4, 0.6]])
        
    @patch('sklearn.ensemble.RandomForestClassifier')
    def test_train_model(self, mock_rf):
        # Configuration du mock
        mock_rf.return_value = self.mock_model
        
        # Test de l'entraînement du modèle
        model = train_model(self.X_train, self.y_train, model_type='random_forest')
        
        # Vérifier que le modèle a été entraîné correctement
        self.mock_model.fit.assert_called_once_with(self.X_train, self.y_train)
        self.assertEqual(model, self.mock_model)
        
        # Tester avec un type de modèle non supporté
        with self.assertRaises(ValueError):
            train_model(self.X_train, self.y_train, model_type='unknown_model')
    
    def test_evaluate_model(self):
        # Test de l'évaluation du modèle
        evaluation = evaluate_model(self.mock_model, self.X_test, self.y_test)
        
        # Vérifier que l'évaluation contient les métriques attendues
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, evaluation)
        
        # Vérifier que le modèle a été utilisé pour prédire
        self.mock_model.predict.assert_called_once_with(self.X_test)
        self.mock_model.predict_proba.assert_called_once_with(self.X_test)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('pickle.dump')
    def test_save_model(self, mock_dump, mock_open):
        # Test de la sauvegarde du modèle
        model_path = 'model.pkl'
        save_model(self.mock_model, model_path)
        
        # Vérifier que le modèle a été sauvegardé correctement
        mock_open.assert_called_once_with(model_path, 'wb')
        mock_dump.assert_called_once()
        self.assertEqual(mock_dump.call_args[0][0], self.mock_model)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('pickle.load')
    @patch('os.path.exists', return_value=True)
    def test_load_model(self, mock_exists, mock_load, mock_open):
        # Configuration du mock
        mock_load.return_value = self.mock_model
        
        # Test du chargement du modèle
        model_path = 'model.pkl'
        model = load_model(model_path)
        
        # Vérifier que le modèle a été chargé correctement
        mock_open.assert_called_once_with(model_path, 'rb')
        mock_load.assert_called_once()
        self.assertEqual(model, self.mock_model)
        
        # Test avec un fichier qui n'existe pas
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            load_model('nonexistent_model.pkl')
    
    @patch('sklearn.model_selection.cross_val_score')
    def test_cross_validate_model(self, mock_cv_score):
        # Configuration du mock
        mock_cv_score.return_value = np.array([0.85, 0.87, 0.82, 0.89, 0.84])
        
        # Test de la validation croisée
        cv_results = cross_validate_model(self.mock_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        # Vérifier que la validation croisée a été effectuée correctement
        mock_cv_score.assert_called_once()
        self.assertEqual(len(cv_results), 5)
        self.assertAlmostEqual(np.mean(cv_results), 0.854)
    
    def test_predict_risk(self):
        # Test de la prédiction de risque
        risk_scores = predict_risk(self.mock_model, self.X_test)
        
        # Vérifier que les scores de risque sont corrects
        self.assertEqual(len(risk_scores), len(self.X_test))
        
        # Vérifier que les scores sont entre 0 et 1
        self.assertTrue(all(0 <= score <= 1 for score in risk_scores))
        
        # Vérifier que le modèle a été utilisé pour prédire les probabilités
        self.mock_model.predict_proba.assert_called_with(self.X_test)


if __name__ == '__main__':
    unittest.main()
