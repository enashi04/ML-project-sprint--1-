import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Supposons que vous avez un module features.py qui contient des fonctions 
# pour l'extraction et la transformation de caractéristiques
from features import (
    extract_time_features,
    calculate_rolling_statistics, 
    create_lag_features, 
    perform_feature_selection,
    detect_anomalies
)


class TestFeatureFunctions(unittest.TestCase):
    
    def setUp(self):
        # Création d'un DataFrame de test avec des données temporelles
        self.time_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'temperature': np.random.normal(37, 2, 100),
            'pression': np.random.normal(103, 1, 100),
            'vibration': np.random.normal(0.4, 0.1, 100),
            'defaillance': [0] * 80 + [1] * 20
        })
        self.time_df.set_index('timestamp', inplace=True)
        
        # DataFrame simple pour les tests
        self.simple_df = pd.DataFrame({
            'temperature': [35.1, 37.2, 39.5, 40.1, 36.8],
            'pression': [102.3, 103.5, 101.2, 104.7, 103.1],
            'vibration': [0.32, 0.28, 0.45, 0.51, 0.38],
            'defaillance': [0, 0, 1, 1, 0]
        })
        
    def test_extract_time_features(self):
        # Test de l'extraction de caractéristiques temporelles
        result_df = extract_time_features(self.time_df)
        
        # Vérifier que les nouvelles colonnes temporelles sont présentes
        expected_cols = ['hour', 'day_of_week', 'is_weekend']
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
        
        # Vérifier que les colonnes d'origine sont préservées
        self.assertIn('temperature', result_df.columns)
        self.assertIn('pression', result_df.columns)
        self.assertIn('vibration', result_df.columns)
        
        # Vérifier que les valeurs extraites sont correctes
        self.assertEqual(result_df['hour'].min(), 0)
        self.assertEqual(result_df['hour'].max(), 23)
        self.assertTrue((result_df['day_of_week'] >= 0).all() and (result_df['day_of_week'] <= 6).all())
        self.assertTrue(set(result_df['is_weekend'].unique()).issubset({0, 1}))
        
    def test_calculate_rolling_statistics(self):
        # Test des fonctions qui calculent des statistiques sur fenêtre glissante
        window_size = 3
        result_df = calculate_rolling_statistics(self.time_df, window_size=window_size)
        
        # Vérifier que les nouvelles colonnes de statistiques glissantes sont présentes
        expected_cols = [
            'temperature_rolling_mean', 'pression_rolling_mean', 'vibration_rolling_mean',
            'temperature_rolling_std', 'pression_rolling_std', 'vibration_rolling_std'
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
        
        # Vérifier que les valeurs calculées sont correctes
        # Les 'window_size-1' premières valeurs devraient être NaN
        self.assertEqual(result_df['temperature_rolling_mean'].iloc[:window_size-1].isna().all(), True)
        
        # Vérifier une valeur calculée manuellement
        temp_values = self.time_df['temperature'].iloc[0:window_size]
        expected_mean = temp_values.mean()
        self.assertAlmostEqual(result_df['temperature_rolling_mean'].iloc[window_size-1], expected_mean)
        
    def test_create_lag_features(self):
        # Test de la création de caractéristiques avec décalage temporel
        lags = [1, 2]
        result_df = create_lag_features(self.time_df, lags=lags)
        
        # Vérifier que les nouvelles colonnes de lag sont présentes
        expected_cols = [
            'temperature_lag_1', 'pression_lag_1', 'vibration_lag_1',
            'temperature_lag_2', 'pression_lag_2', 'vibration_lag_2'
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
        
        # Vérifier que les valeurs de lag sont correctes
        for feature in ['temperature', 'pression', 'vibration']:
            for lag in lags:
                self.assertTrue(np.array_equal(
                    result_df[f'{feature}_lag_{lag}'].iloc[lag:].values,
                    self.time_df[feature].iloc[:-lag].values
                ))
                # Vérifier que les premières valeurs sont NaN
                self.assertEqual(result_df[f'{feature}_lag_{lag}'].iloc[:lag].isna().all(), True)
    
    @patch('sklearn.feature_selection.SelectKBest')
    @patch('sklearn.feature_selection.f_classif')
    def test_perform_feature_selection(self, mock_f_classif, mock_selectkbest):
        # Configuration du mock pour SelectKBest
        mock_selector = MagicMock()
        mock_selector.fit_transform.return_value = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        mock_selector.get_support.return_value = [True, False, True]
        mock_selectkbest.return_value = mock_selector
        
        # Test de la sélection de caractéristiques
        X = self.simple_df.drop('defaillance', axis=1)
        y = self.simple_df['defaillance']
        
        selected_features, feature_scores = perform_feature_selection(X, y, k=2)
        
        # Vérifier que les bonnes caractéristiques sont sélectionnées
        self.assertEqual(len(selected_features), 2)
        mock_selectkbest.assert_called_once_with(mock_f_classif, k=2)
        mock_selector.fit_transform.assert_called_once()
        
    def test_detect_anomalies(self):
        # Test de la détection d'anomalies
        X = self.simple_df.drop('defaillance', axis=1)
        
        # Injecter une anomalie évidente
        X_with_anomaly = X.copy()
        X_with_anomaly.loc[2, 'temperature'] = 100.0  # Valeur très élevée
        
        anomalies = detect_anomalies(X_with_anomaly, contamination=0.2)
        
        # Vérifier que l'anomalie est détectée
        self.assertIn(2, np.where(anomalies == -1)[0])
        
        # Vérifier que le nombre d'anomalies correspond à la contamination
        expected_anomaly_count = int(len(X) * 0.2)
        self.assertEqual(np.sum(anomalies == -1), expected_anomaly_count)


if __name__ == '__main__':
    unittest.main()
