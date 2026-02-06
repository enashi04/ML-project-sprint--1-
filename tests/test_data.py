import unittest
import pandas as pd
import numpy as np
import os
from io import StringIO
from unittest.mock import patch, mock_open

# Supposons que vous avez un module data.py qui contient des fonctions 
# pour charger et prétraiter les données
from data import load_data, preprocess_data, split_data, check_missing_values


class TestDataFunctions(unittest.TestCase):
    
    def setUp(self):
        # Création d'un DataFrame de test
        self.test_df = pd.DataFrame({
            'temperature': [35.1, 37.2, 39.5, 40.1, 36.8],
            'pression': [102.3, 103.5, 101.2, 104.7, 103.1],
            'vibration': [0.32, 0.28, 0.45, 0.51, 0.38],
            'defaillance': [0, 0, 1, 1, 0]
        })
        
        # Mock de données CSV
        self.csv_content = """temperature,pression,vibration,defaillance
35.1,102.3,0.32,0
37.2,103.5,0.28,0
39.5,101.2,0.45,1
40.1,104.7,0.51,1
36.8,103.1,0.38,0"""
    
    @patch('builtins.open', new_callable=mock_open, read_data="temperature,pression,vibration,defaillance\n35.1,102.3,0.32,0\n37.2,103.5,0.28,0")
    @patch('os.path.exists', return_value=True)
    def test_load_data(self, mock_exists, mock_file):
        # Test du chargement de données
        data = load_data('fake_path.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)  # 2 lignes dans notre mock
        self.assertEqual(list(data.columns), ['temperature', 'pression', 'vibration', 'defaillance'])
        
        # Test avec un fichier qui n'existe pas
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')
    
    def test_preprocess_data(self):
        # Test du prétraitement de données
        processed_df = preprocess_data(self.test_df)
        
        # Vérifier que toutes les colonnes sont présentes
        self.assertEqual(set(processed_df.columns), set(['temperature', 'pression', 'vibration', 'defaillance']))
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        self.assertEqual(processed_df.isnull().sum().sum(), 0)
        
        # Vérifier que les valeurs sont normalisées (si applicable)
        # Par exemple, si votre fonction normalise les données entre 0 et 1
        for col in ['temperature', 'pression', 'vibration']:
            if col in processed_df.columns:
                self.assertTrue(processed_df[col].min() >= 0)
                self.assertTrue(processed_df[col].max() <= 1)
    
    def test_split_data(self):
        # Test de la division des données
        X_train, X_test, y_train, y_test = split_data(self.test_df, test_size=0.2, random_state=42)
        
        # Vérifier les dimensions
        self.assertEqual(len(X_train) + len(X_test), len(self.test_df))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        
        # Vérifier que la colonne 'defaillance' n'est pas dans X_train/X_test
        self.assertNotIn('defaillance', X_train.columns)
        self.assertNotIn('defaillance', X_test.columns)
    
    def test_check_missing_values(self):
        # Test de la fonction qui vérifie les valeurs manquantes
        df_with_missing = self.test_df.copy()
        df_with_missing.loc[0, 'temperature'] = np.nan
        
        # Vérifier qu'elle détecte correctement les valeurs manquantes
        self.assertTrue(check_missing_values(df_with_missing))
        
        # Vérifier qu'elle renvoie False quand il n'y a pas de valeurs manquantes
        self.assertFalse(check_missing_values(self.test_df))


if __name__ == '__main__':
    unittest.main()
