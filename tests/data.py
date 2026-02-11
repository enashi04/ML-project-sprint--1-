"""
Module de traitement des données - Fonctions utilitaires
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        DataFrame: Les données chargées
    """
    if not isinstance(file_path, str) or not file_path.endswith('.csv'):
        raise ValueError("Le chemin doit être un fichier CSV valide")
    
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Prétraite les données (nettoyage basique).
    
    Args:
        df (DataFrame): DataFrame à prétraiter
        
    Returns:
        DataFrame: DataFrame prétraité
    """
    df = df.copy()
    
    # Remplir les valeurs manquantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df


def split_data(df, test_size=0.2, random_state=42, target_col='failure_soon'):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        df (DataFrame): DataFrame à diviser
        test_size (float): Proportion des données de test
        random_state (int): Graine aléatoire
        target_col (str): Nom de la colonne cible
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def check_missing_values(df):
    """
    Vérifie la présence de valeurs manquantes.
    
    Args:
        df (DataFrame): DataFrame à vérifier
        
    Returns:
        bool: True s'il y a des valeurs manquantes, False sinon
    """
    return df.isnull().sum().sum() > 0
