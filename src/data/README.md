# src/data/

Ce module gère toute la **chaîne data** du projet :
**extraction → nettoyage → augmentation** des données avant le feature engineering et l'entraînement du modèle.

## Objectif
- Centraliser les étapes de préparation des données
- Garantir une préparation **reproductible** (mêmes entrées → mêmes sorties)
- Éviter la fuite de données (data leakage) entre train et test

## Fichiers
- `extract.py`  
Ce fichier permet d'extraire les données depuis la source (les fichiers de capteurs et de défaillances).   
  Sortie : dataset brut normalisé au format attendu par `clean.py`.

- `clean.py`  
Ce script python va nettoyer et fiabiliser les données (types, valeurs manquantes, doublons, outliers etc ..).  
  Sortie : dataset propre, cohérent pour la suite du projet.

- `augment.py`  
augment.py va enrichir les données avec du windowing temporel, etc.  
  Sortie : dataset enrichi, prêt pour le feature engineering.

## Pipeline 
1. `extract.py` : récupération + format standard
2. `clean.py` : nettoyage + contrôles qualité
3. `augment.py` : augmentation / équilibrage 


## Entrées / sorties 
- Entrées typiques :
  - `data/raw/` 
- Sorties typiques :
  - `data/processed/` 

## Conventions 
- Toujours logger :
  - nombre de lignes avant/après chaque étape
  - nombre de valeurs manquantes
  - distribution de la target (si classification)

## Modifications faites :
- complétion code des fichiers pour la chaîne data.
