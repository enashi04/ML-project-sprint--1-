# src/features/

Ce module contient la logique de **feature engineering** c'est-à-dire la création, transformation et sélection des variables pour l'entrainement dub modle.

## Objectif
- Transformer les données capteurs en variables exploitables par les modèles ML
- Garantir la reproductibilité des transformations entre train et predict

# 

- Features statistiques : moyenne, écart-type, min/max, rolling windows 
- Encodage catégoriel 
- Normalisation / standardisatio
- Sélection de variables / suppression colinéarité 

## Entrées / sorties
- Entrée : fichiers issus de `data/raw/` / `data/processed/`
- Sortie : 
