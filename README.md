# Prédiction de risque de défaillance industrielle

> Pipeline de Machine Learning pour prédire des défaillances industrielles à partir de données capteurs.

## Contexte & objectif
- **Contexte :** entreprise manufacturière souhaitant anticiper les pannes d’équipements.
- **Objectifs :**
  - Réduire les temps d’arrêt (downtime)
  - Optimiser la maintenance préventive / conditionnelle
  - Estimer le risque et, si disponible, le **coût potentiel** de défaillance
- **Livrables :**
  - Pipeline ML reproductible (prétraitement → features → entraînement → évaluation)
  - Modèle(s) entraîné(s) + artefacts (scalers, encoders, features, etc.)
  - Suivi d’expériences avec **Weights & Biases (W&B)**
  - Rapport de résultats (metrics, analyses, recommandations)

## Fonctionnalités
- [x] Ingestion / chargement des données
- [x] Prétraitement & feature engineering
- [x] Entraînement (baseline + modèles avancés)
- [x] Évaluation (classification report, confusion matrix, ROC-AUC)
- [x] Sauvegarde des modèles & artefacts

## Stack
- **Langage :** Python 3.11  
- **Librairies :** pandas, numpy, scikit-learn, xgboost, lightgbm  
- **Outils :** Git, Weights & Biases (W&B)

## Structure du projet
```text
project/
├─ data/
│  ├─ raw/                # données brutes (non modifiées)
│  └─ processed/          # données clean + features prêtes
├─ notebooks/             # exploration / EDA / essais
├─ src/
│  ├─ __init__.py
│  ├─ train.py            # entraînement + tracking W&B + sauvegarde
│  ├─ predict.py          # inférence (batch / fichier)
│  ├─ features.py         # feature engineering
│  └─ utils.py            # fonctions utilitaires (I/O, métriques, etc.)
├─ models/                # modèles sauvegardés (.joblib/.pkl)
├─ reports/               # rapports, figures, exports
├─ requirements.txt
└─ README.md
```
## Installation
**Pré-requis** :
Python 3.11
Compte WandB : wandb.ai
## Setup Local

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

## Données 
Les données sont brutes et déposées dans `data/raw/`
Les données transformées et features seront générées dans `data/processed/`


