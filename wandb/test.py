import os
import wandb
import sys

# #appeler la clé api qui se trouve dans la variable d'environnement
# wandb.login(key=os.getenv("WANDB_API_KEY"))
# # Ajout du répertoire parent dans le PYTHONPATH pour les imports relatifs
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("WANDB_API_KEY exists?", bool(os.getenv("WANDB_API_KEY")))
print("WANDB_API_KEY first chars:", (os.getenv("WANDB_API_KEY") or "")[:6])
