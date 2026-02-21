FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl git cmake libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps d'abord (cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# code ensuite
COPY . /app

# Par défaut : lancer ton pipeline W&B
CMD ["python", "wandb/wandb_tracking.py"]