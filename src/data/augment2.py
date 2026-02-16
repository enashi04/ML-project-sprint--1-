"""
augment.py
Module d'augmentation des données capteurs (feature engineering niveau 1).

Objectifs
- Ajouter des features temporelles, rolling, lag, interactions
- Ajouter des indicateurs de panne "soon" + time_to_failure + next_failure_type (pour construire la cible ensuite)
- Ajouter des indicateurs de "santé" (days_since_last_failure, failures_count_last_30days)
- Sauvegarder un parquet propre pour build_features.py

Notes importantes (anti-fuite)
- Ne PAS scaler ici (le scaling doit être fait après split, ou via pipeline sklearn).
- Rolling features utilisent shift(1) pour n'utiliser que le passé.
- time_to_failure / next_failure_type ne doivent PAS être utilisées comme features d'entraînement.
"""

from __future__ import annotations

import os
import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("augment_log.log"), logging.StreamHandler()],
)
logger = logging.getLogger("augment")


# -------------------------
# Helpers
# -------------------------
SENSOR_NUMERIC_COLS = ("temperature", "vibration", "pressure", "current")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# Feature Engineering
# -------------------------
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_datetime(df, "timestamp")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["year"] = df["timestamp"].dt.year

    df["is_night"] = (df["hour"] < 6) | (df["hour"] >= 18)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df


def create_rolling_features(
    df: pd.DataFrame,
    window_sizes: Iterable[int] = (5, 10, 30),
    group_by: str = "equipment_id",
    numeric_cols: Iterable[str] = SENSOR_NUMERIC_COLS,
) -> pd.DataFrame:
    """
    Rolling anti-fuite: rolling sur la série décalée (shift(1)) => uniquement passé.
    """
    df = df.copy()
    df = ensure_datetime(df, "timestamp")
    df[group_by] = df[group_by].astype(str)
    df = df.sort_values([group_by, "timestamp"])

    df = safe_numeric(df, numeric_cols)

    for window in window_sizes:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            g = df.groupby(group_by, sort=False)[col]
            shifted = g.shift(1)

            df[f"{col}_rolling_mean_{window}"] = shifted.groupby(df[group_by]).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_rolling_std_{window}"] = shifted.groupby(df[group_by]).transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f"{col}_rolling_min_{window}"] = shifted.groupby(df[group_by]).transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f"{col}_rolling_max_{window}"] = shifted.groupby(df[group_by]).transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

    return df


def create_lag_features(
    df: pd.DataFrame,
    lag_periods: Iterable[int] = (1, 3, 5, 10),
    group_by: str = "equipment_id",
    numeric_cols: Iterable[str] = SENSOR_NUMERIC_COLS,
) -> pd.DataFrame:
    df = df.copy()
    df = ensure_datetime(df, "timestamp")
    df[group_by] = df[group_by].astype(str)
    df = df.sort_values([group_by, "timestamp"])

    df = safe_numeric(df, numeric_cols)

    for lag in lag_periods:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            g = df.groupby(group_by, sort=False)[col]
            df[f"{col}_lag_{lag}"] = g.shift(lag)
            df[f"{col}_change_{lag}"] = df[col] - df[f"{col}_lag_{lag}"]
            df[f"{col}_pct_change_{lag}"] = g.pct_change(periods=lag)

    return df


def add_failure_indicators(
    sensor_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    time_window_hours: int = 24,
) -> pd.DataFrame:
    """
    Crée 3 colonnes:
      - failure_soon (0/1) : 1 si timestamp dans [failure_time - window, failure_time]
      - time_to_failure (float) : heures restantes jusqu'à la panne
      - next_failure_type (cat) : type de la prochaine panne

    Ces colonnes servent surtout à fabriquer une cible ensuite.
    """
    sensor_df = sensor_df.copy()
    sensor_df["equipment_id"] = sensor_df["equipment_id"].astype(str)
    sensor_df = ensure_datetime(sensor_df, "timestamp")

    failure_df = failure_df.copy()
    failure_df["equipment_id"] = failure_df["equipment_id"].astype(str)
    failure_df = ensure_datetime(failure_df, "failure_timestamp")

    sensor_df["failure_soon"] = 0
    sensor_df["time_to_failure"] = np.nan
    sensor_df["next_failure_type"] = pd.NA

    failure_df = failure_df.dropna(subset=["equipment_id", "failure_timestamp"])
    if failure_df.empty:
        return sensor_df

    # 23 pannes => boucle OK
    for _, f in failure_df.iterrows():
        eid = f.get("equipment_id")
        ft = f.get("failure_timestamp")
        ftype = f.get("failure_type", pd.NA)
        if pd.isna(eid) or pd.isna(ft):
            continue

        m = (
            (sensor_df["equipment_id"] == eid)
            & (sensor_df["timestamp"] <= ft)
            & (sensor_df["timestamp"] >= (ft - pd.Timedelta(hours=time_window_hours)))
        )
        if not m.any():
            continue

        idx = sensor_df.index[m]
        ttf = (ft - sensor_df.loc[idx, "timestamp"]).dt.total_seconds() / 3600.0

        cur = pd.to_numeric(sensor_df.loc[idx, "time_to_failure"], errors="coerce")
        upd = cur.isna() | (ttf < cur)

        idx_upd = idx[upd.to_numpy()]
        sensor_df.loc[idx_upd, "failure_soon"] = 1
        sensor_df.loc[idx_upd, "time_to_failure"] = ttf.loc[idx_upd].to_numpy()
        sensor_df.loc[idx_upd, "next_failure_type"] = ftype

    logger.info(f"Distribution failure_soon: {sensor_df['failure_soon'].value_counts(dropna=False).to_dict()}")
    return sensor_df


def create_component_health_features(sensor_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features santé SANS merge_asof (évite 'left keys must be sorted'):
      - days_since_last_failure
      - failures_count_last_30days
    """
    df = sensor_df.copy()
    df["equipment_id"] = df["equipment_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    failures = failure_df.copy()
    failures["equipment_id"] = failures["equipment_id"].astype(str)
    failures["failure_timestamp"] = pd.to_datetime(failures["failure_timestamp"], errors="coerce")

    df["days_since_last_failure"] = np.inf
    df["failures_count_last_30days"] = 0

    failures = failures.dropna(subset=["equipment_id", "failure_timestamp"])
    if failures.empty:
        return df

    # per equipment
    for eid, idxs in df.groupby("equipment_id", sort=False).groups.items():
        idxs = np.array(list(idxs))
        sub_ts = df.loc[idxs, "timestamp"]
        valid = sub_ts.notna().to_numpy()
        if not valid.any():
            continue

        idx_valid = idxs[valid]
        ts = sub_ts.loc[idx_valid].values.astype("datetime64[ns]")

        fts = failures.loc[failures["equipment_id"] == eid, "failure_timestamp"].dropna().values.astype("datetime64[ns]")
        if len(fts) == 0:
            continue
        fts = np.sort(fts)

        # last failure before t
        pos = np.searchsorted(fts, ts, side="left") - 1
        has_prev = pos >= 0
        last_ts = np.empty(ts.shape, dtype="datetime64[ns]")
        last_ts[:] = np.datetime64("NaT")
        last_ts[has_prev] = fts[pos[has_prev]]

        delta_days = (ts - last_ts) / np.timedelta64(1, "D")
        days_since = np.where(np.isnat(last_ts), np.inf, delta_days.astype(float))
        df.loc[idx_valid, "days_since_last_failure"] = days_since

        # count failures last 30 days
        c1 = np.searchsorted(fts, ts, side="left")
        t_minus = ts - np.timedelta64(30, "D")
        c0 = np.searchsorted(fts, t_minus, side="left")
        df.loc[idx_valid, "failures_count_last_30days"] = (c1 - c0)

    return df


def create_interaction_features(df: pd.DataFrame, base_cols: Iterable[str] = SENSOR_NUMERIC_COLS) -> pd.DataFrame:
    df = df.copy()
    df = safe_numeric(df, base_cols)
    base = [c for c in base_cols if c in df.columns]

    for i, c1 in enumerate(base):
        for c2 in base[i + 1 :]:
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    for c1 in base:
        for c2 in base:
            if c1 == c2:
                continue
            denom = df[c2].replace(0, np.nan)
            df[f"{c1}_div_{c2}"] = df[c1] / (denom + 1e-6)

    return df


def plot_feature_importances(df: pd.DataFrame, target_col: str = "failure_soon", output_path: str | None = None) -> None:
    if target_col not in df.columns:
        logger.warning(f"Colonne cible '{target_col}' absente. Plot ignoré.")
        return

    work = df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    if work[target_col].isna().all():
        logger.warning(f"Impossible de convertir '{target_col}' en numérique. Plot ignoré.")
        return

    num = work.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        num[target_col] = work[target_col]

    corr = num.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    top = corr.drop(labels=[target_col], errors="ignore").head(20)
    if top.empty:
        logger.warning("Aucune feature numérique pour corrélations.")
        return

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top.values, y=top.index)
    plt.title(f"Top 20 corrélations avec {target_col}")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def augment_data(
    input_dir: str = "cleaned_data",
    output_dir: str = "augmented_data",
    time_window_hours: int = 24,
) -> pd.DataFrame:
    """
    Pipeline:
      1) load cleaned parquet
      2) time features
      3) rolling (anti-fuite)
      4) lag
      5) failure indicators
      6) health features
      7) interactions
      8) save parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    sensor_path = os.path.join(input_dir, "clean_sensor_data.parquet")
    failure_path = os.path.join(input_dir, "clean_failure_data.parquet")

    logger.info(f"Chargement capteurs: {sensor_path}")
    sensor_df = pd.read_parquet(sensor_path)
    logger.info(f"Chargement pannes: {failure_path}")
    failure_df = pd.read_parquet(failure_path)

    sensor_df = ensure_datetime(sensor_df, "timestamp")
    failure_df = ensure_datetime(failure_df, "failure_timestamp")

    if "equipment_id" not in sensor_df.columns or "timestamp" not in sensor_df.columns:
        raise ValueError("sensor_df doit contenir 'equipment_id' et 'timestamp'")

    # FE
    logger.info("Création des features temporelles")
    sensor_df = create_time_features(sensor_df)

    logger.info("Création des rolling features (anti-fuite)")
    sensor_df = create_rolling_features(sensor_df, window_sizes=(5, 10, 30))

    logger.info("Création des lag features")
    sensor_df = create_lag_features(sensor_df, lag_periods=(1, 3, 5, 10))

    logger.info("Ajout des indicateurs de panne (failure_soon / time_to_failure / next_failure_type)")
    sensor_df = add_failure_indicators(sensor_df, failure_df, time_window_hours=time_window_hours)

    logger.info("Création des features santé composants")
    sensor_df = create_component_health_features(sensor_df, failure_df)

    logger.info("Création des interactions")
    sensor_df = create_interaction_features(sensor_df)

    sensor_df = sensor_df.replace([np.inf, -np.inf], np.nan)

    logger.info("Plot corrélations simples (si cible présente)")
    plot_feature_importances(
        sensor_df, target_col="failure_soon", output_path=os.path.join(viz_dir, "feature_corr_with_failure_soon.png")
    )

    out_path = os.path.join(output_dir, "augmented_sensor_data.parquet")
    logger.info(f"Sauvegarde: {out_path}")
    sensor_df.to_parquet(out_path, index=False)

    logger.info(f"OK - dataset augmenté: {sensor_df.shape}")
    return sensor_df


if __name__ == "__main__":
    df_aug = augment_data()
    print("\nRésumé données augmentées:")
    print(df_aug.describe(include="all"))
