"""Fitting the totals model."""
from __future__ import annotations

import logging
import os

import joblib
import pandas as pd

from src.model import TotalsModel, estimate_shrinkage, make_estimator, select_alpha
from src.model_features import build_model_matrix, get_model_features
from config import RIDGE_ALPHA_GRID

logger = logging.getLogger(__name__)

TARGET = "residual"


def build_training_data(game_frame: pd.DataFrame) -> pd.DataFrame:
    """Model matrix restricted to completed games with a posted line."""
    data = build_model_matrix(game_frame)
    data = data[data["total_points"].notna() & data["total_line"].notna()].copy()
    data[TARGET] = data["total_points"] - data["total_line"]
    data["over"] = (data["total_points"] > data["total_line"]).astype(int)

    features = get_model_features()
    # Ridge tolerates a few gaps via median imputation, but a row missing most of
    # its form features is a week-1 game carrying no information.
    coverage = data[features].notna().mean(axis=1)
    dropped = int((coverage < 0.6).sum())
    if dropped:
        logger.info("Dropping %d rows with <60%% feature coverage.", dropped)
    return data[coverage >= 0.6].reset_index(drop=True)


def train_model(
    game_frame: pd.DataFrame,
    model_path: str,
    train_seasons: list[int],
    alpha: float | None = None,
) -> TotalsModel:
    """Fit, calibrate and persist a `TotalsModel`.

    The ridge penalty and the shrinkage applied to every edge are both chosen by
    walk-forward validation *inside* `train_seasons`, so no information from a
    later season leaks into either.
    """
    features = get_model_features()
    data = build_training_data(game_frame)
    data = data[data["season"].isin(train_seasons)].sort_values("date")

    if data.empty:
        raise ValueError(f"No training rows for seasons {train_seasons}.")

    X, y, seasons = data[features], data[TARGET], data["season"]

    if alpha is None:
        alpha = select_alpha(X, y, seasons, features, RIDGE_ALPHA_GRID)

    shrinkage, sigma = estimate_shrinkage(X, y, seasons, alpha, features)

    estimator = make_estimator(alpha)
    estimator.fit(X[features], y)

    model = TotalsModel(
        estimator=estimator,
        features=features,
        sigma=sigma,
        shrinkage=shrinkage,
        alpha=alpha,
        train_seasons=sorted(data["season"].unique().tolist()),
        n_train=len(data),
    )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(
        "Model trained on %d games (%d-%d) and saved to %s.",
        model.n_train, min(model.train_seasons), max(model.train_seasons), model_path,
    )
    logger.info("Strongest effects:\n%s", model.describe().head(12).to_string(index=False))
    return model


def load_model(model_path: str) -> TotalsModel:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python main.py --train-only` first."
        )
    model = joblib.load(model_path)
    logger.info(
        "Loaded model trained %s on %d games (%d-%d).",
        model.trained_at[:10], model.n_train,
        min(model.train_seasons), max(model.train_seasons),
    )
    return model
