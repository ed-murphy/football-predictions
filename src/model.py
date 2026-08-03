"""The totals model.

What changed and why
--------------------
The previous model was an XGBoost classifier fitted directly on the binary
`total_points > total_line` outcome. That framing throws away almost all of the
signal: each game contributes one bit, the base rate is ~48%, and there are only
~2,700 usable rows. With 500 trees at depth 5 the result was a model that scored
worse than a coin flip out of sample (holdout log loss 0.80 against 0.69 for
"always say 50%").

This module models the *residual* instead — how many points a game lands above or
below the closing total — as a regression, and converts that to a probability
through the residual distribution. Three reasons:

  * The residual is continuous, so every game contributes far more information
    than a single over/under bit.
  * The closing line already contains the market's estimate of the level. Asking
    the model only for the disagreement means it never has to relearn "domes are
    high scoring" from scratch.
  * A predicted point total is a more useful output than a bare probability.

Predictions are deliberately shrunk. A `shrinkage` coefficient is estimated by
time-series cross-validation inside the training window (regressing realised
residual on predicted residual) and multiplied into every forecast, so the model
cannot be more confident out of sample than it earned the right to be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import BREAK_EVEN_WIN_RATE, MIN_EDGE_MARGIN_POINTS

logger = logging.getLogger(__name__)

# A tie with the line is a push: stake returned, no win, no loss. Lines sit on
# half-points often enough that ignoring this is a small but free error.
_PUSH_HALF_WIDTH = 0.5


def make_estimator(alpha: float) -> Pipeline:
    """Median-imputed, standardised ridge regression.

    Ridge rather than gradient boosting: with ~2,700 rows and a signal this weak,
    every tree ensemble tested fitted the noise. See `docs/model.md`.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


@dataclass
class TotalsModel:
    """A fitted totals model plus everything needed to interpret its output."""

    estimator: Pipeline
    features: list[str]
    sigma: float                 # SD of residual around the model's prediction
    shrinkage: float             # out-of-sample reliability of the raw edge, in [0, 1]
    alpha: float
    train_seasons: list[int]
    n_train: int
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def predict_edge(self, X: pd.DataFrame) -> np.ndarray:
        """Expected points above the closing total (negative = expect the under)."""
        raw = self.estimator.predict(X[self.features])
        return raw * self.shrinkage

    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        """Expected final combined score."""
        return X["total_line"].to_numpy(dtype=float) + self.predict_edge(X)

    def predict_over_prob(self, X: pd.DataFrame) -> np.ndarray:
        """P(final total finishes above the posted line), excluding pushes."""
        edge = self.predict_edge(X)
        line = X["total_line"].to_numpy(dtype=float)
        # Lines land on whole numbers about half the time; treat +-0.5 around an
        # integer line as the push band so probabilities sum with it, not over it.
        is_integer_line = np.isclose(line, np.round(line))
        half = np.where(is_integer_line, _PUSH_HALF_WIDTH, 0.0)
        p_over = 1.0 - norm.cdf((half - edge) / self.sigma)
        return np.clip(p_over, 1e-6, 1 - 1e-6)

    @property
    def break_even_edge(self) -> float:
        """Smallest edge, in points, that can pay for the vig.

        At -110 a bet needs a 52.38% win rate. Inverting the residual distribution
        says how many points of disagreement with the line that buys — about 0.8
        points at a 13-point sigma. Anything smaller is a losing bet even if the
        model is exactly right, so the threshold is derived here rather than picked
        by eye (or, worse, tuned on the backtest it is then judged by).
        """
        return float(norm.ppf(BREAK_EVEN_WIN_RATE) * self.sigma)

    @property
    def bet_threshold(self) -> float:
        """Break-even edge plus a margin of safety."""
        return self.break_even_edge + MIN_EDGE_MARGIN_POINTS

    def bet_signal(self, X: pd.DataFrame) -> np.ndarray:
        """+1 to bet the over, -1 the under, 0 to pass."""
        edge = self.predict_edge(X)
        threshold = self.bet_threshold
        return np.where(edge >= threshold, 1, np.where(edge <= -threshold, -1, 0))

    def describe(self) -> pd.DataFrame:
        """Standardised ridge coefficients, largest absolute effect first."""
        coefs = self.estimator.named_steps["ridge"].coef_
        return (
            pd.DataFrame({"feature": self.features, "points_per_sd": coefs})
            .assign(abs_effect=lambda d: d["points_per_sd"].abs())
            .sort_values("abs_effect", ascending=False)
            .drop(columns="abs_effect")
            .reset_index(drop=True)
        )


def estimate_shrinkage(
    X: pd.DataFrame, y: pd.Series, seasons: pd.Series, alpha: float, features: list[str]
) -> tuple[float, float]:
    """Measure how much of the model's edge survives out of sample.

    Walks forward one season at a time inside the training window and regresses
    the realised residual on the predicted one (through the origin). A slope of
    1.0 means the model's edges are exactly the right size; 0.3 means only 30% of
    each predicted edge is real and the rest is fitting noise.

    Returns `(shrinkage, sigma)` where sigma is the out-of-sample residual SD.
    """
    order = sorted(seasons.unique())
    preds, actuals = [], []

    for season in order[3:]:               # need a few seasons of history first
        train = seasons < season
        test = seasons == season
        if train.sum() < 200 or test.sum() == 0:
            continue
        est = make_estimator(alpha)
        est.fit(X.loc[train, features], y.loc[train])
        preds.append(est.predict(X.loc[test, features]))
        actuals.append(y.loc[test].to_numpy())

    if not preds:
        logger.warning("Not enough history to estimate shrinkage; defaulting to 0.5.")
        return 0.5, float(np.std(y))

    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)
    n = len(pred)

    denom = float(np.dot(pred, pred))
    if denom <= 0:
        return 0.0, float(np.std(actual))

    slope = float(np.dot(pred, actual) / denom)

    # The slope is itself estimated from a few hundred noisy games, so taking it
    # at face value would swing the model between "bet nothing" and "bet
    # everything" from one season to the next. Shrink it toward zero in
    # proportion to how well determined it is: t^2 / (1 + t^2) leaves a
    # confidently measured slope almost untouched and collapses a coin-flip one.
    residual_var = float(np.mean((actual - slope * pred) ** 2))
    slope_se = float(np.sqrt(residual_var / denom))
    t_stat = slope / slope_se if slope_se > 0 else 0.0
    reliability = t_stat**2 / (1.0 + t_stat**2)

    shrinkage = float(np.clip(slope * reliability, 0.0, 1.0))
    sigma = float(np.std(actual - shrinkage * pred))

    logger.info(
        "Out-of-sample edge reliability: raw slope %.2f (t=%.1f) -> shrinkage %.2f; "
        "sigma %.2f points over %d games.",
        slope, t_stat, shrinkage, sigma, n,
    )
    return shrinkage, sigma


def select_alpha(
    X: pd.DataFrame, y: pd.Series, seasons: pd.Series, features: list[str],
    candidates: list[float],
) -> float:
    """Pick the ridge penalty with the best walk-forward mean squared error."""
    order = sorted(seasons.unique())
    scores = {}

    for alpha in candidates:
        errors = []
        for season in order[3:]:
            train, test = seasons < season, seasons == season
            if train.sum() < 200 or test.sum() == 0:
                continue
            est = make_estimator(alpha)
            est.fit(X.loc[train, features], y.loc[train])
            errors.append(
                np.mean((y.loc[test].to_numpy() - est.predict(X.loc[test, features])) ** 2)
            )
        if errors:
            scores[alpha] = float(np.mean(errors))

    if not scores:
        return candidates[len(candidates) // 2]

    best = min(scores, key=scores.get)
    logger.info(
        "Ridge penalty selected: alpha=%g (walk-forward MSE %.3f). Grid: %s",
        best, scores[best], {a: round(s, 3) for a, s in scores.items()},
    )
    return best
