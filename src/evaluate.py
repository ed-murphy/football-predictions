"""Honest evaluation of the totals model.

The previous version reported "precision at p>0.55" with no reference point, which
reads as an accuracy score. It isn't one: at standard -110 pricing you need
52.38% to break even, so 51% precision is a losing model that looks fine.

Everything here is therefore reported against an explicit benchmark:

  * log loss against the 0.693 coin-flip null — does the model know *anything*?
  * MAE of predicted total against "just repeat the closing line" — does the
    model beat the market at its own job?
  * ROI at -110 with a standard error, so a +4% return on 200 bets is visibly
    indistinguishable from zero.

Backtests use expanding-window walk-forward validation: each season is scored by
a model that only ever saw earlier seasons.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.model import TotalsModel, estimate_shrinkage, make_estimator, select_alpha
from src.model_features import get_model_features
from src.train import TARGET, build_training_data
from config import BREAK_EVEN_WIN_RATE, DECIMAL_PAYOUT, RIDGE_ALPHA_GRID

logger = logging.getLogger(__name__)

COIN_FLIP_LOG_LOSS = float(np.log(2))


def betting_record(signals: np.ndarray, actual_over: np.ndarray,
                   push: np.ndarray | None = None) -> dict:
    """Units staked, won and returned for a set of one-unit bets at -110."""
    placed = signals != 0
    n = int(placed.sum())
    if n == 0:
        return {"n_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_rate": np.nan, "roi": np.nan, "roi_se": np.nan, "profit": 0.0}

    if push is None:
        push = np.zeros_like(actual_over, dtype=bool)

    picked_over = signals == 1
    won = placed & ~push & ((picked_over & actual_over) | (~picked_over & ~actual_over))
    pushed = placed & push
    lost = placed & ~push & ~won

    profit = won.sum() * DECIMAL_PAYOUT - lost.sum()
    decided = int(won.sum() + lost.sum())

    # Per-bet returns are +0.909 / -1 / 0, so the SE of mean return follows directly.
    returns = np.where(won, DECIMAL_PAYOUT, np.where(lost, -1.0, 0.0))[placed]

    return {
        "n_bets": n,
        "wins": int(won.sum()),
        "losses": int(lost.sum()),
        "pushes": int(pushed.sum()),
        "win_rate": float(won.sum() / decided) if decided else np.nan,
        "roi": float(profit / n),
        "roi_se": float(returns.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "profit": float(profit),
    }


def calibration_table(p_over: np.ndarray, actual_over: np.ndarray,
                      bins: int = 5) -> pd.DataFrame:
    """Predicted vs realised over-rate by probability bucket."""
    edges = np.quantile(p_over, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = pd.cut(p_over, np.unique(edges), include_lowest=True)
    return (
        pd.DataFrame({"bucket": bucket, "predicted": p_over, "actual": actual_over})
        .groupby("bucket", observed=True)
        .agg(n=("actual", "size"), predicted=("predicted", "mean"),
             actual=("actual", "mean"))
        .reset_index()
        .round(3)
    )


def score_predictions(p_over: np.ndarray, pred_total: np.ndarray,
                      actual_total: np.ndarray, line: np.ndarray,
                      signals: np.ndarray) -> dict:
    """All headline metrics for one set of scored games."""
    actual_over = actual_total > line
    push = np.isclose(actual_total, line)
    decided = ~push

    record = betting_record(signals, actual_over, push)

    y, p = actual_over[decided], p_over[decided]
    both_outcomes_present = 0 < y.sum() < len(y)

    return {
        "n_games": len(p_over),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "coin_flip_log_loss": COIN_FLIP_LOG_LOSS,
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if both_outcomes_present else np.nan,
        "model_mae": float(np.mean(np.abs(actual_total - pred_total))),
        "market_mae": float(np.mean(np.abs(actual_total - line))),
        **record,
    }


def walk_forward_backtest(
    game_frame: pd.DataFrame,
    start_season: int,
    train_start_season: int,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Score every season from `start_season` with a model blind to it and later.

    Alpha and shrinkage are re-selected within each fold's training window, so the
    backtest reflects the full fitting procedure rather than hyperparameters chosen
    with hindsight.
    """
    features = get_model_features()
    data = build_training_data(game_frame).sort_values("date")
    seasons = [s for s in sorted(data["season"].unique()) if s >= start_season]

    rows, pooled = [], []
    for season in seasons:
        train = data[(data["season"] >= train_start_season) & (data["season"] < season)]
        test = data[data["season"] == season]
        if len(train) < 300 or test.empty:
            continue

        fold_alpha = alpha or select_alpha(
            train[features], train[TARGET], train["season"], features, RIDGE_ALPHA_GRID
        )
        shrinkage, sigma = estimate_shrinkage(
            train[features], train[TARGET], train["season"], fold_alpha, features
        )
        estimator = make_estimator(fold_alpha)
        estimator.fit(train[features], train[TARGET])

        model = TotalsModel(
            estimator=estimator, features=features, sigma=sigma, shrinkage=shrinkage,
            alpha=fold_alpha, train_seasons=sorted(train["season"].unique().tolist()),
            n_train=len(train),
        )

        p_over = model.predict_over_prob(test)
        pred_total = model.predict_total(test)
        actual = test["total_points"].to_numpy(dtype=float)
        line = test["total_line"].to_numpy(dtype=float)

        signals = model.bet_signal(test)
        metrics = score_predictions(p_over, pred_total, actual, line, signals)
        rows.append({"season": season, "alpha": fold_alpha,
                     "bet_threshold": round(model.bet_threshold, 2), **metrics})
        pooled.append(pd.DataFrame({
            "season": season, "p_over": p_over, "pred_total": pred_total,
            "actual_total": actual, "line": line, "signal": signals,
        }))

    if not rows:
        logger.warning("Backtest produced no folds.")
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    _log_backtest(results, pd.concat(pooled, ignore_index=True))
    return results


def _log_backtest(results: pd.DataFrame, pooled: pd.DataFrame) -> None:
    per_season = results[[
        "season", "n_games", "bet_threshold", "log_loss", "auc", "model_mae",
        "market_mae", "n_bets", "win_rate", "roi",
    ]].round(4)
    logger.info("Walk-forward backtest by season:\n%s", per_season.to_string(index=False))

    actual_over = pooled["actual_total"] > pooled["line"]
    push = np.isclose(pooled["actual_total"], pooled["line"])
    decided = ~push
    overall = score_predictions(
        pooled["p_over"].to_numpy(), pooled["pred_total"].to_numpy(),
        pooled["actual_total"].to_numpy(), pooled["line"].to_numpy(),
        pooled["signal"].to_numpy(),
    )
    _log_edge_correlation(pooled)

    logger.info(
        "Pooled %d games | log loss %.4f vs %.4f coin flip | AUC %.3f | "
        "model MAE %.2f vs market MAE %.2f",
        overall["n_games"], overall["log_loss"], COIN_FLIP_LOG_LOSS,
        overall["auc"], overall["model_mae"], overall["market_mae"],
    )

    roi, se, n = overall["roi"], overall["roi_se"], overall["n_bets"]
    if n:
        z = roi / se if se else 0.0
        logger.info(
            "Betting record: %d bets, %d-%d-%d, win rate %.1f%% "
            "(break-even %.1f%%), ROI %+.1f%% +- %.1f%% (z=%.2f)",
            n, overall["wins"], overall["losses"], overall["pushes"],
            overall["win_rate"] * 100, BREAK_EVEN_WIN_RATE * 100,
            roi * 100, se * 100, z,
        )
        if abs(z) < 2:
            logger.info(
                "That ROI is within two standard errors of zero — treat it as "
                "no demonstrated edge, not as a proven one."
            )
    else:
        logger.info("No games cleared the betting filter.")

    logger.info(
        "Calibration:\n%s",
        calibration_table(
            pooled.loc[decided, "p_over"].to_numpy(), actual_over[decided].to_numpy()
        ).to_string(index=False),
    )


def _log_edge_correlation(pooled: pd.DataFrame) -> None:
    """The single number that says whether the model knows anything at all.

    Correlation between the edge the model predicted and the residual that actually
    happened, across every game rather than only the ones bet. It is a far more
    stable read than ROI, which is dominated by which side of a coin flip 200
    filtered games landed on.
    """
    predicted = pooled["pred_total"] - pooled["line"]
    realised = pooled["actual_total"] - pooled["line"]
    n = len(pooled)
    corr = float(np.corrcoef(predicted, realised)[0, 1])
    se = 1.0 / np.sqrt(n - 3)

    logger.info(
        "Edge correlation: %+.4f +- %.4f over %d games (z=%.2f). "
        "Predicted edges span %.1f points of SD against %.1f actual.",
        corr, se, n, corr / se, predicted.std(), realised.std(),
    )


def evaluate_holdout(model: TotalsModel, game_frame: pd.DataFrame,
                     test_seasons: list[int]) -> dict:
    """Score a fitted model on seasons it was not trained on."""
    data = build_training_data(game_frame)
    test = data[data["season"].isin(test_seasons)]
    if test.empty:
        logger.warning("No holdout rows for seasons %s.", test_seasons)
        return {}

    metrics = score_predictions(
        model.predict_over_prob(test), model.predict_total(test),
        test["total_points"].to_numpy(dtype=float),
        test["total_line"].to_numpy(dtype=float),
        model.bet_signal(test),
    )
    logger.info(
        "Holdout %s: %d games | log loss %.4f (coin flip %.4f) | AUC %.3f | "
        "MAE %.2f vs market %.2f | %d bets, ROI %+.1f%% +- %.1f%%",
        test_seasons, metrics["n_games"], metrics["log_loss"], COIN_FLIP_LOG_LOSS,
        metrics["auc"], metrics["model_mae"], metrics["market_mae"],
        metrics["n_bets"], (metrics["roi"] or 0) * 100, (metrics["roi_se"] or 0) * 100,
    )
    return metrics
