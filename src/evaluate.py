import logging
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from config import RANDOM_STATE
from src.train import build_model_data, _make_xgb

logger = logging.getLogger(__name__)

# Confidence bins: abs(p_over - 0.5) in [lo, hi)
_CONF_BINS = [(0.05, 0.10), (0.10, 0.15), (0.15, 1.0)]


def _precision_at_threshold(p_over, actual_over, threshold):
    over_sig  = p_over > threshold
    under_sig = p_over < (1 - threshold)
    signals   = over_sig | under_sig
    n = int(signals.sum())
    if n == 0:
        return None, 0, 0
    correct = int(((actual_over & over_sig) | (~actual_over & under_sig)).sum())
    return correct / n, correct, n


def evaluate_model(model, X_test, y_test, features, test_data, prob_threshold):
    """
    Evaluate a trained XGBClassifier on holdout test data.
    y_test: binary Series (1=over, 0=under).
    """
    p_over     = model.predict_proba(X_test)[:, 1]
    y_true     = y_test.values
    actual_over = y_true.astype(bool)

    ll      = log_loss(y_true, p_over)
    bs      = brier_score_loss(y_true, p_over)
    auc     = roc_auc_score(y_true, p_over)
    rand_ll = log_loss(y_true, np.full_like(p_over, 0.5))

    logger.info(
        "Holdout - LogLoss: %.4f (random=%.4f)  Brier: %.4f  AUC: %.3f",
        ll, rand_ll, bs, auc,
    )

    prec, n_correct, n_signals = _precision_at_threshold(p_over, actual_over, prob_threshold)
    logger.info(
        "Precision at p>%.2f: %d/%d = %.1f%%",
        prob_threshold, n_correct, n_signals, (prec or 0) * 100,
    )

    confidence = np.abs(p_over - 0.5)
    for lo, hi in _CONF_BINS:
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        over_sig_m  = p_over[mask] > 0.5
        under_sig_m = ~over_sig_m
        ec = int(((actual_over[mask] & over_sig_m) | (~actual_over[mask] & under_sig_m)).sum())
        ep = ec / mask.sum()
        logger.info(
            "  Confidence [%.0f%%, %.0f%%): %d/%d = %.1f%%",
            lo * 100, hi * 100 if hi < 1.0 else 100,
            ec, int(mask.sum()), ep * 100,
        )

    results = {
        "LogLoss": ll, "RandomLogLoss": rand_ll, "Brier": bs, "AUC": auc,
        "Num Predictions": n_signals, "Correct Predictions": n_correct, "Precision": prec,
    }
    if hasattr(model, "feature_importances_"):
        results["Feature Importance"] = (
            pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False).to_dict()
        )
    return results


def walk_forward_cv(
    team_games: pd.DataFrame,
    features: list,
    prob_threshold: float = 0.55,
    start_test_season: int = 2018,
    train_start_season: int = 2014,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation for the over/under classifier.

    For each season >= start_test_season, trains on all prior seasons
    >= train_start_season and evaluates on that season.
    """
    model_data = build_model_data(team_games).dropna(
        subset=features + ["total_points", "total_line"]
    )
    model_data = model_data.copy()
    model_data["over"] = (model_data["total_points"] > model_data["total_line"]).astype(int)

    all_seasons  = sorted(model_data["season"].unique())
    test_seasons = [s for s in all_seasons if s >= start_test_season]

    rows = []
    for test_season in test_seasons:
        train_data = model_data[
            (model_data["season"] >= train_start_season) &
            (model_data["season"] < test_season)
        ]
        test_data  = model_data[model_data["season"] == test_season]

        if len(train_data) < 50 or test_data.empty:
            continue

        X_tr, y_tr = train_data[features], train_data["over"]
        X_te, y_te = test_data[features],  test_data["over"]

        m = _make_xgb(RANDOM_STATE)
        m.fit(X_tr, y_tr)

        p_over      = m.predict_proba(X_te)[:, 1]
        y_true      = y_te.values
        actual_over = y_true.astype(bool)

        ll  = log_loss(y_true, p_over)
        bs  = brier_score_loss(y_true, p_over)
        auc = roc_auc_score(y_true, p_over)

        prec, n_correct, n_signals = _precision_at_threshold(p_over, actual_over, prob_threshold)

        rows.append({
            "season":    test_season,
            "log_loss":  round(ll, 4),
            "brier":     round(bs, 4),
            "auc":       round(auc, 3),
            "n_signals": n_signals,
            "n_correct": n_correct,
            "precision": round(prec, 3) if prec is not None else None,
        })
        logger.info(
            "CV %d - LogLoss: %.4f  Brier: %.4f  AUC: %.3f  Precision: %s/%s",
            test_season, ll, bs, auc, n_correct, n_signals,
        )

        confidence = np.abs(p_over - 0.5)
        for lo, hi in _CONF_BINS:
            mask = (confidence >= lo) & (confidence < hi)
            if mask.sum() == 0:
                continue
            over_sig_m  = p_over[mask] > 0.5
            under_sig_m = ~over_sig_m
            ec = int(((actual_over[mask] & over_sig_m) | (~actual_over[mask] & under_sig_m)).sum())
            ep = ec / mask.sum()
            logger.info(
                "  CV %d Confidence [%.0f%%, %.0f%%): %d/%d = %.1f%%",
                test_season, lo * 100, hi * 100 if hi < 1.0 else 100,
                ec, int(mask.sum()), ep * 100,
            )

    cv_results = pd.DataFrame(rows)
    if not cv_results.empty:
        avg = cv_results[["log_loss", "brier", "auc", "precision"]].mean()
        logger.info(
            "Walk-forward CV averages - LogLoss: %.4f  Brier: %.4f  AUC: %.3f  Precision: %.3f",
            avg["log_loss"], avg["brier"], avg["auc"], avg["precision"],
        )
    return cv_results
