import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from config import MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, RANDOM_STATE
from src.train import build_model_data

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, features, test_data, precision_margin):
    """
    Evaluate a trained model on test data and print/return metrics and feature importance.
    """
    # Predict residuals and reconstruct predicted total points
    y_pred_residual = model.predict(X_test)
    vegas_test = X_test['total_line'].values
    y_pred_total = vegas_test + y_pred_residual
    y_test_total = vegas_test + y_test  # y_test is residual

    mae = mean_absolute_error(y_test_total, y_pred_total)
    rmse = root_mean_squared_error(y_test_total, y_pred_total)
    r2 = r2_score(y_test_total, y_pred_total)
    market_mae = mean_absolute_error(y_test_total, vegas_test)
    all_games_directional_accuracy = (
        (y_test_total > vegas_test) == (y_pred_total > vegas_test)
    ).mean()
    logger.info(
        "Holdout MAE: %.2f  RMSE: %.2f  R²: %.3f  VsMarket MAE Δ: %.2f",
        mae,
        rmse,
        r2,
        market_mae - mae,
    )

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Market MAE": market_mae,
        "MAE Improvement vs Market": market_mae - mae,
        "Directional Accuracy (All Games)": all_games_directional_accuracy,
    }

    # Precision
    over_signals = y_pred_total > vegas_test + precision_margin
    under_signals = y_pred_total < vegas_test - precision_margin
    prediction_signals = over_signals | under_signals
    num_predictions = prediction_signals.sum()
    correct_predictions = (
        ((y_test_total > vegas_test) & over_signals) |
        ((y_test_total < vegas_test) & under_signals)
    ).sum()
    precision = correct_predictions / num_predictions if num_predictions > 0 else None
    results.update({
        "Num Predictions": int(num_predictions),
        "Correct Predictions": int(correct_predictions),
        "Precision": precision
    })
    logger.info("Precision at ±%d pts: %s/%s = %.1f%%",
                precision_margin, int(correct_predictions), int(num_predictions),
                (precision or 0) * 100)

    abs_edge = np.abs(y_pred_total - vegas_test)
    edge_bins = [(2, 4), (4, 6), (6, 100)]
    for lo, hi in edge_bins:
        edge_mask = (abs_edge >= lo) & (abs_edge < hi)
        if edge_mask.sum() == 0:
            continue
        edge_correct = (
            ((y_test_total > vegas_test) & (y_pred_total > vegas_test)) |
            ((y_test_total < vegas_test) & (y_pred_total < vegas_test))
        )[edge_mask].sum()
        edge_precision = edge_correct / edge_mask.sum()
        results[f"Precision |edge| {lo}-{hi}"] = edge_precision

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False)
        results["Feature Importance"] = feature_importance.to_dict()

    return results


def walk_forward_cv(
    team_games: pd.DataFrame,
    features: list,
    precision_margin: int = 4,
    start_test_season: int = 2018,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    For each season >= start_test_season, train on ALL prior seasons and
    evaluate on that season.  Returns a DataFrame with per-season metrics so
    you can see whether accuracy is stable across years.

    Parameters
    ----------
    team_games : pd.DataFrame
        Full team-game dataset (same one passed to train_model).
    features : list
        Feature column names used by the model (including interaction terms
        if they should be computed here).
    precision_margin : int
        Minimum |predicted - Vegas| to count as a signal.
    start_test_season : int
        First season to use as a test fold.

    Returns
    -------
    pd.DataFrame
        Columns: season, mae, r2, n_signals, n_correct, precision
    """
    # Build model-ready data (mirrors logic in train_model)
    model_data = build_model_data(team_games).dropna(subset=features + ["total_points"])

    model_data['residual'] = model_data['total_points'] - model_data['total_line']

    all_seasons = sorted(model_data['season'].unique())
    test_seasons = [s for s in all_seasons if s >= start_test_season]

    rows = []
    for test_season in test_seasons:
        train_data = model_data[model_data['season'] < test_season]
        test_data  = model_data[model_data['season'] == test_season]

        if len(train_data) < 50 or test_data.empty:
            continue

        X_tr, y_tr = train_data[features], train_data['residual']
        X_te, y_te = test_data[features],  test_data['residual']

        m = RandomForestRegressor(
            n_estimators=MODEL_N_ESTIMATORS,
            max_depth=MODEL_MAX_DEPTH,
            random_state=RANDOM_STATE
        )
        m.fit(X_tr, y_tr)

        vegas  = X_te['total_line'].values
        y_pred = vegas + m.predict(X_te)
        y_true = vegas + y_te.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        market_mae = mean_absolute_error(y_true, vegas)

        over_sig  = y_pred > vegas + precision_margin
        under_sig = y_pred < vegas - precision_margin
        signals   = over_sig | under_sig
        n_signals = int(signals.sum())
        n_correct = int(
            (((y_true > vegas) & over_sig) | ((y_true < vegas) & under_sig)).sum()
        )
        prec = n_correct / n_signals if n_signals > 0 else None

        rows.append({
            'season':    test_season,
            'mae':       round(mae, 2),
            'rmse':      round(rmse, 2),
            'r2':        round(r2, 3),
            'market_mae': round(market_mae, 2),
            'mae_vs_market': round(market_mae - mae, 2),
            'n_signals': n_signals,
            'n_correct': n_correct,
            'precision': round(prec, 3) if prec is not None else None,
        })
        logger.info("CV %d — MAE: %.2f  R²: %.3f  Precision: %s/%s",
                    test_season, mae, r2, n_correct, n_signals)

    cv_results = pd.DataFrame(rows)
    if not cv_results.empty:
        avg = cv_results[['mae', 'rmse', 'r2', 'mae_vs_market', 'precision']].mean()
        logger.info(
            "Walk-forward CV averages — MAE: %.2f  RMSE: %.2f  R²: %.3f  VsMarket MAE Δ: %.2f  Precision: %.3f",
            avg['mae'], avg['rmse'], avg['r2'], avg['mae_vs_market'], avg['precision']
        )
    return cv_results
