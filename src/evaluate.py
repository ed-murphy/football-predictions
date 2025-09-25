import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

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
    r2 = r2_score(y_test_total, y_pred_total)
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    results = {"MAE": mae, "R2": r2}

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

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False)
        results["Feature Importance"] = feature_importance.to_dict()
    
    print(results)

    return results
