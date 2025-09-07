import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, features, test_data, precision_margin):
    """
    Evaluate a trained model on test data and print/return metrics and feature importance.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    results = {"MAE": mae, "R2": r2}

    # Betting precision if 'total_line' in test_data
    if "total_line" in test_data:
        vegas_test = test_data["total_line"].values
        over_signals = y_pred > vegas_test + precision_margin
        under_signals = y_pred < vegas_test - precision_margin
        prediction_signals = over_signals | under_signals
        num_predictions = prediction_signals.sum()
        correct_predictions = (
            ((y_test > vegas_test) & over_signals) |
            ((y_test < vegas_test) & under_signals)
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
