import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_and_evaluate(
    team_games: pd.DataFrame,
    model_path: str = "model/rf_total_points_model.joblib",
    train_seasons: list[int] = [2021, 2022, 2023],
    test_seasons: list[int] = [2024],
    margin: float = 3.5,
    random_state: int = 42,
):
    """
    Train and evaluate a RandomForestRegressor to predict total points scored in NFL games.
    Includes evaluation for when to predict Over/Under vs. Vegas line.

    Parameters
    ----------
    team_games : pd.DataFrame
        Team-game level dataset (with both home/away features).
    model_path : str
        Where to save the trained model.
    train_seasons : list[int]
        Seasons used for training.
    test_seasons : list[int]
        Seasons used for testing.
    margin : float
        Minimum difference between model prediction and Vegas total to signal a prediction.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestRegressor
        Trained model.
    results : dict
        Evaluation metrics and betting precision.
    """

    # Features for prediction
    features = [
        "total_line",  # Vegas total
        "home_avg_points_for", "home_avg_points_against",
        "away_avg_points_for", "away_avg_points_against",
        "home_qb_avg_epa", "away_qb_avg_epa",
        "home_def_epa", "away_def_epa",
        "home_temperature", "home_wind_speed",
        "home_pace_last5", "away_pace_last5",
    ]

    # Keep one row per game (home team)
    model_data = (
        team_games
        .dropna(subset=features + ["total_points"])
        .loc[lambda df: df["is_home"] == 1, ["game_id", "season", "week", "total_points"] + features]
    )

    # Train/test split
    train_data = model_data[model_data["season"].isin(train_seasons)]
    test_data = model_data[model_data["season"].isin(test_seasons)]

    X_train, y_train = train_data[features], train_data["total_points"]
    X_test, y_test = test_data[features], test_data["total_points"]
    vegas_test = test_data["total_line"].values

    # Fit model
    model = RandomForestRegressor(n_estimators=500, random_state=random_state)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, model_path)

    # Feature importance
    feature_importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    # Predictions
    y_pred = model.predict(X_test)

    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Prediction signals
    over_signals = y_pred > vegas_test + margin
    under_signals = y_pred < vegas_test - margin
    prediction_signals = over_signals | under_signals

    num_predictions = prediction_signals.sum()
    correct_predictions = (
        ((y_test > vegas_test) & over_signals) |
        ((y_test < vegas_test) & under_signals)
    ).sum()
    precision = correct_predictions / num_predictions if num_predictions > 0 else None

    print("Model has been trained.")

    results = {
        "MAE": mae,
        "R2": r2,
        "Num Predictions": int(num_predictions),
        "Correct Predictions": int(correct_predictions),
        "Precision": precision,
        "Feature Importance": feature_importance.to_dict(),
    }

    print(results)

    return model
