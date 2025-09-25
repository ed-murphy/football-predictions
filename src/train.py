import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(
    team_games: pd.DataFrame,
    model_path: str,
    train_seasons: list[int],
    test_seasons: list[int],
    random_state: int
):
    """
    Train a RandomForest model to predict the total points scored in NFL games.

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
    inspection_margin : float
        Proxy for a "strong" scoring prediction relative to Vegas total, used for internal benchmarking.
    random_state : int
        Random seed for reproducibility.
    """

    # Features for prediction
    features = [
        "total_line", # Vegas total for benchmarking
        "home_rolling_avg_points_for",
        "home_rolling_avg_points_against",
        "away_rolling_avg_points_for",
        "away_rolling_avg_points_against",
        "home_rolling_avg_qb_epa",
        "away_rolling_avg_qb_epa",
        "home_rolling_avg_def_epa",
        "away_rolling_avg_def_epa",
        "home_temperature",
        "home_wind_speed",
        "home_rolling_avg_off_pace",
        "away_rolling_avg_off_pace",
        "divisional",
        "regular_season",
        "home_short_rest",
        "away_short_rest",
        "both_short_rest"
    ]

    # Keep one row per game (home team)
    model_data = (
        team_games
        .dropna(subset=features + ["total_points"])
        .loc[lambda df: df["is_home"] == 1, ["game_id", "season", "week", "total_points"] + features]
    )

    model_data['home_x_away_pace'] = model_data['home_rolling_avg_off_pace'] * model_data['away_rolling_avg_off_pace']
    model_data['home_pace_x_wind_speed'] = model_data['home_rolling_avg_off_pace'] * model_data['home_wind_speed']
    model_data['away_pace_x_wind_speed'] = model_data['away_rolling_avg_off_pace'] * model_data['home_wind_speed']
    model_data['home_qb_x_away_def'] = model_data['home_rolling_avg_qb_epa'] * model_data['away_rolling_avg_def_epa']
    model_data['away_qb_x_home_def'] = model_data['away_rolling_avg_qb_epa'] * model_data['home_rolling_avg_def_epa']

    features.append('home_x_away_pace')
    features.append('home_pace_x_wind_speed')
    features.append('away_pace_x_wind_speed')
    features.append('home_qb_x_away_def')
    features.append('away_qb_x_home_def')

    model_data['residual'] = model_data['total_points'] - model_data['total_line']

    # Train/test split
    train_data = model_data[model_data["season"].isin(train_seasons)]
    test_data = model_data[model_data["season"].isin(test_seasons)]
    X_train, y_train = train_data[features], train_data["residual"]
    X_test, y_test = test_data[features], test_data["residual"]

    # Fit model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, model_path)

    print("Model has been trained.")

    return model, X_test, y_test, features, test_data
