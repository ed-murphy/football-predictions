import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH
from src.model_features import BASE_FEATURES, add_engineered_features, get_model_features

logger = logging.getLogger(__name__)


def build_model_data(team_games: pd.DataFrame) -> pd.DataFrame:
    model_data = team_games.loc[
        team_games["is_home"] == 1,
        ["game_id", "season", "week", "total_points"] + BASE_FEATURES
    ].copy()
    return add_engineered_features(model_data)


def train_model(
    team_games: pd.DataFrame,
    model_path: str,
    train_seasons: list[int],
    test_seasons: list[int],
    random_state: int
):
    """
    Train a RandomForest model to predict the total points scored in NFL games.
    """
    features = get_model_features()
    model_data = build_model_data(team_games)
    model_data = model_data.dropna(subset=features + ["total_points"])
    model_data["residual"] = model_data["total_points"] - model_data["total_line"]

    train_data = model_data[model_data["season"].isin(train_seasons)]
    test_data = model_data[model_data["season"].isin(test_seasons)]
    X_train, y_train = train_data[features], train_data["residual"]
    X_test, y_test = test_data[features], test_data["residual"]

    model = RandomForestRegressor(
        n_estimators=MODEL_N_ESTIMATORS,
        max_depth=MODEL_MAX_DEPTH,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    logger.info("Model trained and saved to %s.", model_path)
    return model, X_test, y_test, features, test_data
