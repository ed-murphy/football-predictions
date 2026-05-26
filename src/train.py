import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier
from config import (
    MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_LEARNING_RATE,
    MODEL_SUBSAMPLE, MODEL_COLSAMPLE_BYTREE, MODEL_MIN_CHILD_WEIGHT,
    MODEL_REG_ALPHA, MODEL_REG_LAMBDA,
)
from src.model_features import BASE_FEATURES, add_engineered_features, get_model_features

logger = logging.getLogger(__name__)


def build_model_data(team_games: pd.DataFrame) -> pd.DataFrame:
    model_data = team_games.loc[
        team_games["is_home"] == 1,
        ["game_id", "season", "week", "total_points", "total_line"] + BASE_FEATURES
    ].copy()
    return add_engineered_features(model_data)


def _make_xgb(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        max_depth=MODEL_MAX_DEPTH,
        learning_rate=MODEL_LEARNING_RATE,
        subsample=MODEL_SUBSAMPLE,
        colsample_bytree=MODEL_COLSAMPLE_BYTREE,
        min_child_weight=MODEL_MIN_CHILD_WEIGHT,
        reg_alpha=MODEL_REG_ALPHA,
        reg_lambda=MODEL_REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def train_model(
    team_games: pd.DataFrame,
    model_path: str,
    train_seasons: list[int],
    test_seasons: list[int],
    random_state: int
):
    """
    Train an XGBoost classifier to predict P(actual_total > total_line).
    """
    features = get_model_features()
    model_data = build_model_data(team_games)
    model_data = model_data.dropna(subset=features + ["total_points", "total_line"])

    # Binary target: 1 = over, 0 = under/push
    model_data = model_data.copy()
    model_data["over"] = (model_data["total_points"] > model_data["total_line"]).astype(int)

    train_data = model_data[model_data["season"].isin(train_seasons)]
    test_data  = model_data[model_data["season"].isin(test_seasons)]
    X_train, y_train = train_data[features], train_data["over"]
    X_test,  y_test  = test_data[features],  test_data["over"]

    model = _make_xgb(random_state)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    logger.info("Model trained and saved to %s.", model_path)
    return model, X_test, y_test, features, test_data
