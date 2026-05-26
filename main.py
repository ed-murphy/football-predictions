import logging
import os
import argparse
import sys

import joblib

from src.load import load_data, load_injuries
from src.basic import create_basic_features
from src.pace import create_pace_features
from src.weather import create_weather_features
from src.totals import get_totals
from src.qb import create_qb_features
from src.defense import create_defense_features
from src.train import train_model
from src.weather_forecast import get_forecasted_weather
from src.upcoming import prepare_upcoming_team_games
from src.predictions import save_predictions
from src.rest import create_rest_features
from src.evaluate import evaluate_model, walk_forward_cv
from src.redzone import create_red_zone
from src.old_predictions import get_existing_predictions
from src.turnovers import create_turnover_features
from src.third_down import create_third_down_features
from src.offense import create_offense_features
from src.injuries import create_injury_features
from src.referee import create_referee_features
from config import (
    EVAL_TRAIN_SEASONS, EVAL_TEST_SEASONS, PROD_SEASONS,
    RANDOM_STATE, PROB_THRESHOLD, TRAIN_START_SEASON,
    EVAL_MODEL_PATH, PROD_MODEL_PATH,
)


def _setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def _build_features():
    """Load data and build all historical team-game features."""
    games, plays = load_data()
    injuries = load_injuries()
    team_games = create_basic_features(games)
    team_games, latest_qb_epa = create_qb_features(team_games, plays)
    team_games = create_defense_features(team_games, plays)
    team_games = create_pace_features(team_games, plays)
    team_games = create_rest_features(team_games)
    team_games = create_red_zone(team_games, plays)
    team_games = create_turnover_features(team_games, plays)
    team_games = create_third_down_features(team_games, plays)
    team_games = create_offense_features(team_games, plays)
    if injuries is not None:
        team_games = create_injury_features(team_games, injuries)
    team_games, ref_stats, ref_global_mean = create_referee_features(team_games, games)
    team_games = create_weather_features(team_games)
    return games, team_games, latest_qb_epa, injuries, ref_stats, ref_global_mean


def run_training(team_games):
    """Train eval and prod models, evaluate, and save both to disk."""
    # Eval model (holdout last season)
    model, X_test, y_test, features, test_data = train_model(
        team_games=team_games,
        model_path=EVAL_MODEL_PATH,
        train_seasons=EVAL_TRAIN_SEASONS,
        test_seasons=EVAL_TEST_SEASONS,
        random_state=RANDOM_STATE,
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        features,
        test_data,
        prob_threshold=PROB_THRESHOLD,
    )

    cv_results = walk_forward_cv(
        team_games=team_games,
        features=features,
        prob_threshold=PROB_THRESHOLD,
        start_test_season=2018,
        train_start_season=TRAIN_START_SEASON,
    )
    if not cv_results.empty:
        logging.getLogger(__name__).info(
            "Walk-forward CV summary:\n%s", cv_results.to_string(index=False)
        )

    # Prod model (all available seasons)
    train_model(
        team_games=team_games,
        model_path=PROD_MODEL_PATH,
        train_seasons=PROD_SEASONS,
        test_seasons=[],
        random_state=RANDOM_STATE,
    )
    logging.getLogger(__name__).info("Training complete. Models saved to disk.")


def run_predictions(totals, games, team_games, latest_qb_epa, injuries=None,
                    ref_stats=None, ref_global_mean=None):
    """Load the saved prod model and generate predictions for upcoming games."""
    if not os.path.exists(PROD_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {PROD_MODEL_PATH}. Run with --train_only first."
        )

    prod_model = joblib.load(PROD_MODEL_PATH)
    logging.getLogger(__name__).info("Loaded prod model from %s.", PROD_MODEL_PATH)

    weather_features = get_forecasted_weather(totals)
    existing_predictions = get_existing_predictions()

    upcoming_team_games = prepare_upcoming_team_games(
        totals,
        team_games,
        latest_qb_epa,
        weather_features,
        prod_model,
        existing_predictions,
        injuries=injuries,
        games=games,
        ref_stats=ref_stats,
        ref_global_mean=ref_global_mean,
    )

    predictions = save_predictions(existing_predictions, upcoming_team_games, games)
    return predictions



if __name__ == "__main__":
    _setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_only", action="store_true",
                        help="Train and save models without making predictions")
    parser.add_argument("--use_cached_totals", action="store_true",
                        help="Skip fetching new totals, use cached CSV")
    args = parser.parse_args()

    games, team_games, latest_qb_epa, injuries, ref_stats, ref_global_mean = _build_features()

    if args.train_only:
        run_training(team_games)
        sys.exit(0)

    # Prediction mode — load saved model, no retraining.
    # team_games is passed so get_totals can auto-populate QB/rest/international fields.
    totals = get_totals(use_cache_only=args.use_cached_totals, team_games=team_games)

    run_predictions(totals, games, team_games, latest_qb_epa, injuries=injuries,
                    ref_stats=ref_stats, ref_global_mean=ref_global_mean)

