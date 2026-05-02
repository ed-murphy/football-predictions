import logging
import os
import argparse
import sys

from src.load import load_data
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
from config import (
    EVAL_TRAIN_SEASONS, EVAL_TEST_SEASONS, PROD_SEASONS,
    RANDOM_STATE, PRECISION_MARGIN,
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


def run_analysis(totals):

    # Load historical play-level and game-level data using nfl_data_py
    games, plays = load_data()

    # Add basic historical features like average points for/against
    team_games = create_basic_features(games)

    # Add QB EPA features
    team_games, latest_qb_epa = create_qb_features(team_games, plays)

    # Add defense EPA features
    team_games = create_defense_features(team_games, plays)

    # Add pace features
    team_games = create_pace_features(team_games, plays)

    # Add rest features (short rest + post-bye)
    team_games = create_rest_features(team_games)

    # Add red zone efficiency feature
    team_games = create_red_zone(team_games, plays)

    # Add offensive turnover rate (INTs + fumbles lost)
    team_games = create_turnover_features(team_games, plays)

    # Add third-down conversion rate
    team_games = create_third_down_features(team_games, plays)

    # Add weather features
    team_games = create_weather_features(team_games)

    # Load weather forecasts
    weather_features = get_forecasted_weather(totals)

    # Train model for validation (holdout last season)
    model, X_test, y_test, features, test_data = train_model(
        team_games=team_games,
        model_path=EVAL_MODEL_PATH,
        train_seasons=EVAL_TRAIN_SEASONS,
        test_seasons=EVAL_TEST_SEASONS,
        random_state=RANDOM_STATE,
    )

    # Single-season holdout evaluation
    evaluate_model(
        model,
        X_test,
        y_test,
        features,
        test_data,
        precision_margin=PRECISION_MARGIN,
    )

    # Walk-forward cross-validation for an honest multi-year accuracy picture
    cv_results = walk_forward_cv(
        team_games=team_games,
        features=features,
        precision_margin=PRECISION_MARGIN,
        start_test_season=2018,
    )
    if not cv_results.empty:
        logging.getLogger(__name__).info(
            "Walk-forward CV summary:\n%s", cv_results.to_string(index=False)
        )

    # Production workflow: retrain using all available data
    prod_model, _, _, _, _ = train_model(
        team_games=team_games,
        model_path=PROD_MODEL_PATH,
        train_seasons=PROD_SEASONS,
        test_seasons=[],
        random_state=RANDOM_STATE,
    )

    # Fetch existing predictions
    existing_predictions = get_existing_predictions()

    # Prepare upcoming games data frame and make new predictions
    upcoming_team_games = prepare_upcoming_team_games(
        totals,
        team_games,
        latest_qb_epa,
        weather_features,
        prod_model,
        existing_predictions,
    )

    # Save predictions (and merge actuals for completed games)
    predictions = save_predictions(existing_predictions, upcoming_team_games, games)

    return predictions


if __name__ == "__main__":
    _setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cached_totals", action="store_true",
                        help="Skip fetching new totals, use cached CSV")
    args = parser.parse_args()

    # Fetch totals with or without skipping API
    totals = get_totals(use_cache_only=args.use_cached_totals)

    # Stop here if NOT using cached totals
    if not args.use_cached_totals:
        print("Fetched fresh totals, stopping run so user can add manual-only features.")
        sys.exit(0)

    # Run full analysis using these totals
    predictions = run_analysis(totals)
