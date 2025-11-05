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
from src.evaluate import evaluate_model
from src.redzone import create_red_zone
from src.old_predictions import get_existing_predictions
import argparse
import sys


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

    # Add rest features
    team_games = create_rest_features(team_games)

    # Add red zone efficiency feature
    team_games = create_red_zone(team_games, plays)

    # Add weather features
    team_games = create_weather_features(team_games)

    # Load weather forecasts
    weather_features = get_forecasted_weather(totals)

    # Train model for validation (holdout 2024)
    model, X_test, y_test, features, test_data = train_model(
        team_games = team_games,
        model_path = "model/rf_total_points_model_eval.joblib",
        train_seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        test_seasons = [2024],
        random_state = 42
    )

    # Evaluate model
    evaluate_model(
        model,
        X_test,
        y_test,
        features,
        test_data,
        precision_margin=4
    )

    # Production workflow: retrain using all available data (2021-2024)
    prod_model, _, _, _, _ = train_model(
        team_games = team_games,
        model_path = "model/rf_total_points_model_prod.joblib",
        train_seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        test_seasons = [],  # No test set for production
        random_state = 42
    )

    # fetch existing predictions
    existing_predictions = get_existing_predictions()

    # prepare upcoming games data frame and make new predictions
    upcoming_team_games = prepare_upcoming_team_games(
        totals,
        team_games,
        latest_qb_epa,
        weather_features,
        prod_model,
        existing_predictions
    )

    # Use model to generate predictions for upcoming games (and add points scored in completed games)
    predictions = save_predictions(existing_predictions, upcoming_team_games, games)

    return predictions


if __name__ == "__main__":

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
