from src.load import load_data
from src.basic import create_basic_features
from src.pace import create_pace_features
from src.weather import create_weather_features
from src.totals import get_totals
from src.qb import create_qb_features
from src.defense import create_defense_features
from src.model import train_and_evaluate
from src.weather_forecast import get_forecasted_weather
from src.upcoming import prepare_upcoming_team_games
from src.predictions import save_predictions


def main():

    # Load historical play-level and game-level data using nfl_data_py
    games, plays = load_data()

    # Add basic historical features like average points for/against 
    team_games = create_basic_features(games)

    # Add QB EPA features
    team_games = create_qb_features(team_games, plays)

    # Add defense EPA features
    team_games = create_defense_features(team_games, plays)

    # Add pace features
    team_games = create_pace_features(team_games, plays)

    # Add weather features
    team_games = create_weather_features(team_games)

    # Load Vegas totals for upcoming games
    totals = get_totals()

    # Load weather forecasts
    weather_features = get_forecasted_weather(totals)

    # Train model and print train/test results
    model = train_and_evaluate(team_games)

    # Prepare upcoming games with all features and predict
    upcoming_team_games = prepare_upcoming_team_games(
        totals,
        team_games,
        weather_features,
        model
    )

    # Use model to generate predictions for upcoming games
    save_predictions(upcoming_team_games)

if __name__ == "__main__":
    main()
