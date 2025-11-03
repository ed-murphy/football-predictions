import pandas as pd
import os
from datetime import datetime

def save_predictions(upcoming_team_games: pd.DataFrame, games: pd.DataFrame):
    """
    Generates a table of predictions for upcoming NFL games.
    Appends new predictions to the latest existing predictions file if present.
    Saves output to predictions/predictions_mmddyy[_vN].csv
    """
    upcoming_team_games = upcoming_team_games.loc[:, ~upcoming_team_games.columns.duplicated()]

    # Keep only relevant columns from upcoming predictions
    predictions = upcoming_team_games[['date', 'home_team', 'away_team', 'total_line', 'predicted_total']].copy()
    predictions['date'] = pd.to_datetime(predictions['date']).dt.date

    # Compute actual points from games DataFrame if available
    if {'home_score', 'away_score', 'gameday'}.issubset(games.columns):
        games_subset = games[['home_team', 'away_team', 'gameday', 'home_score', 'away_score']].copy()
        games_subset['date'] = pd.to_datetime(games_subset['gameday']).dt.date
        games_subset['actual_total'] = games_subset['home_score'] + games_subset['away_score']

        # Remove duplicate column names
        games_subset = games_subset.loc[:, ~games_subset.columns.duplicated()]

        # Remove duplicate rows for the same game
        games_subset = games_subset.drop_duplicates(subset=['home_team', 'away_team', 'date'])

        # Merge actual totals into predictions
        predictions = predictions.merge(
            games_subset[['home_team', 'away_team', 'date', 'actual_total']],
            on=['home_team', 'away_team', 'date'],
            how='left'
        )

    else:
        predictions['actual_total'] = pd.NA

    # Ensure no duplicate columns
    predictions = predictions.loc[:, ~predictions.columns.duplicated()]

    # --- build output directory ---
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    today_str = datetime.today().strftime("%Y%m%d")
    base_filename = f"predictions_{today_str}.csv"
    output_path = os.path.join(predictions_dir, base_filename)

    # --- Append to latest existing predictions if available ---
    existing_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    if existing_files:
        latest_file = max(existing_files, key=lambda f: os.path.getmtime(os.path.join(predictions_dir, f)))
        latest_path = os.path.join(predictions_dir, latest_file)
        existing_df = pd.read_csv(latest_path)
        predictions = pd.concat([existing_df, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')

    # --- check for existing file and add version suffix if needed ---
    version = 1
    while os.path.exists(output_path):
        version += 1
        output_path = os.path.join(predictions_dir, f"predictions_{today_str}_v{version}.csv")

    # --- save to CSV ---
    predictions.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    return predictions
