import pandas as pd
import os
from datetime import datetime

def save_predictions(upcoming_team_games: pd.DataFrame, games:pd.DataFrame):
    """
    Generates a table of predictions for upcoming NFL games.
    Appends new predictions to the latest existing predictions file if present.
    Saves output to predictions/predictions_mmddyy[_vN].csv
    """

    # Create simple data frame containing predictions
    predictions = upcoming_team_games[['date', 'home_team', 'away_team', 'total_line', 'predicted_total']]

    # Compute actual points from games DataFrame if available
    if {'home_score', 'away_score', 'gameday'}.issubset(games.columns):
        games['actual_total'] = games['home_score'] + games['away_score']
        # Make sure date columns match
        games['date'] = pd.to_datetime(games['gameday']).dt.date
        predictions['date'] = pd.to_datetime(predictions['date']).dt.date

        # Merge actual totals
        predictions = predictions.merge(
            games[['home_team', 'away_team', 'date', 'actual_total']],
            on=['home_team', 'away_team', 'date'],
            how='left'
        )

        # Remove duplicate columns if any (e.g., from the merge)
        predictions = predictions.loc[:, ~predictions.columns.duplicated()]

    else:
        # If actuals aren't available, fill with NaN
        predictions['actual_total'] = pd.NA

    # Remove duplicate columns if any
    predictions = predictions.loc[:, ~predictions.columns.duplicated()]

    # --- build output directory ---
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)  # create folder if not exists

    today_str = datetime.today().strftime("%Y%m%d")
    base_filename = f"predictions_{today_str}.csv"
    output_path = os.path.join(predictions_dir, base_filename)

    # --- Append to latest existing predictions if available ---
    existing_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    if existing_files:
        latest_file = max(existing_files, key=lambda f: os.path.getmtime(os.path.join(predictions_dir, f)))
        latest_path = os.path.join(predictions_dir, latest_file)
        existing_df = pd.read_csv(latest_path)

        # Combine old and new, keeping last entry for duplicates
        predictions = pd.concat([existing_df, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')

    # --- check for existing file and add version suffix if needed ---
    version = 1
    while os.path.exists(output_path):
        version += 1
        filename = f"predictions_{today_str}_v{version}.csv"
        output_path = os.path.join(predictions_dir, filename)

    # --- save to CSV ---
    predictions.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(predictions)

    return predictions
