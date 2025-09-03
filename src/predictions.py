import pandas as pd
import os
from datetime import datetime

def save_predictions(upcoming_team_games: pd.DataFrame):
    """
    Generates a table of predictions for upcoming NFL games.

    predictions: DataFrame with columns ['date', 'home_team', 'away_team', 'predicted_total']
    Saves output to predictions/predictions_mmddyy[_vN].csv
    """

    # Create simple data frame containing predictions
    predictions = upcoming_team_games[['date', 'home_team', 'away_team', 'predicted_total']]

    # Remove duplicate columns if any
    predictions = predictions.loc[:, ~predictions.columns.duplicated()]

    # --- build output directory ---
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)  # create folder if not exists

    today_str = datetime.today().strftime("%Y%m%d")
    base_filename = f"predictions_{today_str}.csv"
    output_path = os.path.join(predictions_dir, base_filename)

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
