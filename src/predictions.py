import pandas as pd
import os
from datetime import datetime

def save_predictions(existing_predictions: pd.DataFrame,
                     upcoming_team_games: pd.DataFrame,
                     games: pd.DataFrame,
                     predictions_dir: str = "predictions"):
    """
    Generates a table of predictions for upcoming NFL games.
    Appends new predictions to the latest existing predictions file if present.
    Saves output to predictions/predictions_YYYYMMDD[_vN].csv
    """
    if upcoming_team_games is None or upcoming_team_games.empty:
        print("No new upcoming games to predict. Exiting function.")
        return None

    existing_predictions = existing_predictions.copy()
    existing_predictions = existing_predictions.drop(columns=['pred_date'], errors='ignore')

    upcoming_team_games = upcoming_team_games.copy()
    games = games.copy()

    upcoming_team_games['date'] = pd.to_datetime(upcoming_team_games['date']).dt.strftime('%Y-%m-%d')

    col_map = {
        'date' : 'date',
        'team_home' : 'home_team',
        'team_away' : 'away_team',
        'total_line' : 'total_line',
        'predicted_total' : 'predicted_total'
    }

    new_rows = upcoming_team_games[list(col_map.keys())].rename(columns=col_map)

    new_rows = new_rows.loc[:, [c for c in new_rows.columns if c in existing_predictions.columns]]

    combined = pd.concat([existing_predictions, new_rows], ignore_index=True)

    combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')

    combined['predicted_total'] = pd.to_numeric(combined['predicted_total'], errors='coerce')

    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
    games['gameday'] = pd.to_datetime(games['gameday']).dt.strftime('%Y-%m-%d')

    combined = combined.drop(columns=[c for c in combined.columns if 'actual_total' in c], errors='ignore')

    combined = combined.merge(
        games[['gameday', 'home_team', 'away_team', 'total_points']],
        left_on=['date', 'home_team', 'away_team'],
        right_on=['gameday', 'home_team', 'away_team'],
        how='left'
    )

    combined = combined.rename(columns={'total_points': 'actual_total'})

    combined = combined.drop(columns=['gameday'], errors='ignore')

    combined['predicted_total'] = combined['predicted_total'].round(1)

    os.makedirs(predictions_dir, exist_ok=True)

    today_str = datetime.today().strftime('%Y%m%d')
    base_filename = f"predictions_{today_str}.csv"
    filepath = os.path.join(predictions_dir, base_filename)

    version = 1
    while os.path.exists(filepath):
        version += 1
        filepath = os.path.join(predictions_dir, f"predictions_{today_str}_v{version}.csv")

    combined.to_csv(filepath, index=False)

    print(f"Predictions saved to {filepath}")

    return combined
