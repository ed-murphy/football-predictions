import pandas as pd
import os
from datetime import datetime

def save_predictions(upcoming_team_games: pd.DataFrame, games: pd.DataFrame):
    """
    Generates a table of predictions for upcoming NFL games.
    Appends new predictions to the latest existing predictions file if present.
    Saves output to predictions/predictions_YYYYMMDD[_vN].csv
    """
    # Make a safe copy and remove duplicated columns
    upcoming_team_games = upcoming_team_games.loc[:, ~upcoming_team_games.columns.duplicated()].copy()
    
    # Normalize 'date' to datetime.date
    upcoming_team_games['date'] = pd.to_datetime(upcoming_team_games['date']).dt.date
    
    # Keep only relevant columns for predictions
    predictions = upcoming_team_games[['date', 'home_team', 'away_team', 'total_line', 'predicted_total']].copy()
    
    # Remove any duplicate rows
    predictions = predictions.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    
    # Compute actual points from games DataFrame if available
    if {'home_score', 'away_score', 'gameday'}.issubset(games.columns):
        games_subset = games[['home_team', 'away_team', 'gameday', 'home_score', 'away_score']].copy()
        games_subset['date'] = pd.to_datetime(games_subset['gameday']).dt.date
        games_subset['actual_total'] = games_subset['home_score'] + games_subset['away_score']
        
        # Keep only necessary columns and drop duplicates
        games_subset = games_subset[['home_team', 'away_team', 'date', 'actual_total']].drop_duplicates(
            subset=['home_team', 'away_team', 'date']
        ).copy()
        
        # Merge actual totals into predictions
        predictions = predictions.merge(
            games_subset,
            on=['home_team', 'away_team', 'date'],
            how='left'
        )
    else:
        predictions['actual_total'] = pd.NA
    
    # Final deduplication
    predictions = predictions.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
    
    # --- build output directory ---
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    today_str = datetime.today().strftime("%Y%m%d")
    base_filename = f"predictions_{today_str}.csv"
    output_path = os.path.join(predictions_dir, base_filename)

    # --- check for existing file and add version suffix if needed ---
    version = 1
    while os.path.exists(output_path):
        version += 1
        output_path = os.path.join(predictions_dir, f"predictions_{today_str}_v{version}.csv")

    # --- save to CSV ---
    predictions.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    return predictions
