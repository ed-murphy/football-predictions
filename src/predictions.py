import logging
import pandas as pd
import os
from datetime import datetime
from config import PROB_THRESHOLD

logger = logging.getLogger(__name__)


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
        logger.info("No new upcoming games to predict. Exiting function.")
        return None

    existing_predictions = existing_predictions.copy()
    existing_predictions = existing_predictions.drop(columns=['pred_date'], errors='ignore')

    upcoming_team_games = upcoming_team_games.copy()
    games = games.copy()

    upcoming_team_games['date'] = pd.to_datetime(upcoming_team_games['date']).dt.strftime('%Y-%m-%d')

    col_map = {
        'date'            : 'date',
        'team_home'       : 'home_team',
        'team_away'       : 'away_team',
        'total_line'      : 'total_line',
        'p_over'          : 'p_over',
        'line_open'       : 'line_open',
        'line_movement'   : 'line_movement',
        'home_qb_injured' : 'home_qb_injured',
        'away_qb_injured' : 'away_qb_injured',
    }

    new_rows = upcoming_team_games[[
        c for c in col_map.keys() if c in upcoming_team_games.columns
    ]].rename(columns=col_map)

    # Keep all new_rows columns — don't drop informational ones missing from history
    new_rows = new_rows.loc[:, [c for c in new_rows.columns
                                if c in existing_predictions.columns
                                or c in ('line_open', 'line_movement',
                                         'home_qb_injured', 'away_qb_injured',
                                         'p_over')]]

    combined = pd.concat([existing_predictions, new_rows], ignore_index=True)

    combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')

    combined['p_over'] = pd.to_numeric(combined['p_over'], errors='coerce')

    # Bet signal based on probability threshold
    def _bet(p):
        if pd.isna(p):
            return ''
        if p > PROB_THRESHOLD:
            return 'Over'
        if p < 1 - PROB_THRESHOLD:
            return 'Under'
        return ''

    combined['bet'] = combined['p_over'].apply(_bet)

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

    combined['p_over'] = combined['p_over'].round(3)

    os.makedirs(predictions_dir, exist_ok=True)

    today_str = datetime.today().strftime('%Y%m%d')
    base_filename = f"predictions_{today_str}.csv"
    filepath = os.path.join(predictions_dir, base_filename)

    version = 1
    while os.path.exists(filepath):
        version += 1
        filepath = os.path.join(predictions_dir, f"predictions_{today_str}_v{version}.csv")

    combined.to_csv(filepath, index=False)

    logger.info("Predictions saved to %s", filepath)

    return combined
