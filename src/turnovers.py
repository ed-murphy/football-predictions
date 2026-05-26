import logging
import pandas as pd
from config import ROLLING_WINDOW_TURNOVERS

logger = logging.getLogger(__name__)


def create_turnover_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling offensive turnover rate (INTs + fumbles lost) per game.
    Rolling window is ROLLING_WINDOW_TURNOVERS games, shifted to use only prior games.
    Adds: home_rolling_avg_turnovers, away_rolling_avg_turnovers
    """
    available_cols = [c for c in ('interception', 'fumble_lost') if c in plays.columns]

    if not available_cols:
        logger.warning("No turnover columns found in plays data — filling with zeros.")
        team_games['rolling_avg_turnovers']      = 0.0
        team_games['home_rolling_avg_turnovers'] = 0.0
        team_games['away_rolling_avg_turnovers'] = 0.0
        return team_games

    game_turnovers = (
        plays
        .groupby(['game_id', 'posteam'])[available_cols]
        .sum()
        .sum(axis=1)
        .reset_index(name='turnovers')
        .rename(columns={'posteam': 'team'})
    )

    team_games = team_games.merge(game_turnovers, on=['game_id', 'team'], how='left')
    team_games = team_games.sort_values(['team', 'season', 'week'])

    team_games['rolling_avg_turnovers'] = (
        team_games
        .groupby(['team', 'season'])['turnovers']
        .apply(lambda x: x.shift().rolling(ROLLING_WINDOW_TURNOVERS, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    home_feat = (
        team_games.loc[team_games['is_home'] == 1, ['game_id', 'rolling_avg_turnovers']]
        .rename(columns={'rolling_avg_turnovers': 'home_rolling_avg_turnovers'})
    )
    away_feat = (
        team_games.loc[team_games['is_home'] == 0, ['game_id', 'rolling_avg_turnovers']]
        .rename(columns={'rolling_avg_turnovers': 'away_rolling_avg_turnovers'})
    )

    team_games = team_games.merge(home_feat, on='game_id', how='left')
    team_games = team_games.merge(away_feat, on='game_id', how='left')

    logger.info("Turnover features created: home_rolling_avg_turnovers, away_rolling_avg_turnovers")
    return team_games
