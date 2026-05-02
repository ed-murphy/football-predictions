import logging
import pandas as pd
from config import ROLLING_WINDOW_3RD_DOWN

logger = logging.getLogger(__name__)


def create_third_down_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling third-down conversion rate per game.
    Rolling window is ROLLING_WINDOW_3RD_DOWN games, shifted to use only prior games.
    Adds: home_rolling_avg_3rd_pct, away_rolling_avg_3rd_pct
    """
    req = {'third_down_attempt', 'third_down_converted'}
    if not req.issubset(plays.columns):
        missing = req - set(plays.columns)
        logger.warning("Missing third-down columns %s — filling with 0.40 league average.", missing)
        team_games['rolling_avg_3rd_pct']      = 0.40
        team_games['home_rolling_avg_3rd_pct'] = 0.40
        team_games['away_rolling_avg_3rd_pct'] = 0.40
        return team_games

    third_downs = plays[plays['third_down_attempt'] == 1]

    game_3rd = (
        third_downs
        .groupby(['game_id', 'posteam'])
        .agg(attempts=('third_down_attempt', 'sum'),
             conversions=('third_down_converted', 'sum'))
        .reset_index()
        .rename(columns={'posteam': 'team'})
    )
    game_3rd['3rd_pct'] = game_3rd['conversions'] / game_3rd['attempts'].replace(0, float('nan'))

    team_games = team_games.merge(game_3rd[['game_id', 'team', '3rd_pct']], on=['game_id', 'team'], how='left')
    team_games = team_games.sort_values(['team', 'season', 'week'])

    team_games['rolling_avg_3rd_pct'] = (
        team_games
        .groupby(['team', 'season'])['3rd_pct']
        .apply(lambda x: x.shift().rolling(ROLLING_WINDOW_3RD_DOWN, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    home_feat = (
        team_games.loc[team_games['is_home'] == 1, ['game_id', 'rolling_avg_3rd_pct']]
        .rename(columns={'rolling_avg_3rd_pct': 'home_rolling_avg_3rd_pct'})
    )
    away_feat = (
        team_games.loc[team_games['is_home'] == 0, ['game_id', 'rolling_avg_3rd_pct']]
        .rename(columns={'rolling_avg_3rd_pct': 'away_rolling_avg_3rd_pct'})
    )

    team_games = team_games.merge(home_feat, on='game_id', how='left')
    team_games = team_games.merge(away_feat, on='game_id', how='left')

    logger.info("Third-down features created: home_rolling_avg_3rd_pct, away_rolling_avg_3rd_pct")
    return team_games
