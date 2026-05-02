import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_rest_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rest-based flags for home and away teams:
      - short_rest  : <= 6 days since last game (e.g. Thursday games)
      - post_bye    : 11–21 days since last game within the same season
      - both_short_rest : both teams on short rest

    Parameters
    ----------
    team_games : pd.DataFrame
        Team-level game data (must include game_id, team, date, season).

    Returns
    -------
    pd.DataFrame
        Updated with home_short_rest, away_short_rest, both_short_rest,
        home_post_bye, away_post_bye.
    """
    tg = team_games.copy()
    tg['game_date'] = pd.to_datetime(tg['date'])
    tg = tg.sort_values(['team', 'game_date'])

    # Days since last game across all seasons (cross-season gaps will be large)
    tg['days_since_last_game'] = tg.groupby('team')['game_date'].diff().dt.days

    tg['short_rest'] = (tg['days_since_last_game'] <= 6).astype(int)

    # Bye week: exactly one extra week off within a season (11–21 day gap).
    # Cross-season gaps are much larger so they don't false-positive here.
    tg['post_bye'] = (
        (tg['days_since_last_game'] >= 11) & (tg['days_since_last_game'] <= 21)
    ).astype(int)

    # Home team
    home_rest = (
        tg.loc[tg['is_home'] == 1, ['game_id', 'short_rest', 'post_bye']]
        .rename(columns={'short_rest': 'home_short_rest', 'post_bye': 'home_post_bye'})
    )

    # Away team
    away_rest = (
        tg.loc[tg['is_home'] == 0, ['game_id', 'short_rest', 'post_bye']]
        .rename(columns={'short_rest': 'away_short_rest', 'post_bye': 'away_post_bye'})
    )

    team_games = team_games.merge(home_rest, on='game_id', how='left')
    team_games = team_games.merge(away_rest, on='game_id', how='left')

    team_games['both_short_rest'] = (
        (team_games['home_short_rest'] == 1) & (team_games['away_short_rest'] == 1)
    ).astype(int)

    logger.info("Rest features created: home/away short_rest, post_bye, both_short_rest")
    return team_games
