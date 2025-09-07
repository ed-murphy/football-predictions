import pandas as pd

def create_rest_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds short rest flags (<= 6 days between games) for home and away teams.

    Parameters
    ----------
    team_games : pd.DataFrame
        Team-level game data (must include game_id, team, game_date).
    model_data : pd.DataFrame
        Modeling dataset keyed by game_id, home_team, away_team.
    features : list
        Current feature list to be updated in-place.

    Returns
    -------
    model_data : pd.DataFrame
        Updated with home_short_rest, away_short_rest, both_short_rest.
    """
    tg = team_games.copy()
    tg['game_date'] = pd.to_datetime(tg['date'])
    tg = tg.sort_values(['team', 'game_date'])
    tg['days_since_last_game'] = tg.groupby('team')['game_date'].diff().dt.days
    tg['short_rest'] = (tg['days_since_last_game'] <= 6).astype(int)

    # Home team rest features
    home_rest = (
        tg.loc[tg['is_home'] == 1, ['game_id', 'short_rest']]
        .rename(columns={'short_rest': 'home_short_rest'})
    )

    # Away team rest features
    away_rest = (
        tg.loc[tg['is_home'] == 0, ['game_id', 'short_rest']]
        .rename(columns={'short_rest': 'away_short_rest'})
    )

    # Merge into team_games
    team_games = team_games.merge(home_rest, on='game_id', how='left')
    team_games = team_games.merge(away_rest, on='game_id', how='left')

    team_games['both_short_rest'] = ((team_games['home_short_rest'] == 1) & (team_games['away_short_rest'] == 1)).astype(int)

    return team_games
