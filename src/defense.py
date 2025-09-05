import pandas as pd

def create_defense_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Create defensive EPA features and merge them into team_games.

    Parameters
    ----------
    team_games : pd.DataFrame
        Game-level dataset for teams (includes 'game_id', 'team', 'is_home', 'season').
    plays : pd.DataFrame
        Play-by-play dataset (includes 'game_id', 'defteam', 'epa').

    Returns
    -------
    pd.DataFrame
        team_games with added defensive features.
    """

    # defensive EPA (per game, per team)
    def_epa = (
        plays
        .groupby(['game_id', 'defteam'])['epa']
        .mean()
        .reset_index()
        .rename(columns={'defteam': 'team', 'epa': 'def_epa'})
    )
    team_games = team_games.merge(def_epa, on=['game_id', 'team'], how='left')

    # rolling average defensive EPA (last 5 games, per team)
    team_games['def_epa_rolling'] = (
        team_games
        .groupby(['team', 'season'])['def_epa']
        .apply(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=[0,1], drop=True)
    )

    # split into home/away views
    home_def_features = (
        team_games[team_games['is_home'] == 1][['game_id', 'def_epa_rolling']]
        .rename(columns={'def_epa_rolling': 'home_rolling_avg_def_epa'})
    )
    away_def_features = (
        team_games[team_games['is_home'] == 0][['game_id', 'def_epa_rolling']]
        .rename(columns={'def_epa_rolling': 'away_rolling_avg_def_epa'})
    )

    # merge back
    team_games = team_games.merge(home_def_features, on='game_id', how='left')
    team_games = team_games.merge(away_def_features, on='game_id', how='left')

    print("Team defense EPA features created.")

    return team_games
