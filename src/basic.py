import pandas as pd

def create_basic_features(games):
    """Create team-game level features from games and play-by-play data."""

    international_stadiums = ['LON02', 'LON00', 'GER00', 'MEX00', 'FRA00']
    games['international'] = games['stadium_id'].isin(international_stadiums).astype(int)
    
    # total points scored in each game
    games['total_points'] = games['home_score'] + games['away_score']

    games['regular_season'] = (games['game_type'] == 'REG').astype(int)
    games['is_dome'] = games['roof'].str.lower().isin(['dome', 'closed']).astype(int)

    # create home team features
    home = games[['game_id', 'week', 'season', 'home_team', 'home_score', 'away_score', 'total_line', 'total_points', 'gameday', 'div_game', 'regular_season', 'is_dome', 'international']].copy()
    home.columns = ['game_id', 'week', 'season', 'team', 'points_for', 'points_against', 'total_line', 'total_points', 'date', 'divisional', 'regular_season', 'is_dome', 'international']
    home['is_home'] = 1

    # create away team features
    away = games[['game_id', 'week', 'season', 'away_team', 'away_score', 'home_score', 'total_line', 'total_points', 'gameday', 'div_game', 'regular_season', 'is_dome', 'international']].copy()
    away.columns = ['game_id', 'week', 'season', 'team', 'points_for', 'points_against', 'total_line', 'total_points', 'date', 'divisional', 'regular_season', 'is_dome', 'international']
    away['is_home'] = 0

    # combine home and away data frames
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(by=['team', 'season', 'week'])

    # rolling averages for points (shift so it’s prior games only)
    team_games['rolling_avg_points_for'] = (
        team_games.groupby(['team', 'season'])['points_for']
        .shift()
        .rolling(window=3, min_periods=1)
        .mean()
    )
    team_games['rolling_avg_points_against'] = (
        team_games.groupby(['team', 'season'])['points_against']
        .shift()
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # assign home/away features
    team_games['home_rolling_avg_points_for'] = team_games.groupby('game_id')['rolling_avg_points_for'].transform(lambda x: x.where(team_games['is_home']==1).max())
    team_games['home_rolling_avg_points_against'] = team_games.groupby('game_id')['rolling_avg_points_against'].transform(lambda x: x.where(team_games['is_home']==1).max())
    team_games['away_rolling_avg_points_for'] = team_games.groupby('game_id')['rolling_avg_points_for'].transform(lambda x: x.where(team_games['is_home']==0).max())
    team_games['away_rolling_avg_points_against'] = team_games.groupby('game_id')['rolling_avg_points_against'].transform(lambda x: x.where(team_games['is_home']==0).max())

    print("Basic football features created.")
    return team_games
