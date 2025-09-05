import pandas as pd

def create_basic_features(games):
    """Create team-game level features from games and play-by-play data."""
    
    # total points scored in each game
    games['total_points'] = games['home_score'] + games['away_score']

    # create home team features
    home = games[['game_id', 'week', 'season', 'home_team', 'home_score', 'away_score', 'total_line', 'total_points', 'gameday']].copy()
    home.columns = ['game_id', 'week', 'season', 'team', 'points_for', 'points_against', 'total_line', 'total_points', 'date']
    home['is_home'] = 1

    # create away team features
    away = games[['game_id', 'week', 'season', 'away_team', 'away_score', 'home_score', 'total_line', 'total_points', 'gameday']].copy()
    away.columns = ['game_id', 'week', 'season', 'team', 'points_for', 'points_against', 'total_line', 'total_points', 'date']
    away['is_home'] = 0

    # combine home and away data frames
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(by=['team', 'season', 'week'])

    # rolling averages for points
    team_games['rolling_avg_points_for'] = team_games.groupby(['team', 'season'])['points_for'].shift().rolling(window=5, min_periods=1).mean()
    team_games['rolling_avg_points_against'] = team_games.groupby(['team', 'season'])['points_against'].shift().rolling(window=5, min_periods=1).mean()

    # merge rolling average points for/against
    home_features = team_games[team_games['is_home'] == 1][['game_id', 'rolling_avg_points_for', 'rolling_avg_points_against']]
    away_features = team_games[team_games['is_home'] == 0][['game_id', 'rolling_avg_points_for', 'rolling_avg_points_against']]
    team_games = team_games.merge(home_features, on='game_id', how='left')
    team_games = team_games.merge(away_features, on='game_id', how='left')
    team_games.rename(columns={
        'rolling_avg_points_for_x': 'home_rolling_avg_points_for',
        'rolling_avg_points_against_x': 'home_rolling_avg_points_against',
        'rolling_avg_points_for_y': 'away_rolling_avg_points_for',
        'rolling_avg_points_against_y': 'away_rolling_avg_points_against'
    }, inplace=True)

    print("Basic football features created.")

    return team_games
