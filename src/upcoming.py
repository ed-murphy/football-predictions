import pandas as pd

def prepare_upcoming_team_games(upcoming_games, team_games_hist, weather_features, model):
    """
    Build one row per upcoming game with home/away features and forecasted weather.
    """
    # Mapping full team names -> 3-letter codes
    TEAM_ABBREV = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
        "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
        "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
    }

    # Convert team names to abbreviations
    upcoming_games = upcoming_games.copy()
    upcoming_games['home_team'] = upcoming_games['home_team'].map(TEAM_ABBREV)
    upcoming_games['away_team'] = upcoming_games['away_team'].map(TEAM_ABBREV)

    # Create home/away DataFrames
    home = upcoming_games[['commence_time', 'home_team', 'away_team', 'total_line']].copy()
    home.rename(columns={'home_team':'team', 'away_team':'opponent', 'commence_time':'date'}, inplace=True)
    home['is_home'] = 1

    away = upcoming_games[['commence_time', 'away_team', 'home_team', 'total_line']].copy()
    away.rename(columns={'away_team':'team', 'home_team':'opponent', 'commence_time':'date'}, inplace=True)
    away['is_home'] = 0

    # Ensure consistent datetime for merging
    home['date'] = pd.to_datetime(home['date']).dt.tz_localize(None)
    away['date'] = pd.to_datetime(away['date']).dt.tz_localize(None)
    weather_features['kickoff_time'] = pd.to_datetime(weather_features['kickoff_time']).dt.tz_localize(None)

    # Merge rolling averages
    rolling_features = [
        'rolling_avg_points_for', 'rolling_avg_points_against',
        'rolling_avg_qb_epa', 'rolling_avg_def_epa',
        'rolling_avg_off_pace'
    ]

    for feat in rolling_features:
        last_vals = team_games_hist.groupby('team')[feat].last().reset_index()
        last_vals.rename(columns={feat: feat+'_pre_game'}, inplace=True)
        home = home.merge(last_vals, on='team', how='left')
        away = away.merge(last_vals, on='team', how='left')

    # Merge home team weather
    home = home.merge(
        weather_features,
        left_on=['team', 'date'],
        right_on=['home_team', 'kickoff_time'],
        how='left'
    )

    # Combine home + away into single row per game
    game_features = home.merge(
        away,
        left_on=['date', 'opponent'],  # home.opponent = away.team
        right_on=['date', 'team'],
        suffixes=('_home', '_away')
    )

    # Cleanup
    game_features.rename(columns={'team_home': 'home_team', 'team_away': 'away_team', 'total_line_home': 'total_line'}, inplace=True)
    game_features.drop(columns=['opponent_home','opponent_away', 'total_line_away'], inplace=True)

    # Rename columns to match model expectations
    feature_mapping = {
        'total_line' : 'total_line',
        'rolling_avg_points_for_pre_game_home': 'home_rolling_avg_points_for',
        'rolling_avg_points_against_pre_game_home': 'home_rolling_avg_points_against',
        'rolling_avg_points_for_pre_game_away': 'away_rolling_avg_points_for',
        'rolling_avg_points_against_pre_game_away': 'away_rolling_avg_points_against',
        'rolling_avg_qb_epa_pre_game_home': 'home_rolling_avg_qb_epa',
        'rolling_avg_qb_epa_pre_game_away': 'away_rolling_avg_qb_epa',
        'rolling_avg_def_epa_pre_game_home': 'home_rolling_avg_def_epa',
        'rolling_avg_def_epa_pre_game_away': 'away_rolling_avg_def_epa',
        'temperature_F': 'home_temperature',
        'wind_speed_mph': 'home_wind_speed',
        'rolling_avg_off_pace_pre_game_home': 'home_rolling_avg_off_pace',
        'rolling_avg_off_pace_pre_game_away': 'away_rolling_avg_off_pace'
    }
    game_features = game_features.rename(columns=feature_mapping)

    # Predict totals
    feature_cols = list(feature_mapping.values())
    game_features['predicted_total'] = model.predict(game_features[feature_cols])

    return game_features
