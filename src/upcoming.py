import pandas as pd

def prepare_upcoming_team_games(upcoming_games, team_games_hist, latest_qb_epa, weather_features, model):
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
    upcoming_games = upcoming_games.copy()
    upcoming_games['home_team'] = upcoming_games['home_team'].map(TEAM_ABBREV)
    upcoming_games['away_team'] = upcoming_games['away_team'].map(TEAM_ABBREV)

    # adding divisional matchup flag
    DIVISION_MAP = {
        'ARI': 'NFC West', 'ATL': 'NFC South', 'BAL': 'AFC North', 'BUF': 'AFC East',
        'CAR': 'NFC South', 'CHI': 'NFC North', 'CIN': 'AFC North', 'CLE': 'AFC North',
        'DAL': 'NFC East', 'DEN': 'AFC West', 'DET': 'NFC North', 'GB': 'NFC North',
        'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South', 'KC': 'AFC West',
        'LAC': 'AFC West', 'LA': 'NFC West', 'LV': 'AFC West', 'MIA': 'AFC East',
        'MIN': 'NFC North', 'NE': 'AFC East', 'NO': 'NFC South', 'NYG': 'NFC East',
        'NYJ': 'AFC East', 'PHI': 'NFC East', 'PIT': 'AFC North', 'SEA': 'NFC West',
        'SF': 'NFC West', 'TB': 'NFC South', 'TEN': 'AFC South', 'WAS': 'NFC East'
    }
    upcoming_games['home_division'] = upcoming_games['home_team'].map(DIVISION_MAP)
    upcoming_games['away_division'] = upcoming_games['away_team'].map(DIVISION_MAP)
    upcoming_games['divisional'] = (upcoming_games['home_division'] == upcoming_games['away_division']).astype(int)

    # merge in rolling QB EPA for expected home starters
    upcoming_games = upcoming_games.merge(
        latest_qb_epa[['qb_name', 'rolling_avg_qb_epa']],
        left_on=['home_starting_qb'],
        right_on=['qb_name'],
        how='left'
    )
    upcoming_games.rename(columns={'rolling_avg_qb_epa': 'home_rolling_avg_qb_epa'}, inplace=True)

    # merge in rolling QB EPA for expected away starters
    upcoming_games = upcoming_games.merge(
        latest_qb_epa[['qb_name', 'rolling_avg_qb_epa']],
        left_on=['away_starting_qb'],
        right_on=['qb_name'],
        how='left'
    )
    upcoming_games.rename(columns={'rolling_avg_qb_epa': 'away_rolling_avg_qb_epa'}, inplace=True)

    upcoming_games.drop(columns=['qb_name_x', 'qb_name_y'], inplace=True)

    # Add regular season flag: 1 if NOT between Jan 8 and Aug 30, else 0
    game_dates = pd.to_datetime(upcoming_games['commence_time'])
    jan_8 = pd.to_datetime(game_dates.dt.year.astype(str) + '-01-08')
    aug_30 = pd.to_datetime(game_dates.dt.year.astype(str) + '-08-30')
    upcoming_games['regular_season'] = (~((game_dates >= jan_8) & (game_dates <= aug_30))).astype(int)

    # Add rest features
    upcoming_games['both_short_rest'] = ((upcoming_games['home_short_rest'] == 1) & (upcoming_games['away_short_rest'] == 1)).astype(int)

    # Create home/away DataFrames, now including rest features
    home = upcoming_games[['commence_time', 'home_team', 'away_team', 'total_line', 'home_rolling_avg_qb_epa', 'divisional', 'regular_season', 'home_short_rest', 'away_short_rest', 'both_short_rest']].copy()
    home.rename(columns={'home_team':'team', 'away_team':'opponent', 'commence_time':'date'}, inplace=True)
    home['is_home'] = 1

    away = upcoming_games[['commence_time', 'away_team', 'home_team', 'total_line', 'away_rolling_avg_qb_epa', 'divisional', 'regular_season', 'home_short_rest', 'away_short_rest', 'both_short_rest']].copy()
    away.rename(columns={'away_team':'team', 'home_team':'opponent', 'commence_time':'date'}, inplace=True)
    away['is_home'] = 0

    # Ensure consistent datetime for merging
    home['date'] = pd.to_datetime(home['date']).dt.tz_localize(None)
    away['date'] = pd.to_datetime(away['date']).dt.tz_localize(None)
    weather_features['kickoff_time'] = pd.to_datetime(weather_features['kickoff_time']).dt.tz_localize(None)

    # Merge rolling averages not already handled above
    rolling_features = [
        'rolling_avg_points_for', 'rolling_avg_points_against',
        'rolling_avg_def_epa', 'rolling_avg_off_pace'
    ]
    for feat in rolling_features:
        last_vals = team_games_hist.groupby('team')[feat].last().reset_index()
        last_vals.rename(columns={feat: feat+'_pre_game'}, inplace=True)
        home = home.merge(last_vals, on='team', how='left')
        away = away.merge(last_vals, on='team', how='left')
    
    if weather_features['home_team'].isin(TEAM_ABBREV.keys()).any():
        weather_features['home_team'] = weather_features['home_team'].map(TEAM_ABBREV)

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
    game_features.rename(columns={'team_home': 'home_team', 'team_away': 'away_team', 'total_line_home': 'total_line',
                                  'divisional_home': 'divisional', 'regular_season_home': 'regular_season', 'home_short_rest_home': 'home_short_rest',
                                  'away_short_rest_home': 'away_short_rest', 'both_short_rest_home': 'both_short_rest'}, inplace=True)
    game_features.drop(columns=['opponent_home','opponent_away', 'total_line_away', 'divisional_away', 'regular_season_away',
                                'home_short_rest_away', 'away_short_rest_away', 'both_short_rest_away'], inplace=True)

    # Rename columns to match model expectations
    feature_mapping = {
        'total_line': 'total_line',
        'rolling_avg_points_for_pre_game_home': 'home_rolling_avg_points_for',
        'rolling_avg_points_against_pre_game_home': 'home_rolling_avg_points_against',
        'rolling_avg_points_for_pre_game_away': 'away_rolling_avg_points_for',
        'rolling_avg_points_against_pre_game_away': 'away_rolling_avg_points_against',
        'home_rolling_avg_qb_epa': 'home_rolling_avg_qb_epa',
        'away_rolling_avg_qb_epa': 'away_rolling_avg_qb_epa',
        'rolling_avg_def_epa_pre_game_home': 'home_rolling_avg_def_epa',
        'rolling_avg_def_epa_pre_game_away': 'away_rolling_avg_def_epa',
        'temperature': 'home_temperature',
        'wind_speed': 'home_wind_speed',
        'rolling_avg_off_pace_pre_game_home': 'home_rolling_avg_off_pace',
        'rolling_avg_off_pace_pre_game_away': 'away_rolling_avg_off_pace',
        'divisional': 'divisional',
        'regular_season': 'regular_season',
        'home_short_rest': 'home_short_rest',
        'away_short_rest': 'away_short_rest',
        'both_short_rest': 'both_short_rest'
    }
    game_features = game_features.rename(columns=feature_mapping)

    # add interaction terms
    game_features['home_x_away_pace'] = game_features['home_rolling_avg_off_pace'] * game_features['away_rolling_avg_off_pace']
    game_features['home_pace_x_wind_speed'] = game_features['home_rolling_avg_off_pace'] * game_features['home_wind_speed']
    game_features['away_pace_x_wind_speed'] = game_features['away_rolling_avg_off_pace'] * game_features['home_wind_speed']
    game_features['home_qb_x_away_def'] = game_features['home_rolling_avg_qb_epa'] * game_features['away_rolling_avg_def_epa']
    game_features['away_qb_x_home_def'] = game_features['away_rolling_avg_qb_epa'] * game_features['home_rolling_avg_def_epa']

    # Predict totals
    feature_cols = list(feature_mapping.values()) + ['home_x_away_pace'] + ['home_pace_x_wind_speed'] + ['away_pace_x_wind_speed'] + ['home_qb_x_away_def'] + ['away_qb_x_home_def']
    game_features['predicted_total'] = game_features['total_line'] + model.predict(game_features[feature_cols])

    return game_features
