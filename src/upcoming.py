import pandas as pd
import pytz

def prepare_upcoming_team_games(upcoming_games, team_games_hist, latest_qb_epa,
                                weather_features, model, existing_predictions=None):
    """
    Build one row per upcoming game with home/away features and forecasted weather.
    All datetime columns are tz-naive to prevent comparison errors.
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

    # Filter out games that are already predicted
    if existing_predictions is not None and not existing_predictions.empty:
        # Normalize datetimes
        upcoming_games['commence_date'] = pd.to_datetime(upcoming_games['commence_time']).dt.date
        existing_predictions['pred_date'] = pd.to_datetime(existing_predictions['date']).dt.date

        # Merge-based filtering
        merged = upcoming_games.merge(
            existing_predictions,
            left_on=['home_team', 'away_team', 'commence_date'],
            right_on=['home_team', 'away_team', 'pred_date'],
            how='left',
            indicator=True
        )

        # Keep only games not already predicted
        upcoming_games = merged[merged['_merge'] == 'left_only'].copy()

        # Drop right-side total_line if it exists, keep left-side as 'total_line'
        if 'total_line_y' in upcoming_games.columns:
            upcoming_games = upcoming_games.drop(columns=['total_line_y'])
        if 'total_line_x' in upcoming_games.columns:
            upcoming_games = upcoming_games.rename(columns={'total_line_x': 'total_line'})

        # Drop helper columns
        upcoming_games = upcoming_games.drop(columns=['_merge', 'pred_date'], errors='ignore')

        if upcoming_games.empty:
            print("All upcoming games are already in existing predictions.")
            return pd.DataFrame()

    # Divisional matchups
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

    # Merge rolling QB EPA
    upcoming_games = upcoming_games.merge(
        latest_qb_epa[['qb_name', 'rolling_avg_qb_epa']],
        left_on='home_starting_qb',
        right_on='qb_name',
        how='left'
    ).rename(columns={'rolling_avg_qb_epa': 'home_rolling_avg_qb_epa'})
    upcoming_games = upcoming_games.merge(
        latest_qb_epa[['qb_name', 'rolling_avg_qb_epa']],
        left_on='away_starting_qb',
        right_on='qb_name',
        how='left'
    ).rename(columns={'rolling_avg_qb_epa': 'away_rolling_avg_qb_epa'})
    upcoming_games.drop(columns=['qb_name_x', 'qb_name_y'], inplace=True, errors='ignore')

    # --- Ensure tz-naive for date comparisons ---
    game_dates = pd.to_datetime(upcoming_games['commence_time'])
    if game_dates.dt.tz is not None:
        game_dates = game_dates.dt.tz_localize(None)

    jan_8 = pd.to_datetime(game_dates.dt.year.astype(str) + '-01-08')
    aug_30 = pd.to_datetime(game_dates.dt.year.astype(str) + '-08-30')
    upcoming_games['regular_season'] = (~((game_dates >= jan_8) & (game_dates <= aug_30))).astype(int)

    # Rest features
    upcoming_games['both_short_rest'] = ((upcoming_games['home_short_rest'] == 1) & (upcoming_games['away_short_rest'] == 1)).astype(int)

    upcoming_games = upcoming_games.loc[:, ~upcoming_games.columns.duplicated()]

    # Split home/away for features
    home = upcoming_games[['commence_time', 'home_team', 'away_team', 'total_line',
                           'home_rolling_avg_qb_epa', 'divisional', 'regular_season',
                           'international', 'home_short_rest', 'away_short_rest', 'both_short_rest']].copy()
    home.rename(columns={'home_team':'team', 'away_team':'opponent', 'commence_time':'date'}, inplace=True)
    home['is_home'] = 1

    away = upcoming_games[['commence_time', 'away_team', 'home_team', 'total_line',
                           'away_rolling_avg_qb_epa', 'divisional', 'regular_season',
                           'international', 'home_short_rest', 'away_short_rest', 'both_short_rest']].copy()
    away.rename(columns={'away_team':'team', 'home_team':'opponent', 'commence_time':'date'}, inplace=True)
    away['is_home'] = 0

    # Make dates tz-naive
    eastern = pytz.timezone("US/Eastern")
    home['date'] = pd.to_datetime(home['date'], utc=True).dt.tz_convert(eastern).dt.tz_localize(None)
    away['date'] = pd.to_datetime(away['date'], utc=True).dt.tz_convert(eastern).dt.tz_localize(None)
    weather_features['kickoff_time'] = (
        pd.to_datetime(weather_features['kickoff_time'], utc=True)
        .dt.tz_convert(eastern)
        .dt.tz_localize(None)
    )

    # Merge rolling averages from historical team games
    rolling_features = [
        'rolling_avg_points_for', 'rolling_avg_points_against',
        'rolling_avg_def_epa', 'rolling_avg_off_pace', 'rolling_rz_eff'
    ]
    for feat in rolling_features:
        last_vals = team_games_hist.groupby('team')[feat].last().reset_index()
        last_vals.rename(columns={feat: feat+'_pre_game'}, inplace=True)
        home = home.merge(last_vals, on='team', how='left')
        away = away.merge(last_vals, on='team', how='left')

    # Map weather features
    if weather_features['home_team'].isin(TEAM_ABBREV.keys()).any():
        weather_features['home_team'] = weather_features['home_team'].map(TEAM_ABBREV)

    home = home.merge(weather_features,
                      left_on=['team', 'date'],
                      right_on=['home_team', 'kickoff_time'],
                      how='left')

    # Rename weather columns in home before merging
    home.rename(columns={
        'temperature': 'home_temperature',
        'wind_speed': 'home_wind_speed',
        'kickoff_time': 'kickoff_time'
    }, inplace=True)

    # Merge home + away
    game_features = home.merge(
        away,
        left_on=['date', 'opponent'],
        right_on=['date', 'team'],
        suffixes=('_home', '_away')
    )

    # Rename home-side columns to match model expectations
    game_features.rename(columns={
        'divisional_home': 'divisional',
        'regular_season_home': 'regular_season',
        'international_home': 'international',
        'home_short_rest_home': 'home_short_rest',
        'away_short_rest_home': 'away_short_rest',
        'both_short_rest_home': 'both_short_rest'
    }, inplace=True)

    # Rename total_line from home so it exists
    game_features.rename(columns={'total_line_home': 'total_line'}, inplace=True)

    # Cleanup only truly redundant columns
    drop_cols = [
        'opponent_home', 'opponent_away', 'total_line_away', 'divisional_away', 'regular_season_away',
        'international_away','home_short_rest_away','away_short_rest_away','both_short_rest_away'
    ]
    game_features.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Rename for model
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
        'international' : 'international',
        'home_short_rest': 'home_short_rest',
        'away_short_rest': 'away_short_rest',
        'both_short_rest': 'both_short_rest',
        'rolling_rz_eff_pre_game_home' : 'home_rolling_rz_eff',
        'rolling_rz_eff_pre_game_away' : 'away_rolling_rz_eff'
    }
    game_features = game_features.rename(columns=feature_mapping)

    # Add interaction terms
    game_features['home_x_away_pace'] = game_features['home_rolling_avg_off_pace'] * game_features['away_rolling_avg_off_pace']
    game_features['home_pace_x_wind_speed'] = game_features['home_rolling_avg_off_pace'] * game_features['home_wind_speed']
    game_features['away_pace_x_wind_speed'] = game_features['away_rolling_avg_off_pace'] * game_features['home_wind_speed']
    game_features['home_qb_x_away_def'] = game_features['home_rolling_avg_qb_epa'] * game_features['away_rolling_avg_def_epa']
    game_features['away_qb_x_home_def'] = game_features['away_rolling_avg_qb_epa'] * game_features['home_rolling_avg_def_epa']

    # Predict totals
    feature_cols = list(feature_mapping.values()) + [
        'home_x_away_pace','home_pace_x_wind_speed','away_pace_x_wind_speed',
        'home_qb_x_away_def','away_qb_x_home_def'
    ]
    game_features['predicted_total'] = game_features['total_line'] + model.predict(game_features[feature_cols])

    return game_features
