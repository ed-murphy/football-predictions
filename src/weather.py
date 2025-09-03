import os
import pandas as pd
from meteostat import Point, Daily

def create_weather_features(team_games, cache_dir="data/weather_cache"):
    """
    Add weather features to team_games DataFrame.
    Uses Meteostat daily weather data and caches downloads.
    Flags dome teams and sets neutral weather for indoor games.

    Parameters:
        team_games (DataFrame): team-game level data
        cache_dir (str): folder to cache weather CSVs

    Returns:
        DataFrame: team_games with home_weather features added
    """
    os.makedirs(cache_dir, exist_ok=True)

    team_games['date'] = pd.to_datetime(team_games['date'], errors="raise")

    # NFL stadium coordinates (home team -> lat/lon)
    stadium_coords = {
        "ARI": {"lat": 33.5275, "lon": -112.2625},
        "ATL": {"lat": 33.755, "lon": -84.4008},
        "BAL": {"lat": 39.2779, "lon": -76.6227},
        "BUF": {"lat": 42.7738, "lon": -78.7865},
        "CAR": {"lat": 35.2251, "lon": -80.8529},
        "CHI": {"lat": 41.8623, "lon": -87.6167},
        "CIN": {"lat": 39.0955, "lon": -84.5161},
        "CLE": {"lat": 41.5061, "lon": -81.6995},
        "DAL": {"lat": 32.7473, "lon": -97.0945},
        "DEN": {"lat": 39.7439, "lon": -105.0201},
        "DET": {"lat": 42.3400, "lon": -83.0456},
        "GB": {"lat": 44.5013, "lon": -88.0622},
        "HOU": {"lat": 29.6847, "lon": -95.4107},
        "IND": {"lat": 39.7640, "lon": -86.1639},
        "JAX": {"lat": 30.3240, "lon": -81.6375},
        "KC": {"lat": 39.0489, "lon": -94.4839},
        "LV": {"lat": 36.0908, "lon": -115.1830},
        "LAC": {"lat": 33.9535, "lon": -118.3392},
        "LA": {"lat": 34.0141, "lon": -118.2872},
        "MIA": {"lat": 25.9580, "lon": -80.2389},
        "MIN": {"lat": 44.9733, "lon": -93.2572},
        "NE": {"lat": 42.0909, "lon": -71.2643},
        "NO": {"lat": 29.9511, "lon": -90.0812},
        "NYG": {"lat": 40.8135, "lon": -74.0744},
        "NYJ": {"lat": 40.8135, "lon": -74.0744},  # Shares MetLife Stadium
        "PHI": {"lat": 39.9008, "lon": -75.1675},
        "PIT": {"lat": 40.4469, "lon": -80.0158},
        "SEA": {"lat": 47.5952, "lon": -122.3316},
        "SF": {"lat": 37.4030, "lon": -121.9700},
        "TB": {"lat": 27.9759, "lon": -82.5033},
        "TEN": {"lat": 36.1662, "lon": -86.7713},
        "WAS": {"lat": 38.9076, "lon": -77.0209}
    }

    # Dome teams
    dome_teams = {
        "Cardinals", "Falcons", "Cowboys", "Lions", "Texans",
        "Colts", "Raiders", "Chargers", "Rams", "Vikings", "Saints"
    }

    # Initialize weather columns
    for col in ["temperature", "precipitation", "wind_speed"]:
        team_games[f"home_{col}"] = None

    # Loop over each home team
    for team, coords in stadium_coords.items():
        lat = float(coords['lat'])
        lon = float(coords['lon'])
        
        # Filter games for this team
        team_mask = (team_games["team"] == team) & (team_games["is_home"] == 1)
        team_games_subset = team_games.loc[team_mask]

        # Skip dome teams (set neutral weather)
        if team in dome_teams:
            team_games.loc[team_mask, "home_temperature"] = 70
            team_games.loc[team_mask, "home_precipitation"] = 0
            team_games.loc[team_mask, "home_wind_speed"] = 0
            continue

        # Cache filename
        cache_file = os.path.join(cache_dir, f"{team}_weather.csv")

        if os.path.exists(cache_file):
            weather = pd.read_csv(cache_file, parse_dates=["date"])
        else:
            # Build Meteostat Point
            point = Point(float(lat), float(lon))

            start = team_games_subset['date'].min()
            end = team_games_subset['date'].max()

            # Download daily weather
            weather = Daily(point, start, end).fetch().reset_index()
            weather = weather[["time", "tavg", "prcp", "wspd"]]
            weather.columns = ["date", "temperature", "precipitation", "wind_speed"]

            # Cache
            weather.to_csv(cache_file, index=False)

        weather['date'] = pd.to_datetime(weather['date'])
        
        # Merge onto games
        team_games.loc[team_mask, ["home_temperature", "home_precipitation", "home_wind_speed"]] = \
            pd.merge(team_games_subset, weather, left_on="date", right_on="date", how="left")[
                ["temperature", "precipitation", "wind_speed"]
            ].values
        
    print("Weather features created.")

    return team_games
