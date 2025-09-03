import os
import pandas as pd
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

WEATHER_PATH = "data/nfl_weather_forecasts.csv"

# Mapping full team names -> 3-letter abbreviations
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

# Coordinates for stadiums
TEAM_COORDS = {
    "Arizona Cardinals": {"lat": 33.5275, "lon": -112.2625},
    "Atlanta Falcons": {"lat": 33.755, "lon": -84.4008},
    "Baltimore Ravens": {"lat": 39.2779, "lon": -76.6227},
    "Buffalo Bills": {"lat": 42.7738, "lon": -78.7865},
    "Carolina Panthers": {"lat": 35.2251, "lon": -80.8529},
    "Chicago Bears": {"lat": 41.8623, "lon": -87.6167},
    "Cincinnati Bengals": {"lat": 39.0955, "lon": -84.5161},
    "Cleveland Browns": {"lat": 41.5061, "lon": -81.6995},
    "Dallas Cowboys": {"lat": 32.7473, "lon": -97.0945},
    "Denver Broncos": {"lat": 39.7439, "lon": -105.0201},
    "Detroit Lions": {"lat": 42.3400, "lon": -83.0456},
    "Green Bay Packers": {"lat": 44.5013, "lon": -88.0622},
    "Houston Texans": {"lat": 29.6847, "lon": -95.4107},
    "Indianapolis Colts": {"lat": 39.7640, "lon": -86.1639},
    "Jacksonville Jaguars": {"lat": 30.3240, "lon": -81.6375},
    "Kansas City Chiefs": {"lat": 39.0489, "lon": -94.4839},
    "Las Vegas Raiders": {"lat": 36.0908, "lon": -115.1830},
    "Los Angeles Chargers": {"lat": 33.9535, "lon": -118.3392},
    "Los Angeles Rams": {"lat": 34.0141, "lon": -118.2872},
    "Miami Dolphins": {"lat": 25.9580, "lon": -80.2389},
    "Minnesota Vikings": {"lat": 44.9733, "lon": -93.2572},
    "New England Patriots": {"lat": 42.0909, "lon": -71.2643},
    "New Orleans Saints": {"lat": 29.9511, "lon": -90.0812},
    "New York Giants": {"lat": 40.8135, "lon": -74.0744},
    "New York Jets": {"lat": 40.8135, "lon": -74.0744},
    "Philadelphia Eagles": {"lat": 39.9008, "lon": -75.1675},
    "Pittsburgh Steelers": {"lat": 40.4469, "lon": -80.0158},
    "Seattle Seahawks": {"lat": 47.5952, "lon": -122.3316},
    "San Francisco 49ers": {"lat": 37.4030, "lon": -121.9700},
    "Tampa Bay Buccaneers": {"lat": 27.9759, "lon": -82.5033},
    "Tennessee Titans": {"lat": 36.1662, "lon": -86.7713},
    "Washington Commanders": {"lat": 38.9076, "lon": -77.0209}
}

# Dome teams
dome_teams = {
    "Arizona Cardinals", "Atlanta Falcons", "Dallas Cowboys", "Detroit Lions", "Houston Texans",
    "Indianapolis Colts", "Oakland Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Minnesota Vikings", "New Orleans Saints"
}

def fetch_game_weather(home_team, kickoff_time, api_key):
    """Fetch weather for a single game or return dome defaults."""
    if home_team in dome_teams:
        return {
            "home_team": home_team,
            "kickoff_time": kickoff_time,
            "temperature_F": 70,
            "wind_speed_mph": 0,
            "weather_status": "indoor/dome",
        }

    coords = TEAM_COORDS.get(home_team)
    if not coords:
        print(f"[WARN] No coordinates for {home_team}, skipping.")
        return None

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=imperial"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        forecasts = data["list"]

        closest = min(
            forecasts,
            key=lambda f: abs(datetime.fromtimestamp(f["dt"], tz=timezone.utc) - kickoff_time)
        )

        return {
            "home_team": home_team,
            "kickoff_time": kickoff_time,
            "temperature_F": closest["main"]["temp"],
            "wind_speed_mph": closest["wind"].get("speed"),
            "weather_status": closest["weather"][0]["description"],
        }

    except Exception as e:
        print(f"[ERROR] Weather fetch failed for {home_team} at {kickoff_time}: {e}")
        return None

def get_forecasted_weather(upcoming_team_games: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather forecasts for all upcoming games sequentially."""
    load_dotenv()
    api_key = os.getenv("API_KEY_WEATHER")
    if not api_key:
        raise ValueError("Missing API_KEY_WEATHER in .env file")
    
    if os.path.exists(WEATHER_PATH):
        print(f"Reading cached weather forecast for upcoming games from {WEATHER_PATH}...")
        df = pd.read_csv(WEATHER_PATH, parse_dates=['kickoff_time'])
        df['home_team'] = df['home_team'].map(TEAM_ABBREV)
        return df

    upcoming_team_games = upcoming_team_games.copy()
    upcoming_team_games['kickoff_time'] = pd.to_datetime(upcoming_team_games['commence_time'], utc=True)

    weather_data = []
    for _, row in upcoming_team_games.iterrows():
        result = fetch_game_weather(row['home_team'], row['kickoff_time'], api_key)
        if result:
            weather_data.append(result)

    df = pd.DataFrame(weather_data)
    df['home_team'] = df['home_team'].map(TEAM_ABBREV)

    df.to_csv(WEATHER_PATH, index=False)
    print(f"Saved weather forecast for upcoming games to {WEATHER_PATH}")

    return df
