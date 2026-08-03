"""Shared lookup tables.

These were previously duplicated across totals.py, upcoming.py, weather.py and
weather_forecast.py, which meant a team relocation or rename had to be applied in
four places. Import them from here instead.
"""
from __future__ import annotations

# Full team name (as returned by the odds API) -> nflverse 3-letter code
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
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

DIVISION_MAP = {
    "ARI": "NFC West", "ATL": "NFC South", "BAL": "AFC North", "BUF": "AFC East",
    "CAR": "NFC South", "CHI": "NFC North", "CIN": "AFC North", "CLE": "AFC North",
    "DAL": "NFC East", "DEN": "AFC West", "DET": "NFC North", "GB": "NFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "KC": "AFC West",
    "LAC": "AFC West", "LA": "NFC West", "LV": "AFC West", "MIA": "AFC East",
    "MIN": "NFC North", "NE": "AFC East", "NO": "NFC South", "NYG": "NFC East",
    "NYJ": "AFC East", "PHI": "NFC East", "PIT": "AFC North", "SEA": "NFC West",
    "SF": "NFC West", "TB": "NFC South", "TEN": "AFC South", "WAS": "NFC East",
}

# Home stadium lat/lon, used for weather forecast lookups.
STADIUM_COORDS = {
    "ARI": (33.5275, -112.2625), "ATL": (33.7550, -84.4008), "BAL": (39.2779, -76.6227),
    "BUF": (42.7738, -78.7865), "CAR": (35.2251, -80.8529), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7640, -86.1639), "JAX": (30.3240, -81.6375),
    "KC": (39.0489, -94.4839), "LV": (36.0908, -115.1830), "LAC": (33.9535, -118.3392),
    "LA": (34.0141, -118.2872), "MIA": (25.9580, -80.2389), "MIN": (44.9733, -93.2572),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0744),
    "NYJ": (40.8135, -74.0744), "PHI": (39.9008, -75.1675), "PIT": (40.4469, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1662, -86.7713), "WAS": (38.9076, -77.0209),
}

# Teams whose home games are always played indoors (roof column is unreliable for
# upcoming games, so this is the fallback used when building forecast features).
DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LA", "MIN", "NO"}

# nflverse `roof` values that mean "no weather"
INDOOR_ROOFS = {"dome", "closed"}

INTERNATIONAL_STADIUM_IDS = {"LON02", "LON00", "LON01", "GER00", "GER01", "MEX00",
                             "FRA00", "BRA00", "ESP00"}

# Neutral values substituted for indoor games so the model sees a constant
# rather than a missing value.
INDOOR_TEMP_F = 70.0
INDOOR_WIND_MPH = 0.0
