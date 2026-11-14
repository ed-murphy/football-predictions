"""Shared lookup tables."""
from __future__ import annotations

# Full team name (as the odds API returns it) -> nflverse team code
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

# Home stadium (lat, lon) — where the wind forecast is looked up for a home game.
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
