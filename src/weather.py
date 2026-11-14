"""Wind forecasts for upcoming games (Open-Meteo: free, no API key).

The model was fitted on the wind recorded in the gamebook at kickoff. A forecast
is not that number — it is a noisy and slightly biased estimate of it — so the
raw forecast is never fed to the model. `expected_kickoff_wind` translates it
first, using the relationship measured between archived forecasts and what the
gamebooks went on to record.
"""
from __future__ import annotations

import logging
import math

import pandas as pd
import requests

from config import API_TIMEOUT_SECONDS, FORECAST_CALIBRATION, FORECAST_URL

logger = logging.getLogger(__name__)

MAX_LEAD_DAYS = max(FORECAST_CALIBRATION)


def lead_days(kickoff: pd.Timestamp, now: pd.Timestamp) -> int:
    """Whole days until kickoff, rounded up: a game 30 hours away is 2 days out."""
    return max(1, math.ceil((kickoff - now).total_seconds() / 86400))


def expected_kickoff_wind(forecast_mph: float, days_out: int) -> float | None:
    """Best estimate of the gamebook wind, given a forecast made `days_out` ahead.

    Returns None beyond the calibrated horizon, where forecasts say too little
    about game-day wind to act on.
    """
    if forecast_mph is None or pd.isna(forecast_mph) or days_out > MAX_LEAD_DAYS:
        return None
    intercept, slope = FORECAST_CALIBRATION[max(1, int(days_out))]
    return intercept + slope * float(forecast_mph)


def fetch_hourly_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Hourly forecast for one location, indexed by UTC hour. Empty on failure."""
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation_probability",
                "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
                "timezone": "GMT", "forecast_days": 10,
            },
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return parse_hourly(response.json())
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.error("Forecast request failed for (%.2f, %.2f): %s", lat, lon, exc)
        return pd.DataFrame()


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo response into a frame indexed by UTC timestamp."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({
        "wind_mph": hourly["wind_speed_10m"],
        "gust_mph": hourly.get("wind_gusts_10m"),
        "temp_f": hourly.get("temperature_2m"),
        "precip_chance": hourly.get("precipitation_probability"),
    }, index=pd.to_datetime(hourly["time"], utc=True))
    return frame


def forecast_at_kickoff(hourly: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    """The forecast for the hour the game kicks off (empty dict if not covered)."""
    hour = kickoff.floor("h")
    if hourly.empty or hour not in hourly.index:
        return {}
    row = hourly.loc[hour]
    return {key: (None if pd.isna(value) else float(value)) for key, value in row.items()}


def add_wind_forecasts(upcoming: pd.DataFrame, now: pd.Timestamp,
                       fetcher=fetch_hourly_forecast) -> pd.DataFrame:
    """Add forecast and expected kickoff wind to each upcoming game.

    Adds `days_out`, `forecast_wind`, `forecast_gust`, `forecast_temp`,
    `precip_chance` and `expected_wind`. Games under a roof get an expected wind
    of zero without a lookup; open-air games with no usable forecast get NaN.
    One request is made per venue, not per game.
    """
    games = upcoming.copy()
    games["days_out"] = [lead_days(k, now) for k in games["kickoff"]]
    for column in ("forecast_wind", "forecast_gust", "forecast_temp", "precip_chance",
                   "expected_wind"):
        games[column] = float("nan")

    games.loc[games["open_air"] == 0, "expected_wind"] = 0.0

    needs_forecast = games[
        (games["open_air"] == 1) & games["venue_lat"].notna()
        & (games["days_out"] <= MAX_LEAD_DAYS)
    ]
    for (lat, lon), group in needs_forecast.groupby(["venue_lat", "venue_lon"]):
        hourly = fetcher(lat, lon)
        for index, game in group.iterrows():
            at_kickoff = forecast_at_kickoff(hourly, game["kickoff"])
            if not at_kickoff or at_kickoff.get("wind_mph") is None:
                continue
            games.loc[index, "forecast_wind"] = at_kickoff["wind_mph"]
            games.loc[index, "forecast_gust"] = at_kickoff.get("gust_mph")
            games.loc[index, "forecast_temp"] = at_kickoff.get("temp_f")
            games.loc[index, "precip_chance"] = at_kickoff.get("precip_chance")
            games.loc[index, "expected_wind"] = expected_kickoff_wind(
                at_kickoff["wind_mph"], game["days_out"]
            )
    return games
