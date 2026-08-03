"""Kickoff weather forecasts for upcoming games (OpenWeather 5-day/3-hour API).

Units are Fahrenheit and miles per hour throughout, matching the `temp` / `wind`
columns nflverse ships for completed games. Getting this wrong is silent and
expensive: the model would be trained on one scale and served another.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.constants import (
    DOME_TEAMS, INDOOR_TEMP_F, INDOOR_WIND_MPH, STADIUM_COORDS, TEAM_ABBREV,
)
from config import API_BACKOFF_FACTOR, API_MAX_RETRIES, WEATHER_FORECAST_PATH

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
# Beyond this the 5-day forecast has nothing to say; fall back to seasonal normals.
FORECAST_HORIZON_DAYS = 5


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=API_MAX_RETRIES,
        backoff_factor=API_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _to_abbrev(team: str) -> str:
    """Accept either a full team name or an abbreviation."""
    return TEAM_ABBREV.get(team, team)


def fetch_stadium_forecast(team: str, kickoff_times: list[pd.Timestamp], api_key: str,
                           session: requests.Session) -> list[dict]:
    """Forecast at each kickoff for one stadium. One API call per venue, not per game."""
    abbrev = _to_abbrev(team)

    if abbrev in DOME_TEAMS:
        return [
            {"home_team": abbrev, "kickoff_time": t, "temperature": INDOOR_TEMP_F,
             "wind_speed": INDOOR_WIND_MPH, "weather_status": "indoor/dome"}
            for t in kickoff_times
        ]

    coords = STADIUM_COORDS.get(abbrev)
    if coords is None:
        logger.warning("No stadium coordinates for %s; skipping forecast.", team)
        return []

    lat, lon = coords
    try:
        response = session.get(
            FORECAST_URL,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
            timeout=30,
        )
        response.raise_for_status()
        slots = response.json()["list"]
    except Exception as exc:  # network, auth, or schema change — degrade, don't crash
        logger.error("Weather fetch failed for %s: %s", team, exc)
        return []

    rows = []
    for kickoff in kickoff_times:
        closest = min(
            slots,
            key=lambda s: abs(datetime.fromtimestamp(s["dt"], tz=timezone.utc) - kickoff),
        )
        gap_hours = abs(
            datetime.fromtimestamp(closest["dt"], tz=timezone.utc) - kickoff
        ).total_seconds() / 3600
        if gap_hours > 24 * FORECAST_HORIZON_DAYS:
            logger.info(
                "%s kickoff %s is beyond the forecast horizon; leaving weather unknown.",
                abbrev, kickoff,
            )
            continue
        rows.append({
            "home_team": abbrev,
            "kickoff_time": kickoff,
            "temperature": closest["main"]["temp"],          # already Fahrenheit
            "wind_speed": closest["wind"].get("speed", 0.0),  # already mph
            "weather_status": closest["weather"][0]["description"],
        })
    return rows


def get_forecasted_weather(upcoming_games: pd.DataFrame) -> pd.DataFrame:
    """Return one forecast row per upcoming game, using a cached CSV when it is fresh.

    Columns: home_team (abbrev), kickoff_time (UTC), temperature (F),
    wind_speed (mph), weather_status.
    """
    load_dotenv()
    api_key = os.getenv("API_KEY_WEATHER")

    upcoming = upcoming_games.copy()
    upcoming["home_team"] = upcoming["home_team"].map(_to_abbrev)
    upcoming["kickoff_time"] = pd.to_datetime(upcoming["commence_time"], utc=True)

    cached = _read_cache(set(upcoming["home_team"].dropna()))
    if cached is not None:
        return cached

    if not api_key:
        logger.warning(
            "Missing API_KEY_WEATHER — upcoming games will use seasonal-normal weather."
        )
        return _empty_forecast()

    session = _session()
    rows: list[dict] = []
    for team, group in upcoming.groupby("home_team"):
        rows.extend(
            fetch_stadium_forecast(team, list(group["kickoff_time"]), api_key, session)
        )

    forecast = pd.DataFrame(rows, columns=_empty_forecast().columns)
    os.makedirs(os.path.dirname(WEATHER_FORECAST_PATH) or ".", exist_ok=True)
    forecast.to_csv(WEATHER_FORECAST_PATH, index=False)
    logger.info("Saved %d weather forecasts to %s", len(forecast), WEATHER_FORECAST_PATH)
    return forecast


def _empty_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["home_team", "kickoff_time", "temperature", "wind_speed", "weather_status"]
    )


def _read_cache(upcoming_home_teams: set[str]) -> pd.DataFrame | None:
    """Return the cached forecast if it still covers the games we're predicting."""
    if not os.path.exists(WEATHER_FORECAST_PATH):
        return None

    cached = pd.read_csv(WEATHER_FORECAST_PATH, parse_dates=["kickoff_time"])
    if cached.empty:
        return None

    cached["home_team"] = cached["home_team"].map(_to_abbrev)
    if not set(cached["home_team"].dropna()) & upcoming_home_teams:
        logger.info("Weather cache does not cover this slate; re-fetching.")
        os.remove(WEATHER_FORECAST_PATH)
        return None

    if cached["kickoff_time"].dt.tz is None:
        cached["kickoff_time"] = cached["kickoff_time"].dt.tz_localize("UTC")

    logger.info("Using cached weather forecast from %s.", WEATHER_FORECAST_PATH)
    return cached
