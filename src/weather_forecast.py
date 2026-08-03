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


def fetch_venue_forecast(venue_key: str, lat: float, lon: float,
                         kickoff_times: list[pd.Timestamp], api_key: str,
                         session: requests.Session) -> list[dict]:
    """Forecast at each kickoff for one venue. One API call per venue, not per game."""
    try:
        response = session.get(
            FORECAST_URL,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
            timeout=30,
        )
        response.raise_for_status()
        slots = response.json()["list"]
    except Exception as exc:  # network, auth, or schema change: degrade, don't crash
        logger.error("Weather fetch failed for %s: %s", venue_key, exc)
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
                venue_key, kickoff,
            )
            continue
        rows.append({
            "venue_key": venue_key,
            "kickoff_time": kickoff,
            "temperature": closest["main"]["temp"],           # already Fahrenheit
            "wind_speed": closest["wind"].get("speed", 0.0),  # already mph
            "weather_status": closest["weather"][0]["description"],
        })
    return rows


def get_forecasted_weather(venue_requests: pd.DataFrame) -> pd.DataFrame:
    """Kickoff forecasts for a set of venues.

    Takes one row per (venue, kickoff) with columns `venue_key`, `venue_lat`,
    `venue_lon` and `kickoff_time`. Keying on the venue rather than the home team
    is what lets a neutral-site game get the weather where it is actually played.

    Returns `venue_key`, `kickoff_time` (UTC), `temperature` (F), `wind_speed`
    (mph), `weather_status`.
    """
    if venue_requests is None or venue_requests.empty:
        return _empty_forecast()

    load_dotenv()
    api_key = os.getenv("API_KEY_WEATHER")

    requests_df = venue_requests.copy()
    requests_df["kickoff_time"] = pd.to_datetime(requests_df["kickoff_time"], utc=True)

    cached = _read_cache(set(requests_df["venue_key"].dropna()))
    if cached is not None:
        return cached

    if not api_key:
        logger.warning(
            "Missing API_KEY_WEATHER; upcoming games will use seasonal-normal weather."
        )
        return _empty_forecast()

    session = _session()
    rows: list[dict] = []
    for (venue_key, lat, lon), group in requests_df.groupby(
        ["venue_key", "venue_lat", "venue_lon"]
    ):
        rows.extend(fetch_venue_forecast(
            venue_key, lat, lon, list(group["kickoff_time"]), api_key, session
        ))

    forecast = pd.DataFrame(rows, columns=_empty_forecast().columns)
    os.makedirs(os.path.dirname(WEATHER_FORECAST_PATH) or ".", exist_ok=True)
    forecast.to_csv(WEATHER_FORECAST_PATH, index=False)
    logger.info("Saved %d weather forecasts to %s", len(forecast), WEATHER_FORECAST_PATH)
    return forecast


def _empty_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["venue_key", "kickoff_time", "temperature", "wind_speed", "weather_status"]
    )


def _read_cache(wanted_venues: set[str]) -> pd.DataFrame | None:
    """Return the cached forecast if it still covers the venues we're predicting."""
    if not os.path.exists(WEATHER_FORECAST_PATH):
        return None

    cached = pd.read_csv(WEATHER_FORECAST_PATH, parse_dates=["kickoff_time"])
    if cached.empty or "venue_key" not in cached.columns:
        # Written by an older version keyed on home_team; not usable.
        logger.info("Discarding weather cache in an old format.")
        os.remove(WEATHER_FORECAST_PATH)
        return None

    if not set(cached["venue_key"].dropna()) & wanted_venues:
        logger.info("Weather cache does not cover this slate; re-fetching.")
        os.remove(WEATHER_FORECAST_PATH)
        return None

    if cached["kickoff_time"].dt.tz is None:
        cached["kickoff_time"] = cached["kickoff_time"].dt.tz_localize("UTC")

    logger.info("Using cached weather forecast from %s.", WEATHER_FORECAST_PATH)
    return cached
