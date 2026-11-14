"""The schedule: every NFL game since 1999, played and upcoming.

One CSV from nflverse carries everything the project needs about the past —
final scores, closing totals and prices, the wind recorded at kickoff, the roof —
and everything it needs about the future except the live line and the forecast.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

from config import API_TIMEOUT_SECONDS, FIRST_SEASON, GAMES_PATH, GAMES_URL
from src.venues import add_venue_features

logger = logging.getLogger(__name__)

EASTERN = "US/Eastern"
_REQUIRED_COLUMNS = {
    "game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
    "home_score", "away_score", "total_line", "roof", "wind",
}


def download_games(url: str = GAMES_URL, path: str = GAMES_PATH) -> str:
    """Fetch the latest schedule file, replacing the cached copy only if it is sane."""
    logger.info("Downloading schedule and results from nflverse...")
    response = requests.get(url, timeout=API_TIMEOUT_SECONDS * 2)
    response.raise_for_status()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    staging = path + ".download"
    with open(staging, "wb") as handle:
        handle.write(response.content)

    # A truncated or reshaped download must not overwrite a good cache.
    check = pd.read_csv(staging, low_memory=False)
    missing = _REQUIRED_COLUMNS - set(check.columns)
    if missing or len(check) < 5000:
        os.remove(staging)
        raise ValueError(
            f"Downloaded schedule looks wrong ({len(check)} rows, missing {sorted(missing)}); "
            "keeping the existing file."
        )
    os.replace(staging, path)
    logger.info("Saved %d games to %s.", len(check), path)
    return path


def load_games(path: str = GAMES_PATH) -> pd.DataFrame:
    """Read the schedule and add the columns everything else works from.

    Added columns:
      kickoff        kickoff as a timezone-aware UTC timestamp
      is_final       the game has a final score
      total_points   combined final score
      neutral_site, international, open_air, venue_lat, venue_lon   (see venues.py)
      kickoff_wind   wind the field was exposed to, in mph: the gamebook reading
                     for open-air games, 0 indoors, NaN when it was not recorded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No schedule file at {path}. Run `python main.py update` to download it."
        )
    games = pd.read_csv(path, low_memory=False)
    games = games[games["season"] >= FIRST_SEASON].copy()

    local = pd.to_datetime(
        games["gameday"].astype(str) + " " + games["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    games["kickoff"] = (
        local.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    )

    games["is_final"] = games["home_score"].notna() & games["away_score"].notna()
    games["total_points"] = games["home_score"] + games["away_score"]

    games = add_venue_features(games)
    games["kickoff_wind"] = exposed_wind(games)
    return games.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def exposed_wind(games: pd.DataFrame) -> pd.Series:
    """Wind reaching the field: the recorded value outdoors, zero under a roof.

    A retractable roof that was open is left as unknown (NaN). nflverse records no
    wind for those games, and calling them calm would be a guess.
    """
    roof = games["roof"].astype(str).str.lower()
    wind = pd.to_numeric(games["wind"], errors="coerce")
    indoors = roof.isin(["dome", "closed"]) & (games["open_air"] == 0)
    return pd.Series(
        np.where(indoors, 0.0, np.where(games["open_air"] == 1, wind, np.nan)),
        index=games.index,
    )


def completed_games(games: pd.DataFrame) -> pd.DataFrame:
    """Finished games with a closing total: the rows a model can learn from.

    Adds `residual` (points above the closing total), `push`, and `under`
    (1 when the game finished below the line).
    """
    done = games[games["is_final"] & games["total_line"].notna()].copy()
    done["residual"] = done["total_points"] - done["total_line"]
    done["push"] = done["residual"] == 0
    done["under"] = (done["residual"] < 0).astype(int)
    return done


def upcoming_games(games: pd.DataFrame, now: pd.Timestamp, days: int) -> pd.DataFrame:
    """Games that have not kicked off yet and start within `days` of `now`."""
    horizon = now + pd.Timedelta(days=days)
    return games[
        ~games["is_final"] & (games["kickoff"] > now) & (games["kickoff"] <= horizon)
    ].copy()
