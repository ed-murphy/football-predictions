"""Reading the cached nflverse extracts."""
from __future__ import annotations

import logging
import os

import pandas as pd

from config import GAMES_PATH, INJURIES_PATH, PLAYS_PATH

logger = logging.getLogger(__name__)

_DOWNLOAD_HINT = (
    "Run `python download/download_nfl_data.py` to fetch it."
)


def load_data(games_path: str = GAMES_PATH,
              plays_path: str = PLAYS_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load schedule and play-by-play data."""
    missing = [p for p in (games_path, plays_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing data file(s): {missing}. {_DOWNLOAD_HINT}")

    logger.info("Reading schedule data from %s...", games_path)
    games = pd.read_parquet(games_path)

    logger.info("Reading play-by-play data from %s...", plays_path)
    plays = pd.read_parquet(plays_path)

    logger.info(
        "Loaded %d games (%d-%d) and %d plays.",
        len(games), games["season"].min(), games["season"].max(), len(plays),
    )
    return games, plays


def load_injuries(path: str = INJURIES_PATH) -> pd.DataFrame | None:
    """Load weekly injury reports. Returns None when unavailable — the injury
    features degrade to zero rather than failing the run."""
    if not os.path.exists(path):
        logger.warning("Injury data not found at %s. %s", path, _DOWNLOAD_HINT)
        return None

    logger.info("Reading injury reports from %s...", path)
    return pd.read_parquet(path)
