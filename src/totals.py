"""Fetching and caching betting lines from the Odds API.

Keeps a running CSV of every line seen plus a dated snapshot per day, so line
movement can be measured after the fact.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from pytz import timezone as tz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.constants import TEAM_ABBREV
from config import (
    API_BACKOFF_FACTOR, API_MAX_RETRIES, LINE_SNAPSHOTS_DIR, ODDS_BOOKMAKER, TOTALS_PATH,
)

logger = logging.getLogger(__name__)

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
EASTERN = tz("US/Eastern")
# Odds are posted months ahead in the off-season; keep a wide window so early
# season games aren't filtered out in the spring.
LOOKAHEAD_DAYS = 120


def _api_key() -> str | None:
    load_dotenv()
    return os.getenv("API_KEY_TOTALS")


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


def get_totals_from_api(api_key: str) -> pd.DataFrame:
    """Current totals and spreads for upcoming games from the configured book."""
    response = _session().get(
        ODDS_URL,
        params={"apiKey": api_key, "regions": "us", "markets": "totals,spreads",
                "oddsFormat": "american"},
        timeout=30,
    )
    response.raise_for_status()

    cutoff = datetime.now(EASTERN) + timedelta(days=LOOKAHEAD_DAYS)
    rows = []

    for game in response.json():
        book = next(
            (b for b in game.get("bookmakers", [])
             if b["key"].lower() == ODDS_BOOKMAKER or b["title"].lower() == ODDS_BOOKMAKER),
            None,
        )
        if book is None:
            continue

        kickoff = pd.Timestamp(game["commence_time"]).tz_convert(EASTERN)
        if kickoff > cutoff:
            continue

        row = {"home_team": game["home_team"], "away_team": game["away_team"],
               "commence_time": kickoff, "bookmaker": book["title"]}

        for market in book["markets"]:
            if market["key"] == "totals":
                over = next((o for o in market["outcomes"] if o["name"] == "Over"), None)
                if over:
                    row["total_line"] = over["point"]
            elif market["key"] == "spreads":
                home = next(
                    (o for o in market["outcomes"] if o["name"] == game["home_team"]), None
                )
                if home:
                    # nflverse convention: positive spread_line = home favoured.
                    row["spread_line"] = -home["point"]

        if "total_line" in row:
            rows.append(row)

    logger.info("Odds API returned %d games with a posted total.", len(rows))
    return pd.DataFrame(rows)


def enrich_totals(totals: pd.DataFrame, team_games: pd.DataFrame | None) -> pd.DataFrame:
    """Auto-populate the columns a user would otherwise fill in by hand.

    `home/away_starting_qb` come from each team's most recent game and
    `international` from the kickoff hour. Existing non-null values are preserved,
    so manual overrides in the CSV survive a refresh.
    """
    totals = totals.copy()
    for col in ("home_starting_qb", "away_starting_qb", "international"):
        if col not in totals.columns:
            totals[col] = pd.NA

    kickoff = pd.to_datetime(totals["commence_time"], utc=True).dt.tz_convert(EASTERN)
    # No domestic NFL game kicks off before 11am Eastern.
    needs_intl = totals["international"].isna()
    totals.loc[needs_intl, "international"] = (
        kickoff[needs_intl].dt.hour < 11
    ).astype(int)

    if team_games is None or team_games.empty or "starting_qb" not in team_games.columns:
        return totals

    latest_qb = (
        team_games[team_games["starting_qb"].notna()]
        .sort_values("date").groupby("team")["starting_qb"].last()
    )
    for side in ("home", "away"):
        col = f"{side}_starting_qb"
        needs = totals[col].isna()
        abbrev = totals[f"{side}_team"].map(lambda t: TEAM_ABBREV.get(t, t))
        totals.loc[needs, col] = abbrev[needs].map(latest_qb).to_numpy()

    logger.info("Enriched totals with starting quarterbacks and international flags.")
    return totals


def get_totals(path: str = TOTALS_PATH, use_cache_only: bool = False,
               team_games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the current slate, refreshing from the API unless told not to."""
    cached = _read_cache(path)

    if use_cache_only:
        if cached is None:
            raise FileNotFoundError(f"No cached totals at {path} and --use-cached-totals set.")
        logger.info("Using cached totals only.")
        return _finalise(cached, path, team_games, snapshot=False)

    api_key = _api_key()
    if not api_key:
        if cached is None:
            raise ValueError("Missing API_KEY_TOTALS in .env and no cached totals to fall back on.")
        logger.warning("Missing API_KEY_TOTALS in .env; falling back to cached totals.")
        return _finalise(cached, path, team_games, snapshot=False)

    logger.info("Fetching current lines from the Odds API...")
    fresh = get_totals_from_api(api_key)

    if fresh.empty:
        if cached is None:
            raise RuntimeError("Odds API returned no games and there is no cache to fall back on.")
        logger.warning("Odds API returned no games (off-season?). Using cached totals.")
        return _finalise(cached, path, team_games, snapshot=False)

    combined = fresh if cached is None else pd.concat([cached, fresh], ignore_index=True)
    return _finalise(_dedupe(combined), path, team_games, snapshot=True)


def _dedupe(totals: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture, holding the most recently seen line.

    Deduplicating on the exact `commence_time` is not enough: kickoff times shift
    by minutes as the schedule firms up, which used to leave the same game in the
    cache several times over and produce duplicate predictions downstream.
    """
    totals = totals.sort_values("commence_time").copy()

    # Games that kicked off over a week ago are settled; keeping them means every
    # later run re-scores fixtures whose result is already known.
    stale = totals["commence_time"] < pd.Timestamp.now(EASTERN) - pd.Timedelta(days=7)
    if stale.any():
        logger.info("Pruning %d settled fixture(s) from the odds cache.", int(stale.sum()))
        totals = totals[~stale]

    totals["_kickoff_date"] = totals["commence_time"].dt.date
    deduped = totals.drop_duplicates(
        subset=["home_team", "away_team", "_kickoff_date", "bookmaker"], keep="last"
    )
    if len(deduped) < len(totals):
        logger.info("Collapsed %d duplicate fixture rows.", len(totals) - len(deduped))
    return deduped.drop(columns="_kickoff_date").sort_values("commence_time")


def _read_cache(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    logger.info("Reading cached totals from %s...", path)
    cached = pd.read_csv(path)
    cached["commence_time"] = (
        pd.to_datetime(cached["commence_time"], errors="coerce", utc=True)
        .dt.tz_convert(EASTERN)
    )
    return _dedupe(cached[cached["commence_time"].notna()])


def _finalise(totals: pd.DataFrame, path: str, team_games: pd.DataFrame | None,
              snapshot: bool) -> pd.DataFrame:
    result = enrich_totals(totals, team_games)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    result.to_csv(path, index=False)
    if snapshot:
        save_line_snapshot(result)
    return result


def save_line_snapshot(totals: pd.DataFrame, snapshot_dir: str = LINE_SNAPSHOTS_DIR) -> None:
    """Write one dated snapshot of the current lines per day.

    The first snapshot of a week stands in for the opening line, which is what
    `line_movement` is measured against.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"lines_{datetime.today():%Y%m%d}.csv")
    if os.path.exists(path):
        return

    snap = totals[["home_team", "away_team", "total_line"]].copy()
    for col in ("home_team", "away_team"):
        snap[col] = snap[col].map(lambda t: TEAM_ABBREV.get(t, t))
    snap.to_csv(path, index=False)
    logger.info("Saved line snapshot to %s.", path)
