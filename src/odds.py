"""Live totals from the Odds API.

Every sportsbook in the feed is kept, not just one, for two reasons. The median
across books is a better read on the true market line than any single book, and
when two books you can use disagree, the higher total is simply a better under.

Every fetch is also appended to a history file. Free historical opening lines do
not exist for the NFL; this builds your own.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from config import API_TIMEOUT_SECONDS, LINE_HISTORY_PATH, ODDS_URL, STANDARD_ODDS
from src.constants import TEAM_ABBREV
from src.schedule import EASTERN

logger = logging.getLogger(__name__)

OFFER_COLUMNS = ["home_team", "away_team", "kickoff", "book", "line", "under_price", "over_price"]


def fetch_offers(api_key: str | None = None) -> pd.DataFrame:
    """Current total and prices at every book, one row per (game, book).

    Returns an empty frame — never raises — when there is no key or the request
    fails, so the caller can fall back to the line in the schedule file.
    """
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("API_KEY_TOTALS")
    if not api_key:
        logger.warning("No API_KEY_TOTALS in .env; falling back to the lines in the schedule file.")
        return pd.DataFrame(columns=OFFER_COLUMNS)

    try:
        response = requests.get(
            ODDS_URL,
            params={"apiKey": api_key.strip(), "regions": "us", "markets": "totals",
                    "oddsFormat": "american"},
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # Never log the exception text: the request URL in it contains the key.
        logger.error("Odds API request failed (%s); falling back to the schedule file.",
                     type(exc).__name__)
        return pd.DataFrame(columns=OFFER_COLUMNS)

    remaining = response.headers.get("x-requests-remaining")
    offers = parse_offers(response.json())
    logger.info("Odds API: %d offers across %d games from %d books%s.",
                len(offers), offers[["home_team", "away_team"]].drop_duplicates().shape[0],
                offers["book"].nunique(),
                f" ({remaining} requests left this month)" if remaining else "")
    return offers


def parse_offers(payload: list[dict]) -> pd.DataFrame:
    """Flatten the Odds API response into one row per (game, book)."""
    rows = []
    for game in payload:
        home = TEAM_ABBREV.get(game.get("home_team"))
        away = TEAM_ABBREV.get(game.get("away_team"))
        if not home or not away:
            logger.warning("Skipping a game with an unrecognised team: %s at %s.",
                           game.get("away_team"), game.get("home_team"))
            continue

        for book in game.get("bookmakers", []):
            totals = next((m for m in book.get("markets", []) if m.get("key") == "totals"), None)
            if totals is None:
                continue
            sides = {o.get("name"): o for o in totals.get("outcomes", [])}
            if "Under" not in sides or sides["Under"].get("point") is None:
                continue
            rows.append({
                "home_team": home,
                "away_team": away,
                "kickoff": pd.Timestamp(game["commence_time"]).tz_convert("UTC"),
                "book": book["key"],
                "line": float(sides["Under"]["point"]),
                "under_price": float(sides["Under"].get("price", STANDARD_ODDS)),
                "over_price": float(sides.get("Over", {}).get("price", STANDARD_ODDS)),
            })
    return pd.DataFrame(rows, columns=OFFER_COLUMNS)


def offers_from_schedule(upcoming: pd.DataFrame) -> pd.DataFrame:
    """Fallback offers: the total nflverse lists for games that have one posted."""
    listed = upcoming[upcoming["total_line"].notna()]
    return pd.DataFrame({
        "home_team": listed["home_team"].to_numpy(),
        "away_team": listed["away_team"].to_numpy(),
        "kickoff": listed["kickoff"].to_numpy(),
        "book": "nflverse",
        "line": listed["total_line"].to_numpy(dtype=float),
        "under_price": pd.to_numeric(listed.get("under_odds"), errors="coerce")
                         .fillna(STANDARD_ODDS).to_numpy(),
        "over_price": pd.to_numeric(listed.get("over_odds"), errors="coerce")
                        .fillna(STANDARD_ODDS).to_numpy(),
    }, columns=OFFER_COLUMNS)


def attach_game_ids(offers: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Label each offer with the schedule's `game_id`.

    Matched on the two teams and the kickoff date in US Eastern. Kickoff times
    drift by minutes between feeds, and the same teams meet twice a season, so
    neither the exact time nor the teams alone would do.
    """
    if offers.empty:
        return offers.assign(game_id=pd.Series(dtype=object))

    def eastern_date(kickoff: pd.Series) -> pd.Series:
        return pd.to_datetime(kickoff, utc=True).dt.tz_convert(EASTERN).dt.date

    keys = upcoming.assign(game_date=eastern_date(upcoming["kickoff"]))[
        ["game_id", "home_team", "away_team", "game_date"]
    ]
    matched = offers.assign(game_date=eastern_date(offers["kickoff"])).merge(
        keys, on=["home_team", "away_team", "game_date"], how="inner"
    )
    return matched.drop(columns="game_date")


def market_lines(offers: pd.DataFrame) -> pd.Series:
    """The market's total for each game: the median line across every book."""
    return offers.groupby("game_id")["line"].median()


def append_line_history(offers: pd.DataFrame, now: pd.Timestamp,
                        path: str = LINE_HISTORY_PATH) -> None:
    """Add this fetch to the running record of every line seen."""
    if offers.empty:
        return
    stamped = offers.assign(fetched_at=now.isoformat())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stamped.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
