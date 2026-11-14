"""This week's board: every upcoming game, its wind outlook, and whether to bet it.

For each game the steps are:

  1. find the market's total (median across all books) and the best under on offer
     at a book you can use;
  2. forecast the wind at kickoff and translate it into an expected gamebook wind;
  3. ask the model for P(under) and work out the expected value at that price;
  4. flag a bet if the value clears MIN_EV and kickoff is inside BET_WINDOW_DAYS.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import BET_WINDOW_DAYS, MIN_EV, MY_BOOKS, SLATE_DAYS
from src.model import WindModel, expected_value, kelly_stake_pct
from src.odds import attach_game_ids, market_lines, offers_from_schedule
from src.schedule import EASTERN, upcoming_games
from src.weather import add_wind_forecasts, fetch_hourly_forecast

logger = logging.getLogger(__name__)

SLATE_COLUMNS = [
    "game_id", "kickoff_et", "away_team", "home_team", "stadium", "open_air", "days_out",
    "forecast_wind", "forecast_gust", "expected_wind", "market_line", "book", "line",
    "under_price", "p_under", "ev", "bet", "stake_pct", "status",
]


def build_slate(games: pd.DataFrame, model: WindModel, offers: pd.DataFrame,
                now: pd.Timestamp, forecast_fetcher=fetch_hourly_forecast) -> pd.DataFrame:
    """Score every game kicking off in the next `SLATE_DAYS` days."""
    upcoming = upcoming_games(games, now, SLATE_DAYS)
    if upcoming.empty:
        logger.info("No games in the next %d days.", SLATE_DAYS)
        return pd.DataFrame(columns=SLATE_COLUMNS)

    offers = _all_offers(offers, upcoming)
    upcoming = add_wind_forecasts(upcoming, now, forecast_fetcher)
    upcoming["market_line"] = upcoming["game_id"].map(market_lines(offers))

    rows = []
    for game in upcoming.itertuples(index=False):
        quote = best_under(model, game.expected_wind, game.market_line,
                           offers[offers["game_id"] == game.game_id])
        in_window = game.days_out <= BET_WINDOW_DAYS
        is_bet = bool(quote and quote["ev"] >= MIN_EV and in_window and game.open_air == 1)
        rows.append({
            "game_id": game.game_id,
            "kickoff_et": game.kickoff.tz_convert(EASTERN).strftime("%a %Y-%m-%d %H:%M"),
            "away_team": game.away_team,
            "home_team": game.home_team,
            "stadium": game.stadium,
            "open_air": game.open_air,
            "days_out": game.days_out,
            "forecast_wind": game.forecast_wind,
            "forecast_gust": game.forecast_gust,
            "expected_wind": game.expected_wind,
            "market_line": game.market_line,
            **(quote or {"book": "", "line": np.nan, "under_price": np.nan,
                         "p_under": np.nan, "ev": np.nan}),
            "bet": "Under" if is_bet else "",
            "stake_pct": kelly_stake_pct(quote["p_under"], quote["under_price"]) if is_bet else 0.0,
            "status": _status(game, quote, is_bet),
        })

    slate = pd.DataFrame(rows, columns=SLATE_COLUMNS)
    logger.info("Board: %d games, %d open-air with a forecast, %d bet(s) flagged.",
                len(slate), int((slate["forecast_wind"].notna()).sum()),
                int((slate["bet"] != "").sum()))
    return slate


def best_under(model: WindModel, expected_wind: float, market_line: float,
               game_offers: pd.DataFrame) -> dict | None:
    """The most valuable under among the books you can use, or None if there is none."""
    if pd.isna(expected_wind) or pd.isna(market_line):
        return None
    usable = game_offers[game_offers["book"].isin(_usable_books())] if MY_BOOKS else game_offers

    best = None
    for offer in usable.itertuples(index=False):
        p = model.p_under_at(expected_wind, offer.line, market_line)
        ev = expected_value(p, offer.under_price, model.p_push(offer.line))
        if best is None or ev > best["ev"]:
            best = {"book": offer.book, "line": offer.line, "under_price": offer.under_price,
                    "p_under": round(p, 4), "ev": round(ev, 4)}
    return best


def _usable_books() -> set[str]:
    # "nflverse" is the fallback line used when the Odds API is unavailable.
    return {b.lower() for b in MY_BOOKS} | {"nflverse"}


def _all_offers(offers: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Live offers where we have them; the schedule file's line for any game without."""
    live = attach_game_ids(offers, upcoming)
    uncovered = upcoming[~upcoming["game_id"].isin(live["game_id"])]
    fallback = attach_game_ids(offers_from_schedule(uncovered), upcoming)
    if not uncovered.empty and not live.empty:
        logger.info("%d game(s) not in the odds feed; using the schedule file's line.",
                    len(uncovered))
    frames = [f for f in (live, fallback) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else live


def _status(game, quote: dict | None, is_bet: bool) -> str:
    """One plain-English reason for the decision, shown on the board."""
    if game.open_air == 0:
        return "roof: wind can't reach the field"
    if pd.isna(game.expected_wind):
        return "no forecast available for this venue yet"
    if pd.isna(game.market_line):
        return "no line posted yet"
    if quote is None:
        return "no line at your books"
    if is_bet:
        return f"BET: {game.expected_wind:.0f} mph expected at kickoff"
    if quote["ev"] >= MIN_EV:
        return f"watch: windy, but {game.days_out} days out (bets flag within {BET_WINDOW_DAYS})"
    return "not windy enough"
