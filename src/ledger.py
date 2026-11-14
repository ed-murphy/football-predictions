"""The bet log: every bet the model has flagged, and how it turned out.

A bet is written down once, the first time it is flagged, with the line and price
available at that moment. Later runs never rewrite it; they only fill in the
result. That makes the log an honest record of what could actually have been
bet, rather than a table that quietly updates itself with hindsight.

Two results are tracked. Profit is what matters in the end but takes hundreds of
bets to judge. Closing-line value — whether the line moved your way after you
bet — shows up much sooner, and a bettor who consistently beats the closing line
is very likely to be a winning one.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from config import BETS_PATH
from src.model import payout

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "game_id", "flagged_at", "kickoff_et", "away_team", "home_team",
    "book", "line", "under_price", "market_line",
    "forecast_wind", "expected_wind", "p_under", "ev", "stake_pct",
    # Filled in after the game:
    "closing_line", "clv_points", "gamebook_wind", "actual_total", "result", "profit_units",
]
_PICK_COLUMNS = LEDGER_COLUMNS[:LEDGER_COLUMNS.index("closing_line")]


def load_ledger(path: str = BETS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(path)
    ledger["result"] = ledger["result"].fillna("pending")
    return ledger.reindex(columns=LEDGER_COLUMNS)


def add_new_bets(ledger: pd.DataFrame, slate: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Append bets flagged on this slate that are not already in the log."""
    flagged = slate[(slate["bet"] != "") & ~slate["game_id"].isin(ledger["game_id"])]
    if flagged.empty:
        return ledger

    new = flagged.assign(flagged_at=now.strftime("%Y-%m-%d %H:%M UTC"))[_PICK_COLUMNS].copy()
    new["result"] = "pending"
    logger.info("Logged %d new bet(s).", len(new))
    frames = [f for f in (ledger, new.reindex(columns=LEDGER_COLUMNS)) if not f.empty]
    return pd.concat(frames, ignore_index=True)


def grade(ledger: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Fill in the result of every pending bet whose game has finished.

    `clv_points` is the line you bet minus the closing line. For an under,
    positive means the market moved down after you bet — you got the better of it.
    """
    ledger = ledger.copy()
    finals = games[games["is_final"]].set_index("game_id")

    for index, bet in ledger[ledger["result"] == "pending"].iterrows():
        if bet["game_id"] not in finals.index:
            continue
        game = finals.loc[bet["game_id"]]
        total, line = float(game["total_points"]), float(bet["line"])

        if total == line:
            result, profit = "push", 0.0
        elif total < line:
            result, profit = "win", payout(bet["under_price"])
        else:
            result, profit = "loss", -1.0

        ledger.loc[index, ["actual_total", "result", "profit_units"]] = [total, result, round(profit, 3)]
        ledger.loc[index, "gamebook_wind"] = game["wind"]
        if pd.notna(game["total_line"]):
            ledger.loc[index, "closing_line"] = float(game["total_line"])
            ledger.loc[index, "clv_points"] = line - float(game["total_line"])
    return ledger


def save_ledger(ledger: pd.DataFrame, path: str = BETS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ledger.to_csv(path, index=False)


def summarise(ledger: pd.DataFrame) -> dict:
    """The live record: results, return, and closing-line value."""
    settled = ledger[ledger["result"].isin(["win", "loss", "push"])]
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    profit = pd.to_numeric(settled["profit_units"], errors="coerce")
    clv = pd.to_numeric(settled["clv_points"], errors="coerce").dropna()
    return {
        "settled": len(settled),
        "pending": int((ledger["result"] == "pending").sum()),
        "wins": wins,
        "losses": losses,
        "pushes": int((settled["result"] == "push").sum()),
        "units": float(profit.sum()) if len(settled) else 0.0,
        "roi": float(profit.mean()) if len(settled) else np.nan,
        "avg_clv_points": float(clv.mean()) if len(clv) else np.nan,
        "beat_close_share": float((clv > 0).mean()) if len(clv) else np.nan,
    }


def describe(ledger: pd.DataFrame) -> str:
    s = summarise(ledger)
    if s["settled"] == 0:
        return f"Live record: no settled bets yet ({s['pending']} pending)."
    text = (f"Live record: {s['wins']}-{s['losses']}-{s['pushes']}, {s['units']:+.2f} units "
            f"({s['roi']:+.1%} per bet), {s['pending']} pending.")
    if not np.isnan(s["avg_clv_points"]):
        text += (f" Closing-line value: {s['avg_clv_points']:+.2f} points on average; "
                 f"beat the close on {s['beat_close_share']:.0%} of bets.")
    return text
