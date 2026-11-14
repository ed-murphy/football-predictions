"""NFL windy-unders model — command line.

    python main.py              weekly run: refresh data, score upcoming games, update the bet log
    python main.py backtest     walk-forward backtest report
    python main.py record       live record of every bet flagged so far

Weekly-run options:
    --no-download    reuse the cached schedule file instead of refreshing it
    --no-odds-api    use the lines in the schedule file (saves Odds API quota)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from config import BACKTEST_PATH, GAMES_PATH, MIN_EV, SLATE_PATH, STANDARD_ODDS
from src import backtest, ledger
from src.model import fit_wind_model
from src.odds import OFFER_COLUMNS, append_line_history, fetch_offers
from src.schedule import completed_games, download_games, load_games
from src.slate import build_slate

logger = logging.getLogger("main")


def setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def weekly_run(download: bool = True, use_odds_api: bool = True) -> None:
    """Refresh the data, score the upcoming games, and update the bet log."""
    now = pd.Timestamp.now(tz="UTC")

    if download:
        try:
            download_games()
        except Exception as exc:
            if not os.path.exists(GAMES_PATH):
                raise
            logger.warning("Could not refresh the schedule (%s); using the cached copy.", exc)

    games = load_games()
    model = fit_wind_model(completed_games(games))
    logger.info(model.describe())
    logger.info("Bets are flagged at %.1f mph expected kickoff wind or more (EV >= %.0f%% at %d).\n",
                model.wind_needed(min_ev=MIN_EV), MIN_EV * 100, STANDARD_ODDS)

    offers = fetch_offers() if use_odds_api else pd.DataFrame(columns=OFFER_COLUMNS)
    append_line_history(offers, now)

    slate = build_slate(games, model, offers, now)
    os.makedirs(os.path.dirname(SLATE_PATH), exist_ok=True)
    slate.to_csv(SLATE_PATH, index=False)
    print_slate(slate)

    bets = ledger.grade(ledger.add_new_bets(ledger.load_ledger(), slate, now), games)
    ledger.save_ledger(bets)
    logger.info("\n%s", ledger.describe(bets))


def print_slate(slate: pd.DataFrame) -> None:
    if slate.empty:
        logger.info("No upcoming games to show.")
        return
    board = pd.DataFrame({
        "kickoff (ET)": slate["kickoff_et"],
        "game": slate["away_team"] + " @ " + slate["home_team"],
        "line": slate["line"].fillna(slate["market_line"]),
        "fcst mph": slate["forecast_wind"].round(0),
        "exp mph": slate["expected_wind"].round(1),
        "P(under)": (slate["p_under"] * 100).round(1),
        "EV %": (slate["ev"] * 100).round(1),
        "stake %": slate["stake_pct"].where(slate["bet"] != "", None),
        "status": slate["status"],
    })
    logger.info("\n%s", board.to_string(index=False, na_rep="-"))

    flagged = slate[slate["bet"] != ""]
    if flagged.empty:
        logger.info("\nNo bets this run. Most weeks have none or one; that is the model working.")
    for bet in flagged.itertuples():
        logger.info("\n>> UNDER %.1f  %s @ %s  at %s (%+.0f), stake %.2f%% of bankroll",
                    bet.line, bet.away_team, bet.home_team, bet.book, bet.under_price, bet.stake_pct)


def run_backtest() -> None:
    games = load_games()
    logger.info(fit_wind_model(completed_games(games)).describe() + "\n")
    seasons = backtest.report(games)
    os.makedirs(os.path.dirname(BACKTEST_PATH), exist_ok=True)
    seasons.to_csv(BACKTEST_PATH, index=False)
    logger.info("\nBy-season table written to %s.", BACKTEST_PATH)


def show_record() -> None:
    bets = ledger.grade(ledger.load_ledger(), load_games())
    ledger.save_ledger(bets)
    if not bets.empty:
        logger.info(bets[["kickoff_et", "away_team", "home_team", "book", "line", "under_price",
                          "expected_wind", "closing_line", "clv_points", "actual_total",
                          "result", "profit_units"]].to_string(index=False, na_rep="-"))
    logger.info("\n%s", ledger.describe(bets))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="weekly",
                        choices=["weekly", "backtest", "record"])
    parser.add_argument("--no-download", action="store_true",
                        help="reuse the cached schedule file")
    parser.add_argument("--no-odds-api", action="store_true",
                        help="use the schedule file's lines instead of calling the Odds API")
    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "backtest":
        run_backtest()
    elif args.command == "record":
        show_record()
    else:
        weekly_run(download=not args.no_download, use_odds_api=not args.no_odds_api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
