"""Walk-forward backtest: would this have made money, betting only on what was known?

Each season is bet by a model fitted on earlier seasons only. Bets are graded
against the closing total at -110, and again at the price nflverse recorded.

Two versions are reported, because the honest answer sits between them:

  * "Kickoff wind known" uses the wind recorded in the gamebook. It measures how
    wrong the market is, but nobody knows that number when they place the bet.
  * "Forecast" replaces it with what a 1-3 day forecast would have said, using
    the forecast accuracy measured in research/forecast_calibration.py. This is
    the number to plan around.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BACKTEST_START_SEASON, FORECAST_CORRELATION, MIN_EV, STANDARD_ODDS,
)
from src.model import WindModel, break_even, fit_wind_model, payout
from src.schedule import completed_games

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """One season to be bet, and the model that is allowed to bet it."""

    season: int
    model: WindModel
    games: pd.DataFrame      # that season's completed games with known wind
    wind_mean: float         # open-air wind in the training seasons
    wind_sd: float


def make_folds(completed: pd.DataFrame, start_season: int = BACKTEST_START_SEASON) -> list[Fold]:
    """Split into (model fitted on seasons < S, games from season S) pairs."""
    folds = []
    for season in sorted(s for s in completed["season"].unique() if s >= start_season):
        history = completed[completed["season"] < season]
        games = completed[(completed["season"] == season) & completed["kickoff_wind"].notna()]
        if games.empty:
            continue
        outdoor_wind = history.loc[history["open_air"] == 1, "kickoff_wind"].dropna()
        folds.append(Fold(
            season=int(season), model=fit_wind_model(history), games=games,
            wind_mean=float(outdoor_wind.mean()), wind_sd=float(outdoor_wind.std()),
        ))
    return folds


def simulate_forecast(wind: np.ndarray, open_air: np.ndarray, mean: float, sd: float,
                      correlation: float, rng: np.random.Generator) -> np.ndarray:
    """What you would have expected the kickoff wind to be, given only a forecast.

    Draws a forecast that correlates with the true wind as strongly as real
    forecasts do, then returns the best estimate of the true wind given that
    forecast. Two things happen, both of which cost money in real life: the
    estimate is pulled toward the average (a forecast of 20 mph means "probably
    15"), and it is sometimes simply wrong about which games are windy.
    """
    noise = rng.standard_normal(len(wind))
    expected = (
        mean
        + correlation**2 * (wind - mean)
        + correlation * np.sqrt(1.0 - correlation**2) * sd * noise
    )
    return np.where(open_air == 1, np.clip(expected, 0.0, None), 0.0)


def score_fold(fold: Fold, min_ev: float = MIN_EV, forecast_correlation: float | None = None,
               rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Decide and grade every game in one season. Returns one row per game."""
    games = fold.games
    wind = games["kickoff_wind"].to_numpy(dtype=float)

    if forecast_correlation is None:
        wind_seen = wind
    else:
        wind_seen = simulate_forecast(
            wind, games["open_air"].to_numpy(), fold.wind_mean, fold.wind_sd,
            forecast_correlation, rng or np.random.default_rng(0),
        )

    p_under = fold.model.p_under(wind_seen)
    whole_number = (games["total_line"] % 1 == 0).to_numpy()
    p_push = np.where(whole_number, fold.model.push_rate, 0.0)
    ev = (1.0 - p_push) * (p_under * payout(STANDARD_ODDS) - (1.0 - p_under))
    bet = ev >= min_ev

    won = (games["residual"] < 0).to_numpy()
    push = games["push"].to_numpy()
    recorded = pd.to_numeric(games.get("under_odds"), errors="coerce").fillna(STANDARD_ODDS)

    def profit(win_amount):
        return np.where(push, 0.0, np.where(won, win_amount, -1.0))

    return pd.DataFrame({
        "game_id": games["game_id"].to_numpy(),
        "season": fold.season,
        "kickoff_wind": wind,
        "wind_seen": wind_seen,
        "p_under": p_under,
        "ev": ev,
        "bet": bet,
        "result": np.where(push, "push", np.where(won, "win", "loss")),
        "profit": profit(payout(STANDARD_ODDS)),
        "profit_recorded_price": profit(recorded.map(payout).to_numpy()),
    })


def run_backtest(games: pd.DataFrame, start_season: int = BACKTEST_START_SEASON,
                 min_ev: float = MIN_EV, forecast_correlation: float | None = None,
                 seed: int = 0) -> pd.DataFrame:
    """Score every season from `start_season`. One row per game, bets flagged."""
    folds = make_folds(completed_games(games), start_season)
    rng = np.random.default_rng(seed)
    return pd.concat(
        [score_fold(f, min_ev, forecast_correlation, rng) for f in folds], ignore_index=True
    )


def record(scored: pd.DataFrame) -> dict:
    """Betting record for the flagged games: counts, win rate, return and its error."""
    bets = scored[scored["bet"]]
    n = len(bets)
    if n == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": np.nan,
                "units": 0.0, "roi": np.nan, "roi_se": np.nan, "roi_recorded_price": np.nan}

    wins = int((bets["result"] == "win").sum())
    losses = int((bets["result"] == "loss").sum())
    return {
        "bets": n,
        "wins": wins,
        "losses": losses,
        "pushes": int((bets["result"] == "push").sum()),
        "win_rate": wins / (wins + losses) if wins + losses else np.nan,
        "units": float(bets["profit"].sum()),
        "roi": float(bets["profit"].mean()),
        # Standard error of the mean return. An ROI within two of these of zero
        # is not evidence of anything.
        "roi_se": float(bets["profit"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "roi_recorded_price": float(bets["profit_recorded_price"].mean()),
    }


def by_season(scored: pd.DataFrame) -> pd.DataFrame:
    """The betting record for each season separately."""
    rows = [{"season": season, **record(group)} for season, group in scored.groupby("season")]
    return pd.DataFrame(rows)


def forecast_stress_test(games: pd.DataFrame, start_season: int = BACKTEST_START_SEASON,
                         min_ev: float = MIN_EV, correlation: float = FORECAST_CORRELATION,
                         draws: int = 200, seed: int = 0) -> dict:
    """Average betting record when the wind is only known through a forecast.

    The forecast error is random, so one simulated history is not informative;
    this averages `draws` of them.
    """
    folds = make_folds(completed_games(games), start_season)
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(draws):
        scored = pd.concat([score_fold(f, min_ev, correlation, rng) for f in folds])
        records.append(record(scored))
    summary = pd.DataFrame(records)
    return {
        "seasons": len(folds),
        "bets": float(summary["bets"].mean()),
        "win_rate": float(summary["win_rate"].mean()),
        "roi": float(summary["roi"].mean()),
        # Spread across simulated histories of forecast error only; the sampling
        # error of the games themselves is the kickoff-wind roi_se.
        "roi_sd_across_draws": float(summary["roi"].std()),
    }


def calibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Predicted against realised under rate, in bands of predicted probability."""
    decided = scored[scored["result"] != "push"].copy()
    decided["band"] = pd.cut(decided["p_under"], [0, 0.50, 0.52, 0.54, 0.56, 1.0])
    return (
        decided.groupby("band", observed=True)
        .agg(games=("p_under", "size"), predicted=("p_under", "mean"),
             actual=("result", lambda r: (r == "win").mean()))
        .round(3).reset_index()
    )


def report(games: pd.DataFrame, start_season: int = BACKTEST_START_SEASON) -> pd.DataFrame:
    """Run the backtest, log the full report, and return the by-season table."""
    scored = run_backtest(games, start_season)
    seasons = by_season(scored)
    overall = record(scored)
    stress = forecast_stress_test(games, start_season)
    n_seasons = len(seasons)

    logger.info("Walk-forward backtest, %d-%d. Each season bet by a model fitted on earlier "
                "seasons only; unders at %d when expected value >= %.0f%%.",
                seasons["season"].min(), seasons["season"].max(), STANDARD_ODDS, MIN_EV * 100)
    logger.info("By season:\n%s", seasons[
        ["season", "bets", "wins", "losses", "pushes", "win_rate", "units"]
    ].round(3).to_string(index=False))

    z = overall["roi"] / overall["roi_se"]
    logger.info(
        "KICKOFF WIND KNOWN: %d bets (%.0f a season), %d-%d-%d, win rate %.1f%% "
        "(break-even %.1f%%), %+.1f units, ROI %+.1f%% +/- %.1f%% (z = %.1f). "
        "At recorded prices: %+.1f%%. Winning seasons: %d of %d.",
        overall["bets"], overall["bets"] / n_seasons, overall["wins"], overall["losses"],
        overall["pushes"], overall["win_rate"] * 100, break_even() * 100, overall["units"],
        overall["roi"] * 100, overall["roi_se"] * 100, z,
        overall["roi_recorded_price"] * 100, int((seasons["units"] > 0).sum()), n_seasons,
    )
    logger.info(
        "FORECAST (1-3 days out, correlation %.2f with kickoff wind): %.0f bets a season, "
        "win rate %.1f%%, ROI %+.1f%%. This is the realistic expectation; its sampling "
        "error is at least as wide as the +/- %.1f%% above.",
        FORECAST_CORRELATION, stress["bets"] / n_seasons, stress["win_rate"] * 100,
        stress["roi"] * 100, overall["roi_se"] * 100,
    )

    recent = record(scored[scored["season"] >= seasons["season"].max() - 2])
    logger.info(
        "Last three seasons: %d-%d-%d, %+.1f units. A stretch like this is the first "
        "place an edge that has been competed away would show up.",
        recent["wins"], recent["losses"], recent["pushes"], recent["units"],
    )

    sensitivity = []
    for threshold in (0.0, 0.02, 0.04, 0.06):
        r = record(run_backtest(games, start_season, min_ev=threshold))
        sensitivity.append({"min_ev": threshold, "bets": r["bets"], "win_rate": r["win_rate"],
                            "roi": r["roi"], "roi_se": r["roi_se"]})
    logger.info("Sensitivity to the bet threshold (MIN_EV is %.2f):\n%s", MIN_EV,
                pd.DataFrame(sensitivity).round(3).to_string(index=False))
    logger.info("Calibration (all games):\n%s", calibration(scored).to_string(index=False))
    return seasons
