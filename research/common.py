"""Shared helpers for the research scripts.

Nothing in `research/` is used by the weekly run. These scripts exist so that
every claim in docs/model.md can be re-derived, and so new ideas can be tested
with the same discipline the wind signal had to pass.

The discipline is a split-sample rule. A candidate signal is scored on a
discovery era (through 2013) and again on a confirmation era (2014 on). It
counts only if it clears |z| > 1.96 in both, with the same sign. With about 120
candidates tested, a one-era fluke is certain to appear; a two-era fluke of the
same sign has odds of roughly one in 800 per candidate.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from config import GAMES_PATH
from src.constants import STADIUM_COORDS

DISCOVERY_END = 2013
CACHE_DIR = "data/research"

# Where relocated or rebuilt teams played, for looking up historical weather.
_FORMER_HOMES = {
    "OAK": (37.7516, -122.2005), "SD": (32.7831, -117.1196), "STL": (38.6328, -90.1885),
}


def stadium_coords(team: str, season: int) -> tuple[float, float]:
    """Home stadium coordinates for a team in a given season."""
    if team in _FORMER_HOMES:
        return _FORMER_HOMES[team]
    if team == "LV" and season < 2020:
        return _FORMER_HOMES["OAK"]
    if team == "LAC" and season < 2017:
        return _FORMER_HOMES["SD"]
    if team == "LAC" and season < 2020:
        return (33.8644, -118.2611)     # Carson
    if team == "LA" and season < 2016:
        return _FORMER_HOMES["STL"]
    if team == "SF" and season < 2014:
        return (37.7135, -122.3863)     # Candlestick Park
    if team == "CHI" and season == 2002:
        return (40.0993, -88.2360)      # Champaign, during the Soldier Field rebuild
    return STADIUM_COORDS[team]


def load_results(path: str = GAMES_PATH) -> pd.DataFrame:
    """Finished games with closing lines and the residuals every scan is scored on."""
    games = pd.read_csv(path, low_memory=False)
    games = games[games["home_score"].notna()].sort_values(["gameday", "game_id"])
    games = games.reset_index(drop=True)
    games["total_points"] = games["home_score"] + games["away_score"]
    games["margin"] = games["home_score"] - games["away_score"]
    games["total_resid"] = games["total_points"] - games["total_line"]    # + = went over
    games["spread_resid"] = games["margin"] - games["spread_line"]        # + = home covered
    games["kickoff_hour"] = pd.to_numeric(games["gametime"].astype(str).str[:2], errors="coerce")
    games["outdoors"] = (games["roof"] == "outdoors").astype(int)
    games["wind_outdoors"] = games["wind"].where(games["outdoors"] == 1).fillna(0.0)
    return games


def z_score(signal: pd.Series, outcome: pd.Series) -> float:
    """z-statistic of the correlation between a signal and an outcome."""
    ok = signal.notna() & outcome.notna()
    x, y = signal[ok].astype(float), outcome[ok].astype(float)
    if len(x) < 30 or x.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] * np.sqrt(len(x) - 3))


def split_sample_scan(candidates: dict[str, pd.Series], resid: pd.Series,
                      season: pd.Series) -> pd.DataFrame:
    """Score every candidate on the discovery and confirmation eras.

    The outcome is the sign of the residual (which side won), with pushes left
    out. `passes` marks candidates clearing |z| > 1.96 in both eras, same sign.
    """
    decided = resid != 0
    side = np.sign(resid)
    rows = []
    for name, signal in candidates.items():
        early = decided & (season <= DISCOVERY_END)
        late = decided & (season > DISCOVERY_END)
        z1, z2 = z_score(signal[early], side[early]), z_score(signal[late], side[late])
        rows.append({
            "signal": name, "z_discovery": round(z1, 2), "z_confirmation": round(z2, 2),
            "passes": "YES" if min(abs(z1), abs(z2)) > 1.96 and np.sign(z1) == np.sign(z2) else "",
        })
    return pd.DataFrame(rows)


def noise_check(table: pd.DataFrame) -> str:
    """Do the z-scores as a whole look like anything but chance?"""
    z = table[["z_discovery", "z_confirmation"]].to_numpy().ravel()
    z = z[~np.isnan(z)]
    return (f"{len(z)} z-scores: standard deviation {z.std():.2f} (pure noise = 1.00), "
            f"{np.mean(np.abs(z) > 1.96):.0%} beyond +/-1.96 (pure noise = 5%).")


def under_record(games: pd.DataFrame, resid_col: str = "total_resid") -> str:
    """Win rate and flat -110 return from betting the under on every row."""
    decided = games[games[resid_col] != 0]
    if len(decided) < 5:
        return f"n={len(decided)}"
    won = decided[resid_col] < 0
    profit = np.where(won, 100 / 110, -1.0)
    return (f"n={len(decided):4d}  under {won.mean():.1%}  "
            f"ROI {profit.mean():+.1%} +/- {profit.std(ddof=1) / np.sqrt(len(decided)):.1%}")


def fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression with any number of columns (X must include the constant)."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(hessian)))


def get_json(url: str, params: dict, session: requests.Session | None = None) -> dict | None:
    """GET with patience: Open-Meteo's free tier rate-limits by the minute."""
    session = session or requests
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=90)
            if response.status_code == 429:
                print("  rate limited; waiting a minute...", flush=True)
                time.sleep(65)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            print(f"  retrying after {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)
