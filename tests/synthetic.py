"""A synthetic league where the truth is known.

Real NFL data cannot tell you whether the code is right, because nobody knows the
true size of the wind effect. Here it is planted, so the tests can check that the
model finds it, that the backtest never peeks, and that the arithmetic adds up.
"""
import numpy as np
import pandas as pd

from src.schedule import exposed_wind
from src.venues import add_venue_features

TRUE_INTERCEPT, TRUE_SLOPE = -0.10, 0.03   # a wind effect stronger than the real one


def make_games(seasons=range(2000, 2012), per_season=260, slope=TRUE_SLOPE, seed=0) -> pd.DataFrame:
    """A schedule-shaped frame of finished games with a planted wind effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(per_season):
            outdoors = rng.random() < 0.7
            wind = float(rng.gamma(2.5, 3.5)) if outdoors else np.nan
            line = float(rng.choice([40.5, 43.5, 44.0, 47.5]))
            p_under = 1 / (1 + np.exp(-(TRUE_INTERCEPT + slope * (wind if outdoors else 0.0))))
            margin = rng.integers(1, 15) + (0.5 if line % 1 else 0)
            total = line - margin if rng.random() < p_under else line + margin
            home_score = total // 2
            rows.append({
                "game_id": f"{season}_{i:03d}", "season": season, "week": 1 + i // 16,
                "gameday": f"{season}-10-{1 + i % 28:02d}", "gametime": "13:00",
                "home_team": "BUF" if outdoors else "DET", "away_team": "MIA",
                "home_score": home_score, "away_score": total - home_score,
                "location": "Home", "roof": "outdoors" if outdoors else "dome",
                "stadium": "Highmark Stadium" if outdoors else "Ford Field",
                "wind": wind, "total_line": line, "under_odds": -110.0, "over_odds": -110.0,
            })
    return prepare(pd.DataFrame(rows))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns `schedule.load_games` adds, without touching disk."""
    games = raw.copy()
    local = pd.to_datetime(games["gameday"] + " " + games["gametime"])
    games["kickoff"] = local.dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    games["is_final"] = games["home_score"].notna()
    games["total_points"] = games["home_score"] + games["away_score"]
    games = add_venue_features(games)
    games["kickoff_wind"] = exposed_wind(games)
    return games
