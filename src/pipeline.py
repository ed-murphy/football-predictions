"""Feature pipeline: raw nflverse data in, one row per game out.

`build_features` is the only entry point. It returns a `FeatureBundle` carrying
both the wide game-level frame the model trains on and the lookups needed to score
games that haven't been played yet.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.basic import create_basic_features
from src.injuries import create_injury_features, latest_injury_snapshot
from src.load import load_data, load_injuries
from src.model_features import impute_weather, weather_normals
from src.play_features import create_play_features, stat_windows
from src.qb import create_qb_features
from src.referee import create_referee_features
from src.rest import create_rest_features
from config import ROLLING_WINDOW_POINTS

logger = logging.getLogger(__name__)

# Columns that describe the game rather than one team, carried onto the game row.
_GAME_COLUMNS = [
    "game_id", "season", "week", "date", "total_line", "spread_line", "total_points",
    "divisional", "regular_season", "international", "is_dome", "game_temp",
    "game_wind", "both_short_rest", "ref_avg_total", "season_game_num",
    "home_starting_qb", "away_starting_qb",
]


@dataclass
class FeatureBundle:
    """Everything downstream code needs, in one object.

    Passing six positional values around was how `main.py` used to do this; a
    single object means adding a lookup doesn't ripple through every signature.
    """

    games: pd.DataFrame           # raw schedule
    team_games: pd.DataFrame      # two rows per game
    game_frame: pd.DataFrame      # one row per game — the modelling grain
    latest_qb_epa: pd.DataFrame   # qb_name -> rolling_avg_qb_epa
    team_form: pd.DataFrame       # team -> most recent rolling_* values
    injury_snapshot: pd.DataFrame # team -> injury_index, qb_injured
    ref_stats: pd.Series          # referee -> career average total
    ref_global_mean: float
    weather_normals: pd.DataFrame # week -> mean outdoor temp/wind


def to_game_frame(team_games: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two-rows-per-game frame to one row per game.

    Every `home_*` / `away_*` column already carries both teams' values on both
    rows, so keeping the home row loses nothing.
    """
    home = team_games[team_games["is_home"] == 1].copy()
    home = home.rename(columns={"team": "home_team"})

    side_cols = [c for c in home.columns if c.startswith(("home_", "away_"))]
    keep = [c for c in _GAME_COLUMNS if c in home.columns]
    keep += [c for c in side_cols if c not in keep]

    away_names = (
        team_games.loc[team_games["is_home"] == 0, ["game_id", "team"]]
        .rename(columns={"team": "away_team"})
    )
    game_frame = home[keep].merge(away_names, on="game_id", how="left")
    return game_frame.sort_values(["date", "game_id"]).reset_index(drop=True)


def latest_team_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Each team's rolling features as of *after* its most recent game.

    The stored `rolling_avg_*` columns exclude the game they sit on, which is
    correct for training but stale by one game for prediction. Recomputing the
    mean of each team's trailing games gives the value that should be used for the
    next fixture.
    """
    windows = {
        "points_for": ROLLING_WINDOW_POINTS,
        "points_against": ROLLING_WINDOW_POINTS,
        **stat_windows(),
    }
    windows = {s: w for s, w in windows.items() if s in team_games.columns}

    ordered = team_games.sort_values("date")
    form = pd.DataFrame({
        f"rolling_avg_{stat}":
            ordered.groupby("team").tail(window).groupby("team")[stat].mean()
        for stat, window in windows.items()
    })
    form.index.name = "team"
    return form.reset_index()


def build_features(games: pd.DataFrame | None = None,
                   plays: pd.DataFrame | None = None,
                   injuries: pd.DataFrame | None = None) -> FeatureBundle:
    """Build every historical feature. Loads from disk when frames aren't supplied."""
    if games is None or plays is None:
        games, plays = load_data()
    if injuries is None:
        injuries = load_injuries()

    team_games = create_basic_features(games)
    team_games, latest_qb_epa = create_qb_features(team_games, plays)
    team_games = create_play_features(team_games, plays)
    team_games = create_rest_features(team_games)
    team_games = create_injury_features(team_games, injuries)
    team_games, ref_stats, ref_global_mean = create_referee_features(team_games, games)

    game_frame = to_game_frame(team_games)
    normals = weather_normals(game_frame)
    game_frame = impute_weather(game_frame, normals)

    logger.info(
        "Feature build complete: %d games, %d with a final score.",
        len(game_frame), int(game_frame["total_points"].notna().sum()),
    )

    return FeatureBundle(
        games=games,
        team_games=team_games,
        game_frame=game_frame,
        latest_qb_epa=latest_qb_epa,
        team_form=latest_team_form(team_games),
        injury_snapshot=latest_injury_snapshot(injuries),
        ref_stats=ref_stats,
        ref_global_mean=ref_global_mean,
        weather_normals=normals,
    )
