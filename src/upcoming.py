"""Assembling feature rows for games that haven't been played yet.

This is the half of the system most likely to drift away from training. The whole
file exists to produce a frame with the same columns, units and meaning as
`pipeline.to_game_frame` output, which is then handed to the *same*
`build_model_matrix` the trainer uses. When a feature is added to
`model_features.py`, the only change needed here is supplying its raw input.
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.constants import DIVISION_MAP, TEAM_ABBREV
from src.model_features import build_model_matrix, impute_weather
from src.pipeline import FeatureBundle
from src.referee import get_upcoming_referee_feature
from src.rest import BYE_MAX_DAYS, BYE_MIN_DAYS, SHORT_REST_MAX_DAYS
from config import LINE_SNAPSHOTS_DIR

logger = logging.getLogger(__name__)

# Rolling form columns served from `bundle.team_form`, per side.
_FORM_COLUMNS = [
    "rolling_avg_points_for", "rolling_avg_points_against", "rolling_avg_off_plays",
    "rolling_avg_pass_rate", "rolling_avg_explosive_rate", "rolling_avg_success_rate",
    "rolling_avg_pass_epa", "rolling_avg_rush_epa", "rolling_avg_cpoe",
    "rolling_avg_def_epa", "rolling_avg_sack_rate", "rolling_avg_rz_eff",
    "rolling_avg_turnovers", "rolling_avg_third_down_pct",
]


def prepare_upcoming_games(
    totals: pd.DataFrame,
    bundle: FeatureBundle,
    weather_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Build a scored-ready model matrix, one row per upcoming game."""
    if totals is None or totals.empty:
        logger.info("No upcoming games in the odds feed.")
        return pd.DataFrame()

    games = _base_frame(totals)
    games = _add_schedule_context(games, bundle)
    games = _add_team_form(games, bundle)
    games = _add_qb_form(games, totals, bundle)
    games = _add_injuries(games, bundle)
    games = _add_referee(games, bundle)
    games = _add_weather(games, weather_forecast, bundle)
    games = _add_line_movement(games)

    matrix = build_model_matrix(games)
    logger.info("Prepared %d upcoming games for scoring.", len(matrix))
    return matrix


def _base_frame(totals: pd.DataFrame) -> pd.DataFrame:
    """Team codes, kickoff, market numbers."""
    df = totals.copy()
    df["home_team"] = df["home_team"].map(lambda t: TEAM_ABBREV.get(t, t))
    df["away_team"] = df["away_team"].map(lambda t: TEAM_ABBREV.get(t, t))

    kickoff = pd.to_datetime(df["commence_time"], utc=True)
    df["kickoff_utc"] = kickoff
    df["date"] = kickoff.dt.tz_convert("US/Eastern").dt.tz_localize(None)

    unknown = df[df["home_team"].isna() | df["away_team"].isna()]
    if not unknown.empty:
        logger.warning("Dropping %d games with unrecognised team names.", len(unknown))
        df = df.drop(unknown.index)

    df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    df["spread_line"] = _numeric_column(df, "spread_line", default=0.0)
    df["international"] = _numeric_column(df, "international", default=0.0).astype(int)

    df["divisional"] = (
        df["home_team"].map(DIVISION_MAP) == df["away_team"].map(DIVISION_MAP)
    ).astype(int)
    df["regular_season"] = 1

    return df.reset_index(drop=True)


def _numeric_column(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    """Numeric view of an optional column — the odds feed doesn't always carry one."""
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def _add_schedule_context(games: pd.DataFrame, bundle: FeatureBundle) -> pd.DataFrame:
    """Week number and rest days, derived from each team's last completed game."""
    last_game = (
        bundle.team_games.sort_values("date")
        .groupby("team")
        .agg(last_date=("date", "last"), last_season=("season", "last"),
             last_week=("week", "last"))
    )

    for side in ("home", "away"):
        last = games[f"{side}_team"].map(last_game["last_date"])
        rest = (games["date"] - pd.to_datetime(last)).dt.days
        games[f"{side}_rest_days"] = rest.clip(lower=4, upper=21)
        games[f"{side}_short_rest"] = (rest <= SHORT_REST_MAX_DAYS).astype(int)
        games[f"{side}_post_bye"] = rest.between(BYE_MIN_DAYS, BYE_MAX_DAYS).astype(int)

    games["both_short_rest"] = (
        (games["home_short_rest"] == 1) & (games["away_short_rest"] == 1)
    ).astype(int)

    # Week number: continue on from the home team's last played week where the gap
    # looks like a normal turnaround, otherwise start a new season at week 1.
    last_week = games["home_team"].map(last_game["last_week"])
    gap_days = (games["date"] - pd.to_datetime(games["home_team"].map(last_game["last_date"]))).dt.days
    games["week"] = np.where(gap_days <= 30, last_week + (gap_days / 7).round(), 1)
    games["week"] = games["week"].clip(1, 22)
    games["season"] = games["date"].dt.year - (games["date"].dt.month < 3).astype(int)
    return games


def _add_team_form(games: pd.DataFrame, bundle: FeatureBundle) -> pd.DataFrame:
    """Attach each team's latest rolling form to its side of the matchup."""
    form = bundle.team_form.set_index("team")
    available = [c for c in _FORM_COLUMNS if c in form.columns]
    missing = set(_FORM_COLUMNS) - set(available)
    if missing:
        logger.warning("Form columns unavailable for upcoming games: %s", sorted(missing))

    for side in ("home", "away"):
        team = games[f"{side}_team"]
        for col in available:
            games[f"{side}_{col}"] = team.map(form[col]).to_numpy()
        for col in missing:
            games[f"{side}_{col}"] = np.nan

        unknown = sorted(set(team[~team.isin(form.index)]))
        if unknown:
            logger.warning("No recent form on record for: %s", unknown)
    return games


def _add_qb_form(games: pd.DataFrame, totals: pd.DataFrame,
                 bundle: FeatureBundle) -> pd.DataFrame:
    """Rolling EPA for each listed starter, falling back to the league average."""
    qb_epa = bundle.latest_qb_epa.set_index("qb_name")["rolling_avg_qb_epa"]
    league_avg = float(qb_epa.median())

    for side in ("home", "away"):
        col = f"{side}_starting_qb"
        names = totals[col].reindex(games.index) if col in totals.columns else pd.Series(index=games.index, dtype=object)
        values = names.map(qb_epa)
        unknown = values.isna() & names.notna()
        if unknown.any():
            logger.warning(
                "No EPA history for %s starter(s): %s — using league median.",
                side, sorted(set(names[unknown])),
            )
        games[f"{side}_rolling_avg_qb_epa"] = values.fillna(league_avg).to_numpy()
    return games


def _add_injuries(games: pd.DataFrame, bundle: FeatureBundle) -> pd.DataFrame:
    snapshot = bundle.injury_snapshot
    if snapshot.empty:
        for side in ("home", "away"):
            games[f"{side}_injury_index"] = 0.0
            games[f"{side}_qb_injured"] = 0
        return games

    indexed = snapshot.set_index("team")
    for side in ("home", "away"):
        team = games[f"{side}_team"]
        games[f"{side}_injury_index"] = team.map(indexed["injury_index"]).fillna(0.0).to_numpy()
        games[f"{side}_qb_injured"] = (
            team.map(indexed["qb_injured"]).fillna(0).astype(int).to_numpy()
        )
    return games


def _add_referee(games: pd.DataFrame, bundle: FeatureBundle) -> pd.DataFrame:
    games["ref_avg_total"] = get_upcoming_referee_feature(
        games, bundle.games, bundle.ref_stats, bundle.ref_global_mean
    ).to_numpy()
    return games


def _add_weather(games: pd.DataFrame, forecast: pd.DataFrame,
                 bundle: FeatureBundle) -> pd.DataFrame:
    """Forecast temperature/wind in the same units as the training data (F, mph)."""
    from src.constants import DOME_TEAMS, INDOOR_TEMP_F, INDOOR_WIND_MPH

    games["is_dome"] = games["home_team"].isin(DOME_TEAMS).astype(int)
    games["game_temp"] = np.nan
    games["game_wind"] = np.nan

    if forecast is not None and not forecast.empty:
        fc = forecast.copy()
        fc["kickoff_time"] = pd.to_datetime(fc["kickoff_time"], utc=True)
        # Match on home team plus calendar day: forecast slots are three-hourly and
        # kickoff times shift as the schedule firms up.
        fc["match_day"] = fc["kickoff_time"].dt.date
        games["match_day"] = games["kickoff_utc"].dt.date

        merged = games.merge(
            fc[["home_team", "match_day", "temperature", "wind_speed"]]
            .drop_duplicates(["home_team", "match_day"]),
            on=["home_team", "match_day"], how="left",
        )
        games["game_temp"] = merged["temperature"].to_numpy()
        games["game_wind"] = merged["wind_speed"].to_numpy()
        games = games.drop(columns="match_day")

    indoor = games["is_dome"] == 1
    games.loc[indoor, "game_temp"] = INDOOR_TEMP_F
    games.loc[indoor, "game_wind"] = INDOOR_WIND_MPH

    unknown = int(games["game_temp"].isna().sum())
    if unknown:
        logger.info("%d game(s) beyond the forecast horizon; using seasonal normals.", unknown)
    return impute_weather(games, bundle.weather_normals)


def _add_line_movement(games: pd.DataFrame) -> pd.DataFrame:
    """Movement from the earliest line snapshot in the last week (informational)."""
    games["line_open"] = np.nan
    games["line_movement"] = np.nan

    if not os.path.isdir(LINE_SNAPSHOTS_DIR):
        return games

    cutoff = datetime.today() - timedelta(days=7)
    recent = sorted(
        f for f in glob.glob(os.path.join(LINE_SNAPSHOTS_DIR, "lines_*.csv"))
        if _snapshot_date(f) and _snapshot_date(f) >= cutoff
    )
    if not recent:
        return games

    opening = (
        pd.read_csv(recent[0])[["home_team", "away_team", "total_line"]]
        .rename(columns={"total_line": "line_open"})
        .drop_duplicates(["home_team", "away_team"])
    )
    merged = games.merge(opening, on=["home_team", "away_team"], how="left",
                         suffixes=("", "_snap"))
    games["line_open"] = merged["line_open"].to_numpy()
    games["line_movement"] = games["total_line"] - games["line_open"]

    matched = int(games["line_open"].notna().sum())
    logger.info("Line movement matched for %d/%d games from %s.",
                matched, len(games), os.path.basename(recent[0]))
    return games


def _snapshot_date(path: str) -> datetime | None:
    stem = os.path.basename(path)
    try:
        return datetime.strptime(stem[len("lines_"):len("lines_") + 8], "%Y%m%d")
    except ValueError:
        return None
