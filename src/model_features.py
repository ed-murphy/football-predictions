"""The feature contract.

Training and inference build their design matrices through `build_model_matrix`,
so a feature can never be computed one way for training and another way at serve
time. Previously `upcoming.py` carried a 45-entry rename map that had to be kept
in sync with this module by hand.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ── Raw per-side features ─────────────────────────────────────────────────────
# Each entry becomes `home_<name>` and `away_<name>`.
PER_SIDE_FEATURES = [
    "rolling_avg_points_for",
    "rolling_avg_points_against",
    "rolling_avg_qb_epa",
    "rolling_avg_def_epa",
    "rolling_avg_sack_rate",
    "rolling_avg_off_plays",
    "rolling_avg_pass_rate",
    "rolling_avg_explosive_rate",
    "rolling_avg_success_rate",
    "rolling_avg_rush_epa",
    "rolling_avg_pass_epa",
    "rolling_avg_cpoe",
    "rolling_avg_rz_eff",
    "rolling_avg_turnovers",
    "rolling_avg_third_down_pct",
    "injury_index",
    "qb_injured",
    "short_rest",
    "post_bye",
    "rest_days",
]

# ── Game-level context ────────────────────────────────────────────────────────
CONTEXT_FEATURES = [
    "game_temp",
    "game_wind",
    "is_dome",
    "divisional",
    "regular_season",
    "international",   # played outside the US: long travel, unusual body clock
    "neutral_site",    # includes domestic neutral games, where international is 0
    "both_short_rest",
    "ref_avg_total",
    "season_week",
    # Market context. The closing total is the single most informative number
    # available; the model's job is to disagree with it in a specific direction,
    # so it needs to know where it is starting from.
    "total_line",
    "abs_spread",
]

# ── Matchup aggregates ────────────────────────────────────────────────────────
# `combined_*` drives the level of the total, `*_delta` describes the mismatch.
_COMBINED = [
    "rolling_avg_off_plays", "rolling_avg_qb_epa", "rolling_avg_pass_epa",
    "rolling_avg_rush_epa", "rolling_avg_success_rate", "rolling_avg_explosive_rate",
    "rolling_avg_points_for", "rolling_avg_points_against", "rolling_avg_def_epa",
    "rolling_avg_rz_eff", "rolling_avg_third_down_pct", "rolling_avg_turnovers",
    "injury_index",
]
_DELTA = [
    "rolling_avg_off_plays", "rolling_avg_qb_epa", "rolling_avg_def_epa",
    "rolling_avg_points_for", "rolling_avg_points_against", "rolling_avg_sack_rate",
    "rolling_avg_pass_rate", "rolling_avg_cpoe", "rolling_avg_rz_eff",
    "rolling_avg_third_down_pct", "rolling_avg_turnovers",
]

INTERACTION_FEATURES = [
    "qb_epa_vs_def",        # each offence's QB form against the defence it faces
    "plays_x_wind",         # wind suppresses passing most in high-volume games
    "pace_x_efficiency",    # volume only produces points when paired with efficiency
]

_SUMMED = [f"combined_{c}" for c in _COMBINED]
_DIFFED = [f"{d}_delta" for d in _DELTA]


def get_model_features() -> list[str]:
    """The exact ordered column list the model consumes."""
    per_side = [f"{side}_{f}" for f in PER_SIDE_FEATURES for side in ("home", "away")]
    return per_side + CONTEXT_FEATURES + _SUMMED + _DIFFED + INTERACTION_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every combined/delta/interaction column from the per-side columns."""
    df = df.copy()

    for base in _COMBINED:
        df[f"combined_{base}"] = df[f"home_{base}"] + df[f"away_{base}"]
    for base in _DELTA:
        df[f"{base}_delta"] = df[f"home_{base}"] - df[f"away_{base}"]

    # A good quarterback facing a bad defence is worth more than the sum of the
    # two ratings; sign convention is "positive = more expected scoring".
    df["qb_epa_vs_def"] = (
        df["home_rolling_avg_qb_epa"] * df["away_rolling_avg_def_epa"]
        + df["away_rolling_avg_qb_epa"] * df["home_rolling_avg_def_epa"]
    )
    df["plays_x_wind"] = df["combined_rolling_avg_off_plays"] * df["game_wind"]
    df["pace_x_efficiency"] = (
        df["combined_rolling_avg_off_plays"] * df["combined_rolling_avg_success_rate"]
    )
    return df


def build_model_matrix(games_frame: pd.DataFrame) -> pd.DataFrame:
    """Turn one-row-per-game data into the model design matrix.

    Expects the per-side `home_*` / `away_*` columns, the context columns, plus
    `total_line`, `spread_line` and `week`. Returns the same rows with every
    modelling column present and numeric.
    """
    df = games_frame.copy()

    if "abs_spread" not in df.columns:
        df["abs_spread"] = df["spread_line"].abs()
    if "season_week" not in df.columns:
        df["season_week"] = pd.to_numeric(df["week"], errors="coerce")

    missing = [
        c for c in
        [f"{s}_{f}" for f in PER_SIDE_FEATURES for s in ("home", "away")] + CONTEXT_FEATURES
        if c not in df.columns and c not in ("abs_spread", "season_week")
    ]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    df = add_derived_features(df)

    features = get_model_features()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    return df


def impute_weather(df: pd.DataFrame, normals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill unknown kickoff weather with seasonal normals for that week of the year.

    nflverse leaves `temp`/`wind` blank for roughly 5% of outdoor games, and the
    forecast API can't see past five days. Dropping those rows would bias the
    training set toward well-documented stadiums, so fill instead.
    """
    df = df.copy()
    if normals is None:
        normals = weather_normals(df)

    week = pd.to_numeric(df["week"], errors="coerce")
    for col in ("game_temp", "game_wind"):
        fill = week.map(normals[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
        df[col] = df[col].fillna(normals[col].mean())
    return df


def weather_normals(df: pd.DataFrame) -> pd.DataFrame:
    """Mean outdoor temperature and wind by week of season."""
    outdoor = df[df["is_dome"] == 0]
    normals = (
        outdoor.assign(week=pd.to_numeric(outdoor["week"], errors="coerce"))
        .groupby("week")[["game_temp", "game_wind"]]
        .mean()
    )
    if normals.empty:
        normals = pd.DataFrame({"game_temp": [55.0], "game_wind": [8.0]}, index=[1])
    return normals


def describe_features() -> pd.DataFrame:
    """Feature inventory, for documentation and sanity-checking model output."""
    rows = []
    for name in get_model_features():
        if name in CONTEXT_FEATURES:
            group = "context"
        elif name in INTERACTION_FEATURES:
            group = "interaction"
        elif name.startswith("combined_"):
            group = "matchup total"
        elif name.endswith("_delta"):
            group = "matchup gap"
        else:
            group = "team form"
        rows.append({"feature": name, "group": group})
    return pd.DataFrame(rows)


__all__ = [
    "get_model_features", "build_model_matrix", "add_derived_features",
    "impute_weather", "weather_normals", "describe_features",
    "PER_SIDE_FEATURES", "CONTEXT_FEATURES",
]
