"""Team-game features derived from play-by-play data.

Replaces pace.py, defense.py, redzone.py, turnovers.py, third_down.py and
offense.py, which were six copies of the same three steps:

    aggregate plays -> one row per (game_id, team)
    lagged rolling mean over the team's previous N games
    broadcast onto both rows of the game as home_* / away_*

Only the aggregation differs, so that is all a `PlayStat` declares. The rolling and
broadcasting are done once, for every stat at once, which also cuts the number of
merges against `team_games` from ~18 down to 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.rolling import broadcast_home_away, lagged_rolling_mean
from config import (
    ROLLING_WINDOW_3RD_DOWN, ROLLING_WINDOW_DEF, ROLLING_WINDOW_OFFENSE,
    ROLLING_WINDOW_PACE, ROLLING_WINDOW_RZ, ROLLING_WINDOW_TURNOVERS,
)

logger = logging.getLogger(__name__)

Aggregator = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class PlayStat:
    """One per-game team statistic and the window used to smooth it."""

    names: list[str]
    window: int
    compute: Aggregator
    requires: set[str] = field(default_factory=set)


def _offense_agg(plays: pd.DataFrame) -> pd.DataFrame:
    """Snap volume plus per-play offensive efficiency, per (game, offense)."""
    scrimmage = plays[(plays["pass_attempt"] == 1) | (plays["rush_attempt"] == 1)].copy()
    scrimmage["explosive"] = (scrimmage["yards_gained"] >= 15).astype(float)

    keys = ["game_id", "posteam"]
    out = (
        scrimmage.groupby(keys, observed=True)
        .agg(
            off_plays=("pass_attempt", "size"),
            pass_rate=("pass_attempt", "mean"),
            explosive_rate=("explosive", "mean"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )

    for label, subset in [
        ("pass_epa", scrimmage[scrimmage["pass_attempt"] == 1]),
        ("rush_epa", scrimmage[scrimmage["rush_attempt"] == 1]),
        ("cpoe", scrimmage[scrimmage["pass_attempt"] == 1]),
    ]:
        col = "cpoe" if label == "cpoe" else "epa"
        agg = subset.groupby(keys, observed=True)[col].mean().rename(label).reset_index()
        out = out.merge(agg, on=keys, how="left")

    return out


def _defense_agg(plays: pd.DataFrame) -> pd.DataFrame:
    """EPA allowed per play and sacks per opponent dropback, per (game, defense)."""
    def_epa = (
        plays.groupby(["game_id", "defteam"], observed=True)["epa"].mean()
        .rename("def_epa").reset_index()
    )
    dropbacks = plays[plays["qb_dropback"] == 1]
    sack_rate = (
        dropbacks.groupby(["game_id", "defteam"], observed=True)
        .agg(opp_dropbacks=("qb_dropback", "size"), sacks=("sack", "sum"))
        .reset_index()
    )
    sack_rate["sack_rate"] = (
        sack_rate["sacks"] / sack_rate["opp_dropbacks"].replace(0, pd.NA)
    )
    out = def_epa.merge(
        sack_rate[["game_id", "defteam", "sack_rate"]], on=["game_id", "defteam"], how="left"
    )
    return out.rename(columns={"defteam": "posteam"})


def _red_zone_agg(plays: pd.DataFrame) -> pd.DataFrame:
    """Touchdowns per red-zone trip, per (game, offense)."""
    red_zone = plays[(plays["yardline_100"] <= 20) & plays["drive"].notna()]

    drives = (
        red_zone.groupby(["game_id", "posteam", "drive"], observed=True)["touchdown"]
        .max().reset_index(name="td")
    )
    out = (
        drives.groupby(["game_id", "posteam"], observed=True)
        .agg(trips=("td", "size"), tds=("td", "sum"))
        .reset_index()
    )
    out["rz_eff"] = out["tds"] / out["trips"].replace(0, pd.NA)
    return out[["game_id", "posteam", "rz_eff"]]


def _turnover_agg(plays: pd.DataFrame) -> pd.DataFrame:
    """Giveaways (interceptions + fumbles lost) per (game, offense)."""
    cols = [c for c in ("interception", "fumble_lost") if c in plays.columns]
    out = (
        plays.groupby(["game_id", "posteam"], observed=True)[cols]
        .sum().sum(axis=1).reset_index(name="turnovers")
    )
    return out


def _third_down_agg(plays: pd.DataFrame) -> pd.DataFrame:
    """Third-down conversion rate per (game, offense)."""
    attempts = plays[
        (plays["third_down_converted"].fillna(0) + plays["third_down_failed"].fillna(0)) > 0
    ]
    out = (
        attempts.groupby(["game_id", "posteam"], observed=True)
        .agg(attempts=("third_down_converted", "size"),
             conversions=("third_down_converted", "sum"))
        .reset_index()
    )
    out["third_down_pct"] = out["conversions"] / out["attempts"].replace(0, pd.NA)
    return out[["game_id", "posteam", "third_down_pct"]]


PLAY_STATS: list[PlayStat] = [
    PlayStat(
        names=["off_plays", "pass_rate", "explosive_rate", "success_rate",
               "pass_epa", "rush_epa", "cpoe"],
        window=ROLLING_WINDOW_OFFENSE,
        compute=_offense_agg,
        requires={"pass_attempt", "rush_attempt", "yards_gained", "success", "epa", "cpoe"},
    ),
    PlayStat(
        names=["def_epa", "sack_rate"],
        window=ROLLING_WINDOW_DEF,
        compute=_defense_agg,
        requires={"defteam", "epa", "qb_dropback", "sack"},
    ),
    PlayStat(
        names=["rz_eff"],
        window=ROLLING_WINDOW_RZ,
        compute=_red_zone_agg,
        requires={"yardline_100", "drive", "touchdown"},
    ),
    PlayStat(
        names=["turnovers"],
        window=ROLLING_WINDOW_TURNOVERS,
        compute=_turnover_agg,
        requires={"interception", "fumble_lost"},
    ),
    PlayStat(
        names=["third_down_pct"],
        window=ROLLING_WINDOW_3RD_DOWN,
        compute=_third_down_agg,
        requires={"third_down_converted", "third_down_failed"},
    ),
]

# Snap volume stabilises faster than efficiency, so it gets its own (longer)
# window even though the same aggregator produces it.
_WINDOW_OVERRIDES = {"off_plays": ROLLING_WINDOW_PACE}


def stat_windows() -> dict[str, int]:
    """Every play-derived stat mapped to the rolling window it uses."""
    return {
        name: _WINDOW_OVERRIDES.get(name, spec.window)
        for spec in PLAY_STATS for name in spec.names
    }


def create_play_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Attach rolling play-derived features to `team_games`.

    For each stat, adds `rolling_avg_<stat>` (the team's own form) plus
    `home_rolling_avg_<stat>` / `away_rolling_avg_<stat>` (the matchup view).
    """
    plays = plays[plays["posteam"].notna()]

    stats = pd.DataFrame(columns=["game_id", "posteam"])
    windows: dict[str, int] = {}

    for spec in PLAY_STATS:
        missing = spec.requires - set(plays.columns)
        if missing:
            logger.warning(
                "Skipping play stats %s — missing columns %s.", spec.names, sorted(missing)
            )
            continue
        frame = spec.compute(plays)
        stats = stats.merge(frame, on=["game_id", "posteam"], how="outer")
        for name in spec.names:
            windows[name] = _WINDOW_OVERRIDES.get(name, spec.window)

    if not windows:
        logger.warning("No play-derived features could be built.")
        return team_games

    team_games = team_games.merge(
        stats.rename(columns={"posteam": "team"}),
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

    rolled = []
    for name, window in windows.items():
        roll_col = f"rolling_avg_{name}"
        team_games[roll_col] = lagged_rolling_mean(team_games, "team", name, window)
        rolled.append(roll_col)

    team_games = broadcast_home_away(team_games, rolled)

    logger.info("Play-derived features created: %s.", ", ".join(sorted(windows)))
    return team_games
