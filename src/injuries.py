"""Weekly injury-report features.

The injury index is a weighted count of unavailable offensive players: each
listing contributes `P(miss the game) x positional scoring importance`.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.rolling import broadcast_home_away

logger = logging.getLogger(__name__)

# Probability the player is effectively unavailable, by report status.
STATUS_WEIGHTS = {
    "Out": 1.00,
    "Doubtful": 0.75,
    "Questionable": 0.25,
    "Probable": 0.00,
}

# How much that player's absence moves expected scoring.
POSITION_WEIGHTS = {
    "QB": 3.0,
    "WR": 0.8, "RB": 0.5, "TE": 0.5,
    "OL": 0.3, "OT": 0.3, "OG": 0.3, "G": 0.3, "C": 0.3, "T": 0.3,
}


def _team_week_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): injury index and starting-QB-unavailable flag."""
    inj = injuries[injuries["game_type"] == "REG"].copy()
    inj["score"] = (
        inj["report_status"].map(STATUS_WEIGHTS).fillna(0.0)
        * inj["position"].map(POSITION_WEIGHTS).fillna(0.0)
    )

    index = (
        inj.groupby(["season", "week", "team"], as_index=False)["score"]
        .sum().rename(columns={"score": "injury_index"})
    )
    qb_out = (
        inj[(inj["position"] == "QB") & inj["report_status"].isin(["Out", "Doubtful"])]
        .groupby(["season", "week", "team"], as_index=False)
        .size().assign(qb_injured=1)[["season", "week", "team", "qb_injured"]]
    )

    out = index.merge(qb_out, on=["season", "week", "team"], how="left")
    out["qb_injured"] = out["qb_injured"].fillna(0).astype(int)
    return out


def create_injury_features(team_games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add `home/away_injury_index` and `home/away_qb_injured` to `team_games`."""
    if injuries is None or injuries.empty:
        team_games = team_games.assign(injury_index=0.0, qb_injured=0)
        return broadcast_home_away(team_games, ["injury_index", "qb_injured"])

    team_week = _team_week_injuries(injuries)
    team_games = team_games.merge(
        team_week, on=["season", "week", "team"], how="left", validate="many_to_one"
    )
    team_games["injury_index"] = team_games["injury_index"].fillna(0.0)
    team_games["qb_injured"] = team_games["qb_injured"].fillna(0).astype(int)

    team_games = broadcast_home_away(team_games, ["injury_index", "qb_injured"])
    logger.info("Injury features created: injury index and QB-out flag.")
    return team_games


def latest_injury_snapshot(injuries: pd.DataFrame) -> pd.DataFrame:
    """Most recent reported week per team, for scoring upcoming games.

    Returns columns `team`, `injury_index`, `qb_injured`. Empty if unavailable.
    """
    empty = pd.DataFrame(columns=["team", "injury_index", "qb_injured"])
    if injuries is None or injuries.empty:
        return empty

    team_week = _team_week_injuries(injuries)
    if team_week.empty:
        return empty

    # Take each team's own latest report rather than a single global week — teams
    # on a bye have no listing for the current week.
    latest = (
        team_week.sort_values(["season", "week"])
        .groupby("team", as_index=False)
        .tail(1)[["team", "injury_index", "qb_injured", "season", "week"]]
    )
    logger.info(
        "Injury snapshot: %d teams, latest report season %d week %d.",
        len(latest), latest["season"].max(), latest["week"].max(),
    )
    return latest[["team", "injury_index", "qb_injured"]]
