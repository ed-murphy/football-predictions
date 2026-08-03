"""Quarterback form features.

Rolling QB EPA is keyed on the *player*, not the team, so that a mid-season change
of starter immediately changes the feature rather than carrying the previous
starter's form forward.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.rolling import broadcast_home_away, lagged_rolling_mean
from config import ROLLING_WINDOW_QB

logger = logging.getLogger(__name__)


def _qb_game_epa(plays: pd.DataFrame) -> pd.DataFrame:
    """Mean EPA per (game, team, quarterback) across dropbacks and QB runs."""
    qb_names = set(plays["passer_player_name"].dropna())

    qb_plays = plays[
        plays["passer_player_name"].notna()
        | (plays["qb_dropback"] == 1)
        | (plays["rusher_player_name"].isin(qb_names))
    ].copy()
    qb_plays["qb_name"] = qb_plays["passer_player_name"].fillna(qb_plays["rusher_player_name"])
    qb_plays = qb_plays[qb_plays["qb_name"].notna() & qb_plays["posteam"].notna()]

    return (
        qb_plays.groupby(["game_id", "posteam", "qb_name"], observed=True)["epa"]
        .mean().reset_index(name="qb_epa")
        .rename(columns={"posteam": "team"})
    )


def create_qb_features(
    team_games: pd.DataFrame, plays: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add starting-QB rolling EPA to `team_games`.

    Returns the enriched frame plus a `qb_name -> rolling_avg_qb_epa` lookup holding
    each quarterback's most recent value, for scoring upcoming games.
    """
    qb_epa = _qb_game_epa(plays)

    # A team's starter is whoever threw the game's first pass for them.
    starters = (
        plays[plays["passer_player_name"].notna()]
        .sort_values(["game_id", "posteam", "play_id"])
        .groupby(["game_id", "posteam"], observed=True)["passer_player_name"]
        .first().reset_index()
        .rename(columns={"posteam": "team", "passer_player_name": "starting_qb"})
    )

    # Order by kickoff, not game_id: game_id strings don't sort chronologically
    # across seasons once postseason games are included.
    game_dates = team_games[["game_id", "date"]].drop_duplicates("game_id")
    qb_epa = (
        qb_epa.merge(game_dates, on="game_id", how="left")
        .sort_values(["qb_name", "date"])
        .reset_index(drop=True)
    )

    qb_epa["rolling_avg_qb_epa"] = lagged_rolling_mean(
        qb_epa, "qb_name", "qb_epa", ROLLING_WINDOW_QB
    )

    # For an upcoming game a quarterback's most recent outing *is* prior
    # information, so the served value includes it — unlike the training column,
    # which must exclude the game being predicted.
    latest_qb_epa = (
        qb_epa.sort_values("date")
        .groupby("qb_name")
        .tail(ROLLING_WINDOW_QB)
        .groupby("qb_name", as_index=False)["qb_epa"]
        .mean()
        .rename(columns={"qb_epa": "rolling_avg_qb_epa"})
    )

    team_games = team_games.merge(starters, on=["game_id", "team"], how="left")
    team_games = team_games.merge(
        qb_epa[["game_id", "team", "qb_name", "rolling_avg_qb_epa"]],
        left_on=["game_id", "team", "starting_qb"],
        right_on=["game_id", "team", "qb_name"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["qb_name"])

    team_games = broadcast_home_away(team_games, "rolling_avg_qb_epa")
    for side, flag in [("home", 1), ("away", 0)]:
        team_games[f"{side}_starting_qb"] = (
            team_games["starting_qb"].where(team_games["is_home"] == flag)
            .groupby(team_games["game_id"]).transform("first")
        )

    logger.info("QB EPA features created (%d quarterbacks).", qb_epa["qb_name"].nunique())
    return team_games, latest_qb_epa
