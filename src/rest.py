"""Rest and schedule-spot features.

`games.parquet` already ships `home_rest` / `away_rest` (days since each team's
previous game, correct across bye weeks and season boundaries), so these are read
straight off the schedule rather than recomputed from game dates.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.rolling import broadcast_home_away

logger = logging.getLogger(__name__)

SHORT_REST_MAX_DAYS = 6      # Thursday games and the like
BYE_MIN_DAYS = 11            # a full extra week off
BYE_MAX_DAYS = 21


def create_rest_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add short-rest and post-bye flags for both teams.

    Adds `home/away_short_rest`, `home/away_post_bye`, `home/away_rest_days`
    and `both_short_rest`.
    """
    team_games = team_games.copy()
    rest = team_games["rest_days"]

    team_games["short_rest"] = (rest <= SHORT_REST_MAX_DAYS).astype(int)
    team_games["post_bye"] = rest.between(BYE_MIN_DAYS, BYE_MAX_DAYS).astype(int)

    team_games = broadcast_home_away(team_games, ["short_rest", "post_bye", "rest_days"])
    team_games["both_short_rest"] = (
        (team_games["home_short_rest"] == 1) & (team_games["away_short_rest"] == 1)
    ).astype(int)

    logger.info("Rest features created: short rest, post-bye, rest days.")
    return team_games
