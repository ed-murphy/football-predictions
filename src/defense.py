import logging
import pandas as pd
from config import ROLLING_WINDOW_DEF

logger = logging.getLogger(__name__)


def create_defense_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Create defensive EPA and sack-rate features.

    Adds:
      home/away_rolling_avg_def_epa      - rolling mean EPA allowed per play
      home/away_rolling_avg_sack_rate    - rolling sacks per opponent dropback
    """
    # Defensive EPA per game per team
    def_epa = (
        plays
        .groupby(["game_id", "defteam"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"defteam": "team", "epa": "def_epa"})
    )

    # Sacks earned per opponent dropback (defensive quality)
    dropbacks = (
        plays[plays["qb_dropback"] == 1]
        .groupby(["game_id", "defteam"])
        .size()
        .reset_index(name="opp_dropbacks")
        .rename(columns={"defteam": "team"})
    )
    sacks = (
        plays[plays["sack"] == 1]
        .groupby(["game_id", "defteam"])
        .size()
        .reset_index(name="sacks")
        .rename(columns={"defteam": "team"})
    )
    sack_rate = dropbacks.merge(sacks, on=["game_id", "team"], how="left")
    sack_rate["sacks"] = sack_rate["sacks"].fillna(0)
    sack_rate["sack_rate"] = sack_rate["sacks"] / sack_rate["opp_dropbacks"].replace(0, float("nan"))

    team_games = (
        team_games
        .merge(def_epa, on=["game_id", "team"], how="left")
        .merge(sack_rate[["game_id", "team", "sack_rate"]], on=["game_id", "team"], how="left")
    )

    team_games = team_games.sort_values(["team", "season", "week"])

    for col, roll_col in [("def_epa", "rolling_avg_def_epa"), ("sack_rate", "rolling_avg_sack_rate")]:
        team_games[roll_col] = (
            team_games
            .groupby(["team", "season"])[col]
            .apply(lambda x: x.shift().rolling(ROLLING_WINDOW_DEF, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    home_def = (
        team_games[team_games["is_home"] == 1][["game_id", "rolling_avg_def_epa", "rolling_avg_sack_rate"]]
        .rename(columns={
            "rolling_avg_def_epa": "home_rolling_avg_def_epa",
            "rolling_avg_sack_rate": "home_rolling_avg_sack_rate",
        })
    )
    away_def = (
        team_games[team_games["is_home"] == 0][["game_id", "rolling_avg_def_epa", "rolling_avg_sack_rate"]]
        .rename(columns={
            "rolling_avg_def_epa": "away_rolling_avg_def_epa",
            "rolling_avg_sack_rate": "away_rolling_avg_sack_rate",
        })
    )

    team_games = team_games.merge(home_def, on="game_id", how="left")
    team_games = team_games.merge(away_def, on="game_id", how="left")

    logger.info("Team defense EPA and sack rate features created.")
    return team_games

