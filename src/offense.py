import logging
import pandas as pd
from config import ROLLING_WINDOW_OFFENSE

logger = logging.getLogger(__name__)


def create_offense_features(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Create offensive efficiency rolling features per team per game.

    Adds:
      rolling_avg_pass_rate       - fraction of scrimmage plays that are passes
      rolling_avg_explosive_rate  - fraction of plays gaining 15+ yards
      rolling_avg_success_rate    - fraction of plays with positive EPA
      rolling_avg_rush_epa        - mean EPA on rush plays
      rolling_avg_pass_epa        - mean EPA on pass plays
      rolling_avg_cpoe            - mean completion % over expected (pass quality)

    All are shifted to exclude the current game (no leakage).
    """
    # Scrimmage plays only
    scrimmage = plays[(plays["pass_attempt"] == 1) | (plays["rush_attempt"] == 1)].copy()

    game_off = (
        scrimmage.groupby(["game_id", "posteam"])
        .agg(
            n_plays=("play_type", "count"),
            n_passes=("pass_attempt", "sum"),
            n_explosive=("yards_gained", lambda x: (x >= 15).sum()),
            n_success=("success", lambda x: x.fillna(0).sum()),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    game_off["pass_rate"] = game_off["n_passes"] / game_off["n_plays"]
    game_off["explosive_rate"] = game_off["n_explosive"] / game_off["n_plays"]
    game_off["success_rate"] = game_off["n_success"] / game_off["n_plays"]

    rush_epa = (
        plays[plays["rush_attempt"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "epa": "rush_epa"})
    )

    pass_epa = (
        plays[plays["pass_attempt"] == 1]
        .groupby(["game_id", "posteam"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "epa": "pass_epa"})
    )

    cpoe_game = (
        plays[(plays["pass_attempt"] == 1) & plays["cpoe"].notna()]
        .groupby(["game_id", "posteam"])["cpoe"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "cpoe": "cpoe"})
    )

    game_off = (
        game_off
        .merge(rush_epa, on=["game_id", "team"], how="left")
        .merge(pass_epa, on=["game_id", "team"], how="left")
        .merge(cpoe_game, on=["game_id", "team"], how="left")
    )

    team_games = team_games.merge(
        game_off[["game_id", "team", "pass_rate", "explosive_rate",
                  "success_rate", "rush_epa", "pass_epa", "cpoe"]],
        on=["game_id", "team"],
        how="left",
    )

    raw_cols = ["pass_rate", "explosive_rate", "success_rate", "rush_epa", "pass_epa", "cpoe"]
    team_games = team_games.sort_values(["team", "season", "week"])

    for col in raw_cols:
        roll_col = f"rolling_avg_{col}"
        team_games[roll_col] = (
            team_games
            .groupby(["team", "season"])[col]
            .apply(lambda x: x.shift().rolling(ROLLING_WINDOW_OFFENSE, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # Home / away split
    home_cols = ["game_id"] + [f"rolling_avg_{c}" for c in raw_cols]
    away_cols = ["game_id"] + [f"rolling_avg_{c}" for c in raw_cols]

    home_feats = (
        team_games[team_games["is_home"] == 1][home_cols]
        .rename(columns={f"rolling_avg_{c}": f"home_rolling_avg_{c}" for c in raw_cols})
    )
    away_feats = (
        team_games[team_games["is_home"] == 0][away_cols]
        .rename(columns={f"rolling_avg_{c}": f"away_rolling_avg_{c}" for c in raw_cols})
    )

    team_games = team_games.merge(home_feats, on="game_id", how="left")
    team_games = team_games.merge(away_feats, on="game_id", how="left")

    logger.info(
        "Offensive efficiency features created: pass rate, explosive rate, "
        "success rate, rush EPA, pass EPA, CPOE."
    )
    return team_games
