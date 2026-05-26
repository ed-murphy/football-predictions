import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Normalize name variants for the same official
_NAME_CANON = {
    "Ronald Torbert": "Ron Torbert",
}


def _canon(name: str) -> str:
    return _NAME_CANON.get(name, name)


def create_referee_features(
    team_games: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Compute ref_avg_total for each game: the referee's expanding-window average
    total points in all their prior regular-season games (no data leakage).

    Returns
    -------
    team_games  : DataFrame with new `ref_avg_total` column
    ref_stats   : Series {referee → career avg total} for upcoming-game lookup
    global_mean : float fallback for unknown/unannounced referees
    """
    ref_df = (
        games[games["game_type"] == "REG"][["game_id", "gameday", "referee"]]
        .copy()
        .dropna(subset=["referee"])
    )
    ref_df["referee"] = ref_df["referee"].map(_canon)

    # Pull total_points from the home row of team_games (one row per game)
    totals = team_games.loc[team_games["is_home"] == 1, ["game_id", "total_points"]].copy()
    ref_df = ref_df.merge(totals, on="game_id", how="inner")
    ref_df = ref_df.sort_values("gameday").reset_index(drop=True)

    # Expanding per-referee mean, shifted by 1 — prior games only, no leakage
    ref_df["ref_avg_total"] = (
        ref_df.groupby("referee")["total_points"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    global_mean = float(ref_df["total_points"].mean())
    ref_df["ref_avg_total"] = ref_df["ref_avg_total"].fillna(global_mean)

    # Career average per referee (for upcoming-game lookup)
    ref_stats = ref_df.groupby("referee")["total_points"].mean()

    # Merge onto team_games (both home and away rows share the same game_id)
    team_games = team_games.merge(
        ref_df[["game_id", "ref_avg_total"]], on="game_id", how="left"
    )
    team_games["ref_avg_total"] = team_games["ref_avg_total"].fillna(global_mean)

    logger.info("Referee features created: ref_avg_total.")
    return team_games, ref_stats, global_mean


def get_upcoming_referee_feature(
    upcoming_games: pd.DataFrame,
    games: pd.DataFrame,
    ref_stats: pd.Series,
    global_mean: float,
) -> pd.Series:
    """
    Look up ref_avg_total for upcoming games by matching (home_team, away_team)
    in the latest season of games.parquet.

    Returns a Series aligned to upcoming_games.index.
    """
    latest_season = games["season"].max()
    sched = (
        games[games["season"] == latest_season][["home_team", "away_team", "referee"]]
        .copy()
    )
    sched["referee"] = sched["referee"].apply(
        lambda x: _canon(x) if pd.notna(x) else x
    )

    merged = upcoming_games[["home_team", "away_team"]].merge(
        sched, on=["home_team", "away_team"], how="left"
    )
    result = merged["referee"].map(ref_stats).fillna(global_mean)
    result.index = upcoming_games.index
    return result
