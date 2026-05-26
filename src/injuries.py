import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Fractional availability penalty by report status
STATUS_WEIGHTS = {
    "Out":          1.00,
    "Doubtful":     0.75,
    "Questionable": 0.25,
    "Probable":     0.00,
}

# Importance weight per position for totals scoring
POSITION_WEIGHTS = {
    "QB":  3.0,
    "WR":  0.8,
    "RB":  0.5,
    "TE":  0.5,
    "OL":  0.3,
    "OT":  0.3,
    "OG":  0.3,
    "C":   0.3,
}


def _build_team_injury(injuries: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team-week injury index and QB injury flag from raw injury data."""
    inj = injuries[injuries["game_type"] == "REG"].copy()
    inj["status_w"] = inj["report_status"].map(STATUS_WEIGHTS).fillna(0)
    inj["pos_w"] = inj["position"].map(POSITION_WEIGHTS).fillna(0)
    inj["score"] = inj["status_w"] * inj["pos_w"]

    team_inj = (
        inj.groupby(["season", "week", "team"])["score"]
        .sum()
        .reset_index()
        .rename(columns={"score": "injury_index"})
    )

    qb_out = inj[
        (inj["position"] == "QB") &
        (inj["report_status"].isin(["Out", "Doubtful"]))
    ]
    qb_flag = (
        qb_out.groupby(["season", "week", "team"])
        .size()
        .reset_index(name="qb_injury_flag")
    )
    qb_flag["qb_injury_flag"] = 1

    return team_inj.merge(qb_flag, on=["season", "week", "team"], how="left").assign(
        qb_injury_flag=lambda df: df["qb_injury_flag"].fillna(0).astype(int)
    )


def create_injury_features(team_games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weekly injury report features into team_games.

    Adds:
      home/away_injury_index   - weighted sum of skill-position absences (Out/Doubtful/Q)
      home/away_qb_injured     - 1 if starting QB is Out or Doubtful
    """
    team_inj = _build_team_injury(injuries)

    team_games = team_games.merge(team_inj, on=["season", "week", "team"], how="left")
    team_games["injury_index"] = team_games["injury_index"].fillna(0)
    team_games["qb_injury_flag"] = team_games["qb_injury_flag"].fillna(0).astype(int)

    home_inj = (
        team_games[team_games["is_home"] == 1][["game_id", "injury_index", "qb_injury_flag"]]
        .rename(columns={"injury_index": "home_injury_index", "qb_injury_flag": "home_qb_injured"})
    )
    away_inj = (
        team_games[team_games["is_home"] == 0][["game_id", "injury_index", "qb_injury_flag"]]
        .rename(columns={"injury_index": "away_injury_index", "qb_injury_flag": "away_qb_injured"})
    )

    team_games = (
        team_games
        .merge(home_inj, on="game_id", how="left")
        .merge(away_inj, on="game_id", how="left")
    )

    logger.info("Injury features created: home/away injury_index, home/away qb_injured.")
    return team_games


def get_upcoming_injury_features(
    upcoming_games: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the most recent available injury data to upcoming games.

    Looks for the latest season/week in the injury data that overlaps with
    the upcoming games' season/week. Falls back gracefully if no data found.
    """
    if injuries is None or injuries.empty:
        upcoming_games["home_injury_index"] = 0.0
        upcoming_games["away_injury_index"] = 0.0
        upcoming_games["home_qb_injured"] = 0
        upcoming_games["away_qb_injured"] = 0
        return upcoming_games

    team_inj = _build_team_injury(injuries)

    # Determine current season/week from upcoming games (use first game as reference)
    # upcoming_games has 'home_team' (abbrev) and 'commence_time'/'date'
    from datetime import datetime
    now = datetime.now()
    current_season = now.year if now.month >= 9 else now.year - 1
    current_week = team_inj[team_inj["season"] == current_season]["week"].max() if current_season in team_inj["season"].values else None

    if current_week is None or pd.isna(current_week):
        logger.warning("No injury data available for season %d. Injury features will be 0.", current_season)
        upcoming_games["home_injury_index"] = 0.0
        upcoming_games["away_injury_index"] = 0.0
        upcoming_games["home_qb_injured"] = 0
        upcoming_games["away_qb_injured"] = 0
        return upcoming_games

    week_inj = team_inj[
        (team_inj["season"] == current_season) & (team_inj["week"] == current_week)
    ][["team", "injury_index", "qb_injury_flag"]]

    upcoming_games = upcoming_games.merge(
        week_inj.rename(columns={"injury_index": "home_injury_index", "qb_injury_flag": "home_qb_injured"}),
        left_on="home_team", right_on="team", how="left"
    ).drop(columns=["team"], errors="ignore")

    upcoming_games = upcoming_games.merge(
        week_inj.rename(columns={"injury_index": "away_injury_index", "qb_injury_flag": "away_qb_injured"}),
        left_on="away_team", right_on="team", how="left"
    ).drop(columns=["team"], errors="ignore")

    for col in ["home_injury_index", "away_injury_index"]:
        upcoming_games[col] = upcoming_games[col].fillna(0.0)
    for col in ["home_qb_injured", "away_qb_injured"]:
        upcoming_games[col] = upcoming_games[col].fillna(0).astype(int)

    logger.info(
        "Upcoming game injury features populated from season %d week %d.",
        current_season, int(current_week),
    )
    return upcoming_games
