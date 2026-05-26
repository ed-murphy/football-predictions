import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_red_zone(team_games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling red zone efficiency (TDs per red zone trip, last 3 games)
    for both home and away teams.

    Parameters
    ----------
    team_games : pd.DataFrame
        Game-level data with at least ['game_id', 'team', 'is_home', 'date'] columns.
    plays : pd.DataFrame
        Play-by-play data with at least:
        ['game_id', 'posteam', 'drive', 'yardline_100', 'touchdown'].

    Returns
    -------
    pd.DataFrame
        Original team_games DataFrame with two new columns:
        'home_rolling_rZ_eff' and 'away_rolling_rZ_eff'.
    """

    plays = plays.copy()
    plays["drive_id"] = plays["game_id"].astype(str) + "_" + plays["drive"].astype(str)

    # Filter to red zone plays (inside opponent 20)
    red_zone = plays[plays["yardline_100"] <= 20].copy()

    # Identify first entry into red zone per drive
    red_zone["red_zone_entry"] = (
        red_zone.groupby(["game_id", "posteam", "drive_id"])["yardline_100"]
        .transform("min") == red_zone["yardline_100"]
    )

    # Drive-level outcomes
    drive_outcomes = (
        red_zone.groupby(["game_id", "posteam", "drive_id"])
        .agg(
            red_zone_trip=("red_zone_entry", "max"),
            td=("touchdown", "max")
        )
        .reset_index()
    )

    # Keep only drives that actually entered the red zone
    drive_outcomes = drive_outcomes[drive_outcomes["red_zone_trip"]]

    # Game-level efficiency per team
    game_eff = (
        drive_outcomes.groupby(["game_id", "posteam"])
        .agg(
            red_zone_trips=("red_zone_trip", "sum"),
            red_zone_tds=("td", "sum")
        )
        .reset_index()
    )
    game_eff["rz_efficiency"] = game_eff["red_zone_tds"] / game_eff["red_zone_trips"]

    # Merge in game dates
    game_eff = game_eff.merge(
        team_games[["game_id", "team", "date"]].rename(columns={"team": "posteam", "date": "gameday"}),
        on=["game_id", "posteam"],
        how="left"
    )

    # Sort for rolling window
    game_eff = game_eff.sort_values(["posteam", "gameday"]).reset_index(drop=True)

    # Compute lagged rolling efficiency (last 3 games)
    game_eff["rolling_rz_eff"] = (
        game_eff.groupby("posteam")["rz_efficiency"]
        .apply(lambda x: x.shift().rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Merge efficiency back onto team_games (keyed by game_id + team)
    team_games = team_games.merge(
        game_eff[["game_id", "posteam", "rolling_rz_eff"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "posteam"],
        how="left"
    ).drop(columns=["posteam"])

    # Create home/away rolling columns (parallel to QB function)
    home_features = (
        team_games.loc[team_games["is_home"] == 1, ["game_id", "rolling_rz_eff"]]
        .rename(columns={"rolling_rz_eff": "home_rolling_rz_eff"})
    )
    away_features = (
        team_games.loc[team_games["is_home"] == 0, ["game_id", "rolling_rz_eff"]]
        .rename(columns={"rolling_rz_eff": "away_rolling_rz_eff"})
    )

    # Merge home/away rolling values back onto team_games
    team_games = team_games.merge(home_features, on="game_id", how="left")
    team_games = team_games.merge(away_features, on="game_id", how="left")

    logger.info("Red zone rolling efficiency features created: home_rolling_rz_eff, away_rolling_rz_eff")

    return team_games
