import logging

import pandas as pd

from src.constants import INDOOR_ROOFS, INTERNATIONAL_STADIUM_IDS
from src.rolling import broadcast_home_away, lagged_rolling_mean
from config import ROLLING_WINDOW_POINTS

logger = logging.getLogger(__name__)

# Columns carried straight through from the game row onto both team rows.
_GAME_LEVEL = [
    "game_id", "week", "season", "total_line", "total_points", "gameday",
    "div_game", "regular_season", "is_dome", "international", "spread_line",
    "game_temp", "game_wind",
]


def create_basic_features(games: pd.DataFrame) -> pd.DataFrame:
    """Explode `games` into one row per team per game and add scoring-form features.

    Also derives the game-context columns (dome, international, market spread and
    game-time weather) that later modules and the model matrix depend on.
    """
    games = games.copy()

    games["international"] = games["stadium_id"].isin(INTERNATIONAL_STADIUM_IDS).astype(int)
    games["total_points"] = games["home_score"] + games["away_score"]
    games["regular_season"] = (games["game_type"] == "REG").astype(int)
    games["is_dome"] = games["roof"].str.lower().isin(INDOOR_ROOFS).astype(int)

    # nflverse ships kickoff temperature/wind for outdoor games. It is null for
    # indoor games (where the concept doesn't apply) so substitute neutral values;
    # remaining gaps are filled per season in the model matrix.
    games["game_temp"] = games["temp"].where(games["is_dome"] == 0)
    games["game_wind"] = games["wind"].where(games["is_dome"] == 0)
    games.loc[games["is_dome"] == 1, ["game_temp", "game_wind"]] = [70.0, 0.0]

    home = games[_GAME_LEVEL + ["home_team", "home_score", "away_score", "home_rest"]].copy()
    home.columns = _GAME_LEVEL + ["team", "points_for", "points_against", "rest_days"]
    home["is_home"] = 1

    away = games[_GAME_LEVEL + ["away_team", "away_score", "home_score", "away_rest"]].copy()
    away.columns = _GAME_LEVEL + ["team", "points_for", "points_against", "rest_days"]
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.rename(columns={"gameday": "date", "div_game": "divisional"})
    team_games["date"] = pd.to_datetime(team_games["date"])
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

    # How many games this team has already played this season. Rolling features are
    # far noisier in weeks 1-3; giving the model the count lets it discount them.
    team_games["season_game_num"] = team_games.groupby(["team", "season"]).cumcount() + 1

    for src_col, out_col in [
        ("points_for", "rolling_avg_points_for"),
        ("points_against", "rolling_avg_points_against"),
    ]:
        team_games[out_col] = lagged_rolling_mean(
            team_games, "team", src_col, ROLLING_WINDOW_POINTS
        )

    team_games = broadcast_home_away(
        team_games, ["rolling_avg_points_for", "rolling_avg_points_against"]
    )

    logger.info("Basic football features created (%d team-games).", len(team_games))
    return team_games
