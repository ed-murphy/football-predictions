import logging

import numpy as np
import pandas as pd

from src.constants import INDOOR_ROOFS
from src.rolling import broadcast_home_away, lagged_rolling_mean
from src.venues import lookup_venue
from config import ROLLING_WINDOW_POINTS

logger = logging.getLogger(__name__)

# Columns carried straight through from the game row onto both team rows.
_GAME_LEVEL = [
    "game_id", "week", "season", "total_line", "total_points", "gameday",
    "div_game", "regular_season", "is_dome", "international", "neutral_site",
    "spread_line", "game_temp", "game_wind",
]


# No domestic NFL game kicks off before this hour Eastern; a neutral-site game
# that does is being played in Europe for a local afternoon crowd.
_EARLIEST_DOMESTIC_KICKOFF_ET = 11


def _kickoff_hour_et(games: pd.DataFrame) -> pd.Series:
    """Kickoff hour in US Eastern, from whichever column the caller has."""
    if "gametime" in games.columns and games["gametime"].notna().any():
        return pd.to_numeric(
            games["gametime"].astype(str).str.slice(0, 2), errors="coerce"
        )
    if "date" in games.columns:
        return pd.to_datetime(games["date"], errors="coerce").dt.hour
    return pd.Series(np.nan, index=games.index)


def add_venue_features(games: pd.DataFrame) -> pd.DataFrame:
    """Derive `neutral_site`, `international` and `is_dome` from where the game is played.

    nflverse is inconsistent about this and neither venue column can be trusted alone:

    * 2014-2024 `stadium` and `stadium_id` name the real venue.
    * 2025 both name the *home team's* stadium — the London games are listed at
      FirstEnergy Stadium and MetLife.
    * 2026 `stadium` names the real venue but `stadium_id` reverts to the home
      team's code, and `roof` calls the open-air Melbourne Cricket Ground a dome.

    So two independent signals are combined: a lookup of the stadium name against
    the known international venues, and the fact that a neutral-site game kicking
    off before 11am Eastern can only be abroad. The lookup additionally supplies
    the real roof state and coordinates; when only the kickoff-time signal fires,
    the game is known to be international but its venue is not identified, and
    weather falls back to seasonal normals.
    """
    games = games.copy()

    games["neutral_site"] = (
        games["location"].fillna("Home").ne("Home").astype(int)
        if "location" in games.columns else 0
    )

    venues = games.get("stadium", pd.Series(index=games.index, dtype=object)).map(
        lookup_venue
    )
    identified = venues.notna()
    early_kickoff = (
        (games["neutral_site"] == 1)
        & (_kickoff_hour_et(games) < _EARLIEST_DOMESTIC_KICKOFF_ET)
    )
    games["international"] = (identified | early_kickoff).astype(int)

    games["is_dome"] = games["roof"].str.lower().isin(INDOOR_ROOFS).astype(int)
    if identified.any():
        games.loc[identified, "is_dome"] = [
            int(v.indoor) for v in venues[identified]
        ]

    # Where we know the game is abroad but not where, `roof` describes the home
    # team's stadium and is worse than useless — it would call the open-air Berlin
    # Olympiastadion a dome because Lucas Oil has a roof. Nearly every venue the
    # NFL plays at overseas is open-air, so assume that and let the weather fall
    # through to seasonal normals.
    unidentified = early_kickoff & ~identified
    games.loc[unidentified, "is_dome"] = 0

    if identified.any():
        logger.info("International venues identified: %s",
                    sorted(set(venues[identified].map(lambda v: v.city))))
    if unidentified.any():
        logger.info(
            "%d international game(s) detected by kickoff time but with no venue "
            "named in the feed; treated as open-air with seasonal-normal weather.",
            int(unidentified.sum()),
        )

    missed = (games["neutral_site"] == 1) & (games["international"] == 0)
    if missed.any() and "stadium" in games.columns:
        logger.debug(
            "Neutral-site games treated as domestic: %s",
            sorted(set(games.loc[missed, "stadium"].dropna())),
        )
    return games


def create_basic_features(games: pd.DataFrame) -> pd.DataFrame:
    """Explode `games` into one row per team per game and add scoring-form features.

    Also derives the game-context columns (dome, international, market spread and
    game-time weather) that later modules and the model matrix depend on.
    """
    games = games.copy()

    games = add_venue_features(games)
    games["total_points"] = games["home_score"] + games["away_score"]
    games["regular_season"] = (games["game_type"] == "REG").astype(int)

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
