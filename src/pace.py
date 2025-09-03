import pandas as pd

def create_pace_features(team_games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    # --- build pace dataset ---
    plays = pbp[pbp["play_type"].notna()].copy()

    plays_per_game = (
        plays.groupby(["game_id", "posteam"])
        .size()
        .reset_index(name="plays")
    )

    plays_per_game["seconds_per_play"] = 3600 / plays_per_game["plays"]

    plays_per_game = plays_per_game.sort_values(["posteam", "game_id"])

    plays_per_game["pace_last5"] = (
        plays_per_game.groupby("posteam")["seconds_per_play"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # --- merge directly onto team_games ---
    team_games = team_games.merge(
        plays_per_game[["game_id", "posteam", "seconds_per_play", "pace_last5"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "posteam"],
        how="left"
    )

    team_games = team_games.drop(columns=["posteam"])

    # --- create home/away pace columns ---
    team_games["home_pace_last5"] = team_games.loc[team_games["is_home"] == 1, "pace_last5"]
    team_games["away_pace_last5"] = team_games.loc[team_games["is_home"] == 0, "pace_last5"]

    # forward/backfill within each game so both rows have the home/away pace values
    team_games["home_pace_last5"] = team_games.groupby("game_id")["home_pace_last5"].transform("max")
    team_games["away_pace_last5"] = team_games.groupby("game_id")["away_pace_last5"].transform("max")

    print("Team pace features created.")

    return team_games
