"""Does team form from play-by-play data tell you anything the closing line doesn't?

    python -m research.play_by_play_scan          (needs `pip install pyarrow`)

This is the test that retired the previous version of this project, which built
around eighty features from play-by-play data — EPA per play, success rate, pace,
pass rate, explosive plays, sacks, turnovers — and never found an edge.

Each statistic is averaged over a team's previous 5 and previous 12 games, then
combined across the two teams (summed for totals, differenced for spreads) and
put through the split-sample rule in research/common.py.

Result (November 2026): 68 candidates, 136 z-scores, and they are distributed
exactly as pure noise would be. The only ones that pass are wind multiplied by
something. The closing line already contains everything these numbers know.

Downloads about 100 MB of play-by-play the first time; the per-game summary is
cached in data/research/.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

from research.common import cache_path, load_results, noise_check, split_sample_scan

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
OFFENCE_STATS = ["plays", "epa", "success", "pass_rate", "explosive", "turnovers", "no_huddle",
                 "shotgun", "sack", "pass_epa", "rush_epa", "neutral_epa", "neutral_pass_rate"]
DEFENCE_STATS = ["def_epa", "def_success", "def_pass_epa"]


def summarise_season(season: int) -> pd.DataFrame:
    """One row per (game, team): what the offence did and what the defence allowed."""
    raw = pd.read_parquet(io.BytesIO(requests.get(PBP_URL.format(season=season), timeout=300).content))
    plays = raw[raw["posteam"].notna() & ((raw["pass"] == 1) | (raw["rush"] == 1))
                & raw["epa"].notna()].copy()
    plays["explosive"] = (plays["yards_gained"] >= 15).astype(float)
    plays["turnover"] = plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0)
    # "Neutral" plays strip out garbage time and the two-minute drill.
    neutral = plays[plays["wp"].between(0.2, 0.8) & (plays["qtr"] <= 3)
                    & (plays["half_seconds_remaining"] > 120)]
    passes, rushes = plays[plays["pass"] == 1], plays[plays["rush"] == 1]

    offence = plays.groupby(["game_id", "posteam"]).agg(
        plays=("epa", "size"), epa=("epa", "mean"), success=("success", "mean"),
        pass_rate=("pass", "mean"), explosive=("explosive", "mean"),
        turnovers=("turnover", "sum"), no_huddle=("no_huddle", "mean"),
        shotgun=("shotgun", "mean"), sack=("sack", "mean"))
    offence["pass_epa"] = passes.groupby(["game_id", "posteam"])["epa"].mean()
    offence["rush_epa"] = rushes.groupby(["game_id", "posteam"])["epa"].mean()
    offence["neutral_epa"] = neutral.groupby(["game_id", "posteam"])["epa"].mean()
    offence["neutral_pass_rate"] = neutral.groupby(["game_id", "posteam"])["pass"].mean()

    defence = plays.groupby(["game_id", "defteam"]).agg(
        def_epa=("epa", "mean"), def_success=("success", "mean"))
    defence["def_pass_epa"] = passes.groupby(["game_id", "defteam"])["epa"].mean()

    offence.index.names = defence.index.names = ["game_id", "team"]
    return offence.join(defence, how="outer").reset_index()


def team_game_stats(seasons: range, path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    frames = []
    for season in seasons:
        print(f"  summarising {season} play-by-play...", flush=True)
        frames.append(summarise_season(season))
    stats = pd.concat(frames, ignore_index=True)
    stats.to_csv(path, index=False)
    return stats


def rolling_form(games: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each team's average over its previous 5 and 12 games, as home and away frames."""
    sides = pd.concat([
        games[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "team"}).assign(side="home"),
        games[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "team"}).assign(side="away"),
    ]).merge(stats, on=["game_id", "team"], how="left").sort_values(["team", "gameday"])

    columns = []
    for window in (5, 12):
        for stat in OFFENCE_STATS + DEFENCE_STATS:
            name = f"{stat}_last{window}"
            sides[name] = sides.groupby("team")[stat].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean())
            columns.append(name)

    home = sides[sides["side"] == "home"].set_index("game_id")[columns]
    away = sides[sides["side"] == "away"].set_index("game_id")[columns]
    return home, away


def main() -> None:
    pd.set_option("display.width", 200, "display.max_rows", 200)
    games = load_results()
    stats = team_game_stats(range(1999, int(games["season"].max()) + 1),
                            cache_path("team_game_stats.csv"))
    home, away = rolling_form(games, stats)

    games = games[games["total_line"].notna() & games["spread_line"].notna()
                  & (games["season"] >= 2000)].set_index("game_id")
    home, away = home.loc[games.index], away.loc[games.index]

    totals, spreads = {}, {}
    for window in (5, 12):
        for stat in OFFENCE_STATS + DEFENCE_STATS:
            column = f"{stat}_last{window}"
            totals[f"combined {column}"] = home[column] + away[column]
            spreads[f"home minus away {column}"] = home[column] - away[column]
        pass_rate = home[f"pass_rate_last{window}"] + away[f"pass_rate_last{window}"]
        totals[f"wind x combined pass rate (last{window})"] = games["wind_outdoors"] * (pass_rate - 1.15)
        totals[f"matchup epa (last{window})"] = (
            home[f"epa_last{window}"] + away[f"def_epa_last{window}"]
            + away[f"epa_last{window}"] + home[f"def_epa_last{window}"])

    for title, candidates, resid in (
        ("TOTALS  (positive z = predicts the over)", totals, games["total_resid"]),
        ("SPREADS (positive z = predicts the home team covering)", spreads, games["spread_resid"]),
    ):
        table = split_sample_scan(candidates, resid, games["season"])
        print(f"\n{title}\n{table.to_string(index=False)}\n{noise_check(table)}")


if __name__ == "__main__":
    main()
