"""Does the wind effect survive a completely different measurement of the wind?

    python -m research.reanalysis_check

The model is fitted on the wind written in the official gamebook. If that field
were quietly informed by how the game went, the "edge" would be an artefact. So
this script throws the gamebook away and uses ERA5 reanalysis instead: a weather
model's reconstruction of the wind at the stadium's coordinates, produced with no
knowledge that a football game was on.

It also pulls rainfall, which the gamebook does not record, and tests that too.

Result (November 2026): the effect is at least as strong with reanalysis wind
(z = 4.7 against 4.1 for the gamebook on the same games). Rain predicts unders
as well and passes the split-sample rule, but adds nothing out of sample once
wind is in the model, so it is not used.

About 650 API calls the first time (10-15 minutes); cached in data/research/.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from research.common import (
    DISCOVERY_END, cache_path, fit_logistic, get_json, load_results, split_sample_scan,
    stadium_coords, under_record,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["wind_speed_10m", "wind_gusts_10m", "precipitation"]
GAME_HOURS = 4


def fetch_reanalysis(games: pd.DataFrame, path: str) -> pd.DataFrame:
    """Average wind and total rain over each game, one request per (team, season)."""
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["game_id"])
    todo = games[~games["game_id"].isin(done["game_id"])]
    if todo.empty:
        return done

    print(f"Fetching reanalysis weather for {len(todo)} games...")
    session = requests.Session()
    for (team, season), group in todo.groupby(["home_team", "season"]):
        lat, lon = stadium_coords(team, season)
        payload = get_json(ARCHIVE_URL, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(VARIABLES),
            "start_date": group["gameday"].min(), "end_date": group["gameday"].max(),
            "wind_speed_unit": "mph", "timezone": "America/New_York",
        }, session)
        if not payload or "hourly" not in payload:
            continue
        hour_index = {t: i for i, t in enumerate(payload["hourly"]["time"])}
        rows = []
        for game in group.itertuples():
            start = hour_index.get(f"{game.gameday}T{int(game.kickoff_hour):02d}:00")
            if start is None:
                continue
            window = {v: [x for x in payload["hourly"][v][start:start + GAME_HOURS]
                          if x is not None] for v in VARIABLES}
            if not window["wind_speed_10m"]:
                continue      # the last few days are not in the archive yet
            rows.append({
                "game_id": game.game_id,
                "reanalysis_wind": np.mean(window["wind_speed_10m"]),
                "reanalysis_gust": np.mean(window["wind_gusts_10m"] or [np.nan]),
                "rain_mm": np.sum(window["precipitation"]),
            })
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(path, index=False)
        time.sleep(0.4)
    return done


def walk_forward(data: pd.DataFrame, columns: list[str], start: int = 2008) -> str:
    """Out-of-sample record of a logistic model on `columns`, unders at EV >= 2%."""
    data = data.dropna(subset=columns)
    scored = []
    for season in range(start, int(data["season"].max()) + 1):
        train, test = data[data["season"] < season], data[data["season"] == season]
        if test.empty:
            continue
        design = lambda d: np.column_stack([np.ones(len(d))] + [d[c] for c in columns])
        beta, _ = fit_logistic(design(train), train["under"].to_numpy())
        p = 1 / (1 + np.exp(-design(test) @ beta))
        scored.append(pd.DataFrame({"p": p, "under": test["under"].to_numpy()}))
    scored = pd.concat(scored)

    log_loss = -np.mean(scored["under"] * np.log(scored["p"])
                        + (1 - scored["under"]) * np.log(1 - scored["p"]))
    bets = scored[scored["p"] * (100 / 110) - (1 - scored["p"]) >= 0.02]
    profit = np.where(bets["under"] == 1, 100 / 110, -1.0)
    return (f"log loss vs coin flip {log_loss - np.log(2):+.5f} | {len(bets)} bets, "
            f"{bets['under'].mean():.1%}, ROI {profit.mean():+.1%} "
            f"+/- {profit.std(ddof=1) / np.sqrt(len(bets)):.1%}")


def main() -> None:
    pd.set_option("display.width", 200)
    games = load_results()
    games = games[games["total_line"].notna()]
    outdoor = games[(games["outdoors"] == 1) & (games["location"] == "Home")
                    & games["kickoff_hour"].notna()]
    weather = fetch_reanalysis(outdoor, cache_path("reanalysis_weather.csv"))
    data = outdoor.merge(weather, on="game_id")

    both = data[data["wind"].notna()]
    print(f"\n{len(data)} outdoor games with reanalysis weather. Correlation between gamebook "
          f"and reanalysis wind: {np.corrcoef(both['wind'], both['reanalysis_wind'])[0, 1]:.2f}.")

    print("\nSplit-sample test (positive z = predicts the over):")
    data = data.assign(wet=(data["rain_mm"] >= 1).astype(float))
    print(split_sample_scan({
        "gamebook wind": data["wind"],
        "reanalysis wind": data["reanalysis_wind"],
        "reanalysis gusts": data["reanalysis_gust"],
        "rain during game (mm)": data["rain_mm"],
        "wet game (1 mm or more)": data["wet"],
    }, data["total_resid"], data["season"]).to_string(index=False))

    print("\nBlind unders by reanalysis wind:")
    for low, high in ((0, 6), (6, 10), (10, 13), (13, 16), (16, 99)):
        band = data[(data["reanalysis_wind"] >= low) & (data["reanalysis_wind"] < high)]
        print(f"  {low:2d}-{high:<2d} mph   all: {under_record(band)}   |   "
              f"since {DISCOVERY_END + 1}: {under_record(band[band['season'] > DISCOVERY_END])}")

    print("\nBlind unders by rain during the game:")
    for label, mask in (("dry (<0.2 mm)", data["rain_mm"] < 0.2), ("1 mm or more", data["rain_mm"] >= 1),
                        ("3 mm or more", data["rain_mm"] >= 3)):
        print(f"  {label:14s} {under_record(data[mask])}")

    # Walk-forward on every game: roofed games count as still and dry.
    indoor = games[games["roof"].isin(["dome", "closed"])].assign(
        wind=0.0, reanalysis_wind=0.0, rain_mm=0.0, wet=0.0)
    everything = pd.concat([data, indoor])
    everything = everything[everything["total_resid"] != 0].assign(
        under=lambda d: (d["total_resid"] < 0).astype(float))
    print("\nWalk-forward from 2008, unders at -110 when expected value >= 2%:")
    for label, columns in (("gamebook wind", ["wind"]), ("reanalysis wind", ["reanalysis_wind"]),
                           ("reanalysis wind + wet", ["reanalysis_wind", "wet"]),
                           ("wet only", ["wet"])):
        print(f"  {label:22s} {walk_forward(everything, columns)}")


if __name__ == "__main__":
    main()
