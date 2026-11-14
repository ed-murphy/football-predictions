"""How well does a wind forecast predict the wind the gamebook records?

    python -m research.forecast_calibration

Produces the FORECAST_CALIBRATION table in config.py. For every outdoor game
since 2024 it pulls what Open-Meteo was forecasting 1 to 7 days ahead (their
"previous runs" archive, which starts in 2024) and regresses the gamebook wind
on it:

    gamebook_wind = intercept + slope * forecast

Re-run it once a season and paste the new table into config.py. About 450 API
calls the first time (a couple of minutes); results are cached in data/research/.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from research.common import cache_path, get_json, load_results, stadium_coords

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FIRST_SEASON = 2024
LEADS = range(1, 8)


def fetch_archived_forecasts(games: pd.DataFrame, path: str) -> pd.DataFrame:
    """Forecast wind at kickoff, as issued 1-7 days earlier, for each game."""
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["game_id"])
    todo = games[~games["game_id"].isin(done["game_id"])]
    if todo.empty:
        return done

    print(f"Fetching archived forecasts for {len(todo)} games...")
    variables = [f"wind_speed_10m_previous_day{k}" for k in LEADS]
    session, rows = requests.Session(), []
    for game in todo.itertuples():
        lat, lon = stadium_coords(game.home_team, game.season)
        payload = get_json(PREVIOUS_RUNS_URL, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(variables),
            "start_date": game.gameday, "end_date": game.gameday,
            "wind_speed_unit": "mph", "timezone": "America/New_York",
        }, session)
        if not payload or "hourly" not in payload:
            continue
        hour = int(game.kickoff_hour)     # gametime is already US Eastern
        rows.append({"game_id": game.game_id,
                     **{v: payload["hourly"][v][hour] for v in variables}})
        time.sleep(0.15)

    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(path, index=False)
    return done


def calibration_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in LEADS:
        forecast, actual = merged[f"wind_speed_10m_previous_day{k}"], merged["wind"]
        ok = forecast.notna() & actual.notna()
        x, y = forecast[ok], actual[ok]
        slope = np.cov(x, y)[0, 1] / x.var()
        intercept = y.mean() - slope * x.mean()
        rows.append({
            "days_out": k, "games": int(ok.sum()),
            "correlation": round(np.corrcoef(x, y)[0, 1], 2),
            "intercept": round(intercept, 2), "slope": round(slope, 3),
            "error_sd_mph": round((y - intercept - slope * x).std(), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    games = load_results()
    games = games[(games["season"] >= FIRST_SEASON) & (games["outdoors"] == 1)
                  & (games["location"] == "Home") & games["kickoff_hour"].notna()]
    forecasts = fetch_archived_forecasts(games, cache_path("archived_forecasts.csv"))
    merged = games.merge(forecasts, on="game_id")
    merged = merged[merged["wind"].notna()]

    table = calibration_table(merged)
    print(f"\n{len(merged)} outdoor games since {FIRST_SEASON} with a gamebook wind reading "
          f"(mean {merged['wind'].mean():.1f} mph, sd {merged['wind'].std():.1f}).\n")
    print(table.to_string(index=False))
    print("\nFor config.py:\nFORECAST_CALIBRATION = {")
    for row in table.itertuples():
        print(f"    {row.days_out}: ({row.intercept:.2f}, {row.slope:.3f}),   # {row.correlation:.2f}")
    print("}")


if __name__ == "__main__":
    main()
