import os
import nfl_data_py as nfl
import pandas as pd

os.makedirs("data", exist_ok=True)

print("Downloading fresh historical game-level data using nfl_data_py...")
games = nfl.import_schedules([y for y in range(2019, 2025)])
games.to_csv("data/games.csv", index=False)

print("Downloading fresh historical play-level data using nfl_data_py...")
plays = nfl.import_pbp_data([y for y in range(2019, 2025)])
plays.to_csv("data/plays.csv", index=False)

print("Download complete.")
