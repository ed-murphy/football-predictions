import os
import nfl_data_py as nfl
import pandas as pd

# Ensure data directory exists at repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(repo_root, "data")
os.makedirs(data_dir, exist_ok=True)

print("Downloading fresh historical game-level data using nfl_data_py...")
games = nfl.import_schedules([y for y in range(2021, 2025)])
games.to_parquet(os.path.join(data_dir, "games.parquet"), index=False)

print("Downloading fresh historical play-level data using nfl_data_py...")
plays = nfl.import_pbp_data([y for y in range(2021, 2025)])
plays.to_parquet(os.path.join(data_dir, "plays.parquet"), index=False)

print("Download complete. Files saved in:", data_dir)
