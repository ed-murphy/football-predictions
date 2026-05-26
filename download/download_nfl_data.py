import os
import nfl_data_py as nfl

# Ensure data directory exists at repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(repo_root, "data")
os.makedirs(data_dir, exist_ok=True)

SEASONS = [y for y in range(2014, 2026)]

print("Downloading fresh historical game-level data using nfl_data_py...")
games = nfl.import_schedules(SEASONS)
games.to_parquet(os.path.join(data_dir, "games.parquet"), index=False)

print("Downloading fresh historical play-level data using nfl_data_py...")
plays = nfl.import_pbp_data(SEASONS)
plays.to_parquet(os.path.join(data_dir, "plays.parquet"), index=False)

print("Downloading fresh historical injury report data using nfl_data_py...")
injuries = nfl.import_injuries(SEASONS)
injuries.to_parquet(os.path.join(data_dir, "injuries.parquet"), index=False)

print("Download complete. Files saved in:", data_dir)
