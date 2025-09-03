import os
import pandas as pd

import os
import pandas as pd

def load_data():
    """Load cached NFL game-level and play-by-play data from CSV files in the data directory."""
    
    games_path = "data/games.parquet"
    plays_path = "data/plays.parquet"

    if not os.path.exists(games_path) or not os.path.exists(plays_path):
        raise FileNotFoundError(
            "Data files not found. Please run the data download script first to generate games.csv and plays.csv in the data directory."
        )

    print(f"Reading cached historical game-level data from {games_path}...")
    games = pd.read_parquet(games_path)

    print(f"Reading cached historical play-level data from {plays_path}...")
    plays = pd.read_parquet(plays_path)

    return games, plays
