"""Download the nflverse extracts the model needs.

Runs in its own virtualenv: `nfl_data_py` pins older pandas/numpy than the rest of
the project, so mixing them breaks both.

    python download/download_nfl_data.py            # 2014 to the current season
    python download/download_nfl_data.py --full-pbp # every play-by-play column
    python download/download_nfl_data.py --from 2010
"""
import argparse
import os
from datetime import date

import nfl_data_py as nfl

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

DEFAULT_START_SEASON = 2014

# Only the play-by-play columns the feature modules read. The full extract is
# ~380 columns and 250 MB; this is roughly 20 and a fraction of the size, which
# makes every subsequent run of main.py noticeably faster.
#
# If you add a play-derived feature in src/play_features.py, add its inputs here
# too — the feature will otherwise be skipped with a warning at build time.
PBP_COLUMNS = [
    "game_id", "play_id", "posteam", "defteam", "play_type",
    "epa", "success", "yards_gained",
    "pass_attempt", "rush_attempt", "qb_dropback", "sack", "cpoe",
    "passer_player_name", "rusher_player_name",
    "yardline_100", "drive", "touchdown",
    "interception", "fumble_lost",
    "third_down_converted", "third_down_failed",
]


def current_season() -> int:
    """The NFL season currently in progress. A season is named for the year it starts."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=int, default=DEFAULT_START_SEASON,
                        help=f"first season to download (default {DEFAULT_START_SEASON})")
    parser.add_argument("--to", dest="end", type=int, default=None,
                        help="last season to download (default: current season)")
    parser.add_argument("--full-pbp", action="store_true",
                        help="download every play-by-play column instead of the subset used")
    args = parser.parse_args()

    end = args.end or current_season()
    seasons = list(range(args.start, end + 1))
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading seasons {seasons[0]}-{seasons[-1]} into {DATA_DIR}")

    print("  schedules...")
    nfl.import_schedules(seasons).to_parquet(
        os.path.join(DATA_DIR, "games.parquet"), index=False
    )

    print("  play-by-play...", "(all columns)" if args.full_pbp else "(model columns only)")
    plays = (
        nfl.import_pbp_data(seasons)
        if args.full_pbp
        else nfl.import_pbp_data(seasons, columns=PBP_COLUMNS)
    )
    plays.to_parquet(os.path.join(DATA_DIR, "plays.parquet"), index=False)

    print("  injury reports...")
    nfl.import_injuries(seasons).to_parquet(
        os.path.join(DATA_DIR, "injuries.parquet"), index=False
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
