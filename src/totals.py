import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pytz import timezone as tz


DATA_PATH = "data/nfl_over_unders.csv"

load_dotenv()
API_KEY = os.getenv("API_KEY_TOTALS")

if not API_KEY:
    raise ValueError("Missing API_KEY_TOTALS in .env file")


def get_totals_from_api(api_key=API_KEY):
    url = (
        f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
        f"?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    games = []

    eastern = tz("US/Eastern")
    now_eastern = datetime.now(eastern)

    days_ahead = (1 - now_eastern.weekday()) % 7
    next_tuesday = (now_eastern + timedelta(days=days_ahead)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )

    for game in data:
        if not game.get('bookmakers'):
            continue

        # Parse commence_time as UTC
        commence_time_utc = datetime.fromisoformat(game['commence_time'].replace("Z", "+00:00"))
        commence_time_eastern = commence_time_utc.astimezone(eastern)

        if commence_time_eastern > next_tuesday:
            continue

        # Only consider DraftKings
        dk_bookmakers = [b for b in game['bookmakers'] if b['title'].lower() == 'draftkings']
        if not dk_bookmakers:
            continue  # skip if DraftKings not available

        bookmaker = dk_bookmakers[0]  # take the DraftKings bookmaker
        for market in bookmaker['markets']:
            if market['key'] == 'totals':
                for outcome in market['outcomes']:
                    if outcome['name'] == 'Over':
                        # Convert to Eastern timezone, tz-aware
                        commence_time_eastern = commence_time_utc.astimezone(eastern)
                        games.append({
                            'home_team': game['home_team'],
                            'away_team': game['away_team'],
                            'commence_time': commence_time_eastern,
                            'total_line': outcome['point'],
                            'bookmaker': bookmaker['title']
                        })

    return pd.DataFrame(games)


def get_totals(path=DATA_PATH, api_key=API_KEY):
    eastern = tz("US/Eastern")

    if os.path.exists(path):
        print(f"Reading existing Vegas totals from {path}...")
        existing_df = pd.read_csv(path)

        # Normalize datetime
        existing_df['commence_time'] = pd.to_datetime(existing_df['commence_time'], errors='coerce', utc=True)
        existing_df['commence_time'] = existing_df['commence_time'].dt.tz_convert(eastern)

        # Get new data
        print("Fetching new and updated game totals from API...")
        new_df = get_totals_from_api(api_key)

        # Normalize new datetime
        new_df['commence_time'] = pd.to_datetime(new_df['commence_time'], utc=True)
        new_df['commence_time'] = new_df['commence_time'].dt.tz_convert(eastern)
        
        # Combine and drop duplicates
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['home_team', 'away_team', 'commence_time', 'bookmaker'],
            keep='last'
        ).sort_values(by='commence_time')

        combined.to_csv(path, index=False)
        print(f"Appended new games. Saved updated file to {path}.")
        return combined

    else:
        print("No existing file found. Downloading fresh totals...")
        df = get_totals_from_api(api_key)
        df.to_csv(path, index=False)
        print(f"Saved new file to {path}.")
        return df
