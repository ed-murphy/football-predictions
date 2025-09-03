import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
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

    for game in data:
        if not game.get('bookmakers'):
            continue

        # Parse commence_time as UTC
        commence_time_utc = datetime.fromisoformat(game['commence_time'].replace("Z", "+00:00"))

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
        print(f"Reading cached Vegas totals for upcoming games from {path}...")
        df = pd.read_csv(path)
        
        # Ensure datetime, handle tz-aware safely
        df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce', utc=True)
        df['commence_time'] = df['commence_time'].dt.tz_convert(eastern).dt.tz_localize(None)
        
        return df
    else:
        print("Downloading fresh Vegas totals for upcoming games...")
        df = get_totals_from_api(api_key)
        df.to_csv(path, index=False)
        return df
