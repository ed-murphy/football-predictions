import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pytz import timezone as tz
from config import TOTALS_PATH, API_MAX_RETRIES, API_BACKOFF_FACTOR

logger = logging.getLogger(__name__)

DATA_PATH = TOTALS_PATH

load_dotenv()
API_KEY = os.getenv("API_KEY_TOTALS")

if not API_KEY:
    raise ValueError("Missing API_KEY_TOTALS in .env file")


def _session() -> requests.Session:
    """Return a requests Session with automatic retry on transient errors."""
    s = requests.Session()
    retry = Retry(
        total=API_MAX_RETRIES,
        backoff_factor=API_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get_totals_from_api(api_key=API_KEY):
    url = (
        f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
        f"?apiKey={api_key}&regions=us&markets=totals&oddsFormat=american"
    )
    resp = _session().get(url, timeout=30)
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


def get_totals(path=DATA_PATH, api_key=API_KEY, use_cache_only=False):
    eastern = tz("US/Eastern")

    if os.path.exists(path):
        logger.info("Reading existing Vegas totals from %s...", path)
        existing_df = pd.read_csv(path)

        # Normalize datetime
        existing_df['commence_time'] = pd.to_datetime(existing_df['commence_time'], errors='coerce', utc=True)
        existing_df['commence_time'] = existing_df['commence_time'].dt.tz_convert(eastern)

        if use_cache_only:
            logger.info("Using cached totals only. Skipping API fetch.")
            return existing_df

        # Get new data
        logger.info("Fetching new and updated game totals from API...")
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
        logger.info("Appended new games. Saved updated file to %s.", path)
        return combined

    else:
        if use_cache_only:
            raise FileNotFoundError(f"No cached file found at {path}, cannot skip API fetch.")
        logger.info("No existing file found. Downloading fresh totals...")
        df = get_totals_from_api(api_key)
        df.to_csv(path, index=False)
        logger.info("Saved new file to %s.", path)
        return df
