import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pytz import timezone as tz
from config import TOTALS_PATH, API_MAX_RETRIES, API_BACKOFF_FACTOR, LINE_SNAPSHOTS_DIR

logger = logging.getLogger(__name__)

DATA_PATH = TOTALS_PATH

load_dotenv()
API_KEY = os.getenv("API_KEY_TOTALS")

TEAM_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def enrich_totals(totals: pd.DataFrame, team_games: pd.DataFrame | None) -> pd.DataFrame:
    """
    Auto-populate manual columns in totals for rows where they are missing.

    - home/away_starting_qb : most recent starter for each team from historical data
    - home/away_short_rest  : 1 if the team's last game was within 6 days of kickoff
    - international         : 1 if kickoff is before 11:00 AM US/Eastern

    Existing non-null values are preserved so manual overrides in the CSV survive.
    """
    eastern = tz("US/Eastern")
    totals = totals.copy()

    for col in ["home_starting_qb", "away_starting_qb",
                "home_short_rest", "away_short_rest", "international"]:
        if col not in totals.columns:
            totals[col] = pd.NA

    # Kickoff as ET (tz-aware) for the international check, tz-naive for arithmetic.
    kickoff_et = pd.to_datetime(totals["commence_time"], utc=True).dt.tz_convert(eastern)
    kickoff_naive = kickoff_et.dt.tz_localize(None)

    # International: domestic NFL games never kick off before 11 AM ET.
    mask_intl = totals["international"].isna()
    totals.loc[mask_intl, "international"] = (kickoff_et[mask_intl].dt.hour < 11).astype(int)

    if team_games is None or team_games.empty or "starting_qb" not in team_games.columns:
        return totals

    tg = team_games.copy()
    tg["game_date"] = pd.to_datetime(tg["date"])

    # Latest starting QB per team.
    latest_qb: pd.Series = (
        tg[tg["starting_qb"].notna()]
        .sort_values("game_date")
        .groupby("team")["starting_qb"]
        .last()
    )

    # Last game date per team for short-rest calculation.
    last_game_date: pd.Series = (
        tg.sort_values("game_date")
        .groupby("team")["game_date"]
        .last()
    )

    home_abbrev = totals["home_team"].map(TEAM_ABBREV)
    away_abbrev = totals["away_team"].map(TEAM_ABBREV)

    # Starting QBs — only fill missing rows.
    mask_hqb = totals["home_starting_qb"].isna()
    totals.loc[mask_hqb, "home_starting_qb"] = home_abbrev[mask_hqb].map(latest_qb).values

    mask_aqb = totals["away_starting_qb"].isna()
    totals.loc[mask_aqb, "away_starting_qb"] = away_abbrev[mask_aqb].map(latest_qb).values

    # Short rest: days between last game and upcoming kickoff.
    home_last = pd.to_datetime(home_abbrev.map(last_game_date))
    away_last = pd.to_datetime(away_abbrev.map(last_game_date))

    home_gap = (kickoff_naive - home_last).dt.days
    away_gap = (kickoff_naive - away_last).dt.days

    mask_hsr = totals["home_short_rest"].isna()
    totals.loc[mask_hsr, "home_short_rest"] = (home_gap[mask_hsr] <= 6).astype(int)

    mask_asr = totals["away_short_rest"].isna()
    totals.loc[mask_asr, "away_short_rest"] = (away_gap[mask_asr] <= 6).astype(int)

    logger.info("Enriched totals: auto-populated starting QBs, short-rest flags, and international flags.")
    return totals

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


def get_totals(path=DATA_PATH, api_key=API_KEY, use_cache_only=False, team_games=None):
    eastern = tz("US/Eastern")

    if os.path.exists(path):
        logger.info("Reading existing Vegas totals from %s...", path)
        existing_df = pd.read_csv(path)

        # Normalize datetime
        existing_df['commence_time'] = pd.to_datetime(existing_df['commence_time'], errors='coerce', utc=True)
        existing_df['commence_time'] = existing_df['commence_time'].dt.tz_convert(eastern)

        if use_cache_only:
            logger.info("Using cached totals only. Skipping API fetch.")
            result = enrich_totals(existing_df, team_games)
            result.to_csv(path, index=False)
            return result

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

        result = enrich_totals(combined, team_games)
        result.to_csv(path, index=False)
        _save_line_snapshot(result)
        logger.info("Appended new games. Saved updated file to %s.", path)
        return result

    else:
        if use_cache_only:
            raise FileNotFoundError(f"No cached file found at {path}, cannot skip API fetch.")
        logger.info("No existing file found. Downloading fresh totals...")
        df = get_totals_from_api(api_key)
        df['commence_time'] = pd.to_datetime(df['commence_time'], utc=True)
        df['commence_time'] = df['commence_time'].dt.tz_convert(eastern)
        result = enrich_totals(df, team_games)
        result.to_csv(path, index=False)
        _save_line_snapshot(result)
        logger.info("Saved new file to %s.", path)
        return result


def _save_line_snapshot(totals: pd.DataFrame) -> None:
    """Save a dated snapshot of current lines to LINE_SNAPSHOTS_DIR (once per day)."""
    os.makedirs(LINE_SNAPSHOTS_DIR, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    path = os.path.join(LINE_SNAPSHOTS_DIR, f"lines_{today}.csv")
    if not os.path.exists(path):
        snap = totals[["home_team", "away_team", "total_line"]].copy()
        # Store abbreviated team names so they match upcoming.py after TEAM_ABBREV conversion
        snap["home_team"] = snap["home_team"].map(TEAM_ABBREV).fillna(snap["home_team"])
        snap["away_team"] = snap["away_team"].map(TEAM_ABBREV).fillna(snap["away_team"])
        snap.to_csv(path, index=False)
        logger.info("Saved opening line snapshot to %s.", path)
