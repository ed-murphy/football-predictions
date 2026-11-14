"""Every tunable, in one place.

Nothing in here was tuned on the backtest it is judged by. Where a number was
measured from data, the comment says how, and `research/` holds the script that
reproduces it.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
# One file from nflverse: every game since 1999 with final scores, closing lines,
# kickoff weather, roof and stadium. It is the only historical input.
GAMES_URL  = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
GAMES_PATH = "data/games.csv"

LINE_HISTORY_PATH = "data/line_history.csv"   # every line ever fetched, timestamped
SLATE_PATH        = "predictions/slate.csv"   # this week's board, rewritten each run
BETS_PATH         = "predictions/bets.csv"    # permanent log of every flagged bet
BACKTEST_PATH     = "predictions/backtest.csv"

# ── Model ─────────────────────────────────────────────────────────────────────
FIRST_SEASON          = 1999   # earliest season used to fit
BACKTEST_START_SEASON = 2008   # first season scored out of sample (9 seasons of history before it)

# ── Betting rules ─────────────────────────────────────────────────────────────
STANDARD_ODDS = -110   # price assumed in the backtest and when a book's price is unknown

# Flag a bet when its expected return per unit staked is at least this. At -110
# and MIN_EV = 0.02 that works out to "expected kickoff wind of about 13 mph".
MIN_EV = 0.02

# Only flag bets this close to kickoff. Forecast skill falls off quickly with
# lead time (see FORECAST_CALIBRATION); earlier games are shown as "watch".
BET_WINDOW_DAYS = 3

KELLY_FRACTION = 0.25   # fraction of full Kelly used for the suggested stake
MAX_STAKE_PCT  = 2.0    # never suggest more than this share of bankroll on one game

# Sportsbooks you can actually bet at, as the Odds API names them (draftkings,
# fanduel, betmgm, williamhill_us, betrivers, espnbet, ...). The best line among
# these is the one quoted; every book in the feed still informs the market line.
MY_BOOKS = ["draftkings"]

SLATE_DAYS = 8          # how far ahead the weekly board looks

# ── Forecast → expected kickoff wind ──────────────────────────────────────────
# The model is fitted on the wind recorded in the official gamebook at kickoff.
# A forecast is a noisy, slightly biased estimate of that number, so it is
# translated before use:   expected_wind = intercept + slope * forecast_mph
#
# Measured by regressing gamebook wind on archived Open-Meteo forecasts for 405
# outdoor games, 2024-2026 (research/forecast_calibration.py). The slope shrinks
# with lead time because older forecasts are less reliable — a 20 mph forecast
# five days out is worth less than the same forecast the day before.
#
#   days before kickoff: (intercept, slope)      # correlation with gamebook wind
FORECAST_CALIBRATION = {
    1: (1.83, 0.742),   # 0.74
    2: (2.05, 0.704),   # 0.71
    3: (2.28, 0.682),   # 0.70
    4: (2.35, 0.670),   # 0.66
    5: (3.00, 0.611),   # 0.58
    6: (4.04, 0.472),   # 0.47
    7: (4.97, 0.361),   # 0.33
}

# Correlation used by the backtest's forecast stress test: the average skill of a
# forecast made inside BET_WINDOW_DAYS.
FORECAST_CORRELATION = 0.72

# ── APIs ──────────────────────────────────────────────────────────────────────
ODDS_URL     = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"   # free, no key needed
API_TIMEOUT_SECONDS = 30
