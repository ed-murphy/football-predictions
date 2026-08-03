"""Project configuration.

Every tunable lives here so the modelling choices are visible in one place.
"""

# ── Rolling window sizes (games) ──────────────────────────────────────────────
# Short windows track form, long windows track ability. Volume and pace stabilise
# quickly; per-play efficiency is noisier and gets a wider window.
ROLLING_WINDOW_POINTS     = 4
ROLLING_WINDOW_PACE       = 5
ROLLING_WINDOW_QB         = 4
ROLLING_WINDOW_DEF        = 4
ROLLING_WINDOW_RZ         = 5
ROLLING_WINDOW_TURNOVERS  = 5
ROLLING_WINDOW_3RD_DOWN   = 5
ROLLING_WINDOW_OFFENSE    = 5

# ── Model ─────────────────────────────────────────────────────────────────────
# Ridge penalties searched by walk-forward MSE. The useful range is large: with
# ~2,700 rows and ~90 features the honest answer is "shrink almost everything".
RIDGE_ALPHA_GRID = [100.0, 300.0, 1000.0, 3000.0, 10_000.0, 30_000.0]
RANDOM_STATE     = 42

# ── Betting filter ────────────────────────────────────────────────────────────
AMERICAN_ODDS   = -110   # standard totals pricing
DECIMAL_PAYOUT  = 100 / abs(AMERICAN_ODDS)          # profit per unit staked on a win
BREAK_EVEN_WIN_RATE = 1 / (1 + DECIMAL_PAYOUT)      # 52.38% at -110

# A bet is flagged when the model's edge clears the point at which it can pay for
# the vig (derived from the model's own residual sigma — see TotalsModel) plus
# this margin. Raise it to bet less often and more selectively.
MIN_EDGE_MARGIN_POINTS = 0.25

KELLY_FRACTION  = 0.25   # fractional Kelly used for the suggested stake column

# ── Season lists ──────────────────────────────────────────────────────────────
TRAIN_START_SEASON  = 2014          # earliest season in any training fold
BACKTEST_START_SEASON = 2018        # first season scored out of sample
EVAL_TRAIN_SEASONS  = list(range(2014, 2024))   # 2014–2023
EVAL_TEST_SEASONS   = [2024, 2025]
PROD_SEASONS        = list(range(2014, 2027))   # everything available

# ── Paths ─────────────────────────────────────────────────────────────────────
GAMES_PATH             = "data/games.parquet"
PLAYS_PATH             = "data/plays.parquet"
INJURIES_PATH          = "data/injuries.parquet"
TOTALS_PATH            = "data/nfl_over_unders.csv"
WEATHER_FORECAST_PATH  = "data/nfl_weather_forecasts.csv"
LINE_SNAPSHOTS_DIR     = "data/line_snapshots"
EVAL_MODEL_PATH        = "model/totals_model_eval.joblib"
PROD_MODEL_PATH        = "model/totals_model_prod.joblib"
PREDICTIONS_DIR        = "predictions"
BACKTEST_PATH          = "predictions/backtest.csv"

# ── API reliability ───────────────────────────────────────────────────────────
API_MAX_RETRIES    = 3
API_BACKOFF_FACTOR = 1.5   # seconds multiplier between retries
ODDS_BOOKMAKER     = "draftkings"
