# ── Rolling window sizes ──────────────────────────────────────────────────────
ROLLING_WINDOW_POINTS     = 3
ROLLING_WINDOW_PACE       = 5
ROLLING_WINDOW_QB         = 3
ROLLING_WINDOW_DEF        = 3
ROLLING_WINDOW_RZ         = 3
ROLLING_WINDOW_TURNOVERS  = 3
ROLLING_WINDOW_3RD_DOWN   = 3

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODEL_N_ESTIMATORS = 500
MODEL_MAX_DEPTH    = 8
RANDOM_STATE       = 42

# ── Evaluation ────────────────────────────────────────────────────────────────
PRECISION_MARGIN = 4

# ── Season lists ─────────────────────────────────────────────────────────────
EVAL_TRAIN_SEASONS = list(range(2014, 2024))   # 2014–2023
EVAL_TEST_SEASONS  = [2024]
PROD_SEASONS       = list(range(2014, 2026))   # 2014–2025

# ── Paths ─────────────────────────────────────────────────────────────────────
TOTALS_PATH            = "data/nfl_over_unders.csv"
WEATHER_FORECAST_PATH  = "data/nfl_weather_forecasts.csv"
WEATHER_CACHE_DIR      = "data/weather_cache"
EVAL_MODEL_PATH        = "model/rf_total_points_model_eval.joblib"
PROD_MODEL_PATH        = "model/rf_total_points_model_prod.joblib"
PREDICTIONS_DIR        = "predictions"

# ── API reliability ───────────────────────────────────────────────────────────
API_MAX_RETRIES    = 3
API_BACKOFF_FACTOR = 1.5   # seconds multiplier between retries
