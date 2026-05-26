# ── Rolling window sizes ──────────────────────────────────────────────────────
ROLLING_WINDOW_POINTS     = 3
ROLLING_WINDOW_PACE       = 5
ROLLING_WINDOW_QB         = 3
ROLLING_WINDOW_DEF        = 3
ROLLING_WINDOW_RZ         = 3
ROLLING_WINDOW_TURNOVERS  = 3
ROLLING_WINDOW_3RD_DOWN   = 3
ROLLING_WINDOW_OFFENSE    = 4

# ── Model hyperparameters (XGBoost) ───────────────────────────────────────────
MODEL_N_ESTIMATORS       = 500
MODEL_MAX_DEPTH          = 5
MODEL_LEARNING_RATE      = 0.05
MODEL_SUBSAMPLE          = 0.8
MODEL_COLSAMPLE_BYTREE   = 0.8
MODEL_MIN_CHILD_WEIGHT   = 5
MODEL_REG_ALPHA          = 0.1
MODEL_REG_LAMBDA         = 1.0
RANDOM_STATE             = 42

# ── Evaluation ────────────────────────────────────────────────────────────────
PROB_THRESHOLD = 0.55   # min P(over) to flag a bet (mirror: < 1 - PROB_THRESHOLD for under)

# ── Season lists ─────────────────────────────────────────────────────────────
TRAIN_START_SEASON = 2014          # earliest season included in any training fold
EVAL_TRAIN_SEASONS = list(range(2014, 2024))   # 2014–2023
EVAL_TEST_SEASONS  = [2024]
PROD_SEASONS       = list(range(2014, 2026))   # 2014–2025

# ── Paths ─────────────────────────────────────────────────────────────────────
TOTALS_PATH            = "data/nfl_over_unders.csv"
WEATHER_FORECAST_PATH  = "data/nfl_weather_forecasts.csv"
WEATHER_CACHE_DIR      = "data/weather_cache"
INJURIES_PATH          = "data/injuries.parquet"
LINE_SNAPSHOTS_DIR     = "data/line_snapshots"
EVAL_MODEL_PATH        = "model/xgb_total_points_model_eval.joblib"
PROD_MODEL_PATH        = "model/xgb_total_points_model_prod.joblib"
PREDICTIONS_DIR        = "predictions"

# ── API reliability ───────────────────────────────────────────────────────────
API_MAX_RETRIES    = 3
API_BACKOFF_FACTOR = 1.5   # seconds multiplier between retries
