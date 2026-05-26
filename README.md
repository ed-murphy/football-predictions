# NFL Over/Under Predictor

An XGBoost binary classifier that estimates the probability that an NFL game goes **over** the Vegas total. It does not predict raw point totals — it predicts P(over) directly, which is calibrated via log-loss and translates naturally into a bet signal.

---

## Table of Contents

1. [How the Model Works](#how-the-model-works)
2. [Performance](#performance)
3. [Prerequisites](#prerequisites)
4. [Clone & Install](#clone--install)
5. [API Keys](#api-keys)
6. [Weekly Workflow](#weekly-workflow)
   - [Step 1 — Refresh Historical Data](#step-1--refresh-historical-data)
   - [Step 2 — Retrain (optional)](#step-2--retrain-optional)
   - [Step 3 — Generate Predictions](#step-3--generate-predictions)
   - [Step 4 — Override If Needed](#step-4--override-if-needed)
7. [Reading the Output](#reading-the-output)
8. [Key Config Values](#key-config-values)
9. [Feature Reference](#feature-reference)
10. [File Overview](#file-overview)

---

## How the Model Works

**Target:** `1` if `total_points > total_line` (over hit), `0` otherwise.

**Algorithm:** `XGBClassifier` with `objective='binary:logistic'`, trained with log-loss. `predict_proba()[:, 1]` gives P(over). This is meaningfully different from predicting total points with MSE — log-loss trains the model to be well-calibrated on direction rather than penalising magnitude errors.

**Training data:** 2014–2025 regular season and playoff games. Walk-forward CV confirmed that including pre-2018 data improves performance (more data beats recency effects because the model predicts *relative to the line*, and the line already accounts for era-level scoring trends).

**Two models are saved:**
- **Eval model** — trained on 2014–2023, evaluated on 2024 holdout
- **Prod model** — trained on 2014–2025, used for live predictions

**Bet signals** are generated at `PROB_THRESHOLD = 0.55`:
- `p_over > 0.55` → **Over**
- `p_over < 0.45` → **Under**
- Otherwise → no signal (blank)

---

## Performance

Walk-forward CV (one season at a time, 2018–2025, training from 2014):

| Metric | Model | Random baseline |
|--------|-------|----------------|
| AUC | **0.519** | 0.500 |
| Precision at signal | **0.516** | 0.500 |
| Avg log-loss | 0.797 | ~0.693 |
| Avg Brier score | 0.289 | 0.250 |

The edge is real but slim (~1.6% precision above random). Public box-score features are already largely priced into the Vegas line — this model captures residual signal from injury timing, referee tendencies, and line movement direction.

---

## Prerequisites

- Python 3.10+
- Git

---

## Clone & Install

```bash
git clone <repo-url>
cd football-predictions

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## API Keys

Two external APIs are required. Create a `.env` file in the repo root:

```
API_KEY_TOTALS=your_odds_api_key
API_KEY_WEATHER=your_openweathermap_key
```

| Key | Service | Used for |
|-----|---------|----------|
| `API_KEY_TOTALS` | [The Odds API](https://the-odds-api.com) | Fetching DraftKings over/under lines for upcoming games |
| `API_KEY_WEATHER` | [OpenWeatherMap](https://openweathermap.org/api) | Fetching weather forecasts for game locations |

Both services have free tiers sufficient for weekly use.

---

## Weekly Workflow

### Step 1 — Refresh Historical Data

Historical game, play-by-play, and injury data is downloaded via `nfl_data_py` using a **separate** script with its own virtual environment (it requires `pandas<2.0`, which conflicts with the main environment).

**First time only — create the environment:**
```bash
cd download
python -m venv download_env
download_env\Scripts\activate        # Windows
# source download_env/bin/activate   # macOS / Linux
pip install -r download-requirements.txt
python download_nfl_data.py
cd ..
```

**Every week — refresh data:**
```bash
cd download
download_env\Scripts\activate        # Windows
# source download_env/bin/activate   # macOS / Linux
python download_nfl_data.py
cd ..
```

> Re-activate the **main** environment afterward:
> ```bash
> .venv\Scripts\activate       # Windows
> # source .venv/bin/activate  # macOS / Linux
> ```

This saves to `data/`:
- `games.parquet` — game-level schedule and result data (2014–present)
- `plays.parquet` — play-by-play data (2014–present)
- `injuries.parquet` — weekly injury reports (2014–present)

Injury data is the most time-sensitive — it changes through the week as players are added/removed from reports. Re-run the download script on game day for the freshest data.

### Step 2 — Retrain (optional)

Retrain at the start of each new season (or whenever you want to incorporate completed games from the current season):

```bash
python main.py --train_only
```

This:
1. Builds all features from the downloaded parquet files
2. Trains the **eval model** (2014–2023 train / 2024 test) and prints holdout metrics
3. Runs **walk-forward CV** (2018–present) and logs per-season results
4. Trains the **prod model** (2014–present) and saves it to `model/`

Retraining is not required weekly — the prod model already includes the full training window.

### Step 3 — Generate Predictions

```bash
python main.py
```

This fetches current DraftKings lines from The Odds API, auto-populates all context fields, runs the prod model, and writes output to `predictions/predictions_YYYYMMDD.csv`.

To skip the API call and use the lines already saved in `data/nfl_over_unders.csv`:

```bash
python main.py --use_cached_totals
```

### Step 4 — Override If Needed

If a QB is injured or any auto-inferred value is wrong, edit `data/nfl_over_unders.csv` directly, then run with `--use_cached_totals` to use your corrected version.

QB names must match the play-by-play format: `F.Lastname` (e.g. `J.Allen`, `L.Jackson`). Check recent `predictions/` CSVs for examples.

**Auto-populated fields:**

| Field | How it's computed |
|-------|------------------|
| `home_starting_qb` / `away_starting_qb` | Most recent starter for each team from play-by-play history |
| `home_short_rest` / `away_short_rest` | `1` if the team's last game was ≤ 6 days before kickoff |
| `international` | `1` if kickoff is before 11:00 AM US/Eastern (London/Munich games) |

---

## Reading the Output

Output CSV columns:

| Column | Description |
|--------|-------------|
| `date` | Game date |
| `home_team` / `away_team` | NFL team abbreviations |
| `total_line` | Current DraftKings over/under line |
| `p_over` | Model's estimated P(over), 0–1 |
| `bet` | `Over`, `Under`, or blank (no signal) |
| `line_open` | First line snapshot captured this week |
| `line_movement` | `total_line − line_open` (positive = line bet up; negative = bet down) |
| `home_qb_injured` / `away_qb_injured` | `1` if the starting QB is on the injury report |
| `actual_total` | Filled in after the game completes |

**Line movement context:** Every call to `python main.py` saves a timestamped snapshot to `data/line_snapshots/`. `line_open` is the earliest snapshot for the week; `line_movement` shows how many points the market has moved since. Sharp action tends to move lines by 0.5–1.5 points.

---

## Key Config Values

All constants live in `config.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `PROB_THRESHOLD` | `0.55` | Minimum P(over) to flag a bet |
| `TRAIN_START_SEASON` | `2014` | Earliest season used in any training fold |
| `EVAL_TRAIN_SEASONS` | `2014–2023` | Seasons for the holdout eval model |
| `EVAL_TEST_SEASONS` | `[2024]` | Holdout test season |
| `PROD_SEASONS` | `2014–2025` | Seasons for the prod model |
| `ROLLING_WINDOW_QB` | `3` | Games in QB EPA rolling average |
| `ROLLING_WINDOW_OFFENSE` | `4` | Games in offensive efficiency rolling averages |
| `MODEL_N_ESTIMATORS` | `500` | XGBoost trees |
| `MODEL_MAX_DEPTH` | `5` | Max tree depth |
| `MODEL_LEARNING_RATE` | `0.05` | XGBoost eta |

To change the bet threshold, update `PROB_THRESHOLD`. To extend training data when new seasons are available, update `PROD_SEASONS` and `EVAL_TRAIN_SEASONS`.

---

## Feature Reference

### Base features (raw rolling stats)

**Scoring form** — rolling 3-game avg points for/against (home + away)

**QB quality** — rolling 3-game avg QB EPA (home + away)

**Defense** — rolling 3-game avg defensive EPA, sack rate (home + away)

**Offensive efficiency** — rolling 4-game avg pass rate, explosive play rate, success rate, rush EPA, pass EPA, CPOE (home + away)

**Pace** — rolling 5-game avg offensive plays per game (home + away)

**Red zone** — rolling 3-game red zone efficiency (home + away)

**Turnovers** — rolling 3-game avg turnovers (home + away)

**Third down** — rolling 3-game avg third-down conversion rate (home + away)

**Weather** — temperature and wind speed at game site

**Injuries** — injury index (weighted starter availability) and QB injured flag (home + away)

**Referee tendency** — career avg total points allowed by the assigned referee (expanding window, prior games only, to prevent leakage)

**Game context** — divisional flag, regular season flag, international flag, short rest (≤6 days), post-bye (home + away)

### Engineered features (interactions and deltas)

Pace × wind speed interactions, QB EPA × opposing defense matchup, sum and delta versions of pace/QB EPA/pass EPA/rush EPA/success rate/explosive rate/defense EPA/red zone efficiency/turnovers/third-down/sack rate.

---

## File Overview

```
main.py                      Entry point — training and prediction CLI
config.py                    All constants: season ranges, thresholds, hyperparameters, paths
requirements.txt             Main environment dependencies
app.py                       Streamlit dashboard (predictions + performance + model insights)

data/
  games.parquet              Game-level schedule and result data (downloaded)
  plays.parquet              Play-by-play data (downloaded)
  injuries.parquet           Weekly injury reports (downloaded)
  nfl_over_unders.csv        Upcoming game lines (fetched + optionally hand-edited)
  nfl_weather_forecasts.csv  Weather forecast cache (auto-generated)
  weather_cache/             Per-game historical weather cache
  line_snapshots/            Dated opening-line snapshots (auto-generated)

download/
  download_nfl_data.py       Downloads/refreshes all three parquet files
  download-requirements.txt  Separate deps (requires pandas<2.0)

model/
  xgb_total_points_model_eval.joblib   Eval model (trained 2014–2023, tested on 2024)
  xgb_total_points_model_prod.joblib   Prod model (trained 2014–2025, used for predictions)

predictions/                 Output CSVs (predictions_YYYYMMDD[_vN].csv)
old_predictions/             Archive of predictions from previous seasons

src/
  model_features.py          BASE_FEATURES, ENGINEERED_FEATURES, add_engineered_features()
  train.py                   XGBClassifier training, build_model_data()
  evaluate.py                evaluate_model(), walk_forward_cv() — log-loss/AUC/brier/precision
  predictions.py             save_predictions() — merges, adds bet column, writes CSV
  upcoming.py                prepare_upcoming_team_games() — builds feature row for each game
  load.py                    load_data(), load_injuries()
  basic.py                   Rolling points for/against
  qb.py                      QB EPA rolling averages
  defense.py                 Defensive EPA, sack rate
  offense.py                 Pass rate, explosive rate, success rate, rush/pass EPA, CPOE
  pace.py                    Offensive plays per game
  rest.py                    Short rest, post-bye flags
  redzone.py                 Red zone efficiency
  turnovers.py               Turnover rolling averages
  third_down.py              Third-down conversion rate
  injuries.py                Injury index, QB injured flag
  referee.py                 Referee career avg total (expanding window)
  weather.py                 Historical weather lookup
  weather_forecast.py        OpenWeatherMap forecast fetch
  totals.py                  The Odds API fetch + line snapshot saving
  old_predictions.py         Load prior-week predictions for deduplication
```
