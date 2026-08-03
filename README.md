# NFL Totals Model

A model that estimates the combined score of upcoming NFL games and compares it to
the market's posted total.

**Headline result: it does not beat the market.** Across a 2018–2025 walk-forward
backtest the model's average error is 10.41 points against the closing line's 10.42,
and the flagged bets return −3.6% ± 9.7% at −110 pricing. That is a real finding
about how efficient NFL totals markets are, and the repository is built to report it
honestly rather than to manufacture a signal.

---

## Contents

1. [What it does](#what-it-does)
2. [Performance](#performance)
3. [Install](#install)
4. [Weekly workflow](#weekly-workflow)
5. [Reading the output](#reading-the-output)
6. [Configuration](#configuration)
7. [Project layout](#project-layout)
8. [Tests](#tests)

---

## What it does

For each game it predicts **the residual**: how many points the final combined score
will land above or below the posted total.

```
predicted_total = market_total + shrunk_model_edge
P(over)         = P(residual > 0)  under the model's error distribution
```

Modelling the residual rather than the total directly means the model never has to
relearn what the market already knows (that domes are high scoring, that these two
offences are good); it only has to find where the line is wrong. It is a ridge
regression over ~80 features covering team scoring form, pace, per-play efficiency,
quarterback EPA, defensive EPA, red-zone and third-down conversion, turnovers, rest,
injuries, kickoff weather, referee tendency, and the market's own total and spread.

Every predicted edge is multiplied by a **shrinkage coefficient** estimated by
walk-forward validation inside the training window: the model measures how much of
its own edge historically survived out of sample and scales its forecasts down to
match. When there is no measurable signal the coefficient goes to zero and the model
declines to disagree with the line at all.

### Why ridge and not gradient boosting

The previous version of this project was an XGBoost classifier on the binary
over/under outcome. It scored **worse than a coin flip** out of sample — holdout log
loss 0.80 against 0.69 for always predicting 50%. Five hundred trees at depth five,
fitted to ~2,700 near-coinflip labels, memorised noise.

Every tree ensemble tried during the rewrite behaved the same way. With this much
data and this little signal, heavy linear shrinkage is the only thing that
generalises. See [`docs/model.md`](docs/model.md) for the comparison table.

---

## Performance

Walk-forward backtest, 2018–2025, each season scored by a model that saw only
earlier seasons (hyperparameters and shrinkage re-selected per fold):

| Metric | Model | Benchmark |
|---|---|---|
| Log loss | 0.6924 | 0.6931 (coin flip) |
| AUC | 0.522 | 0.500 |
| Mean absolute error of predicted total | 10.41 | 10.42 (repeat the line) |
| Edge correlation | +0.022 ± 0.021 | 0 |
| Flagged bets | 97 | — |
| Win rate | 50.5% | 52.4% needed at −110 |
| ROI | −3.6% ± 9.7% | 0% |

**Edge correlation** — the correlation between the edge the model predicted and the
residual that actually happened, over every game rather than only the ones bet — is
the most stable read on whether the model knows anything. At +0.022 with a standard
error of 0.021 it is not distinguishable from zero over this window. Fixing the
alpha and shrinkage rather than re-selecting them per fold pushes it to about +0.05
(z ≈ 2.5), which is a whisper of signal worth roughly 0.4% of residual variance.

None of these numbers support betting. They are reported with standard errors
precisely so that a run of good luck cannot be mistaken for an edge.

---

## Install

```bash
git clone https://github.com/ed-murphy/football-predictions
cd football-predictions
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Downloading source data needs `nfl_data_py`, which pins older pandas/numpy. Install
it in a separate environment:

```bash
python -m venv download/download_env
download/download_env/Scripts/pip install -r download/download-requirements.txt
download/download_env/Scripts/python download/download_nfl_data.py
```

### API keys

Create a `.env` in the repository root:

```
API_KEY_TOTALS=<the-odds-api.com key>
API_KEY_WEATHER=<openweathermap.org key>
```

Both are optional for training and backtesting; they are needed to score an upcoming
slate. Without them the run falls back to cached lines and seasonal-normal weather.

---

## Weekly workflow

```bash
# 1. Refresh nflverse data (schedules, play-by-play, injuries)
download/download_env/Scripts/python download/download_nfl_data.py

# 2. Retrain, backtest, and score this week's slate
python main.py

# 3. View the output
streamlit run app.py
```

Other entry points:

| Command | Effect |
|---|---|
| `python main.py --train-only` | fit and evaluate, write no predictions |
| `python main.py --predict-only` | score the slate with the saved model |
| `python main.py --backtest` | walk-forward evaluation only |
| `python main.py --use-cached-totals` | skip the odds API, use cached lines |
| `python main.py --skip-backtest` | train without the slow backtest |

### Overriding inputs by hand

`data/nfl_over_unders.csv` is regenerated each run but **existing non-null values are
preserved**, so you can correct a starting quarterback or an international flag and
the correction will survive the next refresh.

---

## Reading the output

Predictions are written to `predictions/predictions_YYYYMMDD[_vN].csv`:

| Column | Meaning |
|---|---|
| `total_line` | the market's posted total |
| `pred_total` | the model's expected combined score |
| `edge` | `pred_total - total_line`, in points |
| `p_over` | probability the game finishes over the line |
| `bet` | `Over`, `Under`, or blank |
| `stake` | quarter-Kelly position size in units |
| `line_open` / `line_movement` | earliest snapshot this week, and the move since |
| `actual_total` / `result` | filled in once the game is played |

`bet` is only populated when the edge clears the point at which it can pay for the
vig. That threshold is **derived, not chosen**: at −110 you need to win 52.38% of
bets, and inverting the model's own residual distribution says how many points of
disagreement that requires (about 0.8 points at a 13-point sigma). Most weeks
nothing qualifies, which is the correct behaviour for a model with no demonstrated
edge.

---

## Configuration

Everything tunable is in [`config.py`](config.py). The choices that matter:

| Setting | Default | Effect |
|---|---|---|
| `RIDGE_ALPHA_GRID` | `100…30000` | penalties searched by walk-forward MSE |
| `MIN_EDGE_MARGIN_POINTS` | `0.25` | margin above break-even required to flag a bet |
| `KELLY_FRACTION` | `0.25` | fraction of full Kelly used for `stake` |
| `ROLLING_WINDOW_*` | 4–5 games | how much history each form feature averages |
| `BACKTEST_START_SEASON` | `2018` | first season scored out of sample |

---

## Project layout

```
main.py                 CLI entry point
config.py               every tunable in one place
app.py                  Streamlit front end

src/
  constants.py          team codes, divisions, stadium coordinates
  rolling.py            lagged rolling means and home/away broadcasting
  load.py               reads the cached nflverse parquet files
  basic.py              schedule -> team-game rows, scoring form
  qb.py                 quarterback rolling EPA (keyed on the player)
  play_features.py      every play-derived feature, from one spec table
  rest.py               short rest and post-bye flags
  injuries.py           weekly injury index
  referee.py            per-official historical totals
  pipeline.py           orchestrates the above into a FeatureBundle
  model_features.py     the feature contract shared by training and serving
  model.py              TotalsModel: probabilities, thresholds, shrinkage
  train.py              fitting and persistence
  evaluate.py           walk-forward backtest and honest metrics
  totals.py             odds API client and line snapshots
  weather_forecast.py   kickoff weather forecasts
  upcoming.py           builds feature rows for unplayed games
  predictions.py        prediction table, grading, and file management

tests/                  pytest suite
download/               separate env for nfl_data_py extraction
docs/model.md           methodology and the model-selection evidence
```

The single most important structural rule: **training and serving share
`model_features.build_model_matrix`.** `upcoming.py` produces raw inputs in the same
units as `pipeline.to_game_frame`, and both go through the same builder. Adding a
feature means editing one file.

---

## Tests

```bash
python -m pytest tests -q
```

The suite concentrates on the things that fail silently: lagged rolling windows
leaking across team and season boundaries, home/away broadcasting duplicating rows,
probability and Kelly arithmetic, bet grading, and train/serve column drift.
