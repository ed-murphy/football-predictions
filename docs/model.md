# Methodology and model selection

This document records the evidence behind the modelling choices, so that they can be
argued with rather than taken on faith.

## The problem with the previous model

The original model was an `XGBClassifier` (500 trees, depth 5) fitted directly to the
binary label `total_points > total_line`, with no access to the line itself.

Holdout evaluation on the 2024 season:

```
LogLoss: 0.7989 (coin flip = 0.6931)   Brier: 0.2925   AUC: 0.500
Precision at p>0.55: 103/216 = 47.7%
```

A log loss meaningfully *above* `ln 2` means the model's confident predictions were
wrong more often than its unconfident ones — it was worse than useless. Three causes:

1. **The label throws away almost everything.** A game contributes one bit, the base
   rate is 48.2%, and there are ~2,700 usable rows. There is not enough information
   in 2,700 coin flips to fit 70 features through 500 trees.
2. **No regularisation that mattered.** `min_child_weight=5` on 2,700 rows permits
   leaves of five games. The model memorised the training seasons.
3. **The reported metric hid it.** "Precision at p>0.55 = 47.7%" reads like an
   accuracy score. The relevant benchmark is 52.38% — the break-even win rate at
   −110 — so 47.7% is a heavy loss, not a near miss.

## Framings compared

Walk-forward CV, test seasons 2018–2025, training from 2014 on the original feature
set. ROI is at −110 with a 0.55 probability filter.

| # | Framing | Log loss | AUC | ROI |
|---|---|---|---|---|
| A | XGB classifier on over/under (original) | 0.7968 | 0.519 | −1.4% |
| B | Heavily regularised XGB classifier | 0.6965 | 0.512 | −1.1% |
| C | L2 logistic regression | 0.6997 | 0.513 | −1.1% |
| D | XGB regression on total points | 0.7398 | 0.508 | −1.9% |
| E | XGB regression on total points, line as a feature | 0.7192 | 0.511 | −2.7% |
| F | Ridge on total points, line as a feature | 0.6990 | 0.527 | +4.1% |
| G | **Ridge on the residual (points − line)** | **0.6959** | **0.529** | +5.0% |

Two conclusions:

* **Every tree ensemble lost to a linear model.** Not marginally — the boosted
  models could not get log loss below the coin-flip null at all.
* **The positive ROIs are not evidence.** At n ≈ 800 bets the standard error of ROI
  is ±3.5%, so +5.0% is 1.4 sigma. It was not used to select the model; log loss and
  edge correlation were.

Framing G was adopted: predict the residual, convert to a probability through the
residual distribution.

## Feature additions that did not help

Tested on top of the final feature set, walk-forward 2019–2025, judged by
out-of-sample correlation between predicted and realised edge:

| Feature set | Edge correlation | MSE |
|---|---|---|
| base | +0.0650 | 172.02 |
| + high-wind indicator (≥15 mph) | +0.0650 | 172.01 |
| + wind² | +0.0664 | 171.98 |
| + freezing indicator (≤32 °F) | +0.0637 | 172.04 |
| + spread² | +0.0642 | 172.03 |
| + line × pace interaction | +0.0648 | 172.02 |
| + all of the above | +0.0639 | 172.02 |

Market baseline MSE (predict the line) is 172.76, so the model improves squared
error by 0.4%. None of the nonlinearities move it. They were left out — adding
features that change the fourth decimal place is noise mining.

## Data corrections made during the rewrite

* **Rolling windows leaked across groups.** The idiom
  `df.groupby(k)[c].shift().rolling(n).mean()` applies `rolling` to the *whole*
  shifted series, so windows spanned team and season boundaries. 11.5% of rows in
  `basic.py` carried wrong values — a team's first game averaged in the previous
  team's results. Fixed in `src/rolling.lagged_rolling_mean`, which uses
  `groupby(...).transform(...)`, and covered by tests.

* **Weather was the wrong quantity.** Historical weather came from Meteostat *daily
  averages* at the stadium's coordinates, while the forecast path fetched *kickoff*
  conditions — and converted them to Celsius and km/h while the training data was
  also metric, but neither matched what nflverse ships. `games.parquet` already
  carries kickoff `temp` (°F) and `wind` (mph) for outdoor games, null only for
  indoor games and ~5% of outdoor ones. Training now uses those, the forecast path
  serves Fahrenheit and mph to match, and gaps are filled from per-week seasonal
  normals rather than dropped.

* **Las Vegas was treated as an outdoor stadium.** The forecast module's dome list
  was keyed on full team names and still contained "Oakland Raiders", so Allegiant
  Stadium games had Las Vegas *outdoor* weather fetched for them. Dome membership now
  lives in `src/constants.DOME_TEAMS`, keyed on the abbreviations used everywhere else.

* **Rest days were recomputed from game dates** when `games.parquet` already ships
  correct `home_rest` / `away_rest` columns.

* **The odds cache accumulated duplicates.** Deduplication keyed on the exact
  `commence_time`, which drifts by minutes as the schedule firms up, leaving the same
  fixture in the cache several times and producing duplicate predictions. It now
  keys on the kickoff *date*, and prunes fixtures that kicked off over a week ago.

## Calibration

Predicted probabilities against realised over-rates, pooled over the backtest:

| Predicted | n | Actual |
|---|---|---|
| 0.480 | 435 | 0.455 |
| 0.486 | 434 | 0.491 |
| 0.494 | 435 | 0.467 |
| 0.500 | 434 | 0.505 |
| 0.515 | 435 | 0.522 |

Monotone and close to the diagonal, but the whole predicted range spans 0.48–0.52.
The model is well calibrated *and* almost completely uninformative — which is the
honest description of a public-data model against an efficient market.

## Shrinkage

The raw ridge prediction is multiplied by a coefficient estimated inside the training
window: walk forward one season at a time, regress the realised residual on the
predicted residual through the origin, and take the slope.

That slope is itself noisy, and taking it at face value made the model swing between
"bet nothing" and "bet everything" between adjacent folds. It is therefore shrunk
toward zero by its own reliability, `t² / (1 + t²)`, where `t` is the slope's
t-statistic. A slope measured with `t = 2.5` survives nearly intact; a coin-flip
slope collapses to zero and the model stops making claims.

This is what produces the empty bet columns in the 2018–2023 backtest folds: at that
point in the walk-forward there genuinely was not enough evidence that the edges
were real.
