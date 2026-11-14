# Methodology and evidence

This document records why the model is what it is, what was tried and thrown
away, and where the result could be wrong. Every table can be regenerated with
the scripts in `research/`. Figures are as of November 2026.

## 1. The starting point: a model with no edge

The previous version of this project predicted NFL totals from about eighty
features — team scoring form, pace, EPA per play, quarterback form, injuries,
rest, referee tendencies — with a heavily regularised ridge regression. Its
walk-forward result against the closing line was a log loss of 0.6924 against
0.6931 for a coin flip, and −3.6% ± 9.7% on 97 bets. No edge.

That was an honest result, and it pointed at the real problem. The model was
asked to learn eighty weak effects at once from about 2,700 games, so the
regularisation needed to stop it fitting noise also flattened anything real. The
question worth asking was not "how do I tune this?" but "is there *any* single
thing the closing line gets wrong, and can it be shown with enough data to
believe?"

## 2. The test every signal had to pass

More data and a stricter standard:

- **27 seasons instead of 12.** nflverse publishes closing totals and spreads back
  to 1999 — about 7,400 games.
- **Split-sample confirmation.** Each candidate is scored on a discovery era
  (1999–2013) and again on a confirmation era (2014–2026). It counts only if it
  clears |z| > 1.96 in *both*, with the same sign.

The second rule matters because of how many things were tested. Try 120 ideas and
six will look significant in any one era by luck alone. The chance of one idea
passing both eras in the same direction by luck is about 1 in 800, so across 120
candidates roughly 0.15 false passes are expected in total.

## 3. What was tested

About 120 candidates in three groups.

### Schedule-based signals — `python -m research.signal_scan`

Twenty-three for totals and twenty-five for spreads: weather, roof and surface,
the size of the line, divisional games, week of season, primetime, Thursday and
Monday games, byes and short rest, neutral sites, and each team's recent results
against the number.

| | Passed |
|---|---|
| Totals (23) | **wind** (z = −3.4 discovery, −2.7 confirmation) |
| Spreads (25) | home underdogs of 7+ (z = +2.5, +2.1) — rejected, see below |

Home underdogs of a touchdown or more covered 56.6% in 1999–2013, **44.8% in
2014–2019**, and 62.1% since 2020. A signal that reverses for six years is not one
to bet, and one false pass among 120 candidates is about what chance predicts.

Moneylines show the familiar favourite–longshot pattern (moderate underdogs
slightly underpriced), shrinking from +13% in 2006–13 to +1% since 2020. Not
bettable.

### Play-by-play team form — `python -m research.play_by_play_scan`

Sixteen statistics (EPA per play, success rate, pass rate, explosive-play rate,
turnovers, sacks, tempo, and their defensive counterparts), each averaged over a
team's last 5 and last 12 games and combined across the two teams. Sixty-eight
candidates, 136 z-scores.

| | z-score standard deviation | Share beyond ±1.96 |
|---|---|---|
| Totals (72 z-scores) | 1.04 | 8% |
| Spreads (64 z-scores) | 1.07 | 6% |
| Pure noise | 1.00 | 5% |

These are the features the old model was built on, and as a group they are
indistinguishable from random numbers. The only one that passed was wind
multiplied by pass rate — wind again. The closing line already contains
everything team form knows. This is why the play-by-play pipeline was removed.

A scores-only team ratings model (opponent-adjusted offence and defence ratings)
was also tried against both totals and spreads. Where it disagreed with the line,
the line was right.

### Weather from an independent source — `python -m research.reanalysis_check`

See section 5.

## 4. The wind effect

Blind unders in open-air games, by the wind recorded in the gamebook at kickoff,
1999–2026, graded at −110:

| Wind | Games | Under wins | ROI |
|---|---|---|---|
| 0–5 mph | 1,057 | 50.0% | −4.6% |
| 5–10 mph | 2,225 | 48.2% | −7.9% |
| 10–13 mph | 879 | 53.6% | +2.3% |
| 13–15 mph | 329 | 54.4% | +3.9% |
| 15–18 mph | 385 | 55.1% | +5.1% |
| 18+ mph | 289 | 59.5% | +13.6% |
| Indoors | 1,833 | 48.6% | −7.3% |

The more wind, the more unders: a dose–response, not a single lucky bucket.

By era, for 13 mph or more:

| Era | Games | Under wins | ROI |
|---|---|---|---|
| 1999–2007 | 372 | 54.6% | +4.2% |
| 2008–2013 | 244 | 55.3% | +5.6% |
| 2014–2019 | 195 | 57.4% | +9.7% |
| 2020–2026 | 192 | 58.9% | +12.4% |
| **All** | **1,003** | **56.1%** | **+7.2% ± 3.0%** |

**Why it exists.** The market does adjust: the average closing total in games with
13+ mph of wind is 42.1, against 43.1 in calm ones (under 8 mph) — one point lower. But those
windy games actually average 41.2 points against 44.4 — three points lower. Books
move the number about a third as far as they should. The usual explanation is
that the public bets overs, so a book has no commercial reason to post a total
low enough to invite more over money.

**It is the weather, not the stadium.** Some stadiums are simply windier, so the
effect could have been "unders in Cleveland and Buffalo". Splitting each game's
wind into its stadium's average and that day's departure from it, the departure
carries the effect (−0.18 points per mph, z = 5.1) and the stadium average adds
nothing significant.

## 5. Is the gamebook wind trustworthy?

The model is fitted on the wind written into the official gamebook. Two worries:
that field might be measured in a way that is not knowable before kickoff, or it
might be subtly shaped by the game itself.

`research/reanalysis_check.py` sets the gamebook aside and uses ERA5 reanalysis:
a weather model's reconstruction of the wind at the stadium's coordinates for the
hours the game was played, made with no knowledge that a game was on.

Open-air home games only, so the gamebook row differs slightly from section 3:

| Signal | z, 1999–2013 | z, 2014–2026 |
|---|---|---|
| Gamebook wind | −2.6 | −3.3 |
| Reanalysis wind | −2.8 | −4.0 |
| Reanalysis gusts | −2.5 | −3.4 |

The two measurements correlate at only 0.66, yet the independent one predicts
unders at least as well (z = 4.7 against 4.1 over the same 4,914 games). Unders hit 57% in the 1,252 games with 10+ mph of
reanalysis wind. The effect is real weather, not a recording artefact.

### Rain: tested, promising, not used

The same script tests rainfall, which gamebooks do not record. Games with 1 mm or
more of rain went under 58.0% of the time (564 games) and it passes the
split-sample rule. It is not in the model, for two reasons:

- **It adds nothing out of sample.** Walk-forward from 2008, wind alone returns
  +6.0% on 697 bets; wind plus rain returns +5.1% on 725. Wet games are mostly
  windy games.
- **Rain forecasts are poor.** When rain is forecast one to three days out, the
  game turns out wet only half to two-thirds of the time; forecast and actual
  rainfall correlate at 0.4–0.5, against 0.70–0.74 for wind.

It is the most likely next addition if someone wants to extend the model.

## 6. From forecast to bet

### Forecasts are not the gamebook — `python -m research.forecast_calibration`

Open-Meteo archives its past forecasts from 2024. For 405 outdoor games since
then, comparing what was forecast with what the gamebook recorded:

| Days before kickoff | Correlation | Expected gamebook wind |
|---|---|---|
| 1 | 0.74 | 1.83 + 0.742 × forecast |
| 2 | 0.71 | 2.05 + 0.704 × forecast |
| 3 | 0.70 | 2.28 + 0.682 × forecast |
| 4 | 0.66 | 2.35 + 0.670 × forecast |
| 5 | 0.58 | 3.00 + 0.611 × forecast |
| 6 | 0.47 | 4.04 + 0.472 × forecast |
| 7 | 0.33 | 4.97 + 0.361 × forecast |

Forecasts hold up for three days and decay quickly after five. So the raw
forecast is never given to the model; it is translated with the table above
first (`FORECAST_CALIBRATION` in `config.py`), and bets are only flagged within
three days of kickoff. In practice a bet needs a raw forecast of about 15 mph.

### What forecast error costs

The backtest's headline uses the true kickoff wind. `backtest.forecast_stress_test`
re-runs it 200 times with the wind known only through a simulated forecast of
realistic accuracy (correlation 0.72):

| | Bets a season | Win rate | ROI |
|---|---|---|---|
| Kickoff wind known | 28 | 56.5% | +7.7% |
| Forecast | 19 | 55.1% | +5.2% |

Forecast error costs about a third of the bets and a third of the edge per bet.

### Pricing a specific line

The model's probability is for the market's total, taken as the median across
every sportsbook in the feed. A book offering a different total is offering a
different bet: each point is worth 3.0% of win probability (measured as the share
of games landing within three points of the total), and a whole-number line has a
2.9% chance of a push. Expected value is computed at the best line among the
books listed in `MY_BOOKS`.

## 7. The backtest

`python main.py backtest`. Each season from 2008 is bet by a model fitted on
earlier seasons only, unders at −110 whenever expected value is at least 2%.

| Season | Bets | W–L–P | Units | | Season | Bets | W–L–P | Units |
|---|---|---|---|---|---|---|---|---|
| 2008 | 46 | 34–11–1 | +19.9 | | 2018 | 19 | 14–5–0 | +7.7 |
| 2009 | 40 | 21–19–0 | +0.1 | | 2019 | 27 | 12–15–0 | −4.1 |
| 2010 | 47 | 20–26–1 | −7.8 | | 2020 | 30 | 12–18–0 | −7.1 |
| 2011 | 32 | 18–14–0 | +2.4 | | 2021 | 34 | 22–12–0 | +8.0 |
| 2012 | 25 | 16–8–1 | +6.5 | | 2022 | 27 | 20–7–0 | +11.2 |
| 2013 | 21 | 11–10–0 | 0.0 | | 2023 | 27 | 18–7–2 | +9.4 |
| 2014 | 22 | 10–11–1 | −1.9 | | 2024 | 38 | 18–20–0 | −3.6 |
| 2015 | 24 | 14–9–1 | +3.7 | | 2025 | 30 | 14–16–0 | −3.3 |
| 2016 | 26 | 11–15–0 | −5.0 | | 2026* | 2 | 1–1–0 | −0.1 |
| 2017 | 23 | 15–8–0 | +5.6 | | **Total** | **540** | **301–232–7** | **+41.6** |

\* through week 10.

ROI +7.7% ± 4.1% (z = 1.9) at −110, +9.7% at the prices nflverse recorded.

The 2% threshold was chosen in advance, not tuned. Other thresholds, for
reference:

| Minimum EV | Bets | Win rate | ROI |
|---|---|---|---|
| 0% | 893 | 56.9% | +8.4% ± 3.1% |
| **2%** | **540** | **56.5%** | **+7.7% ± 4.1%** |
| 4% | 299 | 55.4% | +5.7% ± 5.5% |
| 6% | 155 | 57.8% | +10.3% ± 7.6% |

All positive, none sharply different — which is what a real effect looks like,
and also means there is no magic threshold to find.

The model is also calibrated: across all 4,696 backtest games, those it rated
52–54% to go under did so 56.0% of the time, 54–56% did 53.4%, and above 56% did
60.8%.

## 8. Where this could be wrong

- **The last three seasons lost** (33–37, −7.0 units). For 70 bets at a true 55%
  that is about a one-in-nine outcome: unremarkable, and not distinguishable
  from the start of the market catching up. The average line adjustment for wind
  has not visibly grown, but three seasons cannot settle it.
- **The backtest's own return is 1.9 standard errors from zero.** The case for the
  edge rests more on the z = 4.6 fitted slope, its stability, and the independent
  replication than on the profit figure.
- **Closing lines, not your lines.** Every result here is against the close. If
  you bet earlier and the market then moves down on the same forecast, you do
  better than the backtest; if your book has already shaded the under's price,
  worse. The bet log's closing-line value column measures which.
- **Forecast calibration rests on 405 games** from under three seasons, and on one
  provider's forecasts. Re-run `research/forecast_calibration.py` yearly.
- **Small numbers.** About 20 bets a season. A losing season is nearly as likely
  as a winning one even if everything here is right; the edge only shows over
  several years.
- **Researcher choices.** The functional form (a straight line in wind), the 2%
  threshold and the three-day window were each chosen once and not tuned, but
  they were chosen by someone who had seen the data.

## 9. What is deliberately absent

No team ratings, no quarterback adjustments, no injuries, no pace or efficiency
metrics. They were all tested (section 3) and the closing line already knows
them. Leaving them out is not a simplification of the model — it *is* the model:
trust the market on everything it demonstrably gets right, and disagree only
where there is evidence.

The previous implementation remains in git history (commit `c225912` and earlier).
