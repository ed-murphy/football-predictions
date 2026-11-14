# NFL Windy Unders

A small betting model for NFL totals. It bets one thing: **the under, in open-air
games where a strong wind is forecast.**

Sportsbooks lower the total when it is going to be windy, but not by enough. That
is the only place, out of about 120 signals tested across 27 seasons, where the
closing line could be shown to be wrong.

---

## The result in one table

Walk-forward backtest, 2008–2026. Each season is bet by a model fitted only on
earlier seasons; bets are graded against the closing total at −110.

| | Bets | Record | Win rate | ROI |
|---|---|---|---|---|
| Kickoff wind known exactly | 540 (28 a season) | 301–232–7 | 56.5% | **+7.7% ± 4.1%** |
| With realistic forecast error | ~19 a season | — | 55.1% | **about +5%** |
| Break-even at −110 | | | 52.4% | 0% |

Plan around the second row. The first uses the wind recorded at kickoff, which
nobody knows when they place the bet.

### How much to trust it

**Reasons to believe it**

- The underlying effect is strong: z = 4.6 across 6,996 games, and the fitted
  size has barely moved since 2007.
- It holds in every era tested (1999–2007, 2008–13, 2014–19, 2020–26).
- It replicates with a completely independent wind measurement — weather-model
  reanalysis at the stadium's coordinates, z = 4.8 — so it is not a quirk of how
  gamebooks record wind.
- It has a plain physical cause, and it has been public knowledge among bettors
  for a decade without disappearing.

**Reasons for caution**

- **The last three seasons lost: 33–37, −7.0 units.** That is within normal
  variance for 70 bets, and it is also exactly what a fading edge would look
  like. There is no way to tell which yet.
- Only 10 of 19 backtest seasons were profitable. At ~25 bets a year a losing
  season is close to a coin flip even if the edge is real.
- The backtest's own ROI is 1.9 standard errors from zero at the chosen
  threshold — suggestive, not proof.
- It is graded against closing lines. Your results depend on the line you
  actually get, which is why the bet log tracks closing-line value.

This is a small, real-looking edge with wide error bars. It is not a money
machine, and most weeks it will tell you to do nothing.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Create `.env` in the repository root with a free key from
[the-odds-api.com](https://the-odds-api.com):

```
API_KEY_TOTALS=<your key>
```

The key is optional. Without it the model uses the single line listed in the
nflverse schedule file, which is enough to see the board but not to shop prices.
Weather comes from [Open-Meteo](https://open-meteo.com), which needs no key.

## Weekly workflow

```bash
python main.py
```

That one command refreshes scores and schedules, pulls current lines and wind
forecasts, prints the board, and logs any new bet. Run it Thursday for the
Thursday game and again Friday or Saturday for the weekend — bets are only
flagged within three days of kickoff, when forecasts are reliable.

| Command | What it does |
|---|---|
| `python main.py` | weekly run |
| `python main.py backtest` | full walk-forward report |
| `python main.py record` | every bet flagged so far, graded |
| `streamlit run app.py` | the same board and record in a browser |
| `python main.py --no-odds-api` | weekly run without spending Odds API quota |

### Reading the board

```
        kickoff (ET)      game  line  fcst mph  exp mph  P(under)  EV %  status
Sun 2026-11-15 13:00 HOU @ CLE  39.5       9.0      8.5      49.9  -2.6  not windy enough
Sun 2026-11-15 13:00  CAR @ NO  48.5         -      0.0      47.4  -7.4  roof: wind can't reach the field
```

| Column | Meaning |
|---|---|
| `line` | best under available at your sportsbooks (`MY_BOOKS` in `config.py`) |
| `fcst mph` | raw wind forecast for the kickoff hour |
| `exp mph` | what the gamebook is expected to record, after allowing for forecast error — this is what the model uses |
| `P(under)` | probability the game finishes under that line |
| `EV %` | expected profit per unit staked at the quoted price |
| `status` | why it is or isn't a bet |

A bet needs an open-air stadium, kickoff within three days, and EV of at least
2% — which works out to an expected wind of about 13 mph, or a raw forecast of
about 15.

### The bet log

`predictions/bets.csv` records each bet once, the first time it is flagged, with
the line and price available at that moment. Later runs only fill in the result.
It also records the **closing line**: if the line keeps dropping after you bet an
under, you beat the market, and that shows whether the edge is real long before
the win–loss record can.

---

## How it works

```
log-odds(under) = −0.103 + 0.0187 × kickoff wind in mph
```

| Wind at kickoff | P(under) |
|---|---|
| Indoors / still | 47.4% |
| 10 mph | 52.1% |
| 15 mph | 54.4% |
| 20 mph | 56.8% |

Those two numbers are the entire model. It does not try to predict the score:
the market's total is taken as the best estimate available, and the model only
asks how far wind should tilt the odds of finishing under it.

Three things turn that into a bet:

1. **Forecast translation.** A forecast is a noisy estimate of game-day wind, so
   it is shrunk toward normal before use, and shrunk harder the further out the
   game is. A 20 mph forecast the day before becomes about 17 mph expected; the
   same forecast five days out becomes about 15.
2. **Line shopping.** Every sportsbook in the odds feed sets the market line
   (their median). The bet is quoted at whichever of *your* books offers the
   highest total, and each extra point is worth about 3% of win probability.
3. **Stake.** Quarter-Kelly, capped at 2% of bankroll.

The evidence for each of these, and everything that was tried and rejected, is in
[`docs/model.md`](docs/model.md).

---

## Configuration

Everything tunable is in [`config.py`](config.py).

| Setting | Default | Effect |
|---|---|---|
| `MY_BOOKS` | `["draftkings"]` | sportsbooks you can bet at; add more to shop lines |
| `MIN_EV` | `0.02` | minimum expected value to flag a bet |
| `BET_WINDOW_DAYS` | `3` | how close to kickoff a bet may be flagged |
| `KELLY_FRACTION` | `0.25` | share of full Kelly staked |
| `MAX_STAKE_PCT` | `2.0` | cap on any one stake, % of bankroll |
| `FORECAST_CALIBRATION` | measured | forecast → expected wind, by days out |

---

## Project layout

```
main.py                 command line: weekly run, backtest, record
config.py               every tunable
app.py                  Streamlit front end

src/
  schedule.py           downloads and loads the nflverse schedule file
  venues.py             where each game is played and whether it is open-air
  model.py              the wind model and the betting arithmetic
  backtest.py           walk-forward backtest and forecast stress test
  odds.py               live totals from every sportsbook
  weather.py            wind forecasts, translated to expected kickoff wind
  slate.py              builds the weekly board and decides what is a bet
  ledger.py             the permanent bet log: results and closing-line value
  constants.py          team codes and stadium coordinates

research/               scripts that reproduce the evidence (not used by the weekly run)
  signal_scan.py            ~50 schedule-based signals vs closing totals and spreads
  play_by_play_scan.py      68 team-form signals from play-by-play
  reanalysis_check.py       the wind effect with independent weather data; rain
  forecast_calibration.py   how well forecasts predict game-day wind

tests/                  pytest suite
docs/model.md           methodology, evidence, and what was rejected
predictions/            slate.csv (this week), bets.csv (the log), backtest.csv
```

## Testing a new idea

Add a line to `totals_candidates` or `spread_candidates` in
`research/signal_scan.py` and run `python -m research.signal_scan`. A signal
counts only if it clears |z| > 1.96 in both 1999–2013 and 2014–present with the
same sign. That rule is what separates wind from the dozens of plausible ideas
that looked good in one era and vanished in the other.

## Tests

```bash
python -m pytest tests -q
```

The suite uses a synthetic league with a planted wind effect, so it can check
things real data cannot: that the model recovers a known effect, that it finds
nothing when nothing is there, and that changing a season's results cannot
change the bets placed in that season.

---

*A personal research project, not betting advice.*
