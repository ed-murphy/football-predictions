"""The model: wind makes NFL games go under, by more than the market allows for.

That sentence is the whole model. It has two fitted numbers:

    log-odds(under) = intercept + slope * kickoff_wind_mph

Everything else the market already prices correctly, as far as 27 seasons of data
can tell — see docs/model.md for the 120 or so other signals that were tested and
failed. So rather than re-predict the total from scratch, the model takes the
market's line as the best available estimate and asks only how far wind should
tilt the odds of finishing under it.

The rest of this file is betting arithmetic: turning a probability and a price
into an expected value and a stake.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import KELLY_FRACTION, MAX_STAKE_PCT, STANDARD_ODDS

logger = logging.getLogger(__name__)


# ── Betting arithmetic ────────────────────────────────────────────────────────

def payout(american_odds: float) -> float:
    """Profit per unit staked on a winning bet: -110 -> 0.909, +120 -> 1.2."""
    odds = float(american_odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def break_even(american_odds: float = STANDARD_ODDS) -> float:
    """Win rate needed to break even at a price: 52.4% at -110."""
    return 1.0 / (1.0 + payout(american_odds))


def expected_value(p_win: float, american_odds: float = STANDARD_ODDS,
                   p_push: float = 0.0) -> float:
    """Expected profit per unit staked.

    `p_win` is the chance of winning given the bet is not a push. A push returns
    the stake, so it contributes nothing and just scales the rest down.
    """
    return (1.0 - p_push) * (p_win * payout(american_odds) - (1.0 - p_win))


def kelly_stake_pct(p_win: float, american_odds: float = STANDARD_ODDS) -> float:
    """Suggested stake as a percentage of bankroll (fractional Kelly, capped).

    Full Kelly assumes the win probability is known exactly. It is an estimate,
    so only `KELLY_FRACTION` of it is staked, and never more than `MAX_STAKE_PCT`.
    """
    b = payout(american_odds)
    full_kelly = (p_win * (b + 1.0) - 1.0) / b
    return round(float(np.clip(full_kelly * KELLY_FRACTION * 100.0, 0.0, MAX_STAKE_PCT)), 2)


# ── The model ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WindModel:
    """A fitted wind model plus the two market constants needed to price a bet."""

    intercept: float        # log-odds of the under in still air
    slope: float            # extra log-odds of the under per mph of wind
    slope_se: float         # standard error of the slope
    prob_per_point: float   # win probability one point of line is worth (~3%)
    push_rate: float        # chance a whole-number total lands exactly (~3%)
    n_games: int
    first_season: int
    last_season: int

    def p_under(self, wind_mph) -> np.ndarray:
        """P(game finishes under the market total), given the wind at kickoff."""
        wind = np.asarray(wind_mph, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.intercept + self.slope * wind)))

    def p_under_at(self, wind_mph: float, line: float, market_line: float) -> float:
        """P(under) at a specific book's line, which may differ from the market's.

        The model's probability is for the market line. A book hanging a total one
        point higher is offering an under that wins about 3% more often, and that
        is added here.
        """
        shift = (float(line) - float(market_line)) * self.prob_per_point
        return float(np.clip(self.p_under(wind_mph) + shift, 0.01, 0.99))

    def p_push(self, line: float) -> float:
        """Chance of a push: only whole-number lines can push."""
        return self.push_rate if float(line) == round(float(line)) else 0.0

    def wind_needed(self, american_odds: float = STANDARD_ODDS, min_ev: float = 0.0) -> float:
        """Wind speed at which the under reaches `min_ev` at the given price."""
        target = (1.0 + min_ev) / (1.0 + payout(american_odds))
        return float((np.log(target / (1.0 - target)) - self.intercept) / self.slope)

    def describe(self) -> str:
        probs = {mph: float(self.p_under(mph)) for mph in (0, 10, 15, 20)}
        return (
            f"Wind model fitted on {self.n_games:,} games ({self.first_season}-{self.last_season}): "
            f"each mph adds {self.slope:.4f} +/- {self.slope_se:.4f} to the log-odds of the under "
            f"(z = {self.slope / self.slope_se:.1f}). P(under): "
            + ", ".join(f"{p:.1%} at {mph} mph" for mph, p in probs.items())
            + f". Break-even at {STANDARD_ODDS}: {self.wind_needed():.1f} mph."
        )


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-variable logistic regression by Newton's method.

    Returns `(coefficients, standard_errors)`, each as [intercept, slope]. Written
    out rather than imported so the standard errors are visible: the size of the
    slope relative to its error is the evidence the whole project rests on.
    """
    X = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(2)

    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1.0 - p))[:, None])
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break

    return beta, np.sqrt(np.diag(np.linalg.inv(hessian)))


def fit_wind_model(completed: pd.DataFrame) -> WindModel:
    """Fit the model on completed games (see `schedule.completed_games`).

    Pushes are left out — nobody wins or loses them — as are games whose kickoff
    wind was never recorded.
    """
    usable = completed[~completed["push"] & completed["kickoff_wind"].notna()]
    if len(usable) < 500:
        raise ValueError(f"Only {len(usable)} usable games; need at least 500 to fit.")

    (intercept, slope), (_, slope_se) = fit_logistic(
        usable["kickoff_wind"].to_numpy(), usable["under"].to_numpy()
    )

    # How much probability one point of line is worth: the share of games landing
    # within 3 points of the total, spread over the 6.5 points that covers.
    residual = completed["residual"]
    prob_per_point = float((residual.abs() <= 3).mean() / 6.5)

    whole_number = completed["total_line"] % 1 == 0
    push_rate = float(completed.loc[whole_number, "push"].mean()) if whole_number.any() else 0.0

    return WindModel(
        intercept=float(intercept), slope=float(slope), slope_se=float(slope_se),
        prob_per_point=prob_per_point, push_rate=push_rate, n_games=len(usable),
        first_season=int(usable["season"].min()), last_season=int(usable["season"].max()),
    )
