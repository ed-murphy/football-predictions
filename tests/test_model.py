"""Tests for the probability, threshold and staking logic."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.model import TotalsModel, estimate_shrinkage
from src.predictions import _grade, kelly_stake
from config import BREAK_EVEN_WIN_RATE


class _ConstantEstimator:
    """Stand-in that always predicts a fixed edge, so probabilities are checkable."""

    def __init__(self, edge):
        self.edge = edge

    def predict(self, X):
        return np.full(len(X), self.edge)


def _model(edge=0.0, sigma=13.0, shrinkage=1.0):
    return TotalsModel(
        estimator=_ConstantEstimator(edge), features=["total_line"], sigma=sigma,
        shrinkage=shrinkage, alpha=1.0, train_seasons=[2020], n_train=100,
    )


def _X(lines):
    return pd.DataFrame({"total_line": lines})


def test_no_edge_gives_a_coin_flip_on_a_half_point_line():
    p = _model(edge=0.0).predict_over_prob(_X([44.5]))
    assert p[0] == pytest.approx(0.5, abs=1e-9)


def test_integer_line_leaves_room_for_a_push():
    """On a whole-number line the over and under probabilities must not sum to 1."""
    model = _model(edge=0.0)
    p_over = model.predict_over_prob(_X([44.0]))[0]
    assert p_over < 0.5, "a push band has to come out of the over's share"


def test_positive_edge_raises_the_over_probability():
    assert _model(edge=5.0).predict_over_prob(_X([44.5]))[0] > 0.5
    assert _model(edge=-5.0).predict_over_prob(_X([44.5]))[0] < 0.5


def test_shrinkage_pulls_the_edge_toward_zero():
    full = _model(edge=4.0, shrinkage=1.0).predict_edge(_X([44.5]))[0]
    half = _model(edge=4.0, shrinkage=0.5).predict_edge(_X([44.5]))[0]
    assert half == pytest.approx(full / 2)


def test_predicted_total_is_line_plus_edge():
    model = _model(edge=3.0)
    assert model.predict_total(_X([44.5]))[0] == pytest.approx(47.5)


def test_break_even_edge_matches_the_vig():
    """An edge exactly at the threshold must imply exactly the break-even win rate."""
    model = _model(sigma=13.0)
    edge = model.break_even_edge
    implied = 1 - norm.cdf(-edge / model.sigma)
    assert implied == pytest.approx(BREAK_EVEN_WIN_RATE, abs=1e-6)


def test_break_even_edge_scales_with_uncertainty():
    assert _model(sigma=20.0).break_even_edge > _model(sigma=10.0).break_even_edge


def test_bet_signal_respects_the_threshold():
    model = _model(sigma=13.0)
    below = _model(edge=model.bet_threshold - 0.1, sigma=13.0)
    above = _model(edge=model.bet_threshold + 0.1, sigma=13.0)
    assert below.bet_signal(_X([44.5]))[0] == 0
    assert above.bet_signal(_X([44.5]))[0] == 1
    assert _model(edge=-(model.bet_threshold + 0.1), sigma=13.0).bet_signal(_X([44.5]))[0] == -1


def test_shrinkage_is_zero_when_predictions_are_noise():
    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    y = pd.Series(rng.normal(scale=13.0, size=n))
    seasons = pd.Series(np.repeat(np.arange(2015, 2027), n // 12))

    shrinkage, sigma = estimate_shrinkage(X, y, seasons, alpha=100.0,
                                          features=list(X.columns))
    assert 0.0 <= shrinkage < 0.2, "pure noise must not earn confidence"
    assert sigma == pytest.approx(13.0, rel=0.15)


def test_shrinkage_is_high_when_the_signal_is_real():
    rng = np.random.default_rng(1)
    n = 1200
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    y = pd.Series(4.0 * X["f0"] + rng.normal(scale=2.0, size=n))
    seasons = pd.Series(np.repeat(np.arange(2015, 2027), n // 12))

    shrinkage, _ = estimate_shrinkage(X, y, seasons, alpha=1.0, features=list(X.columns))
    assert shrinkage > 0.8


def test_kelly_is_zero_when_no_bet_is_flagged():
    assert kelly_stake(np.array([0.9]), np.array([0]))[0] == 0.0


def test_kelly_grows_with_confidence():
    signal = np.array([1, 1])
    stakes = kelly_stake(np.array([0.55, 0.65]), signal)
    assert stakes[1] > stakes[0] > 0


def test_kelly_is_zero_at_break_even():
    stake = kelly_stake(np.array([BREAK_EVEN_WIN_RATE]), np.array([1]))[0]
    assert stake == pytest.approx(0.0, abs=1e-3)


def test_kelly_uses_the_under_probability_for_under_bets():
    over_side = kelly_stake(np.array([0.65]), np.array([1]))[0]
    under_side = kelly_stake(np.array([0.35]), np.array([-1]))[0]
    assert over_side == pytest.approx(under_side)


@pytest.mark.parametrize("bet,line,actual,expected", [
    ("Over", 44.5, 50, "win"),
    ("Over", 44.5, 40, "loss"),
    ("Under", 44.5, 40, "win"),
    ("Under", 44.5, 50, "loss"),
    ("Over", 44.0, 44, "push"),
    ("Over", 44.5, None, "pending"),
    ("", 44.5, 50, ""),
])
def test_grading(bet, line, actual, expected):
    table = pd.DataFrame({"bet": [bet], "total_line": [line], "actual_total": [actual]})
    assert _grade(table).iloc[0] == expected
