"""The wind model and the betting arithmetic around it."""
import numpy as np
import pytest

from src.model import (
    WindModel, break_even, expected_value, fit_logistic, fit_wind_model, kelly_stake_pct, payout,
)
from src.schedule import completed_games
from tests.synthetic import TRUE_SLOPE, make_games


def _model(**overrides) -> WindModel:
    values = dict(intercept=-0.10, slope=0.02, slope_se=0.004, prob_per_point=0.03,
                  push_rate=0.03, n_games=7000, first_season=1999, last_season=2026)
    values.update(overrides)
    return WindModel(**values)


# ── Prices ────────────────────────────────────────────────────────────────────

def test_payout_for_favourite_and_underdog_prices():
    assert payout(-110) == pytest.approx(0.9091, abs=1e-4)
    assert payout(-200) == pytest.approx(0.5)
    assert payout(+120) == pytest.approx(1.2)


def test_break_even_at_standard_price():
    assert break_even(-110) == pytest.approx(0.5238, abs=1e-4)
    assert break_even(+100) == pytest.approx(0.5)


def test_a_bet_at_the_break_even_rate_has_zero_value():
    assert expected_value(break_even(-110), -110) == pytest.approx(0.0, abs=1e-12)


def test_expected_value_matches_hand_calculation():
    # 55% at -110: 0.55 * 0.9091 - 0.45 = +0.05
    assert expected_value(0.55, -110) == pytest.approx(0.05, abs=1e-3)


def test_a_possible_push_shrinks_value_toward_zero():
    assert 0 < expected_value(0.55, -110, p_push=0.03) < expected_value(0.55, -110)
    assert expected_value(0.50, -110, p_push=0.03) > expected_value(0.50, -110)


def test_kelly_stakes_nothing_without_an_edge():
    assert kelly_stake_pct(0.50, -110) == 0.0
    assert kelly_stake_pct(break_even(-110), -110) == 0.0


def test_kelly_is_fractional_and_capped():
    # Full Kelly at 55% / -110 is 5.5% of bankroll; a quarter of that is ~1.4%.
    assert kelly_stake_pct(0.55, -110) == pytest.approx(1.38, abs=0.02)
    assert kelly_stake_pct(0.90, -110) == 2.0   # MAX_STAKE_PCT


# ── Fitting ───────────────────────────────────────────────────────────────────

def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 25, 200_000)
    y = rng.random(200_000) < 1 / (1 + np.exp(-(-0.3 + 0.05 * x)))
    (intercept, slope), (_, slope_se) = fit_logistic(x, y)
    assert intercept == pytest.approx(-0.3, abs=0.02)
    assert slope == pytest.approx(0.05, abs=0.002)
    assert 0 < slope_se < 0.001


def test_fit_finds_the_planted_wind_effect(games):
    model = fit_wind_model(completed_games(games))
    assert abs(model.slope - TRUE_SLOPE) < 3 * model.slope_se
    assert model.slope / model.slope_se > 3


def test_fit_reports_no_effect_when_there_is_none():
    model = fit_wind_model(completed_games(make_games(slope=0.0, seed=5)))
    assert abs(model.slope) < 3 * model.slope_se


def test_pushes_and_unknown_wind_are_left_out_of_the_fit(games):
    completed = completed_games(games)
    model = fit_wind_model(completed)
    usable = (~completed["push"] & completed["kickoff_wind"].notna()).sum()
    assert model.n_games == usable


def test_fit_refuses_too_little_data(games):
    with pytest.raises(ValueError):
        fit_wind_model(completed_games(games).head(100))


def test_push_rate_and_point_value_are_measured_from_the_data(games):
    model = fit_wind_model(completed_games(games))
    assert 0.0 <= model.push_rate < 0.2
    assert 0.0 < model.prob_per_point < 0.2


# ── Predicting ────────────────────────────────────────────────────────────────

def test_more_wind_means_more_unders():
    p = _model().p_under([0, 5, 10, 15, 20, 30])
    assert np.all(np.diff(p) > 0)


def test_still_air_probability_comes_from_the_intercept():
    assert _model(intercept=0.0).p_under(0) == pytest.approx(0.5)


def test_wind_needed_is_where_value_turns_positive():
    model = _model()
    wind = model.wind_needed(-110, min_ev=0.0)
    assert expected_value(float(model.p_under(wind)), -110) == pytest.approx(0.0, abs=1e-9)
    assert model.wind_needed(-110, min_ev=0.02) > wind


def test_a_higher_line_is_a_better_under():
    model = _model()
    at_market = model.p_under_at(15, line=44.5, market_line=44.5)
    one_higher = model.p_under_at(15, line=45.5, market_line=44.5)
    assert at_market == pytest.approx(float(model.p_under(15)))
    assert one_higher == pytest.approx(at_market + 0.03)
    assert model.p_under_at(15, line=43.5, market_line=44.5) < at_market


def test_only_whole_number_lines_can_push():
    model = _model()
    assert model.p_push(44.0) == 0.03
    assert model.p_push(44.5) == 0.0
