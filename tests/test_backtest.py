"""The backtest: no peeking, correct grading, and honest handling of forecast error."""
import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.model import payout
from src.schedule import completed_games
from tests.synthetic import make_games


def test_each_season_is_bet_by_a_model_that_never_saw_it(games):
    for fold in backtest.make_folds(completed_games(games), start_season=2005):
        assert fold.model.last_season < fold.season
        assert (fold.games["season"] == fold.season).all()


def test_changing_a_seasons_results_cannot_change_its_own_bets(games):
    """The strongest leak test: rewrite 2010's outcomes and 2010's bets must not move."""
    tampered = games.copy()
    in_2010 = tampered["season"] == 2010
    tampered.loc[in_2010, "home_score"] = tampered.loc[in_2010, "home_score"] + 30
    tampered["total_points"] = tampered["home_score"] + tampered["away_score"]

    before = backtest.run_backtest(games, start_season=2005)
    after = backtest.run_backtest(tampered, start_season=2005)
    pick = lambda scored: scored.loc[scored["season"] == 2010, ["game_id", "p_under", "bet"]]
    pd.testing.assert_frame_equal(pick(before).reset_index(drop=True),
                                  pick(after).reset_index(drop=True))


def test_backtest_profits_from_a_planted_edge(games):
    result = backtest.record(backtest.run_backtest(games, start_season=2005))
    assert result["bets"] > 100
    assert result["roi"] > 0.03
    assert result["win_rate"] > 0.54


def test_backtest_does_not_invent_an_edge():
    no_edge = make_games(slope=0.0, seed=11)
    result = backtest.record(backtest.run_backtest(no_edge, start_season=2005))
    # With no real effect any bets placed should lose about the vig, not win.
    assert result["bets"] == 0 or result["roi"] < 0.03


def test_only_windy_open_air_games_are_bet(games):
    scored = backtest.run_backtest(games, start_season=2005)
    bets = scored[scored["bet"]]
    assert (bets["kickoff_wind"] > 0).all()
    assert bets["kickoff_wind"].min() > scored.loc[~scored["bet"], "kickoff_wind"].median()


def test_record_arithmetic():
    scored = pd.DataFrame({
        "bet": [True, True, True, True, False],
        "result": ["win", "win", "loss", "push", "win"],
        "profit": [payout(-110), payout(-110), -1.0, 0.0, payout(-110)],
        "profit_recorded_price": [1.0, 1.0, -1.0, 0.0, 1.0],
    })
    result = backtest.record(scored)
    assert (result["bets"], result["wins"], result["losses"], result["pushes"]) == (4, 2, 1, 1)
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["units"] == pytest.approx(2 * payout(-110) - 1)
    assert result["roi"] == pytest.approx((2 * payout(-110) - 1) / 4)
    assert result["roi_recorded_price"] == pytest.approx(0.25)


def test_record_with_no_bets_is_empty_not_an_error():
    result = backtest.record(pd.DataFrame({"bet": [False], "result": ["win"], "profit": [0.9],
                                           "profit_recorded_price": [0.9]}))
    assert result["bets"] == 0 and np.isnan(result["roi"])


def test_simulated_forecast_is_shrunk_and_leaves_roofed_games_alone():
    rng = np.random.default_rng(0)
    wind = np.clip(rng.normal(8.5, 5.0, 20_000), 0, None)
    open_air = np.ones_like(wind)
    seen = backtest.simulate_forecast(wind, open_air, mean=wind.mean(), sd=wind.std(),
                                      correlation=0.7, rng=rng)

    # A forecast-based estimate is pulled toward the average...
    assert seen[wind > 20].mean() < wind[wind > 20].mean()
    assert seen[wind < 2].mean() > wind[wind < 2].mean()
    assert seen.std() < wind.std()
    # ...and tracks the truth about as well as the stated correlation says.
    assert np.corrcoef(seen, wind)[0, 1] == pytest.approx(0.7, abs=0.03)

    roofed = backtest.simulate_forecast(wind, np.zeros_like(wind), 8.5, 5.0, 0.7, rng)
    assert (roofed == 0).all()


def test_forecast_error_costs_money_but_does_not_erase_a_real_edge(games):
    perfect = backtest.record(backtest.run_backtest(games, start_season=2005))
    noisy = backtest.forecast_stress_test(games, start_season=2005, draws=20)
    assert 0 < noisy["roi"] < perfect["roi"]
    assert noisy["win_rate"] < perfect["win_rate"]


def test_a_useless_forecast_cannot_tell_games_apart():
    """With no forecast skill, every open-air game looks like an average one."""
    wind = np.array([0.0, 8.0, 30.0])
    seen = backtest.simulate_forecast(wind, np.ones(3), mean=8.5, sd=5.0, correlation=0.0,
                                      rng=np.random.default_rng(0))
    assert seen == pytest.approx([8.5, 8.5, 8.5])


def test_calibration_bands_are_roughly_honest(games):
    table = backtest.calibration(backtest.run_backtest(games, start_season=2005))
    big = table[table["games"] >= 300]
    assert (abs(big["predicted"] - big["actual"]) < 0.06).all()
