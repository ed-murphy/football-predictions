"""Tests for the rolling helpers.

These guard the bug this refactor was built around: `groupby(k)[c].shift().rolling(n)`
applies the window to the whole shifted series rather than per group, so a team's
first game silently averaged in the previous team's results.
"""
import numpy as np
import pandas as pd
import pytest

from src.rolling import broadcast_home_away, lagged_rolling_mean


def _frame():
    return pd.DataFrame({
        "team": ["A"] * 4 + ["B"] * 4,
        "value": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
    })


def test_first_row_of_each_group_is_nan():
    result = lagged_rolling_mean(_frame(), "team", "value", window=3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[4]), "team B's first game must not see team A's data"


def test_window_does_not_span_groups():
    result = lagged_rolling_mean(_frame(), "team", "value", window=3)
    # B's second row averages only B's first game.
    assert result.iloc[5] == pytest.approx(100.0)
    # B's fourth row averages B's games 1-3, not any of A's.
    assert result.iloc[7] == pytest.approx((100 + 200 + 300) / 3)


def test_excludes_the_current_row():
    result = lagged_rolling_mean(_frame(), "team", "value", window=3)
    assert result.iloc[3] == pytest.approx((10 + 20 + 30) / 3)
    assert result.iloc[3] != pytest.approx((20 + 30 + 40) / 3)


def test_window_is_bounded():
    df = pd.DataFrame({"team": ["A"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = lagged_rolling_mean(df, "team", "value", window=2)
    assert result.iloc[4] == pytest.approx((3 + 4) / 2)


def test_multiple_group_columns():
    df = pd.DataFrame({
        "team": ["A", "A", "A", "A"],
        "season": [2023, 2023, 2024, 2024],
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    result = lagged_rolling_mean(df, ["team", "season"], "value", window=3)
    assert np.isnan(result.iloc[2]), "a new season starts a new window"
    assert result.iloc[3] == pytest.approx(30.0)


def _team_games():
    return pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "team": ["A", "B", "C", "D"],
        "is_home": [1, 0, 1, 0],
        "rating": [1.0, 2.0, 3.0, 4.0],
    })


def test_broadcast_puts_both_sides_on_both_rows():
    out = broadcast_home_away(_team_games(), "rating")
    g1 = out[out["game_id"] == "g1"]
    assert (g1["home_rating"] == 1.0).all()
    assert (g1["away_rating"] == 2.0).all()


def test_broadcast_does_not_add_rows():
    df = _team_games()
    assert len(broadcast_home_away(df, "rating")) == len(df)


def test_broadcast_rejects_unknown_column():
    with pytest.raises(KeyError):
        broadcast_home_away(_team_games(), "not_a_column")
