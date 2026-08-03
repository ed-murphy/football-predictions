"""Tests for the feature contract shared by training and inference."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import betting_record
from src.model_features import (
    CONTEXT_FEATURES, PER_SIDE_FEATURES, build_model_matrix, get_model_features,
    impute_weather,
)
from config import DECIMAL_PAYOUT


# Derived inside build_model_matrix, so a caller supplies their inputs instead.
_DERIVED_CONTEXT = {"abs_spread", "season_week"}


def _game_row(**overrides):
    row = {f"{side}_{feat}": 1.0
           for feat in PER_SIDE_FEATURES for side in ("home", "away")}
    row.update({c: 1.0 for c in CONTEXT_FEATURES if c not in _DERIVED_CONTEXT})
    row.update({"total_line": 44.5, "spread_line": -3.0, "week": 7, "is_dome": 0})
    row.update(overrides)
    return pd.DataFrame([row])


def test_feature_list_has_no_duplicates():
    features = get_model_features()
    assert len(features) == len(set(features))


def test_matrix_contains_every_declared_feature():
    matrix = build_model_matrix(_game_row())
    missing = set(get_model_features()) - set(matrix.columns)
    assert not missing


def test_matrix_is_entirely_numeric():
    matrix = build_model_matrix(_game_row())
    assert matrix[get_model_features()].dtypes.apply(
        lambda d: np.issubdtype(d, np.number)
    ).all()


def test_missing_column_is_reported_rather_than_silently_imputed():
    row = _game_row().drop(columns=["home_rolling_avg_qb_epa"])
    with pytest.raises(KeyError, match="home_rolling_avg_qb_epa"):
        build_model_matrix(row)


def test_abs_spread_is_symmetric():
    favoured_home = build_model_matrix(_game_row(spread_line=-7.0))["abs_spread"].iloc[0]
    favoured_away = build_model_matrix(_game_row(spread_line=7.0))["abs_spread"].iloc[0]
    assert favoured_home == favoured_away == 7.0


def test_combined_and_delta_features_use_both_sides():
    row = _game_row(home_rolling_avg_qb_epa=0.3, away_rolling_avg_qb_epa=0.1)
    matrix = build_model_matrix(row)
    assert matrix["combined_rolling_avg_qb_epa"].iloc[0] == pytest.approx(0.4)
    assert matrix["rolling_avg_qb_epa_delta"].iloc[0] == pytest.approx(0.2)


def test_weather_imputation_fills_from_seasonal_normals():
    df = pd.DataFrame({
        "week": [1, 1, 2],
        "is_dome": [0, 0, 0],
        "game_temp": [70.0, 80.0, np.nan],
        "game_wind": [5.0, 7.0, np.nan],
    })
    filled = impute_weather(df)
    assert filled["game_temp"].notna().all()
    assert filled["game_wind"].notna().all()


def test_weather_imputation_leaves_known_values_alone():
    df = pd.DataFrame({"week": [1, 2], "is_dome": [0, 0],
                       "game_temp": [70.0, 40.0], "game_wind": [5.0, 12.0]})
    filled = impute_weather(df)
    assert filled["game_temp"].tolist() == [70.0, 40.0]


def test_betting_record_counts_pushes_separately():
    signals = np.array([1, 1, 1])
    actual_over = np.array([True, False, False])
    push = np.array([False, False, True])

    record = betting_record(signals, actual_over, push)
    assert (record["wins"], record["losses"], record["pushes"]) == (1, 1, 1)
    assert record["win_rate"] == pytest.approx(0.5)


def test_betting_record_profit_uses_minus_110_pricing():
    record = betting_record(np.array([1, 1]), np.array([True, True]))
    assert record["profit"] == pytest.approx(2 * DECIMAL_PAYOUT)
    assert record["roi"] == pytest.approx(DECIMAL_PAYOUT)


def test_betting_record_handles_no_bets():
    record = betting_record(np.array([0, 0]), np.array([True, False]))
    assert record["n_bets"] == 0
    assert np.isnan(record["roi"])


def test_break_even_win_rate_is_flat():
    """52.38% of bets at -110 returns exactly the stake."""
    n = 10_000
    wins = round(n / (1 + DECIMAL_PAYOUT))
    signals = np.ones(n, dtype=int)
    actual_over = np.array([True] * wins + [False] * (n - wins))
    record = betting_record(signals, actual_over)
    assert record["roi"] == pytest.approx(0.0, abs=1e-3)
