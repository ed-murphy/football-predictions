"""The bet log: written once, graded later, never rewritten with hindsight."""
import numpy as np
import pandas as pd
import pytest

from src import ledger
from src.model import payout

NOW = pd.Timestamp("2026-11-13 12:00", tz="UTC")


def _slate(line=44.5, price=-110.0, bet="Under") -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": "2026_10_MIA_BUF", "kickoff_et": "Sun 2026-11-15 13:00",
        "away_team": "MIA", "home_team": "BUF", "book": "draftkings", "line": line,
        "under_price": price, "market_line": 44.5, "forecast_wind": 20.0,
        "expected_wind": 16.7, "p_under": 0.558, "ev": 0.065, "bet": bet, "stake_pct": 1.6,
    }])


def _final(total: float, closing_line=43.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": "2026_10_MIA_BUF", "is_final": True, "total_points": total,
        "total_line": closing_line, "wind": 18.0,
    }])


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ledger.LEDGER_COLUMNS)


def test_a_flagged_bet_is_logged_as_pending():
    log = ledger.add_new_bets(_empty(), _slate(), NOW)
    assert len(log) == 1
    assert log["result"].iloc[0] == "pending" and log["line"].iloc[0] == 44.5
    assert log["flagged_at"].iloc[0].startswith("2026-11-13")


def test_unflagged_games_are_not_logged():
    assert ledger.add_new_bets(_empty(), _slate(bet=""), NOW).empty


def test_a_bet_is_never_rewritten_by_a_later_run():
    """Saturday's better line must not replace the line that was available on Friday."""
    log = ledger.add_new_bets(_empty(), _slate(line=44.5), NOW)
    log = ledger.add_new_bets(log, _slate(line=47.5), NOW + pd.Timedelta(days=1))
    assert len(log) == 1 and log["line"].iloc[0] == 44.5


@pytest.mark.parametrize("total, result, profit", [
    (38.0, "win", payout(-110)),
    (51.0, "loss", -1.0),
])
def test_grading_an_under(total, result, profit):
    log = ledger.grade(ledger.add_new_bets(_empty(), _slate(), NOW), _final(total))
    assert log["result"].iloc[0] == result
    assert log["profit_units"].iloc[0] == pytest.approx(profit, abs=1e-3)
    assert log["actual_total"].iloc[0] == total


def test_landing_exactly_on_a_whole_number_line_is_a_push():
    log = ledger.grade(ledger.add_new_bets(_empty(), _slate(line=44.0), NOW), _final(44.0))
    assert log["result"].iloc[0] == "push" and log["profit_units"].iloc[0] == 0.0


def test_profit_uses_the_price_that_was_actually_available():
    log = ledger.grade(ledger.add_new_bets(_empty(), _slate(price=+105.0), NOW), _final(38.0))
    assert log["profit_units"].iloc[0] == pytest.approx(1.05)


def test_closing_line_value_is_positive_when_the_line_drops_after_an_under():
    log = ledger.grade(ledger.add_new_bets(_empty(), _slate(line=44.5), NOW),
                       _final(51.0, closing_line=42.5))
    assert log["closing_line"].iloc[0] == 42.5
    assert log["clv_points"].iloc[0] == pytest.approx(2.0)
    assert log["gamebook_wind"].iloc[0] == 18.0


def test_unfinished_games_stay_pending():
    not_final = _final(40.0).assign(is_final=False)
    log = ledger.grade(ledger.add_new_bets(_empty(), _slate(), NOW), not_final)
    assert log["result"].iloc[0] == "pending" and pd.isna(log["profit_units"].iloc[0])


def test_ledger_survives_a_round_trip_to_disk(tmp_path):
    path = str(tmp_path / "bets.csv")
    ledger.save_ledger(ledger.add_new_bets(_empty(), _slate(), NOW), path)
    reloaded = ledger.load_ledger(path)
    assert list(reloaded.columns) == ledger.LEDGER_COLUMNS
    assert reloaded["result"].iloc[0] == "pending"

    graded = ledger.grade(reloaded, _final(38.0))
    assert graded["result"].iloc[0] == "win"


def test_missing_ledger_file_is_an_empty_log(tmp_path):
    assert ledger.load_ledger(str(tmp_path / "nothing.csv")).empty


def test_summary_counts_units_and_closing_line_value():
    log = pd.DataFrame({
        "result": ["win", "loss", "push", "pending"],
        "profit_units": [0.909, -1.0, 0.0, np.nan],
        "clv_points": [1.0, -0.5, 0.5, np.nan],
    }).reindex(columns=ledger.LEDGER_COLUMNS)
    summary = ledger.summarise(log)
    assert (summary["wins"], summary["losses"], summary["pushes"], summary["pending"]) == (1, 1, 1, 1)
    assert summary["units"] == pytest.approx(-0.091)
    assert summary["avg_clv_points"] == pytest.approx(1 / 3)
    assert summary["beat_close_share"] == pytest.approx(2 / 3)
    assert "1-1-1" in ledger.describe(log)


def test_summary_of_an_empty_log():
    assert "no settled bets" in ledger.describe(_empty())
