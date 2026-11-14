"""Odds parsing, forecast translation, and the weekly board built from them."""
import numpy as np
import pandas as pd
import pytest

from config import BET_WINDOW_DAYS, FORECAST_CALIBRATION
from src import odds, weather
from src.model import WindModel
from src.schedule import upcoming_games
from src.slate import best_under, build_slate
from tests.synthetic import prepare

NOW = pd.Timestamp("2026-11-13 12:00", tz="UTC")   # a Friday
MODEL = WindModel(intercept=-0.10, slope=0.02, slope_se=0.004, prob_per_point=0.03,
                  push_rate=0.03, n_games=7000, first_season=1999, last_season=2026)


def _schedule() -> pd.DataFrame:
    """Three unplayed games: outdoors on Sunday, a dome, and outdoors far in the future."""
    rows = [
        ("2026_10_MIA_BUF", "2026-11-15", "BUF", "MIA", "outdoors", "Highmark Stadium"),
        ("2026_10_CAR_NO", "2026-11-15", "NO", "CAR", "dome", "Caesars Superdome"),
        ("2026_11_NE_CHI", "2026-11-19", "CHI", "NE", "outdoors", "Soldier Field"),
    ]
    return prepare(pd.DataFrame([{
        "game_id": gid, "season": 2026, "week": 10, "gameday": day, "gametime": "13:00",
        "home_team": home, "away_team": away, "home_score": np.nan, "away_score": np.nan,
        "location": "Home", "roof": roof, "stadium": stadium, "wind": np.nan,
        "total_line": 44.5, "under_odds": -110.0, "over_odds": -110.0,
    } for gid, day, home, away, roof, stadium in rows]))


def _forecast(mph: float):
    """A stand-in forecast service that reports the same wind for every hour."""
    def fetch(lat, lon):
        hours = pd.date_range("2026-11-13", periods=24 * 9, freq="h", tz="UTC")
        return pd.DataFrame({"wind_mph": mph, "gust_mph": mph * 1.6, "temp_f": 40.0,
                             "precip_chance": 10.0}, index=hours)
    return fetch


def _payload(lines: dict[str, float], home="Buffalo Bills", away="Miami Dolphins") -> list[dict]:
    return [{
        "home_team": home, "away_team": away, "commence_time": "2026-11-15T18:00:00Z",
        "bookmakers": [{
            "key": book,
            "markets": [{"key": "totals", "outcomes": [
                {"name": "Over", "point": line, "price": -110},
                {"name": "Under", "point": line, "price": -110},
            ]}],
        } for book, line in lines.items()],
    }]


# ── Odds ──────────────────────────────────────────────────────────────────────

def test_offers_are_parsed_one_row_per_book():
    offers = odds.parse_offers(_payload({"draftkings": 44.5, "fanduel": 45.5}))
    assert list(offers["book"]) == ["draftkings", "fanduel"]
    assert list(offers["line"]) == [44.5, 45.5]
    assert offers["home_team"].iloc[0] == "BUF" and offers["away_team"].iloc[0] == "MIA"


def test_unknown_teams_and_books_without_totals_are_skipped():
    payload = _payload({"draftkings": 44.5}, home="London Monarchs")
    assert odds.parse_offers(payload).empty

    no_totals = _payload({"draftkings": 44.5})
    no_totals[0]["bookmakers"][0]["markets"][0]["key"] = "spreads"
    assert odds.parse_offers(no_totals).empty


def test_offers_match_the_schedule_on_teams_and_eastern_date():
    upcoming = upcoming_games(_schedule(), NOW, days=8)
    # A 20:20 ET Sunday kickoff is already Monday in UTC; it must still match Sunday's game.
    late = odds.parse_offers(_payload({"draftkings": 44.5}))
    late["kickoff"] = pd.Timestamp("2026-11-16 01:20", tz="UTC")
    assert list(odds.attach_game_ids(late, upcoming)["game_id"]) == ["2026_10_MIA_BUF"]

    wrong_week = late.assign(kickoff=pd.Timestamp("2026-12-20 18:00", tz="UTC"))
    assert odds.attach_game_ids(wrong_week, upcoming).empty


def test_market_line_is_the_median_across_books():
    offers = odds.parse_offers(_payload({"a": 44.5, "b": 44.5, "c": 46.5})).assign(game_id="g")
    assert odds.market_lines(offers)["g"] == 44.5


def test_line_history_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "history.csv"
    offers = odds.parse_offers(_payload({"draftkings": 44.5}))
    odds.append_line_history(offers, NOW, str(path))
    odds.append_line_history(offers, NOW + pd.Timedelta(hours=1), str(path))
    history = pd.read_csv(path)
    assert len(history) == 2 and history["fetched_at"].nunique() == 2


def test_missing_api_key_falls_back_quietly(monkeypatch):
    monkeypatch.delenv("API_KEY_TOTALS", raising=False)
    monkeypatch.setattr(odds, "load_dotenv", lambda: None)
    assert odds.fetch_offers().empty


# ── Forecast translation ──────────────────────────────────────────────────────

def test_forecast_is_translated_not_used_raw():
    intercept, slope = FORECAST_CALIBRATION[1]
    assert weather.expected_kickoff_wind(20.0, 1) == pytest.approx(intercept + slope * 20.0)
    assert weather.expected_kickoff_wind(20.0, 1) < 20.0


def test_older_forecasts_are_trusted_less():
    estimates = [weather.expected_kickoff_wind(25.0, d) for d in sorted(FORECAST_CALIBRATION)]
    assert estimates == sorted(estimates, reverse=True)


def test_no_estimate_beyond_the_calibrated_horizon_or_without_a_forecast():
    assert weather.expected_kickoff_wind(25.0, max(FORECAST_CALIBRATION) + 1) is None
    assert weather.expected_kickoff_wind(None, 1) is None


def test_lead_days_round_up():
    kickoff = pd.Timestamp("2026-11-15 18:00", tz="UTC")
    assert weather.lead_days(kickoff, kickoff - pd.Timedelta(hours=30)) == 2
    assert weather.lead_days(kickoff, kickoff - pd.Timedelta(hours=3)) == 1


def test_open_meteo_payload_is_read_at_the_kickoff_hour():
    payload = {"hourly": {"time": ["2026-11-15T17:00", "2026-11-15T18:00"],
                          "wind_speed_10m": [5.0, 17.0], "wind_gusts_10m": [9.0, 30.0],
                          "temperature_2m": [40.0, 41.0], "precipitation_probability": [0, 20]}}
    at_kickoff = weather.forecast_at_kickoff(
        weather.parse_hourly(payload), pd.Timestamp("2026-11-15 18:25", tz="UTC"))
    assert at_kickoff["wind_mph"] == 17.0
    assert weather.forecast_at_kickoff(
        weather.parse_hourly(payload), pd.Timestamp("2026-11-20 18:00", tz="UTC")) == {}


# ── The board ─────────────────────────────────────────────────────────────────

def _slate(mph: float, lines=None) -> pd.DataFrame:
    offers = odds.parse_offers(_payload(lines or {"draftkings": 44.5}))
    return build_slate(_schedule(), MODEL, offers, NOW, _forecast(mph)).set_index("game_id")


def test_a_windy_open_air_game_inside_the_window_is_a_bet():
    game = _slate(mph=25.0).loc["2026_10_MIA_BUF"]
    assert game["bet"] == "Under" and game["stake_pct"] > 0
    assert game["book"] == "draftkings" and game["line"] == 44.5
    assert game["expected_wind"] < game["forecast_wind"]
    assert game["status"].startswith("BET")


def test_a_calm_game_is_not_a_bet():
    game = _slate(mph=4.0).loc["2026_10_MIA_BUF"]
    assert game["bet"] == "" and game["stake_pct"] == 0 and game["ev"] < 0


def test_a_roofed_game_is_never_a_bet_however_windy_the_city():
    game = _slate(mph=40.0).loc["2026_10_CAR_NO"]
    assert game["bet"] == "" and game["expected_wind"] == 0
    assert np.isnan(game["forecast_wind"])


def test_a_windy_game_outside_the_window_is_watched_not_bet():
    game = _slate(mph=40.0).loc["2026_11_NE_CHI"]
    assert game["days_out"] > BET_WINDOW_DAYS
    assert game["bet"] == "" and game["status"].startswith("watch")


def test_games_missing_from_the_odds_feed_use_the_schedule_line():
    game = _slate(mph=40.0).loc["2026_11_NE_CHI"]
    assert game["book"] == "nflverse" and game["line"] == 44.5


def test_only_your_books_are_quoted_but_every_book_sets_the_market():
    # The market is 46.5; draftkings hangs 44.5, two points worse for an under.
    game = _slate(mph=25.0, lines={"draftkings": 44.5, "fanduel": 46.5, "betmgm": 46.5}
                  ).loc["2026_10_MIA_BUF"]
    assert game["market_line"] == 46.5
    assert game["book"] == "draftkings" and game["line"] == 44.5
    fair = float(MODEL.p_under(game["expected_wind"]))
    assert game["p_under"] == pytest.approx(fair - 2 * 0.03, abs=1e-3)


def test_best_under_prefers_the_higher_line_and_better_price():
    offers = pd.DataFrame({
        "book": ["draftkings", "draftkings2", "draftkings3"],
        "line": [44.5, 45.5, 45.5], "under_price": [-110.0, -110.0, -105.0],
    })
    import src.slate as slate
    original, slate.MY_BOOKS = slate.MY_BOOKS, []          # allow every book for this test
    try:
        quote = best_under(MODEL, 18.0, 44.5, offers)
    finally:
        slate.MY_BOOKS = original
    assert quote["book"] == "draftkings3" and quote["line"] == 45.5


def test_no_forecast_means_no_bet_and_says_why():
    offers = odds.parse_offers(_payload({"draftkings": 44.5}))
    slate = build_slate(_schedule(), MODEL, offers, NOW, lambda lat, lon: pd.DataFrame())
    game = slate.set_index("game_id").loc["2026_10_MIA_BUF"]
    assert game["bet"] == "" and "no forecast" in game["status"]
