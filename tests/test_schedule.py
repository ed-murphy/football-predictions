"""Loading the schedule file and deriving the columns the model reads."""
import numpy as np
import pandas as pd
import pytest

from src import schedule

_ROWS = [
    # game_id, gameday, gametime, home, away, home_score, away_score, roof, wind, total_line
    ("2025_10_MIA_BUF", "2025-11-09", "13:00", "BUF", "MIA", 20.0, 17.0, "outdoors", 18.0, 44.5),
    ("2025_10_CAR_NO", "2025-11-09", "13:00", "NO", "CAR", 30.0, 14.0, "dome", np.nan, 44.0),
    ("2025_10_SF_DAL", "2025-11-09", "16:25", "DAL", "SF", 24.0, 21.0, "open", np.nan, 47.5),
    ("2025_10_NE_CHI", "2025-11-09", "13:00", "CHI", "NE", 10.0, 9.0, "outdoors", np.nan, 40.5),
    ("2026_10_NYJ_BUF", "2026-11-15", "20:20", "BUF", "NYJ", np.nan, np.nan, "outdoors", np.nan, 43.5),
]


@pytest.fixture
def games(tmp_path) -> pd.DataFrame:
    frame = pd.DataFrame([{
        "game_id": gid, "season": int(gid[:4]), "week": 10, "gameday": day, "gametime": time,
        "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
        "location": "Home", "roof": roof, "stadium": "", "wind": wind, "total_line": line,
    } for gid, day, time, home, away, hs, as_, roof, wind, line in _ROWS])
    path = tmp_path / "games.csv"
    frame.to_csv(path, index=False)
    return schedule.load_games(str(path)).set_index("game_id")


def test_kickoff_is_converted_from_eastern_to_utc(games):
    assert games.loc["2025_10_MIA_BUF", "kickoff"] == pd.Timestamp("2025-11-09 18:00", tz="UTC")
    # 20:20 ET in mid-November is 01:20 UTC the next day.
    assert games.loc["2026_10_NYJ_BUF", "kickoff"] == pd.Timestamp("2026-11-16 01:20", tz="UTC")


def test_outdoor_games_carry_the_recorded_wind(games):
    assert games.loc["2025_10_MIA_BUF", "kickoff_wind"] == 18.0


def test_indoor_games_have_zero_wind(games):
    assert games.loc["2025_10_CAR_NO", "kickoff_wind"] == 0.0


def test_unrecorded_wind_stays_unknown_rather_than_becoming_calm(games):
    """An outdoor game with no reading, and an open retractable roof, are both unknown."""
    assert np.isnan(games.loc["2025_10_NE_CHI", "kickoff_wind"])
    assert np.isnan(games.loc["2025_10_SF_DAL", "kickoff_wind"])


def test_completed_games_have_residual_and_outcome(games):
    done = schedule.completed_games(games.reset_index()).set_index("game_id")
    assert "2026_10_NYJ_BUF" not in done.index
    assert done.loc["2025_10_MIA_BUF", "residual"] == pytest.approx(-7.5)
    assert done.loc["2025_10_MIA_BUF", "under"] == 1
    assert done.loc["2025_10_CAR_NO", "push"] and done.loc["2025_10_CAR_NO", "under"] == 0


def test_upcoming_games_are_unplayed_and_inside_the_window(games):
    frame = games.reset_index()
    now = pd.Timestamp("2026-11-13 12:00", tz="UTC")
    assert list(schedule.upcoming_games(frame, now, days=8)["game_id"]) == ["2026_10_NYJ_BUF"]
    assert schedule.upcoming_games(frame, now, days=1).empty
    after_kickoff = pd.Timestamp("2026-11-17", tz="UTC")
    assert schedule.upcoming_games(frame, after_kickoff, days=8).empty


def test_missing_file_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="main.py update|main.py"):
        schedule.load_games(str(tmp_path / "absent.csv"))


def test_a_bad_download_does_not_replace_a_good_cache(tmp_path, monkeypatch):
    path = tmp_path / "games.csv"
    path.write_text("good cache")

    class Truncated:
        content = b"game_id,season\n2026_01_A_B,2026\n"
        def raise_for_status(self): pass

    monkeypatch.setattr(schedule.requests, "get", lambda *a, **k: Truncated())
    with pytest.raises(ValueError, match="looks wrong"):
        schedule.download_games(path=str(path))
    assert path.read_text() == "good cache"
