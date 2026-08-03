"""End-to-end pipeline tests on a small synthetic season.

The point of these is train/serve drift: the historical path and the upcoming-game
path must produce the same columns with the same meaning. A unit test on either
half alone would not notice them diverging.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_features, to_game_frame
from src.model_features import get_model_features
from src.train import build_training_data

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
N_WEEKS = 8


def _synthetic_games(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2023, 2024):
        for week in range(1, N_WEEKS + 1):
            for home, away in [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])]:
                if week % 2 == 0:
                    home, away = away, home
                rows.append({
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season, "week": week, "game_type": "REG",
                    "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(weeks=week - 1),
                    "home_team": home, "away_team": away,
                    "home_score": rng.integers(10, 35), "away_score": rng.integers(10, 35),
                    "total_line": 44.5, "spread_line": rng.choice([-3.0, 3.0, -7.0]),
                    "div_game": 1, "roof": "outdoors", "stadium_id": "AAA00",
                    "temp": 60.0, "wind": 8.0, "home_rest": 7, "away_rest": 7,
                    "referee": "Ref One",
                })
    return pd.DataFrame(rows)


def _synthetic_plays(games, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _, game in games.iterrows():
        for team in (game["home_team"], game["away_team"]):
            defteam = game["away_team"] if team == game["home_team"] else game["home_team"]
            for play_id in range(60):
                is_pass = play_id % 2 == 0
                rows.append({
                    "game_id": game["game_id"], "play_id": play_id,
                    "posteam": team, "defteam": defteam,
                    "play_type": "pass" if is_pass else "run",
                    "pass_attempt": int(is_pass), "rush_attempt": int(not is_pass),
                    "qb_dropback": int(is_pass), "sack": 0,
                    "epa": rng.normal(), "success": int(rng.random() > 0.5),
                    "yards_gained": rng.integers(-2, 25), "cpoe": rng.normal(),
                    "passer_player_name": f"QB.{team}" if is_pass else None,
                    "rusher_player_name": None if is_pass else f"RB.{team}",
                    "yardline_100": rng.integers(1, 99), "drive": play_id // 6 + 1,
                    "touchdown": int(rng.random() > 0.9),
                    "interception": 0, "fumble_lost": 0,
                    "third_down_converted": int(is_pass and rng.random() > 0.6),
                    "third_down_failed": int(not is_pass and rng.random() > 0.7),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bundle():
    games = _synthetic_games()
    return build_features(games=games, plays=_synthetic_plays(games), injuries=None)


def test_game_frame_has_one_row_per_game(bundle):
    assert len(bundle.game_frame) == len(bundle.team_games) / 2
    assert bundle.game_frame["game_id"].is_unique


def test_game_frame_carries_both_teams(bundle):
    row = bundle.game_frame.iloc[0]
    assert row["home_team"] != row["away_team"]
    assert row["home_team"] in TEAMS and row["away_team"] in TEAMS


def test_training_matrix_has_every_model_feature(bundle):
    data = build_training_data(bundle.game_frame)
    assert not data.empty
    assert not set(get_model_features()) - set(data.columns)


def test_no_future_information_in_rolling_features(bundle):
    """A team's first game of the dataset can have no prior form."""
    first = (
        bundle.team_games.sort_values("date").groupby("team").head(1)
    )
    assert first["rolling_avg_points_for"].isna().all()


def test_rolling_points_matches_a_hand_calculation(bundle):
    team = TEAMS[0]
    history = bundle.team_games[bundle.team_games["team"] == team].sort_values("date")
    expected = history["points_for"].iloc[:4].mean()
    assert history["rolling_avg_points_for"].iloc[4] == pytest.approx(expected)


def test_team_form_covers_every_team(bundle):
    assert set(bundle.team_form["team"]) == set(TEAMS)


def test_team_form_uses_the_most_recent_games(bundle):
    """`team_form` must include a team's last game; the stored column excludes it."""
    team = TEAMS[0]
    history = bundle.team_games[bundle.team_games["team"] == team].sort_values("date")
    served = bundle.team_form.set_index("team").loc[team, "rolling_avg_points_for"]
    stored = history["rolling_avg_points_for"].iloc[-1]
    assert served != pytest.approx(stored)
    assert served == pytest.approx(history["points_for"].tail(4).mean())


def test_to_game_frame_is_idempotent_on_columns(bundle):
    assert set(to_game_frame(bundle.team_games).columns) == set(bundle.game_frame.columns)


def test_qb_features_are_present(bundle):
    assert bundle.latest_qb_epa["qb_name"].nunique() == len(TEAMS)
    assert bundle.game_frame["home_rolling_avg_qb_epa"].notna().any()
