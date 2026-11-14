"""Which pre-game facts does the closing line get wrong?

    python -m research.signal_scan

Scores about fifty candidate signals built from the schedule file alone — weather,
rest, schedule spot, recent results against the number — against closing totals
and closing spreads, using the split-sample rule described in research/common.py.

Result (November 2026): wind passes for totals. Nothing passes for spreads except
big home underdogs, which reverses sign in 2014-2019 and is not used. Moneylines
show the usual favourite-longshot pattern, fading over time and never enough to
beat the vig.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.common import load_results, noise_check, split_sample_scan

pd.set_option("display.width", 200)


def add_recent_form(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's results against the number over its previous games this season."""
    shared = ["game_id", "season", "gameday", "total_points", "total_line",
              "total_resid", "spread_resid"]
    home = games[shared + ["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}).assign(side="home")
    away = games[shared + ["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}).assign(side="away")
    away["spread_resid"] = -away["spread_resid"]       # from the away team's point of view

    team_games = pd.concat([home, away]).sort_values(["team", "gameday"])
    team_games["margin"] = team_games["pf"] - team_games["pa"]
    by_team_season = team_games.groupby(["team", "season"])

    def previous(column: str, n: int) -> pd.Series:
        return by_team_season[column].transform(
            lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    team_games["over_by_last1"] = previous("total_resid", 1)
    team_games["over_by_last3"] = previous("total_resid", 3)
    team_games["points_last3"] = previous("total_points", 3)
    team_games["cover_by_last1"] = previous("spread_resid", 1)
    team_games["cover_by_last3"] = previous("spread_resid", 3)
    team_games["margin_last1"] = previous("margin", 1)

    games = games.copy()
    for side in ("home", "away"):
        rows = team_games[team_games["side"] == side].set_index("game_id")
        for column in ("over_by_last1", "over_by_last3", "points_last3", "cover_by_last1",
                       "cover_by_last3", "margin_last1"):
            games[f"{side}_{column}"] = games["game_id"].map(rows[column])
    return games


def totals_candidates(g: pd.DataFrame) -> dict[str, pd.Series]:
    outdoor_temp = g["temp"].where(g["outdoors"] == 1)
    return {
        "wind (mph, outdoors)": g["wind_outdoors"],
        "wind above 10 mph": (g["wind_outdoors"] - 10).clip(lower=0),
        "freezing (<= 32F)": (outdoor_temp <= 32).astype(float),
        "temperature": outdoor_temp.fillna(65),
        "outdoors": g["outdoors"],
        "grass": g["surface"].astype(str).str.contains("grass").astype(float),
        "total line": g["total_line"],
        "size of spread": g["spread_line"].abs(),
        "divisional": g["div_game"],
        "week": g["week"],
        "week 1": (g["week"] == 1).astype(float),
        "weeks 1-4": (g["week"] <= 4).astype(float),
        "week 13 on": (g["week"] >= 13).astype(float),
        "playoffs": (g["game_type"] != "REG").astype(float),
        "primetime": (g["kickoff_hour"] >= 19).astype(float),
        "Thursday": (g["weekday"] == "Thursday").astype(float),
        "Monday": (g["weekday"] == "Monday").astype(float),
        "both off a bye": ((g["home_rest"] >= 13) & (g["away_rest"] >= 13)).astype(float),
        "either on short rest": ((g["home_rest"] <= 5) | (g["away_rest"] <= 5)).astype(float),
        "neutral site": (g["location"] != "Home").astype(float),
        "both went over last game (pts)": g["home_over_by_last1"] + g["away_over_by_last1"],
        "both went over last 3 (pts)": g["home_over_by_last3"] + g["away_over_by_last3"],
        "recent scoring above the line": g["home_points_last3"] + g["away_points_last3"]
                                         - 2 * g["total_line"],
    }


def spread_candidates(g: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "spread (home favoured +)": g["spread_line"],
        "home underdog": (g["spread_line"] < 0).astype(float),
        "home favoured by 7+": (g["spread_line"] >= 7).astype(float),
        "home underdog by 7+": (g["spread_line"] <= -7).astype(float),
        "divisional": g["div_game"],
        "rest advantage (days)": g["home_rest"] - g["away_rest"],
        "home off a bye": (g["home_rest"] >= 13).astype(float),
        "away off a bye": (g["away_rest"] >= 13).astype(float),
        "away on short rest": (g["away_rest"] <= 5).astype(float),
        "covered by more last game": g["home_cover_by_last1"] - g["away_cover_by_last1"],
        "covered by more last 3": g["home_cover_by_last3"] - g["away_cover_by_last3"],
        "won by more last game": g["home_margin_last1"] - g["away_margin_last1"],
        "home blown out last game": (g["home_margin_last1"] <= -17).astype(float),
        "away blown out last game": (g["away_margin_last1"] <= -17).astype(float),
        "home won big last game": (g["home_margin_last1"] >= 17).astype(float),
        "away won big last game": (g["away_margin_last1"] >= 17).astype(float),
        "primetime": (g["kickoff_hour"] >= 19).astype(float),
        "total line": g["total_line"],
        "wind (mph, outdoors)": g["wind_outdoors"],
        "wind x spread": g["wind_outdoors"] * g["spread_line"],
        "week": g["week"],
        "week 1": (g["week"] == 1).astype(float),
        "late divisional underdog": ((g["week"] >= 13) & (g["div_game"] == 1))
                                    * -np.sign(g["spread_line"]),
        "playoff favourite": (g["game_type"] != "REG") * g["spread_line"],
        "outdoors": g["outdoors"],
    }


def moneyline_table(g: pd.DataFrame) -> pd.DataFrame:
    """Return from blindly backing the underdog or favourite, by underdog price."""
    def profit(odds):
        return np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))

    ml = g[g["home_moneyline"].notna() & (g["margin"] != 0)].copy()
    home, away = profit(ml["home_moneyline"]), profit(ml["away_moneyline"])
    ml["dog_pays"], ml["fav_pays"] = np.maximum(home, away), np.minimum(home, away)
    ml["dog_won"] = np.where(home > away, ml["margin"] > 0, ml["margin"] < 0)
    ml["dog_price"] = pd.cut(ml["dog_pays"], [0.9, 1.3, 1.8, 2.5, 3.5, 20],
                             labels=["+100 to +130", "+130 to +180", "+180 to +250",
                                     "+250 to +350", "+350 and up"])
    ml["era"] = pd.cut(ml["season"], [2005, 2013, 2019, 2030],
                       labels=["2006-13", "2014-19", "2020 on"])
    return ml.groupby(["era", "dog_price"], observed=True).apply(
        lambda d: pd.Series({
            "games": len(d),
            "dog ROI": np.where(d["dog_won"], d["dog_pays"], -1).mean(),
            "fav ROI": np.where(~d["dog_won"], d["fav_pays"], -1).mean(),
        }), include_groups=False).round(3)


def main() -> None:
    games = add_recent_form(load_results())
    games = games[games["total_line"].notna() & games["spread_line"].notna()]

    for title, candidates, resid in (
        ("TOTALS  (positive z = predicts the over)", totals_candidates(games), games["total_resid"]),
        ("SPREADS (positive z = predicts the home team covering)",
         spread_candidates(games), games["spread_resid"]),
    ):
        table = split_sample_scan(candidates, resid, games["season"])
        print(f"\n{title}\n{table.to_string(index=False)}\n{noise_check(table)}")

    print("\nMONEYLINES: blind return by underdog price")
    print(moneyline_table(games).to_string())


if __name__ == "__main__":
    main()
