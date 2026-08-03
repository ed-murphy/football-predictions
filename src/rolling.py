"""Rolling-window helpers shared by every feature module.

Two things were being re-implemented (and re-broken) in each of basic.py, pace.py,
qb.py, defense.py, redzone.py, turnovers.py, third_down.py and offense.py:

1. A *lagged* rolling mean — the average over a team's previous N games, which must
   exclude the current game or the model trains on the outcome it is predicting.
2. Broadcasting a per-team column onto both rows of a game as `home_x` / `away_x`.

Doing them once here removes ~200 lines of duplication and, more importantly, fixes
a real leakage bug: `df.groupby(k)[c].shift().rolling(n).mean()` applies `rolling`
to the *whole* shifted series, so the window silently spans group boundaries. On
this dataset that produced wrong values for 11.5% of rows.
"""
from __future__ import annotations

import pandas as pd


def lagged_rolling_mean(
    df: pd.DataFrame,
    group_cols: list[str] | str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Mean of `value_col` over the previous `window` rows within each group.

    The current row is excluded (shift before rolling), and the window never spans
    group boundaries. Rows are consumed in their existing order, so the caller must
    sort chronologically first.

    Returns a Series aligned to ``df.index``.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift().rolling(window, min_periods=min_periods).mean())
    )


def broadcast_home_away(
    team_games: pd.DataFrame,
    columns: list[str] | str,
) -> pd.DataFrame:
    """Add `home_<col>` / `away_<col>` for each of `columns`.

    `team_games` holds two rows per game (one per team). For every game this copies
    the home team's value into `home_<col>` and the away team's into `away_<col>` on
    *both* rows, so a single row carries the full matchup.

    Uses a groupby-transform rather than the merge-on-game_id pattern the feature
    modules used previously: merging silently duplicates rows whenever `game_id`
    isn't unique on the right-hand side, which is easy to trip over.
    """
    if isinstance(columns, str):
        columns = [columns]

    out = team_games.copy()
    is_home = out["is_home"] == 1

    for col in columns:
        if col not in out.columns:
            raise KeyError(f"{col!r} not present in team_games")
        home_vals = out[col].where(is_home)
        away_vals = out[col].where(~is_home)
        out[f"home_{col}"] = home_vals.groupby(out["game_id"]).transform("max")
        out[f"away_{col}"] = away_vals.groupby(out["game_id"]).transform("max")

    return out


def merge_team_game_stat(
    team_games: pd.DataFrame,
    stat: pd.DataFrame,
    stat_cols: list[str],
    team_col: str = "posteam",
) -> pd.DataFrame:
    """Left-join a play-derived per-(game, team) stat frame onto team_games.

    Guards against the silent row-duplication that a many-to-one join would cause.
    """
    right = stat.rename(columns={team_col: "team"})[["game_id", "team"] + stat_cols]
    return team_games.merge(right, on=["game_id", "team"], how="left", validate="one_to_one")
