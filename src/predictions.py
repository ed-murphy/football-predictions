"""Building, scoring and persisting the prediction table.

The output carries the model's *reasoning*, not just a verdict: the total it
expects, how far that is from the line, the probability that implies, and the
stake the probability justifies. A row that says "Over, 56%" is much harder to
audit after the fact than one that says "we expect 47.8 against a line of 44.5".
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.model import TotalsModel
from config import DECIMAL_PAYOUT, KELLY_FRACTION, PREDICTIONS_DIR

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "date", "home_team", "away_team", "total_line", "pred_total", "edge",
    "p_over", "bet", "stake", "line_open", "line_movement",
    "home_qb_injured", "away_qb_injured", "actual_total", "result",
]

_FILENAME_RE = re.compile(r"^predictions_(\d{8})(?:_v(\d+))?\.csv$")


def kelly_stake(p_over: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Fractional-Kelly stake in units, given a win probability and a side.

    Full Kelly on a -110 bet is `(p*(b+1) - 1) / b`. It is scaled by
    `KELLY_FRACTION` because the win probability is itself an estimate: betting
    full Kelly on a mis-estimated edge is how bankrolls die.
    """
    p_win = np.where(signal == 1, p_over, np.where(signal == -1, 1 - p_over, np.nan))
    b = DECIMAL_PAYOUT
    full = (p_win * (b + 1) - 1) / b
    return np.where(signal == 0, 0.0, np.clip(full, 0, None) * KELLY_FRACTION).round(3)


def build_prediction_table(model: TotalsModel, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Score upcoming games and format them for output."""
    if upcoming is None or upcoming.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    p_over = model.predict_over_prob(upcoming)
    pred_total = model.predict_total(upcoming)
    edge = pred_total - upcoming["total_line"].to_numpy(dtype=float)
    signal = model.bet_signal(upcoming)

    table = pd.DataFrame({
        "date": pd.to_datetime(upcoming["date"]).dt.strftime("%Y-%m-%d"),
        "home_team": upcoming["home_team"].to_numpy(),
        "away_team": upcoming["away_team"].to_numpy(),
        "total_line": upcoming["total_line"].to_numpy(dtype=float).round(1),
        "pred_total": pred_total.round(1),
        "edge": edge.round(1),
        "p_over": p_over.round(3),
        "bet": np.where(signal == 1, "Over", np.where(signal == -1, "Under", "")),
        "stake": kelly_stake(p_over, signal),
        "line_open": upcoming.get("line_open", pd.Series(index=upcoming.index, dtype=float)).to_numpy(),
        "line_movement": upcoming.get("line_movement", pd.Series(index=upcoming.index, dtype=float)).to_numpy(),
        "home_qb_injured": upcoming["home_qb_injured"].to_numpy(),
        "away_qb_injured": upcoming["away_qb_injured"].to_numpy(),
    })

    n_bets = int((table["bet"] != "").sum())
    logger.info(
        "Scored %d games; %d cleared the %.1f-point edge threshold.",
        len(table), n_bets, model.bet_threshold,
    )
    return table


def attach_results(table: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Fill in `actual_total` and grade each bet once the game has been played."""
    table = table.copy()
    finals = games.loc[
        games["home_score"].notna(),
        ["gameday", "home_team", "away_team", "home_score", "away_score"],
    ].copy()
    finals["date"] = pd.to_datetime(finals["gameday"]).dt.strftime("%Y-%m-%d")
    finals["actual_total"] = finals["home_score"] + finals["away_score"]

    table = table.drop(columns=["actual_total", "result"], errors="ignore").merge(
        finals[["date", "home_team", "away_team", "actual_total"]],
        on=["date", "home_team", "away_team"], how="left",
    )

    table["result"] = _grade(table)
    return table


def _grade(table: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(table["actual_total"], errors="coerce")
    line = pd.to_numeric(table["total_line"], errors="coerce")
    bet = table["bet"].fillna("")

    won = ((bet == "Over") & (actual > line)) | ((bet == "Under") & (actual < line))
    lost = ((bet == "Over") & (actual < line)) | ((bet == "Under") & (actual > line))
    push = (bet != "") & actual.notna() & np.isclose(actual, line)

    return pd.Series(
        np.select([bet == "", actual.isna(), push, won, lost],
                  ["", "pending", "push", "win", "loss"], default=""),
        index=table.index,
    )


def save_predictions(
    new_rows: pd.DataFrame,
    existing: pd.DataFrame,
    games: pd.DataFrame,
    predictions_dir: str = PREDICTIONS_DIR,
    keep_days: int = 21,
) -> pd.DataFrame:
    """Merge new predictions into the running table, grade it, and write a new file.

    Rows are keyed on (date, home_team, away_team); a re-run replaces the previous
    forecast for a game rather than appending a duplicate. Graded history is kept
    for `keep_days` so the app can show a track record.
    """
    frames = [f for f in (existing, new_rows) if f is not None and not f.empty]
    if not frames:
        logger.info("Nothing to save: no new or existing predictions.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined[combined["date"].notna()]

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=keep_days)
    combined = combined[combined["date"] >= cutoff]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    )

    combined = attach_results(combined, games)
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLUMNS].sort_values(["date", "home_team"])

    os.makedirs(predictions_dir, exist_ok=True)
    path = _next_filename(predictions_dir)
    combined.to_csv(path, index=False)

    graded = combined[combined["result"].isin(["win", "loss", "push"])]
    if not graded.empty:
        wins = int((graded["result"] == "win").sum())
        losses = int((graded["result"] == "loss").sum())
        logger.info("Recent graded record in this file: %d-%d-%d.",
                    wins, losses, int((graded["result"] == "push").sum()))

    logger.info("Predictions saved to %s (%d rows).", path, len(combined))
    return combined


def load_latest_predictions(predictions_dir: str = PREDICTIONS_DIR) -> pd.DataFrame:
    """Most recent prediction file, or an empty frame if there is none."""
    path = latest_prediction_path(predictions_dir)
    if path is None:
        logger.info("No existing prediction files found.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_csv(path)
    logger.info("Loaded %d existing predictions from %s.", len(df), path)
    return df


def latest_prediction_path(predictions_dir: str = PREDICTIONS_DIR) -> str | None:
    """Newest `predictions_YYYYMMDD[_vN].csv`, by date then version."""
    if not os.path.isdir(predictions_dir):
        return None

    keyed = []
    for name in os.listdir(predictions_dir):
        match = _FILENAME_RE.match(name)
        if match:
            keyed.append(((match.group(1), int(match.group(2) or 0)), name))

    if not keyed:
        return None
    return os.path.join(predictions_dir, max(keyed)[1])


def _next_filename(predictions_dir: str) -> str:
    today = datetime.today().strftime("%Y%m%d")
    path = os.path.join(predictions_dir, f"predictions_{today}.csv")
    version = 1
    while os.path.exists(path):
        version += 1
        path = os.path.join(predictions_dir, f"predictions_{today}_v{version}.csv")
    return path
