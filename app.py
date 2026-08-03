"""Streamlit front end for the NFL totals model.

Shows what the model thinks about every game on the slate, not only the handful
it would bet, plus the running record so the claims can be checked.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.predictions import latest_prediction_path
from config import BACKTEST_PATH, BREAK_EVEN_WIN_RATE, DECIMAL_PAYOUT, PREDICTIONS_DIR

st.set_page_config(page_title="NFL Scoring Predictions", page_icon="🏈", layout="wide")


@st.cache_data(ttl=600)
def load_predictions() -> tuple[pd.DataFrame, str | None]:
    path = latest_prediction_path(PREDICTIONS_DIR)
    if path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("bet", "result"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df, path


@st.cache_data(ttl=600)
def load_backtest() -> pd.DataFrame:
    if not os.path.exists(BACKTEST_PATH):
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_PATH)


predictions, source_path = load_predictions()

st.title("🏈 NFL Scoring Predictions")

if predictions.empty:
    st.warning("No predictions available yet. Run `python main.py` to generate some.")
    st.stop()

updated = datetime.fromtimestamp(os.path.getmtime(source_path))
st.caption(f"Model output last refreshed {updated:%d %b %Y, %H:%M}")

# ── Slate ─────────────────────────────────────────────────────────────────────
upcoming = predictions[predictions["result"].isin(["", "pending"])].copy()
settled = predictions[predictions["result"].isin(["win", "loss", "push"])].copy()

slate_dates = sorted(upcoming["date"].dt.date.unique())
if slate_dates:
    # Group the slate into weeks anchored on the first upcoming kickoff.
    anchor = pd.Timestamp(slate_dates[0]).normalize()
    upcoming["slate"] = ((upcoming["date"] - anchor).dt.days // 7).clip(lower=0)
    labels = {
        n: f"{grp['date'].min():%d %b} – {grp['date'].max():%d %b}"
        for n, grp in upcoming.groupby("slate")
    }
    choice = st.selectbox(
        "Slate", list(labels), format_func=lambda n: labels[n], index=0
    )
    board = upcoming[upcoming["slate"] == choice].sort_values(["date", "home_team"])
else:
    board = upcoming

flagged = board[board["bet"] != ""]

col1, col2, col3 = st.columns(3)
col1.metric("Games on this slate", len(board))
col2.metric("Bets flagged", len(flagged))
col3.metric(
    "Largest disagreement",
    f"{board['edge'].abs().max():.1f} pts" if not board.empty else "—",
)

if flagged.empty and not board.empty:
    st.info(
        "The model does not disagree with the market by enough to justify a bet on "
        "this slate. That is the normal outcome — see *How to read this* below."
    )

display = board.assign(
    Matchup=board["away_team"] + " @ " + board["home_team"],
    Date=board["date"].dt.strftime("%a %d %b"),
    Line=board["total_line"].round(1),
    Model=board["pred_total"].round(1),
    Edge=board["edge"].round(1),
    **{"P(over)": (board["p_over"] * 100).round(1)},
)[["Date", "Matchup", "Line", "Model", "Edge", "P(over)", "bet", "stake"]].rename(
    columns={"bet": "Signal", "stake": "Stake (u)"}
)

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Line": st.column_config.NumberColumn("Market total", format="%.1f"),
        "Model": st.column_config.NumberColumn("Model total", format="%.1f"),
        "Edge": st.column_config.NumberColumn(
            "Edge", format="%+.1f", help="Model total minus market total, in points."
        ),
        "P(over)": st.column_config.NumberColumn("P(over)", format="%.1f%%"),
        "Stake (u)": st.column_config.NumberColumn(
            "Stake", format="%.2f", help="Quarter-Kelly stake in units."
        ),
    },
)

# ── Track record ──────────────────────────────────────────────────────────────
st.subheader("Track record")

record_col, backtest_col = st.columns(2)

with record_col:
    st.markdown("**Live flagged bets**")
    graded = settled[settled["bet"] != ""]
    if graded.empty:
        st.caption("No flagged bets have been settled yet.")
    else:
        wins = int((graded["result"] == "win").sum())
        losses = int((graded["result"] == "loss").sum())
        pushes = int((graded["result"] == "push").sum())
        decided = wins + losses
        profit = wins * DECIMAL_PAYOUT - losses
        st.metric(
            f"{wins}–{losses}–{pushes}",
            f"{profit:+.2f} u",
            delta=f"{(wins / decided - BREAK_EVEN_WIN_RATE) * 100:+.1f}pp vs break-even"
            if decided else None,
        )
        st.caption(
            f"Only the last {len(settled)} graded games are retained in the "
            "prediction file, so this is a recent record rather than a full history."
        )

with backtest_col:
    st.markdown("**Walk-forward backtest**")
    backtest = load_backtest()
    if backtest.empty:
        st.caption("Run `python main.py --backtest` to generate one.")
    else:
        st.dataframe(
            backtest[["season", "n_games", "model_mae", "market_mae",
                      "n_bets", "win_rate", "roi"]].round(3),
            hide_index=True, use_container_width=True,
            column_config={
                "n_games": "Games", "model_mae": "Model MAE",
                "market_mae": "Market MAE", "n_bets": "Bets",
                "win_rate": st.column_config.NumberColumn("Win rate", format="%.3f"),
                "roi": st.column_config.NumberColumn("ROI", format="%+.3f"),
            },
        )

# ── Explanation ───────────────────────────────────────────────────────────────
with st.expander("How to read this"):
    st.markdown(
        f"""
**Model total** is the combined score the model expects, built from each team's
recent scoring, pace, efficiency and quarterback form, plus kickoff weather, rest,
injuries and the referee's historical tendency.

**Edge** is that number minus the market total. The market is very good at this:
across the backtest the model's average error is within a tenth of a point of
simply repeating the line. So the edge is almost always small, and a small edge is
not a bet.

**Signal** only appears when the edge is large enough to pay for the vig. At -110
you need to win **{BREAK_EVEN_WIN_RATE:.1%}** of bets to break even; the model
translates that into the minimum number of points it must disagree by, using its
own error distribution. Most weeks nothing qualifies.

**Stake** is a quarter-Kelly position size for the probability shown — deliberately
small, because the probability is itself an estimate.
        """
    )

st.markdown(
    """
    <p style='font-size:12px; color:gray;'>
    ⚠️ <b>Disclaimer:</b> this is a statistics and engineering portfolio project, not
    betting advice. The backtested edge is not statistically distinguishable from
    zero. Predictions may be inaccurate, outdated, or simply wrong. Do not bet on them.
    View the code on
    <a href='https://github.com/ed-murphy/football-predictions' target='_blank'>GitHub</a>.
    </p>
    """,
    unsafe_allow_html=True,
)
