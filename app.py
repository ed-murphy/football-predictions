"""Streamlit front end: this week's board, the live record, and the backtest.

    streamlit run app.py

Reads the files `python main.py` writes; it never fetches or computes anything
itself, so what you see is exactly what the last run produced.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from config import (
    BACKTEST_PATH, BET_WINDOW_DAYS, BETS_PATH, MIN_EV, MY_BOOKS, SLATE_PATH, STANDARD_ODDS,
)
from src import ledger
from src.model import break_even

st.set_page_config(page_title="NFL windy unders", page_icon="🏈", layout="wide")


@st.cache_data(ttl=300)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


st.title("🏈 NFL windy unders")

slate = read_csv(SLATE_PATH)
if slate.empty:
    st.warning("No board yet. Run `python main.py` to build one.")
    st.stop()

slate["bet"] = slate["bet"].fillna("")
updated = datetime.fromtimestamp(os.path.getmtime(SLATE_PATH))
st.caption(f"Board last built {updated:%a %d %b %Y, %H:%M}. Run `python main.py` to refresh.")

# ── This week ─────────────────────────────────────────────────────────────────
flagged = slate[slate["bet"] != ""]
open_air = slate[slate["open_air"] == 1]

left, middle, right = st.columns(3)
left.metric("Games on the board", len(slate))
middle.metric("Bets flagged", len(flagged))
right.metric(
    "Windiest forecast",
    f"{open_air['forecast_wind'].max():.0f} mph" if open_air["forecast_wind"].notna().any() else "—",
)

if flagged.empty:
    st.info("Nothing to bet right now. Most weeks have no bet or one — the model only "
            "acts when a strong wind is forecast at an open-air stadium.")
else:
    for bet in flagged.itertuples():
        st.success(
            f"**Under {bet.line:g}** — {bet.away_team} @ {bet.home_team}, {bet.kickoff_et} ET · "
            f"{bet.book} {bet.under_price:+.0f} · expected wind {bet.expected_wind:.0f} mph · "
            f"P(under) {bet.p_under:.1%} · EV {bet.ev:+.1%} · stake {bet.stake_pct:.2f}% of bankroll"
        )

board = pd.DataFrame({
    "Kickoff (ET)": slate["kickoff_et"],
    "Game": slate["away_team"] + " @ " + slate["home_team"],
    "Total": slate["line"].fillna(slate["market_line"]),
    "Book": slate["book"],
    "Forecast wind": slate["forecast_wind"],
    "Expected wind": slate["expected_wind"],
    "P(under)": slate["p_under"] * 100,
    "EV": slate["ev"] * 100,
    "Status": slate["status"],
})
st.dataframe(
    board, hide_index=True, width="stretch",
    column_config={
        "Total": st.column_config.NumberColumn(format="%.1f"),
        "Forecast wind": st.column_config.NumberColumn(
            format="%.0f mph", help="Raw forecast for the kickoff hour."),
        "Expected wind": st.column_config.NumberColumn(
            format="%.1f mph",
            help="Best estimate of the wind the gamebook will record, after allowing for "
                 "forecast error. This is what the model uses."),
        "P(under)": st.column_config.NumberColumn(format="%.1f%%"),
        "EV": st.column_config.NumberColumn(
            format="%+.1f%%", help="Expected profit per unit staked at the quoted price."),
    },
)

# ── Track record ──────────────────────────────────────────────────────────────
st.subheader("Track record")
live_col, backtest_col = st.columns(2)

with live_col:
    st.markdown("**Live bets**")
    bets = ledger.load_ledger(BETS_PATH)
    summary = ledger.summarise(bets)
    if bets.empty:
        st.caption("No bets flagged yet.")
    else:
        a, b, c = st.columns(3)
        a.metric("Record", f"{summary['wins']}–{summary['losses']}–{summary['pushes']}",
                 f"{summary['pending']} pending", delta_color="off")
        b.metric("Units", f"{summary['units']:+.2f}")
        c.metric("Closing-line value",
                 "—" if pd.isna(summary["avg_clv_points"]) else f"{summary['avg_clv_points']:+.2f} pts",
                 help="Line you bet minus the closing line. Positive means the market moved "
                      "your way afterwards — the earliest sign an edge is real.")
        st.dataframe(
            bets[["kickoff_et", "away_team", "home_team", "line", "under_price", "expected_wind",
                  "closing_line", "actual_total", "result", "profit_units"]],
            hide_index=True, width="stretch",
        )

with backtest_col:
    st.markdown("**Backtest, by season**")
    seasons = read_csv(BACKTEST_PATH)
    if seasons.empty or "units" not in seasons.columns:
        st.caption("Run `python main.py backtest` to generate it.")
    else:
        total_bets = int(seasons["bets"].sum())
        wins, losses = int(seasons["wins"].sum()), int(seasons["losses"].sum())
        a, b, c = st.columns(3)
        a.metric("Bets", total_bets)
        b.metric("Win rate", f"{wins / (wins + losses):.1%}", f"break-even {break_even():.1%}",
                 delta_color="off")
        c.metric("ROI", f"{seasons['units'].sum() / total_bets:+.1%}")
        st.bar_chart(seasons.set_index("season")["units"], height=220)
        st.caption("Units won per season with the kickoff wind known exactly. Expect less "
                   "with real forecasts, and expect losing seasons: about half of them are.")

# ── Explanation ───────────────────────────────────────────────────────────────
with st.expander("How this works"):
    books = ", ".join(MY_BOOKS) if MY_BOOKS else "any book in the feed"
    st.markdown(f"""
**The idea.** Sportsbooks lower the total when it is going to be windy, but not by
enough. Since 1999, games played in 13 mph of wind or more have finished under the
closing total about 56% of the time; {break_even():.1%} is break-even at {STANDARD_ODDS}.
Nothing else tested — about 120 other signals — beat the closing line.

**Forecast vs expected wind.** The model was fitted on the wind recorded at kickoff.
A forecast is a noisy estimate of that, so it is shrunk toward normal before use, and
shrunk harder the further away the game is.

**When a bet is flagged.** Open-air stadium, kickoff within {BET_WINDOW_DAYS} days,
and expected value of at least {MIN_EV:.0%} at the best under available at {books}.

**How much to expect.** A handful of bets a month, a few percent return per bet, and
long losing runs. See `docs/model.md` for the evidence and its limits.
    """)

st.caption("A personal research project, not betting advice. The edge is small, the sample "
           "behind it is finite, and the last three seasons of the backtest lost money.")
