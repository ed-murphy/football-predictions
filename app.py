import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

st.set_page_config(layout='wide')

st.markdown(
    """
    <style>
    .main {
        max-width: 400px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NFL Scoring Predictions")

pred_dir = "predictions"
csv_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No predictions files found.")

latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(pred_dir, x)))
latest_path = os.path.join(pred_dir, latest_csv)
predictions = pd.read_csv(latest_path)

predictions['date'] = pd.to_datetime(predictions['date'])

match = re.search(r'_(\d{8})', latest_csv)
if match:
    file_date = datetime.strptime(match.group(1), "%Y%m%d")
    formatted_date = file_date.strftime("%#m/%#d/%y")
    st.markdown(f"<p style='color:red; font-weight:bold;'>Predictions last updated on {formatted_date}</p>", unsafe_allow_html=True)
else:
    st.warning("Could not parse update date from filename.")

start_date = predictions['date'].min().to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
predictions['week'] = ((predictions['date'] - start_date).dt.days // 7 + 1).clip(lower=1)

today = datetime.today()
current_week = max(1, ((today - start_date).days // 7 + 1))

week_options = sorted(predictions['week'].unique())
default_index = week_options.index(current_week) if current_week in week_options else len(week_options) - 1
selected_week = st.selectbox('Select Week', week_options, index=default_index)

display_df = predictions[predictions['week'] == selected_week].sort_values(['date', 'home_team']).copy()

display_df['date'] = display_df['date'].dt.strftime("%b %d, %Y")
display_df['total_line'] = display_df['total_line'].round(1)
display_df['p_over'] = display_df['p_over'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
display_df['actual_total'] = display_df['actual_total'].apply(
    lambda x: '' if pd.isnull(x) else str(int(x)) if float(x).is_integer() else str(x)
)

display_df = display_df.rename(columns={
    "date":            "Game Date",
    "home_team":       "Home",
    "away_team":       "Away",
    "total_line":      "Total Line",
    "p_over":          "P(Over)",
    "bet":             "Bet",
    "actual_total":    "Actual",
    "home_qb_injured": "Home QB Inj",
    "away_qb_injured": "Away QB Inj",
})

cols_to_show = ["Game Date", "Home", "Away", "Total Line", "P(Over)", "Bet", "Home QB Inj", "Away QB Inj", "Actual"]
cols_to_show = [c for c in cols_to_show if c in display_df.columns]

st.markdown(
    """
    <style>
    .stDataFrame div[data-testid="stDataFrameContainer"] div[role="gridcell"],
    .stDataFrame div[data-testid="stDataFrameContainer"] th {
        text-align: center !important;
        justify-content: center !important;
    }
    .stDataFrame div[data-testid="stDataFrameContainer"] {
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)

st.markdown(
    """
    <p style='font-size:12px; color:gray;'>
    ⚠️ Disclaimer: <br>
    This site is not a source of betting advice.<br>
    It is intended as a coding/statistics portfolio project and monitored for entertainment purposes.<br>
    Predictions may be inaccurate, outdated, or completely wrong.<br>
    Do not use this information for placing bets.<br>
    View the code on <a href='https://github.com/ed-murphy/football-predictions' target='_blank'>GitHub</a>.
    </p>
    """,
    unsafe_allow_html=True
)
