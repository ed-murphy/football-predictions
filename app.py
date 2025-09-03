import pandas as pd
import streamlit as st
from datetime import datetime
import os

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

# Compute NFL week
start_date = datetime(2025, 9, 4)  # Season start
predictions['date'] = pd.to_datetime(predictions['date'])
predictions['week'] = ((predictions['date'] - start_date).dt.days // 7 + 1).clip(lower=1)

# Dropdown to select week
week_options = sorted(predictions['week'].unique())
selected_week = st.selectbox('Select Week', week_options)

# Filter by selected week
week_df = predictions[predictions['week'] == selected_week]

# Sort and prepare display dataframe
display_df = week_df.sort_values(['date', 'home_team']).copy()
display_df['date'] = display_df['date'].dt.strftime("%b %d, %Y")
display_df['predicted_total'] = display_df['predicted_total'].round(1)
display_df = display_df.rename(
    columns={
        "date": "Game Date",
        "home_team": "Home team",
        "away_team": "Away team",
        "predicted_total": "Predicted Points"
    }
)

# Inject CSS for table formatting
st.markdown(
    """
    <style>
    .stDataFrame div[data-testid="stDataFrameContainer"] div[role="gridcell"] {
        text-align: left;
    }
    .stDataFrame div[data-testid="stDataFrameContainer"] {
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display dataframe
st.dataframe(
    display_df[["Game Date", "Home team", "Away team", "Predicted Points"]],
    use_container_width=True,
    hide_index=True
)

# Disclaimer
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
