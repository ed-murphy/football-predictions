import pandas as pd
import streamlit as st
from datetime import datetime
import os
import glob

st.set_page_config(layout='centered')
st.title("NFL Scoring Predictions")
st.subheader("Predictions for future weeks are subject to change.")

# --- Find the latest-dated predictions file ---
PREDICTIONS_DIR = "predictions"

# Look for files matching predictions_*.csv in the subfolder
files = glob.glob(os.path.join(PREDICTIONS_DIR, "predictions_*.csv"))

if not files:
    st.write("No scoring predictions available.")
else:
    # Sort files by modification time (latest first)
    latest_file = max(files, key=os.path.getmtime)

    # Load predictions CSV
    predictions = pd.read_csv(latest_file, parse_dates=['date'])

    if predictions.empty:
        st.write("No scoring predictions at this time.")
    else:
        # Compute NFL week based on a reference start date
        start_date = datetime(2025, 9, 4)  # Season start
        predictions['week'] = ((predictions['date'] - start_date).dt.days // 7 + 1).clip(lower=1)

        # Dropdown to select week
        week_options = sorted(predictions['week'].unique())
        selected_week = st.selectbox('Select Week', week_options)

        # Filter by selected week
        week_df = predictions[predictions['week'] == selected_week]

        if week_df.empty:
            st.write(f"No scoring predictions for week {selected_week}.")
        else:
            # Sort and prepare display dataframe
            display_df = week_df.sort_values(['date', 'home_team']).copy()
            display_df['date'] = display_df['date'].dt.strftime("%b %d, %Y")
            display_df['predicted_total'] = display_df['predicted_total'].round(1)
            display_df = display_df.rename(
                columns={
                    "date": "Game Date",
                    "home_team": "Home team",
                    "away_team": "Away team",
                    "predicted_total": "Predicted Points Scored"
                }
            )

            # Inject CSS to left-align all columns and auto-adjust table height
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
                display_df[["Game Date", "Home team", "Away team", "Predicted Points Scored"]],
                use_container_width=True,
                hide_index=True
            )

    st.markdown(
        """
        <p style='font-size:12px; color:gray;'>
        ⚠️ Disclaimer: <br>
        This site is not a source of betting advice.<br>
        It is intended as a coding/statistics portfolio project and monitored for entertainment purposes.<br>
        Predictions may be inaccurate, outdated, or completely wrong.<br>
        Do not use this information for placing bets.<br>
        View the code on <a href='https://github.com/your-username/your-repo' target='_blank'>GitHub</a>.
        </p>
        """,
        unsafe_allow_html=True
    )
