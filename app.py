import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import joblib
from src.load import load_data

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

tab1, tab2, tab3, tab4 = st.tabs(['Predictions', 'About', 'Performance', 'Model Insights'])

with tab1:
    pred_dir = "predictions"
    csv_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No predictions files found.")

    latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(pred_dir, x)))
    latest_path = os.path.join(pred_dir, latest_csv)
    predictions = pd.read_csv(latest_path)

    predictions['date'] = pd.to_datetime(predictions['date']).dt.date
    if 'actual_total' not in predictions.columns:
        predictions['actual_total'] = np.nan
    else:
        predictions['actual_total'] = predictions['actual_total'].replace('', np.nan).astype(float)

    match = re.search(r'_(\d{8})', latest_csv)
    if match:
        file_date_str = match.group(1)
        file_date = datetime.strptime(file_date_str, "%Y%m%d")
        formatted_date = file_date.strftime("%#m/%#d/%y")
        st.markdown(f"<p style='color:red; font-weight:bold;'>Predictions last updated on {formatted_date}</p>", unsafe_allow_html=True)
    else:
        st.warning("Could not parse update date from filename.")

    start_date = datetime(2025, 9, 4)
    predictions['date'] = pd.to_datetime(predictions['date'])
    predictions['week'] = ((predictions['date'] - start_date).dt.days // 7 + 1).clip(lower=1)

    today = datetime.today()
    current_week = max(1, ((today - start_date).days // 7 + 1))

    week_options = sorted(predictions['week'].unique())
    default_index = week_options.index(current_week) if current_week in week_options else 0
    selected_week = st.selectbox('Select Week', week_options, index=default_index)

    week_df = predictions[predictions['week'] == selected_week]

    display_df = week_df.sort_values(['date', 'home_team']).copy()
    display_df['date'] = display_df['date'].dt.strftime("%b %d, %Y")

    if 'total_line' in display_df.columns:
        display_df['total_line'] = display_df['total_line'].round(1)

    # Format p_over as percentage and derive a confidence string
    if 'p_over' in display_df.columns:
        display_df['p_over_pct'] = display_df['p_over'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
        )

    display_df = display_df.rename(
        columns={
            "date": "Game Date",
            "home_team": "Home",
            "away_team": "Away",
            "p_over_pct": "P(Over)",
            "bet": "Bet",
            "actual_total": "Actual",
            "total_line": "DraftKings Total",
        }
    )

    display_df['Actual'] = display_df['Actual'].apply(
        lambda x: '' if pd.isnull(x) else int(x) if float(x).is_integer() else x
    )

    # Build styled table
    cols_to_show = ["Game Date", "Home", "Away", "DraftKings Total", "P(Over)", "Bet", "Actual"]
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    styled = display_df[cols_to_show].rename(columns={"edge_str": "Edge"})

    # Color the Edge column: green = lean over (positive), red = lean under (negative)
    PRECISION = 4.0

    def _color_edge(val):
        try:
            num = float(val.replace("+", ""))
        except (ValueError, AttributeError):
            return ""
        if num >= PRECISION:
            return "background-color: #d4edda; color: #155724; font-weight: bold"  # green
        if num <= -PRECISION:
            return "background-color: #f8d7da; color: #721c24; font-weight: bold"  # red
        return ""

    styled_df = styled.style.applymap(_color_edge, subset=["Edge"])

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

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

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

with tab2:
    st.markdown(
        """
        This app shows predictions for the total amount of points scored in every NFL game.

        The predictions are generated by a statistical model trained on data for every NFL snap since 2014.

        Variables include:
        - Betting market over/under
        - Recent team points scored/allowed
        - QB and defense EPA
        - Weather (temperature, wind)
        - Pace, divisional status, short rest, and red zone efficiency

        When trained on 2014–2023 data and tested on 2024 games, the model picked the over/under correctly 70% of the time when its prediction was 4+ points different from the betting market.

        ---
        [View on GitHub](https://github.com/ed-murphy/football-predictions)
        """
    )

with tab3:
    perf_df = predictions.copy()

    if 'actual_total' in perf_df.columns:
        perf_df = perf_df.replace('', np.nan)
        perf_df['actual_total'] = pd.to_numeric(perf_df['actual_total'], errors='coerce')
        perf_df['p_over'] = pd.to_numeric(perf_df['p_over'], errors='coerce')
        if 'total_line' in perf_df.columns:
            perf_df['total_line'] = pd.to_numeric(perf_df['total_line'], errors='coerce')

        perf_df = perf_df.dropna(subset=['actual_total', 'p_over'])

        if perf_df.empty:
            st.info("No completed games available yet to evaluate performance.")
        else:
            def quadrant_label(row):
                if 'total_line' not in row or pd.isna(row['total_line']):
                    return None
                p_over = row['p_over']
                actual = row['actual_total']
                market = row['total_line']
                pred_over = p_over > 0.5
                if actual > market:
                    return 'Predicted Over, was Correct' if pred_over else 'Predicted Under, was Incorrect'
                elif actual < market:
                    return 'Predicted Under, was Correct' if not pred_over else 'Predicted Over, was Incorrect'
                return None

            perf_df['quadrant'] = perf_df.apply(quadrant_label, axis=1)
            perf_df = perf_df.dropna(subset=['quadrant'])

            # --- Section 1: All Games ---
            st.header("All NFL Games So Far This Season")

            labels = [
                "Predicted Over,\nwas Correct",
                "Predicted Under,\nwas Correct",
                "Predicted Over,\nwas Incorrect",
                "Predicted Under,\nwas Incorrect"
            ]
            counts = perf_df['quadrant'].value_counts().reindex([
                'Predicted Over, was Correct',
                'Predicted Under, was Correct',
                'Predicted Over, was Incorrect',
                'Predicted Under, was Incorrect'
            ], fill_value=0)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, counts.values, color=['green','blue','red','orange'])
            ax.set_ylabel("Number of Games")
            plt.xticks(rotation=0)

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, height * 0.95, str(int(height)),
                    ha='center', va='top', fontsize=10, fontweight='bold', color='white'
                )

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            st.image(buf.getvalue(), width=600)

            # --- Section 2: Bet signals (p_over > 55% or < 45%) ---
            diff_df = perf_df.copy()
            diff_df['confidence'] = (diff_df['p_over'] - 0.5).abs()
            diff_df = diff_df[diff_df['confidence'] >= 0.05]  # p_over > 55% or < 45%

            if diff_df.empty:
                st.info("No games where model confidence exceeded 55%.")
            else:
                st.header("NFL Games So Far This Season Where Model Confidence > 55%")

                counts_diff = diff_df['quadrant'].value_counts().reindex([
                    'Predicted Over, was Correct',
                    'Predicted Under, was Correct',
                    'Predicted Over, was Incorrect',
                    'Predicted Under, was Incorrect'
                ], fill_value=0)

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(labels, counts_diff.values, color=['green','blue','red','orange'])
                ax.set_ylabel("Number of Games")
                plt.xticks(rotation=0)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2, height * 0.95, str(int(height)),
                        ha='center', va='top', fontsize=10, fontweight='bold', color='white'
                    )

                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                st.image(buf.getvalue(), width=600)

            # --- Section 3: Weekly accuracy trend ---
            if 'week' in perf_df.columns or 'date' in perf_df.columns:
                st.header("Weekly Accuracy Trend")

                trend_df = perf_df.copy()
                if 'week' not in trend_df.columns:
                    start_date = datetime(2025, 9, 4)
                    trend_df['date'] = pd.to_datetime(trend_df['date'])
                    trend_df['week'] = ((trend_df['date'] - start_date).dt.days // 7 + 1).clip(lower=1)

                def is_correct(row):
                    if pd.isna(row.get('total_line')):
                        return None
                    return int(
                        (row['actual_total'] > row['total_line']) == (row['p_over'] > 0.5)
                    )

                trend_df['correct'] = trend_df.apply(is_correct, axis=1)
                trend_df = trend_df.dropna(subset=['correct'])

                weekly = (
                    trend_df.groupby('week')['correct']
                    .agg(accuracy='mean', n='count')
                    .reset_index()
                )
                weekly['accuracy_pct'] = (weekly['accuracy'] * 100).round(1)

                if not weekly.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(weekly['week'], weekly['accuracy_pct'], marker='o', linewidth=2, color='steelblue')
                    ax.axhline(50, linestyle='--', color='gray', linewidth=1, label='50% baseline')
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title("Over/Under Accuracy by Week")
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
                    ax.legend()
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    st.image(buf.getvalue(), width=700)

            # --- Section 4: Probability calibration ---
            st.header("Probability Calibration")
            cal_df = perf_df.dropna(subset=['p_over', 'actual_total', 'total_line'])
            if not cal_df.empty:
                cal_df = cal_df.copy()
                cal_df['actual_over'] = (cal_df['actual_total'] > cal_df['total_line']).astype(int)
                cal_df['bin'] = pd.cut(cal_df['p_over'], bins=np.arange(0.3, 0.75, 0.05))
                calib = (
                    cal_df.groupby('bin', observed=True)['actual_over']
                    .agg(empirical_rate='mean', n='count')
                    .reset_index()
                )
                calib['bin_mid'] = calib['bin'].apply(lambda b: b.mid)
                calib = calib.dropna()
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot([0.3, 0.7], [0.3, 0.7], 'r--', linewidth=1, label='Perfect calibration')
                ax.scatter(calib['bin_mid'], calib['empirical_rate'],
                           s=calib['n'] * 5, alpha=0.7, color='steelblue', zorder=3)
                for _, row in calib.iterrows():
                    ax.annotate(f"n={int(row['n'])}",
                                (row['bin_mid'], row['empirical_rate']),
                                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
                ax.set_xlim(0.3, 0.7)
                ax.set_ylim(0.3, 0.7)
                ax.set_xlabel("Predicted P(Over)")
                ax.set_ylabel("Empirical Over Rate")
                ax.set_title("Calibration: Predicted Probability vs Actual Over Rate")
                ax.legend()
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                st.image(buf.getvalue(), width=550)


with tab4:
    st.header("Model Insights")

    MODEL_PATH = "model/xgb_total_points_model_prod.joblib"

    @st.cache_resource
    def _load_model(path):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    prod_model = _load_model(MODEL_PATH)

    if prod_model is None:
        st.info("Production model not found. Run `main.py` first to train and save the model.")
    elif not hasattr(prod_model, "feature_importances_"):
        st.info("Loaded model does not expose feature importances.")
    else:
        importances = prod_model.feature_importances_
        feature_names = prod_model.feature_names_in_ if hasattr(prod_model, "feature_names_in_") else [
            f"feature_{i}" for i in range(len(importances))
        ]
        fi_series = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=True)
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(fi_series.index, fi_series.values, color='steelblue')
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances (Production Model)")
        for bar, val in zip(bars, fi_series.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va='center', fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        st.image(buf.getvalue(), width=650)

        st.caption(
            "Feature importance is computed as mean decrease in impurity across all 500 decision trees. "
            "Higher = the model leans on that feature more when making splits."
        )
