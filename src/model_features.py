from __future__ import annotations

import pandas as pd


BASE_FEATURES = [
    "total_line",
    "home_rolling_avg_points_for",
    "home_rolling_avg_points_against",
    "away_rolling_avg_points_for",
    "away_rolling_avg_points_against",
    "home_rolling_avg_qb_epa",
    "away_rolling_avg_qb_epa",
    "home_rolling_avg_def_epa",
    "away_rolling_avg_def_epa",
    "home_temperature",
    "home_wind_speed",
    "home_rolling_avg_off_pace",
    "away_rolling_avg_off_pace",
    "divisional",
    "regular_season",
    "international",
    "home_short_rest",
    "away_short_rest",
    "both_short_rest",
    "home_post_bye",
    "away_post_bye",
    "home_rolling_rz_eff",
    "away_rolling_rz_eff",
    "home_rolling_avg_turnovers",
    "away_rolling_avg_turnovers",
    "home_rolling_avg_3rd_pct",
    "away_rolling_avg_3rd_pct",
]


ENGINEERED_FEATURES = [
    "home_x_away_pace",
    "home_pace_x_wind_speed",
    "away_pace_x_wind_speed",
    "home_qb_x_away_def",
    "away_qb_x_home_def",
    "pace_delta",
    "qb_epa_delta",
    "def_epa_delta",
    "offense_form_delta",
    "defense_form_delta",
    "rz_eff_delta",
    "turnover_delta",
    "third_down_delta",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction and matchup-delta features used by model training and scoring.
    """
    df = df.copy()
    df["home_x_away_pace"] = df["home_rolling_avg_off_pace"] * df["away_rolling_avg_off_pace"]
    df["home_pace_x_wind_speed"] = df["home_rolling_avg_off_pace"] * df["home_wind_speed"]
    df["away_pace_x_wind_speed"] = df["away_rolling_avg_off_pace"] * df["home_wind_speed"]
    df["home_qb_x_away_def"] = df["home_rolling_avg_qb_epa"] * df["away_rolling_avg_def_epa"]
    df["away_qb_x_home_def"] = df["away_rolling_avg_qb_epa"] * df["home_rolling_avg_def_epa"]

    df["pace_delta"] = df["home_rolling_avg_off_pace"] - df["away_rolling_avg_off_pace"]
    df["qb_epa_delta"] = df["home_rolling_avg_qb_epa"] - df["away_rolling_avg_qb_epa"]
    df["def_epa_delta"] = df["home_rolling_avg_def_epa"] - df["away_rolling_avg_def_epa"]
    df["offense_form_delta"] = df["home_rolling_avg_points_for"] - df["away_rolling_avg_points_for"]
    df["defense_form_delta"] = df["home_rolling_avg_points_against"] - df["away_rolling_avg_points_against"]
    df["rz_eff_delta"] = df["home_rolling_rz_eff"] - df["away_rolling_rz_eff"]
    df["turnover_delta"] = df["home_rolling_avg_turnovers"] - df["away_rolling_avg_turnovers"]
    df["third_down_delta"] = df["home_rolling_avg_3rd_pct"] - df["away_rolling_avg_3rd_pct"]
    return df


def get_model_features() -> list[str]:
    return BASE_FEATURES + ENGINEERED_FEATURES
