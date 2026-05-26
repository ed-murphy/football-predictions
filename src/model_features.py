from __future__ import annotations

import pandas as pd


BASE_FEATURES = [
    # Scoring form
    "home_rolling_avg_points_for",
    "home_rolling_avg_points_against",
    "away_rolling_avg_points_for",
    "away_rolling_avg_points_against",
    # QB quality
    "home_rolling_avg_qb_epa",
    "away_rolling_avg_qb_epa",
    # Defense
    "home_rolling_avg_def_epa",
    "away_rolling_avg_def_epa",
    "home_rolling_avg_sack_rate",
    "away_rolling_avg_sack_rate",
    # Weather
    "home_temperature",
    "home_wind_speed",
    # Pace
    "home_rolling_avg_off_pace",
    "away_rolling_avg_off_pace",
    # Game context
    "divisional",
    "regular_season",
    "international",
    "home_short_rest",
    "away_short_rest",
    "both_short_rest",
    "home_post_bye",
    "away_post_bye",
    # Injuries
    "home_injury_index",
    "away_injury_index",
    "home_qb_injured",
    "away_qb_injured",
    # Efficiency
    "home_rolling_rz_eff",
    "away_rolling_rz_eff",
    "home_rolling_avg_turnovers",
    "away_rolling_avg_turnovers",
    "home_rolling_avg_3rd_pct",
    "away_rolling_avg_3rd_pct",
    # Offensive efficiency (new)
    "home_rolling_avg_pass_rate",
    "away_rolling_avg_pass_rate",
    "home_rolling_avg_explosive_rate",
    "away_rolling_avg_explosive_rate",
    "home_rolling_avg_success_rate",
    "away_rolling_avg_success_rate",
    "home_rolling_avg_rush_epa",
    "away_rolling_avg_rush_epa",
    "home_rolling_avg_pass_epa",
    "away_rolling_avg_pass_epa",
    "home_rolling_avg_cpoe",
    "away_rolling_avg_cpoe",
    # Referee tendency
    "ref_avg_total",
]


ENGINEERED_FEATURES = [
    # Pace interactions
    "home_x_away_pace",
    "home_pace_x_wind_speed",
    "away_pace_x_wind_speed",
    # QB vs defence matchup
    "home_qb_x_away_def",
    "away_qb_x_home_def",
    # Matchup deltas
    "pace_delta",
    "qb_epa_delta",
    "def_epa_delta",
    "offense_form_delta",
    "defense_form_delta",
    "rz_eff_delta",
    "turnover_delta",
    "third_down_delta",
    # New combined / matchup features
    "combined_pace",
    "combined_qb_epa",
    "combined_pass_epa",
    "combined_rush_epa",
    "combined_success_rate",
    "combined_explosive_rate",
    "sack_rate_delta",
    "pass_rate_delta",
    "cpoe_delta",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Existing interactions
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

    # New combined/matchup features
    df["combined_pace"] = df["home_rolling_avg_off_pace"] + df["away_rolling_avg_off_pace"]
    df["combined_qb_epa"] = df["home_rolling_avg_qb_epa"] + df["away_rolling_avg_qb_epa"]
    df["combined_pass_epa"] = df["home_rolling_avg_pass_epa"] + df["away_rolling_avg_pass_epa"]
    df["combined_rush_epa"] = df["home_rolling_avg_rush_epa"] + df["away_rolling_avg_rush_epa"]
    df["combined_success_rate"] = df["home_rolling_avg_success_rate"] + df["away_rolling_avg_success_rate"]
    df["combined_explosive_rate"] = df["home_rolling_avg_explosive_rate"] + df["away_rolling_avg_explosive_rate"]
    df["sack_rate_delta"] = df["home_rolling_avg_sack_rate"] - df["away_rolling_avg_sack_rate"]
    df["pass_rate_delta"] = df["home_rolling_avg_pass_rate"] - df["away_rolling_avg_pass_rate"]
    df["cpoe_delta"] = df["home_rolling_avg_cpoe"] - df["away_rolling_avg_cpoe"]

    return df


def get_model_features() -> list[str]:
    return BASE_FEATURES + ENGINEERED_FEATURES

