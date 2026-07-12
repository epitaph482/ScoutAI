"""
scoutai_common.py -- Shared config, feature engineering, and model-routing
logic for the Scout AI project.

Every notebook should import from this module instead of redefining
LEAGUE_WEIGHTS / engineer_features() / FULL_FEATURES / etc. locally.
This is the fix for a real bug found during development: several
notebooks had their own copy of the league-weights dictionary, and one
of them silently drifted out of sync (missing Ligue 1, Eredivisie,
Liga Portugal), giving those leagues a neutral coefficient instead of
the intended one. A single shared module makes that kind of drift
impossible.
"""

import os
import numpy as np
import pandas as pd
import sqlalchemy
import joblib
from dotenv import load_dotenv

# ==========================================================
# CONFIG
# ==========================================================

# Loads variables from a .env file in the project root (see .env.example
# for the expected format). The .env file itself is gitignored and never
# committed -- only .env.example (with placeholder values) is.
load_dotenv()

DB_URL = os.getenv("SCOUTAI_DB_URL")
if not DB_URL:
    raise RuntimeError(
        "SCOUTAI_DB_URL is not set. Copy .env.example to .env in the "
        "project root and fill in your real database credentials."
    )

WONDERKID_AGE_THRESHOLD = 22
PASSPORT_TIER_1_MAX_RANK = 15
PASSPORT_TIER_2_MAX_RANK = 60

LEAGUE_WEIGHTS = {
    "Premier League": 1.5,
    "LaLiga": 1.4,
    "Serie A": 1.3,
    "Bundesliga": 1.3,
    "Ligue 1": 1.2,
    "Eredivisie": 1.15,
    "Liga Portugal": 1.15,
    "Süper Lig": 1.05,
}

MODEL_PATHS = {
    "full": "models/scout_model_full.pkl",
    "performance_only": "models/scout_model_performance_only.pkl",
}

# Feature lists -- MUST exactly match what 02_scout_ai_model.ipynb trains
# scout_model_full.pkl / scout_model_performance_only.pkl on. Any script
# that calls model_pipeline.predict(...) must subset to one of these two
# lists, in this exact form -- never pass a raw/whole dataframe to
# predict(), since the pipeline expects precisely these columns.
FULL_FEATURES = [
    "age", "height_in_cm", "total_appearances", "minutes_per_match",
    "total_goals", "total_assists", "goals_per_90", "assists_per_90",
    "total_yellow_cards", "total_red_cards", "international_caps", "international_goals",
    "club_squad_size", "club_avg_age", "stadium_seats", "foreigners_percentage",
    "contract_months_remaining", "wonderkid_hype", "league_coefficient",
    "has_transfer_history", "max_career_transfer_fee", "most_recent_transfer_fee",
    "detailed_position", "foot", "passport_tier",
]

PERFORMANCE_FEATURES = [
    "age", "height_in_cm", "total_appearances", "minutes_per_match",
    "total_goals", "total_assists", "goals_per_90", "assists_per_90",
    "total_yellow_cards", "total_red_cards", "international_caps", "international_goals",
    "club_squad_size", "club_avg_age", "stadium_seats", "foreigners_percentage",
    "wonderkid_hype", "league_coefficient", "detailed_position", "foot", "passport_tier",
]

NUMERIC_FEATURES = [
    "age", "total_appearances", "international_caps", "international_goals",
    "max_career_transfer_fee", "most_recent_transfer_fee", "height_in_cm",
    "minutes_per_match", "total_goals", "total_assists", "goals_per_90",
    "assists_per_90", "total_yellow_cards", "total_red_cards", "club_squad_size",
    "club_avg_age", "stadium_seats", "foreigners_percentage", "contract_months_remaining",
]


# ==========================================================
# DATABASE
# ==========================================================

def get_engine() -> sqlalchemy.engine.Engine:
    return sqlalchemy.create_engine(DB_URL)


def load_master_view(engine: sqlalchemy.engine.Engine) -> pd.DataFrame:
    """Load view_scout_master -- the single source of truth."""
    return pd.read_sql("SELECT * FROM view_scout_master", engine)


# ==========================================================
# FEATURE ENGINEERING
# (identical logic used everywhere -- training, inference, analysis)
# ==========================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "passport_power_rank" in df.columns:
        df["passport_power_rank"] = pd.to_numeric(df["passport_power_rank"], errors="coerce")

    if "log_current_market_value" not in df.columns and "current_market_value" in df.columns:
        df["current_market_value"] = pd.to_numeric(df["current_market_value"], errors="coerce").fillna(0)
        df["log_current_market_value"] = np.log1p(df["current_market_value"])

    df["wonderkid_hype"] = np.where(
        df["age"] <= WONDERKID_AGE_THRESHOLD,
        df["total_appearances"].fillna(0) + (df["international_caps"].fillna(0) * 3),
        0,
    )

    conditions = [
        df["passport_power_rank"].fillna(999) <= PASSPORT_TIER_1_MAX_RANK,
        (df["passport_power_rank"].fillna(999) > PASSPORT_TIER_1_MAX_RANK)
        & (df["passport_power_rank"].fillna(999) <= PASSPORT_TIER_2_MAX_RANK),
    ]
    df["passport_tier"] = np.select(conditions, ["Tier_1", "Tier_2"], default="Tier_3")

    df["league_coefficient"] = df["competition_name"].map(LEAGUE_WEIGHTS).fillna(1.0)
    df["detailed_position"] = df["sub_position"].fillna(df["position_group"]).astype(str)

    # has_transfer_history / max_career_transfer_fee / most_recent_transfer_fee
    # are already computed correctly in view_scout_master -- never
    # recompute them here.

    return df


# ==========================================================
# MODEL LOADING & ROUTING
# ==========================================================

_loaded_models = {}


def get_model(label: str):
    """Lazy-load a model by label ('full' or 'performance_only')."""
    if label not in _loaded_models:
        _loaded_models[label] = joblib.load(MODEL_PATHS[label])
    return _loaded_models[label]


def route_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Predict market value for every row in df, routing each player to
    scout_model_full.pkl or scout_model_performance_only.pkl based on
    has_transfer_history. Adds 'predicted_value' and 'model_used'
    columns and returns the dataframe.
    """
    df = df.copy()
    df["predicted_value"] = np.nan
    df["model_used"] = ""

    has_history_mask = df["has_transfer_history"] == 1
    no_history_mask = ~has_history_mask

    if has_history_mask.any():
        model_full = get_model("full")
        log_preds = model_full.predict(df.loc[has_history_mask, FULL_FEATURES])
        df.loc[has_history_mask, "predicted_value"] = np.expm1(log_preds)
        df.loc[has_history_mask, "model_used"] = "full"

    if no_history_mask.any():
        model_perf = get_model("performance_only")
        log_preds = model_perf.predict(df.loc[no_history_mask, PERFORMANCE_FEATURES])
        df.loc[no_history_mask, "predicted_value"] = np.expm1(log_preds)
        df.loc[no_history_mask, "model_used"] = "performance_only"

    return df
