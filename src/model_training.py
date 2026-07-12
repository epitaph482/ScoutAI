"""
model_training.py -- Shared model training logic for Scout AI.
Contains the full dual-model (full / performance_only) training
pipeline. notebooks/02_scout_ai_model.ipynb is just a thin wrapper that
calls train_all_models() from here -- keeping the notebook readable
while the actual logic lives in one place, importable and testable.
"""
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Resolve absolute path and locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Automatically create required directories at the project root level if they do not exist
(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "images").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.scoutai_common import (
    get_engine, load_master_view, engineer_features,
    FULL_FEATURES, PERFORMANCE_FEATURES, MODEL_PATHS,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ScoutAI")

TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "xgb_params": {
        "n_estimators": 1200, "learning_rate": 0.03, "max_depth": 7,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
    },
}

CATEGORICAL_FEATURES = ["detailed_position", "foot", "passport_tier"]

def build_pipeline(existing_categorical):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), existing_categorical)],
        remainder="passthrough",
    )
    xgb_model = xgb.XGBRegressor(**TRAIN_CONFIG["xgb_params"])
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", xgb_model)])

def train_model(df, feature_list, model_label):
    existing_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    X = df[feature_list]
    y = df["log_current_market_value"]
    
    model_pipeline = build_pipeline(existing_categorical)
    logger.info(f"[{model_label}] Running {TRAIN_CONFIG['cv_folds']}-fold cross-validation...")
    
    kfold = KFold(n_splits=TRAIN_CONFIG["cv_folds"], shuffle=True, random_state=TRAIN_CONFIG["random_state"])
    cv_scores = cross_val_score(model_pipeline, X, y, cv=kfold, scoring="r2", n_jobs=1)
    logger.info(f"[{model_label}] CV R2 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_CONFIG["test_size"], random_state=TRAIN_CONFIG["random_state"]
    )
    logger.info(f"[{model_label}] Training final model...")
    model_pipeline.fit(X_train, y_train)
    test_r2 = model_pipeline.score(X_test, y_test)
    logger.info(f"[{model_label}] Holdout R2: {test_r2:.4f}")
    
    # Ensure model artifacts are saved using absolute paths
    model_save_path = PROJECT_ROOT / MODEL_PATHS[model_label]
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_save_path)
    logger.info(f"[{model_label}] Model saved to '{model_save_path}'.")
    
    return model_pipeline, X, existing_categorical, cv_scores, test_r2

def plot_feature_importance(model_pipeline, feature_list, existing_categorical, model_label):
    fitted_xgb = model_pipeline.named_steps["regressor"]
    cat_out = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(existing_categorical)
    remainder_features = [f for f in feature_list if f not in existing_categorical]
    all_features = list(cat_out) + remainder_features
    importances = fitted_xgb.feature_importances_
    
    fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    fi_df["Feature"] = fi_df["Feature"].str.replace("_", " ").str.title()
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#2c3e50")
    plt.title(f"Scout AI ({model_label}): Key Drivers of Market Value", fontsize=16)
    plt.tight_layout()
    
    # Save the chart using an absolute path to prevent working directory issues
    save_path = PROJECT_ROOT / "images" / f"feature_importance_{model_label}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"[{model_label}] Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:<30} {row['Importance']:.4f}")

def run_pipeline_for_model(df, feature_list, model_label):
    model_pipeline, X, existing_categorical, cv_scores, test_r2 = train_model(df, feature_list, model_label)
    plot_feature_importance(model_pipeline, feature_list, existing_categorical, model_label)
    
    predictions = model_pipeline.predict(X)
    df_out = df.copy()
    df_out[f"predicted_value_{model_label}"] = np.expm1(predictions)
    df_out[f"value_gap_{model_label}"] = df_out[f"predicted_value_{model_label}"] - df_out["current_market_value"]
    df_out[f"value_gap_pct_{model_label}"] = (
        df_out[f"value_gap_{model_label}"] / df_out["current_market_value"].replace(0, np.nan)
    ) * 100
    
    export_cols = [
        "player_id", "player_name", "club_name", "competition_name", "age",
        "current_market_value", f"predicted_value_{model_label}",
        f"value_gap_{model_label}", f"value_gap_pct_{model_label}"
    ]
    
    # Export predictions using absolute path to the data folder
    csv_path = PROJECT_ROOT / "data" / f"scout_predictions_export_{model_label}.csv"
    df_out[export_cols].sort_values(
        f"value_gap_pct_{model_label}",
        ascending=False
    ).to_csv(csv_path, index=False)
    
    logger.info(f"[{model_label}] Predictions exported to {csv_path}.")
    return cv_scores, test_r2

def train_all_models():
    """Train both the full and performance_only models end to end."""
    engine = get_engine()
    df = load_master_view(engine)
    df = engineer_features(df)
    
    cv_full, r2_full = run_pipeline_for_model(df, FULL_FEATURES, "full")
    cv_perf, r2_perf = run_pipeline_for_model(df, PERFORMANCE_FEATURES, "performance_only")
    
    logger.info("=" * 60)
    logger.info(f"Full model            -> CV R2: {cv_full.mean():.4f}  | Holdout R2: {r2_full:.4f}")
    logger.info(f"Performance-only model -> CV R2: {cv_perf.mean():.4f}  | Holdout R2: {r2_perf:.4f}")
    logger.info("=" * 60)
    
    return {
        "full": {"cv_scores": cv_full, "holdout_r2": r2_full},
        "performance_only": {"cv_scores": cv_perf, "holdout_r2": r2_perf},
    }