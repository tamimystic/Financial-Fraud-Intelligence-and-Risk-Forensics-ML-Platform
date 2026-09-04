"""
Global Platform Settings and Constants.
"""

import os
from dataclasses import dataclass
from typing import List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "test_features.parquet")

MODELS_DIR = os.path.join(BASE_DIR, "models")
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_xgboost_model.json")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "calibrator_isotonic.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

COST_CHARGEBACK = 15.00
COST_FRICTION = 10.00
COST_REVIEW = 5.00

THRESHOLD_OPTIMAL = 0.0800
THRESHOLD_FRICTIONLESS = 0.0200
THRESHOLD_STEP_UP_MAX = 0.3000
THRESHOLD_MANUAL_REVIEW_MAX = 0.7000
HIGH_VALUE_THRESHOLD = 10000.00

RAW_PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]

ENGINEERED_FEATURE_NAMES: List[str] = RAW_PCA_FEATURES + [
    "Amount",
    "Hour_of_Day",
    "Time_Sin",
    "Time_Cos",
    "Time_Delta_Sec",
    "Amount_Log1p",
    "Amount_YJ",
    "Amount_ZScore",
    "Amount_to_Mean_Ratio",
    "V14_x_V17",
    "V12_x_V10",
    "V14_x_V4",
    "V11_x_V12",
    "V16_x_V17",
    "V3_x_V14",
    "V14_per_LogAmount",
    "V17_per_LogAmount",
    "V12_per_LogAmount",
    "V10_per_LogAmount",
    "V14_Squared",
    "V17_Squared",
    "V12_Squared",
    "V10_Squared",
    "iForest_Anomaly_Score"
]

@dataclass(frozen=True)
class PlatformSettings:
    base_dir: str = BASE_DIR
    champion_model_path: str = CHAMPION_MODEL_PATH
    calibrator_path: str = CALIBRATOR_PATH
    feature_names_path: str = FEATURE_NAMES_PATH
    threshold_optimal: float = THRESHOLD_OPTIMAL
    threshold_frictionless: float = THRESHOLD_FRICTIONLESS
    threshold_step_up_max: float = THRESHOLD_STEP_UP_MAX
    threshold_manual_review_max: float = THRESHOLD_MANUAL_REVIEW_MAX
    high_value_threshold: float = HIGH_VALUE_THRESHOLD
