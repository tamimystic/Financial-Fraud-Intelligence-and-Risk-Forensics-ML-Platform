"""
Champion XGBoost Inference Engine with Integrated Calibration.
"""

import os
import json
import numpy as np
import xgboost as xgb
from typing import Optional, List
from src.config.settings import (
    CHAMPION_MODEL_PATH,
    CALIBRATOR_PATH,
    FEATURE_NAMES_PATH
)
from src.models.calibrator import ProbabilityCalibrator

class ChampionModelEngine:
    def __init__(
        self,
        model_path: str = CHAMPION_MODEL_PATH,
        calibrator_path: str = CALIBRATOR_PATH,
        feature_names_path: str = FEATURE_NAMES_PATH
    ):
        self.model_path = model_path
        self.calibrator_path = calibrator_path
        self.feature_names_path = feature_names_path
        self.model: Optional[xgb.Booster] = None
        self.calibrator = ProbabilityCalibrator(self.calibrator_path)
        self.feature_names: List[str] = []

    def load_artifacts(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        self.calibrator.load(self.calibrator_path)
        
        if os.path.exists(self.feature_names_path):
            with open(self.feature_names_path, "r") as f:
                self.feature_names = json.load(f)

    def predict_raw_probability(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_artifacts()
        dmat = xgb.DMatrix(X, feature_names=self.feature_names if len(self.feature_names) == X.shape[1] else None)
        return self.model.predict(dmat)

    def predict_risk_probability(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.predict_raw_probability(X)
        return self.calibrator.calibrate(raw_probs)
