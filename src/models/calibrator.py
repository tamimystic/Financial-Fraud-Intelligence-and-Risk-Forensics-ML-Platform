"""
Non-Parametric Probability Calibrator Wrapper.
"""

import os
import joblib
import numpy as np
from typing import Optional

class ProbabilityCalibrator:
    def __init__(self, calibrator_path: Optional[str] = None):
        self.calibrator_path = calibrator_path
        self.calibrator = None

    def load(self, path: Optional[str] = None) -> None:
        target_path = path or self.calibrator_path
        if target_path and os.path.exists(target_path):
            self.calibrator = joblib.load(target_path)
        else:
            self.calibrator = None

    def calibrate(self, raw_probabilities: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probabilities)
        return raw_probabilities
