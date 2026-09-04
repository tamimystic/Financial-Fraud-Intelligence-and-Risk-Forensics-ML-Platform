"""
Decile-Level Population Stability Index (PSI) Drift Detector.
"""

import numpy as np
from typing import Dict, Any, List

class DriftDetector:
    def __init__(self, num_buckets: int = 10, epsilon: float = 1e-4):
        self.num_buckets = num_buckets
        self.epsilon = epsilon

    def compute_psi(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        b = np.asarray(baseline, dtype=np.float64)
        c = np.asarray(current, dtype=np.float64)
        
        percentiles = np.linspace(0, 100, self.num_buckets + 1)
        breakpoints = np.percentile(b, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        b_counts = np.histogram(b, bins=breakpoints)[0]
        c_counts = np.histogram(c, bins=breakpoints)[0]
        
        b_pct = (b_counts + self.epsilon) / (len(b) + self.epsilon * self.num_buckets)
        c_pct = (c_counts + self.epsilon) / (len(c) + self.epsilon * self.num_buckets)
        
        psi_per_bucket = (c_pct - b_pct) * np.log(c_pct / b_pct)
        total_psi = float(np.sum(psi_per_bucket))
        
        if total_psi < 0.10:
            stability_tier = "STABLE"
            action = "No intervention required. Distribution is consistent."
        elif total_psi < 0.25:
            stability_tier = "MODERATE_DRIFT"
            action = "Monitor closely; scheduled recalibration recommended."
        else:
            stability_tier = "SIGNIFICANT_DRIFT"
            action = "Immediate champion-challenger retraining and recalibration required."
            
        return {
            "psi_score": round(total_psi, 5),
            "stability_tier": stability_tier,
            "operational_action": action,
            "decile_baseline_percentages": [round(float(x), 4) for x in b_pct],
            "decile_current_percentages": [round(float(x), 4) for x in c_pct]
        }
