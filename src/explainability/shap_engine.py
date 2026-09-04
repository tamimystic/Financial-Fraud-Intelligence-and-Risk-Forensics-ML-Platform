"""
Sub-Millisecond TreeSHAP Explainer Engine.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List
from src.models.inference import ChampionModelEngine

class TreeSHAPExplainer:
    def __init__(self, model_engine: ChampionModelEngine):
        self.model_engine = model_engine
        if self.model_engine.model is None:
            self.model_engine.load_artifacts()
        self.explainer = shap.TreeExplainer(self.model_engine.model)

    def calculate_shap_values(self, df_features: pd.DataFrame) -> np.ndarray:
        X_mat = np.ascontiguousarray(df_features[self.model_engine.feature_names].values)
        return self.explainer.shap_values(X_mat)

    def get_top_contributors(self, df_features: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        shap_vals = self.calculate_shap_values(df_features)
        single_shap = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        feature_names = self.model_engine.feature_names
        
        pairs = list(zip(feature_names, single_shap))
        sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        return [
            {
                "feature": str(k),
                "shap_value": round(float(v), 5),
                "direction": "INCREASES_FRAUD_RISK" if v > 0 else "DECREASES_FRAUD_RISK"
            }
            for k, v in sorted_pairs
        ]
