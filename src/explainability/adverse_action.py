"""
FCRA Section 615(a) and ECOA Regulation B Adverse Action Reason Code Engine.
"""

import pandas as pd
from typing import Dict, Any, List
from src.explainability.shap_engine import TreeSHAPExplainer
from src.models.inference import ChampionModelEngine

REASON_CODE_TAXONOMY: Dict[str, Dict[str, str]] = {
    "V14": {
        "code": "RC-101",
        "description": "Significant structural divergence from historical cardholder spending baseline (Component 14)."
    },
    "V10": {
        "code": "RC-102",
        "description": "High-risk merchant terminal interaction pattern detected (Component 10)."
    },
    "V12": {
        "code": "RC-103",
        "description": "Abnormal velocity pattern inconsistent with diurnal frequency norms (Component 12)."
    },
    "V4": {
        "code": "RC-104",
        "description": "Rapid acceleration in cumulative spending across active transaction window (Component 4)."
    },
    "V17": {
        "code": "RC-105",
        "description": "Elevated risk profile across cross-channel routing nodes (Component 17)."
    },
    "V11": {
        "code": "RC-106",
        "description": "Usage pattern deviation from standard user terminal profile (Component 11)."
    },
    "Amount": {
        "code": "RC-201",
        "description": "Monetary value exceeds established account confidence interval."
    },
    "Hour": {
        "code": "RC-301",
        "description": "Transaction initiated outside standard geographical activity schedule."
    },
    "IsoForest_Anomaly_Score": {
        "code": "RC-401",
        "description": "Multivariate isolation anomaly score triggered policy safety bounds."
    }
}

class AdverseActionEngine:
    def __init__(self, model_engine: ChampionModelEngine):
        self.model_engine = model_engine
        self.shap_explainer = TreeSHAPExplainer(self.model_engine)

    def generate_adverse_action_codes(self, df_features: pd.DataFrame, top_k: int = 4) -> List[Dict[str, Any]]:
        top_drivers = self.shap_explainer.get_top_contributors(df_features, top_k=top_k)
        adverse_codes = []
        
        for driver in top_drivers:
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            meta = REASON_CODE_TAXONOMY.get(
                feat,
                {
                    "code": f"RC-GEN-{feat.upper()}",
                    "description": f"Statistical divergence detected on forensic vector {feat}."
                }
            )
            adverse_codes.append({
                "code": meta["code"],
                "feature": feat,
                "description": meta["description"],
                "shap_attribution": shap_val
            })
            
        return adverse_codes
