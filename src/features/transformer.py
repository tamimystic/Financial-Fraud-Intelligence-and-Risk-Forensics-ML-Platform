"""
Production 53-Feature Leak-Free Transformer Pipeline.
"""

import numpy as np
import pandas as pd
from typing import List
from src.features.constants import ALL_EXPECTED_FEATURES

class FeatureTransformer:
    def __init__(self, expected_features: List[str] = ALL_EXPECTED_FEATURES):
        self.expected_features = expected_features
        self.amount_mean = 88.3496
        self.amount_std = 250.1201

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        
        time_vals = df_out["Time"].values if "Time" in df_out.columns else np.zeros(len(df_out))
        hour_vals = (time_vals / 3600.0) % 24.0
        df_out["Hour_of_Day"] = hour_vals
        df_out["Time_Sin"] = np.sin(2.0 * np.pi * hour_vals / 24.0)
        df_out["Time_Cos"] = np.cos(2.0 * np.pi * hour_vals / 24.0)
        df_out["Time_Delta_Sec"] = df_out["Time"].diff().fillna(1.0) if "Time" in df_out.columns else np.ones(len(df_out))
        
        amt = df_out["Amount"].values if "Amount" in df_out.columns else np.zeros(len(df_out))
        df_out["Amount_Log1p"] = np.log1p(np.maximum(amt, 0.0))
        df_out["Amount_YJ"] = np.log1p(np.maximum(amt, 0.0))
        df_out["Amount_ZScore"] = (amt - self.amount_mean) / (self.amount_std + 1e-6)
        df_out["Amount_to_Mean_Ratio"] = amt / (self.amount_mean + 1e-6)
        
        v14 = df_out["V14"].values if "V14" in df_out.columns else np.zeros(len(df_out))
        v10 = df_out["V10"].values if "V10" in df_out.columns else np.zeros(len(df_out))
        v12 = df_out["V12"].values if "V12" in df_out.columns else np.zeros(len(df_out))
        v4 = df_out["V4"].values if "V4" in df_out.columns else np.zeros(len(df_out))
        v17 = df_out["V17"].values if "V17" in df_out.columns else np.zeros(len(df_out))
        v11 = df_out["V11"].values if "V11" in df_out.columns else np.zeros(len(df_out))
        v16 = df_out["V16"].values if "V16" in df_out.columns else np.zeros(len(df_out))
        v3 = df_out["V3"].values if "V3" in df_out.columns else np.zeros(len(df_out))
        
        df_out["V14_x_V17"] = v14 * v17
        df_out["V12_x_V10"] = v12 * v10
        df_out["V14_x_V4"] = v14 * v4
        df_out["V11_x_V12"] = v11 * v12
        df_out["V16_x_V17"] = v16 * v17
        df_out["V3_x_V14"] = v3 * v14
        
        log_amt = df_out["Amount_Log1p"].values
        df_out["V14_per_LogAmount"] = v14 / (log_amt + 0.1)
        df_out["V17_per_LogAmount"] = v17 / (log_amt + 0.1)
        df_out["V12_per_LogAmount"] = v12 / (log_amt + 0.1)
        df_out["V10_per_LogAmount"] = v10 / (log_amt + 0.1)
        
        df_out["V14_Squared"] = v14 ** 2
        df_out["V17_Squared"] = v17 ** 2
        df_out["V12_Squared"] = v12 ** 2
        df_out["V10_Squared"] = v10 ** 2
        
        risk_norm = np.sqrt(v14**2 + v10**2 + v12**2 + v17**2)
        df_out["iForest_Anomaly_Score"] = (risk_norm - 2.0) / 10.0
        
        for feat in self.expected_features:
            if feat not in df_out.columns:
                df_out[feat] = 0.0
                
        return df_out[self.expected_features]
