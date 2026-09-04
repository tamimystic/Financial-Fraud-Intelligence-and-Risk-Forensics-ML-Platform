"""
Feature Matrix Column Definitions and Mapping.
"""

from typing import List

ALL_EXPECTED_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)] + [
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
