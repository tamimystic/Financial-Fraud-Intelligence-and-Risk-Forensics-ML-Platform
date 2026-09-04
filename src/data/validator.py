"""
Data Integrity and Schema Validator.
"""

import pandas as pd
from typing import List, Dict, Any

class DataValidator:
    def __init__(self, required_columns: List[str] = None):
        if required_columns is None:
            self.required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        else:
            self.required_columns = required_columns

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        null_counts = int(df[self.required_columns].isnull().sum().sum()) if not missing_cols else -1
        is_valid = len(missing_cols) == 0 and null_counts == 0
        return {
            "is_valid": is_valid,
            "missing_columns": missing_cols,
            "total_null_entries": null_counts,
            "row_count": len(df)
        }
