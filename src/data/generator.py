"""
Transaction Stream Generator for Live Simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Generator

class TransactionStreamGenerator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def generate_stream(self) -> Generator[Dict[str, Any], None, None]:
        for _, row in self.dataframe.iterrows():
            yield row.to_dict()

    def get_sample_by_type(self, sample_type: str = "legit") -> Dict[str, Any]:
        if "Class" in self.dataframe.columns:
            if sample_type == "fraud":
                filtered = self.dataframe[self.dataframe["Class"] == 1]
            else:
                filtered = self.dataframe[self.dataframe["Class"] == 0]
            if len(filtered) > 0:
                return filtered.sample(n=1, random_state=42).iloc[0].to_dict()
        return self.dataframe.iloc[0].to_dict()
