"""
Leak-Free Data Loader and Temporal Splitter.
"""

import os
import pandas as pd
from typing import Tuple
from src.config.settings import RAW_DATA_PATH

class DataLoader:
    def __init__(self, raw_data_path: str = RAW_DATA_PATH):
        self.raw_data_path = raw_data_path

    def load_raw_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_data_path}")
        return pd.read_csv(self.raw_data_path)

    def temporal_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sorted_df = df.sort_values(by="Time").reset_index(drop=True)
        total_len = len(sorted_df)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        train_df = sorted_df.iloc[:train_end].copy()
        val_df = sorted_df.iloc[train_end:val_end].copy()
        test_df = sorted_df.iloc[val_end:].copy()
        return train_df, val_df, test_df
