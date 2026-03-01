import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .indicators import TechnicalIndicators


class DataLoader:
    def __init__(self, ticker: str, start: str, end: str, cache_dir: Optional[str] = None):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._raw: Optional[pd.DataFrame] = None
        self._processed: Optional[pd.DataFrame] = None

    def fetch(self) -> pd.DataFrame:
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self.ticker}_{self.start}_{self.end}.csv"
            if cache_file.exists():
                self._raw = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return self._raw

        self._raw = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True, progress=False)
        if self._raw.empty:
            raise ValueError(f"No data returned for ticker {self.ticker}")

        if isinstance(self._raw.columns, pd.MultiIndex):
            self._raw.columns = self._raw.columns.get_level_values(0)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._raw.to_csv(cache_file)

        return self._raw

    def process(self, feature_columns: Optional[list] = None) -> pd.DataFrame:
        if self._raw is None:
            self.fetch()

        df = TechnicalIndicators.compute_all(self._raw)
        df.dropna(inplace=True)

        if feature_columns:
            df = df[feature_columns]

        self._processed = df
        return df

    def get_feature_columns(self) -> list:
        if self._processed is None:
            self.process()
        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        indicator_cols = [c for c in self._processed.columns if c not in price_cols]
        return price_cols + indicator_cols

    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._processed is None:
            self.process()

        n = len(self._processed)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = self._processed.iloc[:train_end]
        val = self._processed.iloc[train_end:val_end]
        test = self._processed.iloc[val_end:]

        return train, val, test

    @staticmethod
    def normalize(df: pd.DataFrame, stats: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
        if stats is None:
            stats = {"mean": df.mean(), "std": df.std().replace(0, 1e-8)}
        normalized = (df - stats["mean"]) / stats["std"]
        return normalized, stats
