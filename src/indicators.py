import numpy as np
import pandas as pd


class TechnicalIndicators:
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
        middle = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df["Close"]
        high = df.get("High", close)
        low = df.get("Low", close)
        volume = df.get("Volume", pd.Series(np.ones(len(close)), index=close.index))

        result["sma_10"] = TechnicalIndicators.sma(close, 10)
        result["sma_20"] = TechnicalIndicators.sma(close, 20)
        result["sma_50"] = TechnicalIndicators.sma(close, 50)
        result["ema_12"] = TechnicalIndicators.ema(close, 12)
        result["ema_26"] = TechnicalIndicators.ema(close, 26)
        result["rsi"] = TechnicalIndicators.rsi(close, 14)

        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = histogram

        bb_upper, bb_mid, bb_lower = TechnicalIndicators.bollinger_bands(close)
        result["bb_upper"] = bb_upper
        result["bb_mid"] = bb_mid
        result["bb_lower"] = bb_lower
        result["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        result["atr"] = TechnicalIndicators.atr(high, low, close, 14)
        result["obv"] = TechnicalIndicators.obv(close, volume)

        k, d = TechnicalIndicators.stochastic_oscillator(high, low, close)
        result["stoch_k"] = k
        result["stoch_d"] = d

        result["williams_r"] = TechnicalIndicators.williams_r(high, low, close)

        result["price_change"] = close.pct_change()
        result["volatility"] = close.pct_change().rolling(10).std()

        return result
