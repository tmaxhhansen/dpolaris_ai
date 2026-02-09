"""
Feature Engineering for dPolaris ML Models

Generates features from market data for model training and prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("dpolaris.ml.features")


class FeatureEngine:
    """
    Generate features for ML models from market data.

    Features include:
    - Price-based: returns, moving averages, momentum
    - Volatility: historical vol, ATR, Bollinger width
    - Volume: volume trends, OBV
    - Technical: RSI, MACD, stochastic
    - Time: day of week, month, days to expiration
    """

    def __init__(self):
        self.feature_names: list[str] = []

    def generate_features(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        target_horizon: int = 5,
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume, date
            include_targets: Whether to include target variables
            target_horizon: Days ahead for target calculation

        Returns:
            DataFrame with features (and targets if requested)
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Price features
        df = self._add_price_features(df)

        # Volatility features
        df = self._add_volatility_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Technical indicators
        df = self._add_technical_features(df)

        # Time features
        df = self._add_time_features(df)

        # Target variables
        if include_targets:
            df = self._add_targets(df, horizon=target_horizon)

        # Drop rows with NaN (from lookback calculations)
        df = df.dropna()

        # Store feature names
        self.feature_names = [
            col for col in df.columns
            if col not in ["date", "open", "high", "low", "close", "volume", "symbol"]
            and not col.startswith("target_")
        ]

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        close = df["close"]

        # Returns
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)
        df["return_20d"] = close.pct_change(20)

        # Moving averages
        df["sma_5"] = close.rolling(5).mean()
        df["sma_10"] = close.rolling(10).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()

        # Price relative to MAs
        df["price_sma5_ratio"] = close / df["sma_5"]
        df["price_sma20_ratio"] = close / df["sma_20"]
        df["price_sma50_ratio"] = close / df["sma_50"]
        df["price_sma200_ratio"] = close / df["sma_200"]

        # MA crossovers
        df["sma5_sma20_ratio"] = df["sma_5"] / df["sma_20"]
        df["sma20_sma50_ratio"] = df["sma_20"] / df["sma_50"]
        df["sma50_sma200_ratio"] = df["sma_50"] / df["sma_200"]

        # Exponential moving averages
        df["ema_12"] = close.ewm(span=12).mean()
        df["ema_26"] = close.ewm(span=26).mean()

        # Momentum
        df["momentum_5"] = close - close.shift(5)
        df["momentum_10"] = close - close.shift(10)
        df["momentum_20"] = close - close.shift(20)

        # Rate of change
        df["roc_5"] = (close - close.shift(5)) / close.shift(5)
        df["roc_10"] = (close - close.shift(10)) / close.shift(10)
        df["roc_20"] = (close - close.shift(20)) / close.shift(20)

        # High/Low ranges
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Historical volatility (annualized)
        df["hvol_5"] = df["return_1d"].rolling(5).std() * np.sqrt(252)
        df["hvol_10"] = df["return_1d"].rolling(10).std() * np.sqrt(252)
        df["hvol_20"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
        df["hvol_50"] = df["return_1d"].rolling(50).std() * np.sqrt(252)

        # Volatility ratio (short vs long term)
        df["hvol_ratio_5_20"] = df["hvol_5"] / (df["hvol_20"] + 1e-8)
        df["hvol_ratio_10_50"] = df["hvol_10"] / (df["hvol_50"] + 1e-8)

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr_14"] = true_range.rolling(14).mean()
        df["atr_percent"] = df["atr_14"] / close

        # Bollinger Bands
        df["bb_middle"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

        # Keltner Channel width
        df["kc_width"] = df["atr_14"] * 2 / close

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features"""
        volume = df["volume"]
        close = df["close"]

        # Volume moving averages
        df["vol_sma_5"] = volume.rolling(5).mean()
        df["vol_sma_20"] = volume.rolling(20).mean()
        df["vol_sma_50"] = volume.rolling(50).mean()

        # Volume ratios
        df["vol_ratio_5"] = volume / (df["vol_sma_5"] + 1)
        df["vol_ratio_20"] = volume / (df["vol_sma_20"] + 1)

        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()
        df["obv"] = obv
        df["obv_sma_20"] = obv.rolling(20).mean()
        df["obv_trend"] = obv / (df["obv_sma_20"] + 1e-8)

        # Volume-weighted price
        df["vwap_20"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        df["price_vwap_ratio"] = close / (df["vwap_20"] + 1e-8)

        # Accumulation/Distribution
        clv = ((close - df["low"]) - (df["high"] - close)) / (df["high"] - df["low"] + 1e-8)
        df["ad_line"] = (clv * volume).cumsum()

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # RSI zones
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_cross_up"] = ((df["macd"] > df["macd_signal"]) &
                               (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-8)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R
        df["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-8)

        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = (-low).diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-8)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-8)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # Trend strength
        df["trend_strength"] = df["adx"]
        df["trend_direction"] = (df["plus_di"] > df["minus_di"]).astype(int) * 2 - 1

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])

            # Day of week (0=Monday, 4=Friday)
            df["day_of_week"] = dates.dt.dayofweek

            # Day of month
            df["day_of_month"] = dates.dt.day

            # Month
            df["month"] = dates.dt.month

            # Quarter
            df["quarter"] = dates.dt.quarter

            # Week of year
            df["week_of_year"] = dates.dt.isocalendar().week.astype(int)

            # Is month end/start
            df["is_month_end"] = dates.dt.is_month_end.astype(int)
            df["is_month_start"] = dates.dt.is_month_start.astype(int)

            # Days since year start
            df["days_into_year"] = dates.dt.dayofyear

        return df

    def _add_targets(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Add target variables for supervised learning"""
        close = df["close"]

        # Future return (classification target)
        future_return = close.shift(-horizon) / close - 1
        df["target_return"] = future_return

        # Direction (1=up, 0=down)
        df["target_direction"] = (future_return > 0).astype(int)

        # Magnitude buckets
        df["target_magnitude"] = pd.cut(
            future_return,
            bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
            labels=[0, 1, 2, 3, 4]  # Strong down, down, flat, up, strong up
        ).astype(float)

        # Future volatility
        future_vol = df["return_1d"].shift(-horizon).rolling(horizon).std() * np.sqrt(252)
        df["target_volatility"] = future_vol

        # Max drawdown in next N days
        future_max = close.shift(-1).rolling(horizon).max()
        future_min = close.shift(-1).rolling(horizon).min()
        df["target_max_gain"] = (future_max / close) - 1
        df["target_max_loss"] = (future_min / close) - 1

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of feature names (excluding targets)"""
        return self.feature_names

    def get_feature_importance_baseline(self) -> dict[str, float]:
        """Return baseline feature importance estimates"""
        # Based on typical importance in financial ML
        return {
            "momentum": ["roc_5", "roc_10", "roc_20", "momentum_5", "momentum_10"],
            "trend": ["price_sma20_ratio", "price_sma50_ratio", "sma5_sma20_ratio"],
            "volatility": ["hvol_20", "hvol_ratio_5_20", "bb_width", "atr_percent"],
            "mean_reversion": ["rsi_14", "bb_position", "stoch_k"],
            "volume": ["vol_ratio_20", "obv_trend"],
        }
