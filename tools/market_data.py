"""
Market Data Service for dPolaris

Fetches stock and options data from various sources:
- Yahoo Finance (free, no API key)
- Polygon.io (optional, better data)
- Alpha Vantage (optional, fundamental data)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger("dpolaris.tools.market_data")

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


async def fetch_quote(symbol: str) -> Optional[dict]:
    """
    Fetch current quote for a symbol.

    Returns:
        Dict with price, change, volume, etc.
    """
    if not HAS_YFINANCE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
            "open": info.get("open") or info.get("regularMarketOpen"),
            "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        return None


async def fetch_historical_data(
    symbol: str,
    days: int = 365,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data.

    Args:
        symbol: Stock ticker
        days: Number of days of history
        interval: Data interval ('1d', '1h', '5m', etc.)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if not HAS_YFINANCE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )

        if df.empty:
            return None

        # Standardize column names
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Rename 'date' column if needed
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        # Ensure required columns
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {symbol}")
                return None

        return df[required]

    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None


async def fetch_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
) -> Optional[dict]:
    """
    Fetch options chain data.

    Args:
        symbol: Stock ticker
        expiration: Optional specific expiration date (YYYY-MM-DD)

    Returns:
        Dict with calls, puts, and metadata
    """
    if not HAS_YFINANCE:
        return None

    try:
        ticker = yf.Ticker(symbol)

        # Get available expirations
        expirations = ticker.options

        if not expirations:
            return None

        # Use specified expiration or nearest
        if expiration and expiration in expirations:
            exp_date = expiration
        else:
            exp_date = expirations[0]  # Nearest expiration

        # Fetch options chain
        opt = ticker.option_chain(exp_date)

        # Get current price for reference
        info = ticker.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        def process_options_df(df: pd.DataFrame, option_type: str) -> list[dict]:
            """Process options DataFrame to list of dicts"""
            records = []
            for _, row in df.iterrows():
                record = {
                    "strike": row["strike"],
                    "last_price": row.get("lastPrice"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "volume": row.get("volume"),
                    "open_interest": row.get("openInterest"),
                    "implied_volatility": row.get("impliedVolatility"),
                    "in_the_money": row.get("inTheMoney"),
                    "type": option_type,
                }
                records.append(record)
            return records

        calls = process_options_df(opt.calls, "call")
        puts = process_options_df(opt.puts, "put")

        return {
            "symbol": symbol,
            "expiration": exp_date,
            "expirations_available": list(expirations),
            "current_price": current_price,
            "calls": calls,
            "puts": puts,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        return None


async def calculate_iv_metrics(symbol: str) -> Optional[dict]:
    """
    Calculate IV metrics for a symbol.

    Returns:
        IV rank, IV percentile, current IV, HV comparison
    """
    if not HAS_YFINANCE:
        return None

    try:
        # Get historical data for HV calculation
        df = await fetch_historical_data(symbol, days=365)
        if df is None:
            return None

        # Calculate historical volatility
        returns = df["close"].pct_change()
        hv_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        hv_50 = returns.rolling(50).std().iloc[-1] * np.sqrt(252)

        # Get options data for IV
        options = await fetch_options_chain(symbol)
        if not options or not options["calls"]:
            return None

        # Calculate average ATM IV
        current_price = options["current_price"]
        atm_calls = [
            c for c in options["calls"]
            if abs(c["strike"] - current_price) / current_price < 0.05
            and c["implied_volatility"]
        ]

        if not atm_calls:
            return None

        current_iv = np.mean([c["implied_volatility"] for c in atm_calls])

        # Calculate IV rank (simplified - would need historical IV data for accurate calculation)
        # Using HV as proxy for historical IV range
        iv_rank = min(100, max(0, (current_iv - hv_50) / (hv_20 - hv_50 + 0.01) * 100))

        return {
            "symbol": symbol,
            "current_iv": current_iv,
            "hv_20": hv_20,
            "hv_50": hv_50,
            "iv_hv_spread": current_iv - hv_20,
            "iv_rank": iv_rank,  # Simplified
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error calculating IV metrics for {symbol}: {e}")
        return None


class MarketDataService:
    """
    Service for fetching and caching market data.
    """

    def __init__(self, cache_ttl: int = 60):
        self.cache_ttl = cache_ttl
        self._cache: dict = {}

    async def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[dict]:
        """Get quote with caching"""
        cache_key = f"quote_{symbol}"
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        data = await fetch_quote(symbol)
        if data:
            self._set_cache(cache_key, data)
        return data

    async def get_historical(
        self,
        symbol: str,
        days: int = 365,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
        cache_key = f"hist_{symbol}_{days}"
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        data = await fetch_historical_data(symbol, days)
        if data is not None:
            self._set_cache(cache_key, data)
        return data

    async def get_options(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[dict]:
        """Get options chain with caching"""
        cache_key = f"opt_{symbol}_{expiration}"
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        data = await fetch_options_chain(symbol, expiration)
        if data:
            self._set_cache(cache_key, data)
        return data

    async def get_iv_metrics(self, symbol: str, use_cache: bool = True) -> Optional[dict]:
        """Get IV metrics with caching"""
        cache_key = f"iv_{symbol}"
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        data = await calculate_iv_metrics(symbol)
        if data:
            self._set_cache(cache_key, data)
        return data

    async def get_multiple_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get quotes for multiple symbols concurrently"""
        tasks = [self.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result
            for symbol, result in zip(symbols, results)
            if not isinstance(result, Exception) and result is not None
        }

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        entry = self._cache[key]
        age = (datetime.now() - entry["timestamp"]).total_seconds()
        return age < self.cache_ttl

    def _set_cache(self, key: str, data):
        """Set cache entry"""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now(),
        }

    def clear_cache(self):
        """Clear all cached data"""
        self._cache = {}


# Screening functions

async def screen_high_iv_rank(
    symbols: list[str],
    min_iv_rank: float = 50,
) -> list[dict]:
    """
    Screen for stocks with high IV rank.

    Args:
        symbols: List of symbols to screen
        min_iv_rank: Minimum IV rank threshold

    Returns:
        List of symbols meeting criteria with IV data
    """
    service = MarketDataService()
    results = []

    for symbol in symbols:
        try:
            iv_data = await service.get_iv_metrics(symbol)
            if iv_data and iv_data.get("iv_rank", 0) >= min_iv_rank:
                results.append(iv_data)
        except Exception as e:
            logger.debug(f"Skipping {symbol}: {e}")
            continue

    # Sort by IV rank descending
    results.sort(key=lambda x: x.get("iv_rank", 0), reverse=True)
    return results


async def screen_momentum(
    symbols: list[str],
    min_roc: float = 0.05,
    lookback: int = 20,
) -> list[dict]:
    """
    Screen for stocks with strong momentum.

    Args:
        symbols: List of symbols to screen
        min_roc: Minimum rate of change
        lookback: Lookback period for ROC calculation

    Returns:
        List of symbols with positive momentum
    """
    service = MarketDataService()
    results = []

    for symbol in symbols:
        try:
            df = await service.get_historical(symbol, days=lookback + 10)
            if df is None or len(df) < lookback:
                continue

            close = df["close"]
            roc = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]

            if roc >= min_roc:
                results.append({
                    "symbol": symbol,
                    "roc": roc,
                    "price": close.iloc[-1],
                    "lookback": lookback,
                })
        except Exception as e:
            logger.debug(f"Skipping {symbol}: {e}")
            continue

    # Sort by ROC descending
    results.sort(key=lambda x: x["roc"], reverse=True)
    return results


async def get_market_overview() -> dict:
    """
    Get market overview (SPY, QQQ, VIX, sector ETFs).
    """
    service = MarketDataService()

    indices = ["SPY", "QQQ", "IWM", "DIA"]
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]
    volatility = ["VIX", "UVXY", "SVXY"]

    # Fetch all quotes
    all_symbols = indices + sectors + volatility
    quotes = await service.get_multiple_quotes(all_symbols)

    return {
        "indices": {s: quotes.get(s) for s in indices if quotes.get(s)},
        "sectors": {s: quotes.get(s) for s in sectors if quotes.get(s)},
        "volatility": {s: quotes.get(s) for s in volatility if quotes.get(s)},
        "timestamp": datetime.now().isoformat(),
    }
