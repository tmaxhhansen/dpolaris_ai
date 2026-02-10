from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from api import server


class _StubMarketService:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    async def get_historical(self, symbol: str, days: int = 365):
        _ = symbol, days
        return self.frame.copy()


class _Cfg:
    class ml:
        training_data_days = 3650


def _synthetic_ohlcv() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=320, freq="B", tz="UTC")
    base = np.linspace(100.0, 140.0, len(ts))
    return pd.DataFrame(
        {
            "date": ts,
            "open": base * 0.999,
            "high": base * 1.005,
            "low": base * 0.995,
            "close": base,
            "volume": np.linspace(1_000_000, 2_000_000, len(ts)).astype(int),
        }
    )


def test_predict_inspect_endpoint_uses_asof_causal_data(monkeypatch):
    source = _synthetic_ohlcv()
    inspect_time = pd.Timestamp(source.iloc[220]["date"]).tz_convert("UTC")

    # Corrupt only future rows; causal inspect should ignore these values.
    future_mask = pd.to_datetime(source["date"], utc=True) > inspect_time
    source.loc[future_mask, "close"] = source.loc[future_mask, "close"] * 50.0

    monkeypatch.setattr(server, "market_service", _StubMarketService(source))
    monkeypatch.setattr(server, "config", _Cfg())
    monkeypatch.setattr(
        server,
        "_predict_symbol_direction",
        lambda symbol, df: {
            "source": "classic_ml",
            "model_name": symbol,
            "model_type": "logistic",
            "prediction_label": "UP",
            "confidence": 0.71,
            "probability_up": 0.67,
            "probability_down": 0.33,
        },
    )

    response = asyncio.run(
        server.inspect_prediction(
            ticker="SPY",
            time=inspect_time.isoformat(),
            horizon=5,
            run_id=None,
        )
    )

    resolved = pd.Timestamp(response["resolved_time"]).tz_convert("UTC")
    assert resolved <= inspect_time
    assert response["trace_meta"]["causal_asof"] is True

    expected_close = float(source.loc[source["date"] <= inspect_time, "close"].iloc[-1])
    inspected_close = float(response["raw_input_snapshot"]["ohlcv"]["close"])
    assert inspected_close == expected_close

    # Future corruption must not dominate feature snapshot at inspect timestamp.
    raw_features = response["feature_vector"]["raw"]
    assert "return_1d" in raw_features
    assert "price_sma20_ratio" in raw_features
