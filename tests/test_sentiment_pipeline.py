from __future__ import annotations

import numpy as np
import pandas as pd

from features import FeatureSpec, TechnicalFeatureLibrary, build_default_registry, generate_sentiment_features
from sentiment.pipeline import SentimentPipelineConfig, process_sentiment_stream


def _price_timeline() -> pd.DataFrame:
    ts = pd.date_range("2024-04-01 09:00:00+00:00", periods=6, freq="1h")
    close = np.array([100.0, 100.4, 100.2, 100.6, 100.8, 101.0], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.array([1000, 1100, 900, 1300, 1050, 1200], dtype=float),
        }
    )


def test_sentiment_dedupe_near_identical_headlines():
    headlines = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-04-01 09:10:00+00:00"),
                pd.Timestamp("2024-04-01 09:12:00+00:00"),
                pd.Timestamp("2024-04-01 09:14:00+00:00"),
                pd.Timestamp("2024-04-01 10:00:00+00:00"),
            ],
            "headline": [
                "Apple stock jumps after earnings beat",
                "Apple stock jumps after earnings beat!",
                "APPLE stock jumps after earnings beat",
                "Apple faces lawsuit after product issue",
            ],
            "source": ["reuters", "reuters", "cnbc", "cnbc"],
        }
    )

    events, meta = process_sentiment_stream(
        headlines_df=headlines,
        config=SentimentPipelineConfig(dedupe_similarity_threshold=0.9, dedupe_time_window="12H"),
        ticker_universe={"AAPL"},
    )

    assert meta["dedupe"]["input_count"] == 4
    assert meta["dedupe"]["deduped_count"] == 2
    assert len(events) == 2


def test_sentiment_features_are_causally_aligned_to_timestamps():
    price = _price_timeline()
    headlines = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-04-01 10:30:00+00:00"),
                pd.Timestamp("2024-04-01 11:30:00+00:00"),
            ],
            "headline": [
                "AAPL beats estimates and raises guidance",
                "AAPL faces investigation and profit warning",
            ],
            "source": ["reuters", "benzinga"],
        }
    )

    sent = generate_sentiment_features(
        price,
        headlines_df=headlines,
        symbol="AAPL",
        ticker_universe={"AAPL"},
        params={
            "windows": ["2H"],
            "recency_half_life_minutes": 180.0,
            "spike_lookback": 3,
            "source_credibility": {"reuters": 1.0, "benzinga": 0.7},
        },
    )

    # Before first event, attention should be zero and no lookahead sentiment.
    pre = sent.loc[price["timestamp"] <= pd.Timestamp("2024-04-01 10:00:00+00:00")]
    assert (pre["sent_attention_count_w2H"] == 0).all()
    assert pre["sent_sentiment_mean_w2H"].isna().all()

    # After first event, sentiment appears.
    at_11 = sent.loc[price["timestamp"] == pd.Timestamp("2024-04-01 11:00:00+00:00"), "sent_sentiment_mean_w2H"].iloc[0]
    assert at_11 > 0

    # After second (negative) event, mean should decline versus previous bar.
    at_12 = sent.loc[price["timestamp"] == pd.Timestamp("2024-04-01 12:00:00+00:00"), "sent_sentiment_mean_w2H"].iloc[0]
    assert at_12 < at_11


def test_sentiment_plugin_via_registry():
    price = _price_timeline()
    headlines = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-04-01 10:15:00+00:00"),
                pd.Timestamp("2024-04-01 12:15:00+00:00"),
            ],
            "headline": ["MSFT strong growth outlook", "MSFT downgrade on weak demand"],
            "source": ["reuters", "marketwatch"],
        }
    )

    library = TechnicalFeatureLibrary(build_default_registry())
    out, meta = library.generate(
        price,
        specs=[FeatureSpec(name="sentiment", params={"windows": ["4H"], "spike_lookback": 3})],
        headlines_df=headlines,
        symbol="MSFT",
        ticker_universe={"MSFT"},
        include_base_columns=False,
        drop_na=False,
    )

    assert "sent_sentiment_mean_w4H" in out.columns
    assert "sent_attention_count_w4H" in out.columns
    assert "sentiment" in [item["plugin"] for item in meta["catalog"]]

