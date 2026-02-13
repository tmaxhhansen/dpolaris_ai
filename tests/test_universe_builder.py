from __future__ import annotations

from datetime import datetime, timezone

from universe.builder import build_combined_universe, build_nasdaq_top_500, build_wsb_top_500


def test_nasdaq_builder_uses_avg_dollar_volume_fallback_when_market_cap_missing(tmp_path):
    out_path = tmp_path / "nasdaq_top_500.json"

    symbol_rows = [
        {"symbol": "AAA", "company_name": "AAA Co"},
        {"symbol": "BBB", "company_name": "BBB Co"},
        {"symbol": "CCC", "company_name": "CCC Co"},
    ]

    profiles = {
        "AAA": {"market_cap": None, "avg_dollar_volume": 100.0, "company_name": "AAA Co"},
        "BBB": {"market_cap": None, "avg_dollar_volume": 500.0, "company_name": "BBB Co"},
        "CCC": {"market_cap": None, "avg_dollar_volume": 250.0, "company_name": "CCC Co"},
    }

    payload = build_nasdaq_top_500(
        output_path=out_path,
        top_n=2,
        min_avg_dollar_volume=0.0,
        symbol_rows=symbol_rows,
        profile_fetcher=lambda symbol: profiles[symbol],
        now=datetime(2026, 2, 10, tzinfo=timezone.utc),
    )

    assert payload["criteria"]["ranking"] == "avg_dollar_volume_desc_fallback"
    assert [row["symbol"] for row in payload["tickers"]] == ["BBB", "CCC"]
    assert out_path.exists()


def test_wsb_builder_dedupes_posts_and_extracts_mentions(tmp_path):
    out_path = tmp_path / "wsb_top_500.json"

    posts = [
        {
            "id": "p1",
            "title": "AAPL to the moon",
            "body": "AAPL calls look strong",
            "created_at": "2026-02-09T10:00:00+00:00",
        },
        {
            "id": "p2",
            "title": "AAPL to the moon!!!",
            "body": "AAPL calls look strong",
            "created_at": "2026-02-09T10:05:00+00:00",
        },
        {
            "id": "p3",
            "title": "TSLA and AAPL setup",
            "body": "TSLA momentum, AAPL follow-through",
            "created_at": "2026-02-09T11:00:00+00:00",
        },
    ]

    payload = build_wsb_top_500(
        output_path=out_path,
        top_n=10,
        window_days=7,
        posts=posts,
        valid_tickers={"AAPL", "TSLA"},
        now=datetime(2026, 2, 10, tzinfo=timezone.utc),
    )

    by_symbol = {row["symbol"]: row for row in payload["tickers"]}
    assert by_symbol["AAPL"]["mention_count"] == 2
    assert by_symbol["TSLA"]["mention_count"] == 1
    assert payload["data_sources"][0]["deduped_posts"] == 2
    assert out_path.exists()


def test_combined_universe_merges_duplicates_and_tracks_sources(tmp_path):
    out_path = tmp_path / "combined_1000.json"

    nasdaq_payload = {
        "schema_version": "1.0.0",
        "generated_at": "2026-02-10T00:00:00+00:00",
        "universe_hash": "nasdaqhash",
        "tickers": [
            {"symbol": "AAPL", "company_name": "Apple", "market_cap": 1_000.0, "avg_dollar_volume": 100.0},
            {"symbol": "MSFT", "company_name": "Microsoft", "market_cap": 900.0, "avg_dollar_volume": 95.0},
        ],
    }

    wsb_payload = {
        "schema_version": "1.0.0",
        "generated_at": "2026-02-10T00:00:00+00:00",
        "universe_hash": "wsbhash",
        "tickers": [
            {"symbol": "AAPL", "mention_count": 8, "mention_velocity": 1.1, "sentiment_score": 0.2},
            {"symbol": "TSLA", "mention_count": 6, "mention_velocity": 0.8, "sentiment_score": -0.1},
        ],
    }

    payload = build_combined_universe(
        output_path=out_path,
        nasdaq_payload=nasdaq_payload,
        wsb_payload=wsb_payload,
        top_n=10,
        now=datetime(2026, 2, 10, tzinfo=timezone.utc),
    )

    by_symbol = {row["symbol"]: row for row in payload["merged"]}
    assert set(by_symbol.keys()) == {"AAPL", "MSFT", "TSLA"}
    assert set(by_symbol["AAPL"]["sources"]) == {"nasdaq_top_500", "wsb_top_500"}
    assert by_symbol["AAPL"]["mention_count"] == 8
    assert payload["universe_hash"]
    assert out_path.exists()
