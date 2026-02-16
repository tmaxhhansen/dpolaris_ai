from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_universe_endpoints_generate_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("DPOLARIS_UNIVERSE_DIR", str(tmp_path / "universe"))
    monkeypatch.setenv("DPOLARIS_UNIVERSE_METADATA_REFRESH_LIMIT", "0")

    from api.server import get_scan_universe, list_universe_names

    names = asyncio.run(list_universe_names())
    assert isinstance(names, list)
    assert {"nasdaq500", "watchlist", "combined"}.issubset(set(names))

    for name in ("nasdaq500", "wsb100", "combined"):
        payload = asyncio.run(get_scan_universe(name))
        assert payload["name"] == name
        assert int(payload.get("count") or 0) > 0
        tickers = payload.get("tickers")
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        first = tickers[0]
        assert "symbol" in first
        assert "name" in first
        assert "sector" in first
        assert "market_cap" in first
        assert "avg_volume_7d" in first
        assert "change_pct_1d" in first
        assert "last_analysis_date" in first

    watchlist_payload = asyncio.run(get_scan_universe("watchlist"))
    assert watchlist_payload["name"] == "watchlist"
    assert int(watchlist_payload.get("count") or 0) == 0

    custom_alias_payload = asyncio.run(get_scan_universe("custom"))
    assert custom_alias_payload["name"] == "watchlist"

    combined_alias = asyncio.run(get_scan_universe("combined400"))
    assert combined_alias["name"] == "combined"


def test_watchlist_endpoints_add_remove_symbol(monkeypatch, tmp_path):
    monkeypatch.setenv("DPOLARIS_UNIVERSE_DIR", str(tmp_path / "universe"))

    from api import server

    previous_config = server.config
    server.config = SimpleNamespace(data_dir=tmp_path)
    try:
        added = asyncio.run(server.add_watchlist_symbol("TSLA"))
        assert added.get("status") == "ok"
        assert added.get("symbol") == "TSLA"

        watchlist_payload = asyncio.run(server.get_watchlist_universe_endpoint())
        watchlist_symbols = {row.get("symbol") for row in watchlist_payload.get("tickers") or [] if isinstance(row, dict)}
        assert "TSLA" in watchlist_symbols

        combined_payload = asyncio.run(server.get_scan_universe("combined"))
        combined_symbols = {row.get("symbol") for row in combined_payload.get("tickers") or [] if isinstance(row, dict)}
        assert "TSLA" in combined_symbols

        removed = asyncio.run(server.remove_watchlist_symbol("TSLA"))
        assert removed.get("status") == "ok"
    finally:
        server.config = previous_config
