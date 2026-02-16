from __future__ import annotations

import asyncio


def test_universe_endpoints_generate_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("DPOLARIS_UNIVERSE_DIR", str(tmp_path / "universe"))

    from api.server import get_scan_universe, list_universe_names

    names = asyncio.run(list_universe_names())
    assert isinstance(names, list)
    assert {"nasdaq300", "wsb100", "combined"}.issubset(set(names))

    for name in ("nasdaq300", "wsb100", "combined"):
        payload = asyncio.run(get_scan_universe(name))
        assert payload["name"] == name
        assert int(payload.get("count") or 0) > 0
        tickers = payload.get("tickers")
        assert isinstance(tickers, list)
        assert len(tickers) > 0
