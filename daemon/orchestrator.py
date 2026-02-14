"""
Windows-friendly orchestrator daemon with backend self-healing.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

from daemon.backend_process import (
    BackendProcessConfig,
    BackendProcessManager,
    PortInUseByUnknownProcessError,
)
from tools.slack_notifier import SlackNotifier

logger = logging.getLogger("dpolaris.orchestrator")


@dataclass
class OrchestratorConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    interval_health_seconds: int = 60
    interval_scan_seconds: int = 30 * 60
    scan_timeout_seconds: int = 20 * 60
    scan_poll_seconds: int = 5
    top_n: int = 5
    horizon_days: int = 5
    signal_threshold: float = 0.70
    dry_run: bool = False
    max_backoff_seconds: int = 15 * 60


def parse_duration_seconds(value: str) -> int:
    raw = str(value).strip().lower()
    if not raw:
        raise ValueError("empty duration")
    if raw.endswith("ms"):
        return max(1, int(float(raw[:-2]) / 1000.0))
    if raw.endswith("s"):
        return max(1, int(float(raw[:-1])))
    if raw.endswith("m"):
        return max(1, int(float(raw[:-1]) * 60))
    if raw.endswith("h"):
        return max(1, int(float(raw[:-1]) * 3600))
    return max(1, int(float(raw)))


class OrchestratorDaemon:
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        process_manager: Optional[BackendProcessManager] = None,
        notifier: Optional[SlackNotifier] = None,
    ):
        self.config = config or OrchestratorConfig()
        if process_manager is None:
            process_manager = BackendProcessManager(
                BackendProcessConfig(host=self.config.host, port=self.config.port)
            )
        self.process_manager = process_manager
        self.notifier = notifier or SlackNotifier(dry_run=self.config.dry_run)

        self._base_url = f"http://{self.config.host}:{self.config.port}"
        self._running = False
        self._started_at: Optional[dt.datetime] = None
        self._last_health_check: Optional[dt.datetime] = None
        self._last_scan_run: Optional[dt.datetime] = None
        self._consecutive_health_failures = 0
        self._health_backoff_seconds = self.config.interval_health_seconds
        self._next_health_due = 0.0
        self._next_scan_due = 0.0
        self._last_unhealthy_alert_at: Optional[dt.datetime] = None
        self._lock = threading.RLock()

    def run_forever(self) -> None:
        self._running = True
        now = time.monotonic()
        self._next_health_due = now
        self._next_scan_due = now
        self._started_at = dt.datetime.utcnow()
        logger.info("Orchestrator started for %s", self._base_url)

        while self._running:
            now = time.monotonic()
            if now >= self._next_health_due:
                self._run_health_cycle()
            if now >= self._next_scan_due:
                if self._is_market_hours():
                    self._run_scan_cycle()
                    self._last_scan_run = dt.datetime.utcnow()
                self._next_scan_due = time.monotonic() + self.config.interval_scan_seconds
            time.sleep(1)

    def stop(self) -> None:
        self._running = False

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            backend = self.process_manager.get_state()
            return {
                "running": self._running,
                "uptime_seconds": self._uptime_seconds(),
                "last_health_check": self._iso(self._last_health_check),
                "last_scan_run": self._iso(self._last_scan_run),
                "consecutive_health_failures": self._consecutive_health_failures,
                "health_backoff_seconds": self._health_backoff_seconds,
                "backend_state": backend,
                "last_restart": backend.get("last_restart"),
                "restart_count_24h": backend.get("restart_count_24h", 0),
            }

    def force_restart_backend(self, reason: str = "api") -> dict[str, Any]:
        try:
            self._restart_backend(reason=reason)
            healthy = self.process_manager.wait_until_healthy(timeout_seconds=20)
            return {"status": "ok", "healthy": healthy, "backend": self.process_manager.get_state()}
        except Exception as exc:
            return {"status": "error", "detail": str(exc), "backend": self.process_manager.get_state()}

    def _run_health_cycle(self) -> None:
        self._last_health_check = dt.datetime.utcnow()
        healthy = self.process_manager.check_health(timeout_seconds=2.0)
        if healthy:
            self._consecutive_health_failures = 0
            self._health_backoff_seconds = self.config.interval_health_seconds
            self._next_health_due = time.monotonic() + self.config.interval_health_seconds
            return

        self._consecutive_health_failures += 1
        reason = f"health check failed ({self._consecutive_health_failures})"
        logger.warning("Backend unhealthy: %s", reason)

        try:
            self._restart_backend(reason=reason)
            if not self.process_manager.wait_until_healthy(timeout_seconds=20):
                raise RuntimeError("backend failed to become healthy after restart")
            self._send_backend_restarted_alert(reason=reason)
            self._consecutive_health_failures = 0
            self._health_backoff_seconds = self.config.interval_health_seconds
        except PortInUseByUnknownProcessError as exc:
            self._send_unhealthy_alert(
                title="Backend Unhealthy (Port Ownership Conflict)",
                detail=str(exc),
                include_logs=True,
            )
            self._health_backoff_seconds = min(
                self.config.max_backoff_seconds,
                max(self.config.interval_health_seconds, self._health_backoff_seconds * 2),
            )
        except Exception as exc:
            self._send_unhealthy_alert(
                title="Backend Unhealthy (Restart Failed)",
                detail=str(exc),
                include_logs=True,
            )
            self._health_backoff_seconds = min(
                self.config.max_backoff_seconds,
                max(self.config.interval_health_seconds, self._health_backoff_seconds * 2),
            )

        self._next_health_due = time.monotonic() + self._health_backoff_seconds

    def _restart_backend(self, reason: str) -> None:
        self.process_manager.restart_backend(reason=reason)

    def _run_scan_cycle(self) -> None:
        if not self.process_manager.check_health(timeout_seconds=2.0):
            logger.warning("Skipping scan cycle because backend is unhealthy")
            return
        try:
            run_id = self._start_scan()
            if not run_id:
                return
            completed = self._wait_for_scan(run_id)
            if not completed:
                self._send_unhealthy_alert(
                    title="Scan Timeout",
                    detail=f"Scan run {run_id} did not complete in time.",
                    include_logs=False,
                )
                return
            rows = self._fetch_scan_results(run_id, self.config.top_n)
            self._process_signals_and_alert(rows)
        except Exception as exc:
            logger.exception("Scan cycle failed: %s", exc)
            self._send_unhealthy_alert(
                title="Scan Cycle Failure",
                detail=str(exc),
                include_logs=False,
            )

    def _start_scan(self) -> Optional[str]:
        payload = {"universe": "all"}
        body, status = self._request_json("POST", "/api/scan/start", payload, timeout=10)
        if status >= 400:
            fallback_payload = {"universe": "combined_1000"}
            body, status = self._request_json("POST", "/api/scan/start", fallback_payload, timeout=10)
        if status >= 400:
            raise RuntimeError(f"scan start failed: status={status} body={body}")
        run_id = str((body or {}).get("runId") or "").strip()
        if not run_id:
            raise RuntimeError(f"scan start returned no runId: {body}")
        return run_id

    def _wait_for_scan(self, run_id: str) -> bool:
        deadline = time.monotonic() + self.config.scan_timeout_seconds
        final_states = {"completed", "failed", "error", "cancelled"}
        while time.monotonic() < deadline:
            body, status = self._request_json("GET", f"/api/scan/status/{run_id}", timeout=8)
            if status >= 400:
                raise RuntimeError(f"scan status failed: status={status} body={body}")
            state = str((body or {}).get("status") or "").strip().lower()
            if state in final_states:
                return state == "completed"
            time.sleep(max(1, self.config.scan_poll_seconds))
        return False

    def _fetch_scan_results(self, run_id: str, limit: int) -> list[dict[str, Any]]:
        path = f"/api/scan/results/{run_id}?page=1&pageSize={max(1, int(limit))}"
        body, status = self._request_json("GET", path, timeout=10)
        if status >= 400:
            raise RuntimeError(f"scan results failed: status={status} body={body}")
        items = body.get("items") if isinstance(body, dict) else []
        if not isinstance(items, list):
            return []
        return [x for x in items if isinstance(x, dict)]

    def _process_signals_and_alert(self, rows: list[dict[str, Any]]) -> None:
        alerts: list[dict[str, Any]] = []
        for row in rows[: self.config.top_n]:
            symbol = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            path = f"/api/signals/{urlparse.quote(symbol)}?horizon_days={int(self.config.horizon_days)}"
            signal, status = self._request_json("POST", path, timeout=20)
            if status >= 400 or not isinstance(signal, dict):
                continue
            confidence = self._safe_float(signal.get("confidence"), 0.0)
            bias = str(signal.get("bias") or "UNKNOWN")
            if confidence >= float(self.config.signal_threshold) and bias != "NO_TRADE":
                alerts.append(
                    {
                        "symbol": symbol,
                        "confidence": round(confidence, 4),
                        "bias": bias,
                        "setup_type": signal.get("setup_type"),
                    }
                )

        if not alerts:
            return

        lines = [
            f"{item['symbol']} | bias={item['bias']} | confidence={item['confidence']:.3f} | setup={item.get('setup_type')}"
            for item in alerts
        ]
        self.notifier.send(
            "Scan Alerts",
            "Top signals above threshold:\n" + "\n".join(lines),
            fields={"threshold": self.config.signal_threshold, "count": len(alerts)},
        )

    def _send_backend_restarted_alert(self, reason: str) -> None:
        tail = self.process_manager.get_tail_lines(20)
        self.notifier.send(
            "Backend Restarted",
            f"Reason: {reason}\nRecent logs:\n" + "\n".join(tail[-20:]),
            fields={
                "host": self.config.host,
                "port": self.config.port,
                "restart_count_24h": self.process_manager.get_state().get("restart_count_24h", 0),
            },
        )

    def _send_unhealthy_alert(self, title: str, detail: str, include_logs: bool) -> None:
        now = dt.datetime.utcnow()
        if self._last_unhealthy_alert_at and (now - self._last_unhealthy_alert_at).total_seconds() < 30:
            return
        self._last_unhealthy_alert_at = now

        body = detail
        if include_logs:
            tail = self.process_manager.get_tail_lines(20)
            if tail:
                body += "\nRecent logs:\n" + "\n".join(tail)

        self.notifier.send(
            title,
            body,
            fields={
                "failures": self._consecutive_health_failures,
                "backoff_seconds": self._health_backoff_seconds,
            },
            level="error",
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout: int = 10,
    ) -> tuple[dict[str, Any], int]:
        url = f"{self._base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            import json

            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urlrequest.Request(url, method=method.upper(), headers=headers, data=data)
        try:
            with urlrequest.urlopen(req, timeout=max(1, int(timeout))) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                body = self._safe_json(raw)
                return body, int(resp.status)
        except urlerror.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return self._safe_json(raw), int(exc.code)
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            return {"error": str(exc)}, 599

    @staticmethod
    def _safe_json(raw: str) -> dict[str, Any]:
        import json

        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            return {"data": data}
        except Exception:
            return {"raw": raw}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _iso(value: Optional[dt.datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    def _uptime_seconds(self) -> int:
        if self._started_at is None:
            return 0
        return max(0, int((dt.datetime.utcnow() - self._started_at).total_seconds()))

    @staticmethod
    def _is_market_hours() -> bool:
        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now_et <= close_t


_ORCHESTRATOR_SINGLETON: Optional[OrchestratorDaemon] = None
_ORCH_LOCK = threading.RLock()


def get_orchestrator_singleton() -> OrchestratorDaemon:
    global _ORCHESTRATOR_SINGLETON
    with _ORCH_LOCK:
        if _ORCHESTRATOR_SINGLETON is None:
            _ORCHESTRATOR_SINGLETON = OrchestratorDaemon()
        return _ORCHESTRATOR_SINGLETON

