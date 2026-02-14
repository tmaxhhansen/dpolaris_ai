"""
Slack notifier utility for dPolaris.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml

logger = logging.getLogger("dpolaris.slack")


def _settings_path() -> Path:
    raw = os.getenv("DPOLARIS_SETTINGS_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path("~/dpolaris_data/config/settings.yaml").expanduser()


def _webhook_from_settings(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    direct = data.get("slack_webhook_url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    notifications = data.get("notifications")
    if isinstance(notifications, dict):
        value = notifications.get("slack_webhook_url")
        if isinstance(value, str) and value.strip():
            return value.strip()

        slack = notifications.get("slack")
        if isinstance(slack, dict):
            value = slack.get("webhook_url")
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def resolve_slack_webhook_url() -> Optional[str]:
    env_value = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if env_value:
        return env_value
    return _webhook_from_settings(_settings_path())


class SlackNotifier:
    def __init__(self, webhook_url: Optional[str] = None, dry_run: bool = False):
        self.webhook_url = webhook_url or resolve_slack_webhook_url()
        self.dry_run = bool(dry_run)

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(
        self,
        title: str,
        body: str,
        *,
        fields: Optional[dict[str, Any]] = None,
        level: str = "info",
    ) -> bool:
        payload = self._build_payload(title=title, body=body, fields=fields)

        if self.dry_run:
            logger.info("[dry-run] Slack message: %s", payload.get("text", ""))
            return True

        if not self.webhook_url:
            logger.warning("Slack webhook URL is not configured; dropping message: %s", title)
            return False

        body_bytes = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.webhook_url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=6) as resp:
                ok = 200 <= int(resp.status) < 300
                if not ok:
                    logger.warning("Slack webhook returned status %s", resp.status)
                return ok
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            if level.lower() == "error":
                logger.error("Failed posting Slack alert: %s", exc)
            else:
                logger.warning("Failed posting Slack alert: %s", exc)
            return False

    @staticmethod
    def _build_payload(
        *,
        title: str,
        body: str,
        fields: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        lines = [f"*{title}*", body.strip()]
        if fields:
            for key, value in fields.items():
                lines.append(f"*{key}:* {value}")
        text = "\n".join(x for x in lines if x)
        return {"text": text}

