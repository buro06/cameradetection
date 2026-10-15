"""ntfy (https://ntfy.sh or self-hosted) alert delivery: a push notification per alert, with the clip attached."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from .config import NtfyConfig
from .events import EventResult
from .timefmt import clock

if TYPE_CHECKING:
    from .engine import Engine

log = logging.getLogger(__name__)

ALERT_TAGS = "rotating_light"  # shown as 🚨


class NtfyError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def _auth(cfg: NtfyConfig, password: str) -> tuple[str, str] | None:
    return (cfg.username, password) if cfg.username else None


def _raise_for(r: requests.Response) -> None:
    if r.ok:
        return
    try:
        detail = r.json().get("error") or r.text
    except ValueError:
        detail = r.text
    raise NtfyError(f"HTTP {r.status_code}: {str(detail).strip()[:200]}", status=r.status_code)


def publish(cfg: NtfyConfig, password: str, message: str, *, title: str | None = None, tags: str | None = None,
            priority: int | None = None, attachment: Path | None = None, http_timeout: float = 30) -> None:
    """Publish one message to the configured topic. Metadata goes in query parameters, which (unlike
    headers) carry UTF-8 safely; with `attachment` the file is the request body."""
    if not cfg.topic:
        raise NtfyError("no topic configured")
    url = f"{cfg.server.rstrip('/')}/{cfg.topic}"
    params = {k: v for k, v in {"title": title, "tags": tags, "priority": priority}.items() if v is not None}
    if attachment is not None:
        params.update(message=message, filename=attachment.name)
        with open(attachment, "rb") as f:
            r = requests.put(url, data=f, params=params, auth=_auth(cfg, password), timeout=http_timeout)
    else:
        r = requests.post(url, data=message.encode("utf-8"), params=params, auth=_auth(cfg, password),
                          timeout=http_timeout)
    _raise_for(r)


def check_login(cfg: NtfyConfig, password: str) -> str:
    """Verify the server is reachable and the credentials work, without publishing. Returns the user name."""
    r = requests.get(f"{cfg.server.rstrip('/')}/v1/account", auth=_auth(cfg, password), timeout=15)
    _raise_for(r)
    try:
        return r.json().get("username") or "anonymous"
    except ValueError:
        raise NtfyError("not an ntfy server (no JSON from /v1/account)") from None


def clip_rejected(e: NtfyError) -> bool:
    """The server refused the attachment itself (too large, or attachments not enabled)."""
    return e.status == 413 or (e.status == 400 and "attachment" in str(e).lower())


def retryable(e: NtfyError) -> bool:
    return e.status is None or e.status == 429 or e.status >= 500


class NtfyNotifier:
    def __init__(self, engine: Engine):
        self.cfg = engine.cfg
        self._halt = threading.Event()
        self._outbox: queue.Queue = queue.Queue(maxsize=50)
        self.ok = True
        self.last_error = ""

    def start(self) -> None:
        threading.Thread(target=self._send_loop, name="ntfy-send", daemon=True).start()

    def stop(self) -> None:
        self._halt.set()

    def publish(self, message: str, **kw) -> None:
        publish(self.cfg.ntfy, self.cfg.ntfy_password(), message, **kw)

    # ---- outgoing -------------------------------------------------------------
    def _enqueue(self, fn, *args) -> None:
        try:
            self._outbox.put_nowait((fn, args))
        except queue.Full:
            log.error("ntfy outbox full, dropping message")

    def _send_loop(self) -> None:
        while not self._halt.is_set():
            try:
                fn, args = self._outbox.get(timeout=1)
            except queue.Empty:
                continue
            self._deliver(fn, *args)

    def _deliver(self, fn, *args) -> None:
        for attempt in range(4):
            try:
                fn(*args)
                self.ok, self.last_error = True, ""
                return
            except NtfyError as e:
                self.ok, self.last_error = False, str(e)
                if not retryable(e):
                    log.error("ntfy send failed (%s); check server, topic and credentials", e)
                    return
                log.warning("ntfy send failed (%s), retrying", e)
            except requests.RequestException as e:
                self.ok, self.last_error = False, str(e)
                log.warning("ntfy network error (%s), retrying", e)
            except Exception:
                log.exception("ntfy send error")
                return
            if attempt < 3:
                time.sleep(2 ** attempt * 3)

    def send_text(self, text: str, title: str | None = None) -> None:
        self._enqueue(lambda: self.publish(text, title=title))

    def send_alert(self, result: EventResult, clip: Path | None) -> None:
        when = datetime.fromtimestamp(result.wall_time)
        title = f"{result.camera} — {clock(when)} ({when:%a %d %b})"
        lines = [f"{'🟢' if p.trusted else ('🟠' if p.name else '🔴')} {p.scored_label}" for p in result.people]
        message = "\n".join(lines) or "nobody identified"
        if not (self.cfg.ntfy.attach_clip and clip is not None and clip.exists()):
            clip = None
        self._enqueue(self._send_alert, message, title, clip)

    def _send_alert(self, message: str, title: str, clip: Path | None) -> None:
        kw = dict(title=title, tags=ALERT_TAGS, priority=self.cfg.ntfy.priority)
        if clip is not None:
            try:
                self.publish(message, attachment=clip, **kw)
                return
            except NtfyError as e:
                if not clip_rejected(e):
                    raise
                log.warning("ntfy refused the clip (%s), sending the alert without it", e)
                message += f"\n(clip not attached: {e})"
        self.publish(message, **kw)
