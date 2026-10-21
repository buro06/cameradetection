import numpy as np
import pytest

from camwatch import ntfy
from camwatch.camera import BufferedFrame
from camwatch.events import EventResult, PersonInfo
from camwatch.ntfy import NtfyError, NtfyNotifier

CLIP_URL = "https://ntfy.example/file/m1.mp4"


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self.ok, self._body = status, status < 400, body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.fixture
def http(monkeypatch):
    """Records requests; `http.responses` is a list of statuses (or FakeResponses) served in order, then 200.
    A 200 answers like ntfy: message id m<n>, and an attachment URL when a file was uploaded."""
    class Http:
        calls = []
        responses = []

        def _reply(self, method, url, data=None, **kw):
            body = data.read() if hasattr(data, "read") else data
            self.calls.append({"method": method, "url": url, "body": body, **kw})
            r = self.responses.pop(0) if self.responses else 200
            if isinstance(r, FakeResponse):
                return r
            if r != 200:
                return FakeResponse(r, {"error": "attachment too large"})
            n = len(self.calls)
            filename = (kw.get("params") or {}).get("filename")
            att = {"attachment": {"url": f"https://ntfy.example/file/m{n}.{filename.rsplit('.', 1)[-1]}"}} if filename else {}
            return FakeResponse(200, {"id": f"m{n}", **att})

    h = Http()
    for method in ("put", "post", "get"):
        monkeypatch.setattr(ntfy.requests, method, lambda url, _m=method, **kw: h._reply(_m.upper(), url, **kw))
    monkeypatch.setattr(ntfy.time, "sleep", lambda s: None)
    return h


@pytest.fixture
def notifier(cfg, monkeypatch):
    monkeypatch.delenv("CAMWATCH_NTFY_PASSWORD", raising=False)
    cfg.ntfy.enabled, cfg.ntfy.server, cfg.ntfy.topic = True, "https://ntfy.example/", "camwatch"
    cfg.ntfy.username, cfg.ntfy.password = "camuser", "secret"
    n = NtfyNotifier(type("Engine", (), {"cfg": cfg})())
    n._enqueue = n._deliver  # deliver synchronously, with the real retry policy
    return n


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "door.mp4"
    p.write_bytes(b"mp4data")
    return p


def alert(people=(), frames=True):
    fr = [BufferedFrame(1.0, np.zeros((48, 64, 3), np.uint8), 1.0, None)] if frames else []
    return EventResult("Front door", 1_700_000_000.0, "alert", list(people), fr, 0.0, 5.0, 0.0)


def person(name=None, trusted=False, unknown_id=None):
    return PersonInfo(1, name, trusted, name or "u", unknown_id, None)


def summary(http):
    return [(c["method"], (c.get("params") or {}).get("filename")) for c in http.calls]


def test_alert_is_one_message_with_snapshot_linking_the_clip(notifier, http, clip):
    notifier.send_alert(alert([person("Alice"), person(unknown_id=7)]), clip)
    assert summary(http) == [("PUT", "door.mp4"), ("PUT", "snapshot.jpg")]
    up, snap = http.calls
    assert up["url"] == "https://ntfy.example/camwatch-clips" and up["body"] == b"mp4data"
    assert snap["url"] == "https://ntfy.example/camwatch"
    assert up["params"]["priority"] == 1 and up["auth"] == ("camuser", "secret")
    p = snap["params"]
    assert snap["body"][:2] == b"\xff\xd8"  # JPEG
    assert p["title"].startswith("Front door — ") and p["message"] == "🟠 Alice\n🔴 Unknown #7"
    assert (p["tags"], p["priority"]) == ("rotating_light", 4)
    assert "click" not in p and p["actions"] == f"view, ▶ Play clip, {CLIP_URL}"


def test_clip_topic_can_be_configured(notifier, http, clip):
    notifier.cfg.ntfy.clip_topic = "hidden"
    notifier.send_alert(alert(), clip)
    assert http.calls[0]["url"] == "https://ntfy.example/hidden"


@pytest.mark.parametrize("status", [413, 403])  # too large, or no write access to the clips topic
def test_rejected_clip_still_sends_the_snapshot_alert(notifier, http, clip, status):
    http.responses = [status]
    notifier.send_alert(alert(), clip)
    assert summary(http) == [("PUT", "door.mp4"), ("PUT", "snapshot.jpg")]
    p = http.calls[1]["params"]
    assert "click" not in p and "clip not attached" in p["message"]
    assert notifier.ok


def test_without_frames_the_alert_is_text_with_the_clip_link(notifier, http, clip):
    notifier.send_alert(alert(frames=False), clip)
    assert summary(http) == [("PUT", "door.mp4"), ("POST", None)]
    assert http.calls[1]["body"] == "nobody identified".encode() and CLIP_URL in http.calls[1]["params"]["actions"]


def test_attach_clip_off_sends_only_the_snapshot(notifier, http, clip):
    notifier.cfg.ntfy.attach_clip = False
    notifier.send_alert(alert(), clip)
    assert summary(http) == [("PUT", "snapshot.jpg")]
    assert "click" not in http.calls[0]["params"]


def test_rejected_snapshot_falls_back_to_text(notifier, http):
    http.responses = [413]
    notifier.send_alert(alert(), None)
    assert summary(http) == [("PUT", "snapshot.jpg"), ("POST", None)]


def test_auth_failure_is_not_retried(notifier, http):
    http.responses = [401]
    notifier.send_alert(alert(frames=False), None)
    assert len(http.calls) == 1 and not notifier.ok and "401" in notifier.last_error


def test_server_errors_are_retried(notifier, http):
    http.responses = [503, 503]
    notifier.send_alert(alert(frames=False), None)
    assert len(http.calls) == 3 and notifier.ok


def test_env_password_overrides_config(notifier, http, monkeypatch):
    monkeypatch.setenv("CAMWATCH_NTFY_PASSWORD", "from-env")
    notifier.send_text("hi")
    assert http.calls[0]["auth"] == ("camuser", "from-env")


def test_check_login_returns_server_username(notifier, http):
    http.responses = [FakeResponse(200, {"username": "camuser"})]
    assert ntfy.check_login(notifier.cfg.ntfy, "secret") == "camuser"
    assert http.calls[0]["url"] == "https://ntfy.example/v1/account"
    http.responses = [401]
    with pytest.raises(NtfyError):
        ntfy.check_login(notifier.cfg.ntfy, "wrong")
