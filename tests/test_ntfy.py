import pytest

from camwatch import ntfy
from camwatch.events import EventResult, PersonInfo
from camwatch.ntfy import NtfyError, NtfyNotifier


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self.ok, self._body = status, status < 400, body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.fixture
def http(monkeypatch):
    """Records requests; `http.responses` is a list of statuses (or FakeResponses) served in order, then 200."""
    class Http:
        calls = []
        responses = []

        def _reply(self, method, url, data=None, **kw):
            body = data.read() if hasattr(data, "read") else data
            self.calls.append({"method": method, "url": url, "body": body, **kw})
            r = self.responses.pop(0) if self.responses else 200
            return r if isinstance(r, FakeResponse) else FakeResponse(r, {"error": "attachment too large"})

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


def alert(people=()):
    return EventResult("Front door", 1_700_000_000.0, "alert", list(people), [], 0.0, 5.0, 0.0)


def person(name=None, trusted=False, unknown_id=None):
    return PersonInfo(1, name, trusted, name or "u", unknown_id, None)


def test_alert_uploads_clip_with_metadata_and_basic_auth(notifier, http, tmp_path):
    clip = tmp_path / "door.mp4"
    clip.write_bytes(b"mp4data")
    notifier.send_alert(alert([person("Alice"), person(unknown_id=7)]), clip)
    (call,) = http.calls
    assert call["method"] == "PUT" and call["url"] == "https://ntfy.example/camwatch"
    assert call["auth"] == ("camuser", "secret") and call["body"] == b"mp4data"
    p = call["params"]
    assert p["title"].startswith("Front door — ") and p["filename"] == "door.mp4"
    assert p["message"] == "🟠 Alice\n🔴 Unknown #7"
    assert (p["tags"], p["priority"]) == ("rotating_light", 4)


def test_alert_without_clip_posts_text(notifier, http):
    notifier.send_alert(alert(), None)
    (call,) = http.calls
    assert call["method"] == "POST" and call["body"] == "nobody identified".encode()
    assert "filename" not in call["params"] and "message" not in call["params"]


def test_attach_clip_off_sends_text_only(notifier, http, tmp_path):
    clip = tmp_path / "door.mp4"
    clip.write_bytes(b"x")
    notifier.cfg.ntfy.attach_clip = False
    notifier.send_alert(alert(), clip)
    assert [c["method"] for c in http.calls] == ["POST"]


def test_rejected_clip_falls_back_to_text(notifier, http, tmp_path):
    clip = tmp_path / "door.mp4"
    clip.write_bytes(b"x")
    http.responses = [413]
    notifier.send_alert(alert(), clip)
    assert [c["method"] for c in http.calls] == ["PUT", "POST"]
    assert b"clip not attached" in http.calls[1]["body"]
    assert notifier.ok


def test_auth_failure_is_not_retried(notifier, http):
    http.responses = [401]
    notifier.send_alert(alert(), None)
    assert len(http.calls) == 1 and not notifier.ok and "401" in notifier.last_error


def test_server_errors_are_retried(notifier, http):
    http.responses = [503, 503]
    notifier.send_alert(alert(), None)
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
