import time

import numpy as np
import pytest

from camwatch.telegram import TelegramBot
from conftest import unit

CHAT, USER, STRANGER = 111, 222, 999


class FakeEngine:
    def __init__(self, cfg, db):
        self.cfg, self.db = cfg, db
        self.armed_calls = []
        self.streams = {}

    def set_armed(self, cam, armed):
        self.armed_calls.append((cam, armed))

    def arm_text(self):
        return "armed?"

    def status_text(self):
        return "status"


@pytest.fixture
def bot(cfg, db):
    cfg.telegram.chat_ids = [CHAT]
    cfg.telegram.allowed_user_ids = [USER]
    b = TelegramBot(FakeEngine(cfg, db))
    b.sent = []
    b.calls = []
    b._enqueue = lambda fn, *a: b.sent.append((fn, a))
    b.call = lambda method, **kw: b.calls.append((method, kw))
    return b


def msg(text, chat=CHAT, user=USER, reply_caption=None, age=0):
    m = {"message_id": 1, "date": int(time.time() - age), "text": text, "chat": {"id": chat}, "from": {"id": user}}
    if reply_caption is not None:
        m["reply_to_message"] = {"message_id": 2, "caption": reply_caption}
    return {"update_id": 1, "message": m}


def unknown(db, seed=3):
    return db.record_unknown(unit(seed), np.zeros((20, 20, 3), np.uint8), 0.9, "cam")


def test_unauthorized_chat_is_ignored(bot):
    bot._handle_update(msg("/disarm", chat=STRANGER, user=STRANGER))
    assert bot.engine.armed_calls == [] and bot.sent == []


def test_allowed_user_can_command_from_private_chat(bot):
    bot._handle_update(msg("/disarm front", chat=USER, user=USER))
    assert bot.engine.armed_calls == [("front", False)]


def test_stale_commands_are_ignored(bot):
    bot._handle_update(msg("/disarm", age=3600))
    assert bot.engine.armed_calls == []


def test_name_by_replying_to_face_photo(bot, db):
    uid = unknown(db)
    bot._handle_update(msg("/name Alice", reply_caption=f"Unknown #{uid} — reply /name <Name>"))
    p = db.get_person("Alice")
    assert p and not p["trusted"] and not db.unknown_exists(uid)


def test_trust_by_number(bot, db):
    uid = unknown(db)
    bot._handle_update(msg(f"/trust {uid} Bob"))
    assert db.get_person("Bob")["trusted"]


def test_trust_existing_person_without_reply(bot, db):
    db.add_person("Carol", trusted=False)
    bot._handle_update(msg("/trust Carol"))
    assert db.get_person("Carol")["trusted"]


def test_reply_to_alert_with_several_unknowns_asks_for_face(bot, db):
    a, b = unknown(db, 3), unknown(db, 4)
    bot._handle_update(msg("/name Dan", reply_caption=f"🚨 cam\n🔴 Unknown #{a}\n🔴 Unknown #{b}"))
    assert db.get_person("Dan") is None
    assert db.unknown_exists(a) and db.unknown_exists(b)


def test_callback_button_assigns_face(bot, db):
    pid = db.add_person("Erin", trusted=True)
    uid = unknown(db)
    bot._handle_update({"update_id": 2, "callback_query": {
        "id": "cb", "data": f"as:{uid}:{pid}", "from": {"id": USER},
        "message": {"message_id": 5, "chat": {"id": CHAT}}}})
    assert not db.unknown_exists(uid)
    assert db.get_person("Erin")["samples"] == 1


def test_callback_from_stranger_rejected(bot, db):
    uid = unknown(db)
    bot._handle_update({"update_id": 2, "callback_query": {
        "id": "cb", "data": f"ig:{uid}", "from": {"id": STRANGER},
        "message": {"message_id": 5, "chat": {"id": STRANGER}}}})
    assert db.unknown_exists(uid)


def test_whitelist_applies_inside_alert_group(bot):
    bot._handle_update(msg("/disarm", chat=CHAT, user=STRANGER))
    assert bot.engine.armed_calls == [] and bot.sent == []
    bot._handle_update(msg("/disarm", chat=CHAT, user=USER))
    assert bot.engine.armed_calls == [(None, False)]


def test_group_member_cannot_press_face_buttons_when_not_whitelisted(bot, db):
    uid = unknown(db)
    bot._handle_update({"update_id": 2, "callback_query": {
        "id": "cb", "data": f"ig:{uid}", "from": {"id": STRANGER},
        "message": {"message_id": 5, "chat": {"id": CHAT}}}})
    assert db.unknown_exists(uid)
    assert ("answerCallbackQuery", {"callback_query_id": "cb", "text": "Not authorized"}) in bot.calls


def test_empty_whitelist_lets_alert_chat_members_command(bot):
    bot.cfg.telegram.allowed_user_ids = []
    bot._handle_update(msg("/disarm", chat=CHAT, user=STRANGER))
    assert bot.engine.armed_calls == [(None, False)]
    bot._handle_update(msg("/disarm", chat=STRANGER, user=STRANGER))
    assert len(bot.engine.armed_calls) == 1


def test_disk_command_replies_with_report(bot, monkeypatch):
    import camwatch.telegram as tg

    class SyncThread:
        def __init__(self, target, args=(), **kw):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(tg.threading, "Thread", SyncThread)
    monkeypatch.setattr(tg, "disk_report", lambda cfg, esc: "💾 report")
    bot._handle_update(msg("/disk"))
    texts = [fn for fn, _ in bot.sent]
    assert len(texts) == 2  # typing indicator + report
    texts[1]()
    assert bot.calls[-1] == ("sendMessage", {"chat_id": CHAT, "text": "💾 report", "parse_mode": "HTML"})


def test_disk_command_requires_authorization(bot):
    bot._handle_update(msg("/disk", chat=STRANGER, user=STRANGER))
    assert bot.sent == []
