"""Telegram Bot API client: alert delivery, face labeling and remote control commands."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import requests

from .diskusage import disk_report
from .events import EventResult
from .timefmt import clock

if TYPE_CHECKING:
    from .engine import Engine

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
STALE_UPDATE_SECONDS = 300
MAX_NAME_BUTTONS = 8
HELP = """<b>camwatch commands</b>
/status – camera health and arm state
/arm [camera] – enable alerts (all cameras if none given)
/disarm [camera] – pause alerts
/snapshot [camera] – live picture
/disk – space used by camwatch vs free disk space
/people – known people
/unknowns – recent unknown faces to label
/name &lt;Name&gt; – reply to a face to label it
/trust &lt;Name&gt; – reply to a face to label it as trusted (no alerts), or trust an existing person
/untrust &lt;Name&gt; – alerts again for that person
/ignore – reply to a face to discard it
Faces can also be referenced by number: /name 17 Alice"""


class TelegramError(RuntimeError):
    pass


def api_call(token: str, method: str, files: dict | None = None, http_timeout: float = 30, **params) -> Any:
    data = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items() if v is not None}
    r = requests.post(API.format(token=token, method=method), data=data, files=files, timeout=http_timeout)
    try:
        body = r.json()
    except ValueError:
        raise TelegramError(f"{method}: HTTP {r.status_code}") from None
    if not body.get("ok"):
        retry = body.get("parameters", {}).get("retry_after")
        err = TelegramError(f"{method}: {body.get('description', r.status_code)}")
        err.retry_after = retry  # type: ignore[attr-defined]
        raise err
    return body["result"]


def discover_chats(token: str) -> list[dict]:
    """Chats/users that recently messaged the bot (used by the setup wizard)."""
    seen: dict[int, dict] = {}
    for upd in api_call(token, "getUpdates", http_timeout=15):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat, user = msg.get("chat") or {}, msg.get("from") or {}
        if chat:
            title = chat.get("title") or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")])) or chat.get("username")
            seen[chat["id"]] = {"chat_id": chat["id"], "type": chat.get("type"), "title": title,
                                "user_id": user.get("id"), "user": user.get("username") or user.get("first_name")}
    return list(seen.values())


class TelegramBot:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.cfg = engine.cfg
        self._halt = threading.Event()
        self._outbox: queue.Queue = queue.Queue(maxsize=200)
        self._offset = 0
        self.connected = False
        self.last_error = ""
        self.bot_name = ""

    @property
    def token(self) -> str:
        return self.cfg.telegram_token()

    def call(self, method: str, **kw) -> Any:
        return api_call(self.token, method, **kw)

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True).start()
        threading.Thread(target=self._send_loop, name="tg-send", daemon=True).start()

    def stop(self) -> None:
        self._halt.set()

    # ---- outgoing -------------------------------------------------------------
    def _enqueue(self, fn, *args) -> None:
        try:
            self._outbox.put_nowait((fn, args))
        except queue.Full:
            log.error("Telegram outbox full, dropping message")

    def _send_loop(self) -> None:
        while not self._halt.is_set():
            try:
                fn, args = self._outbox.get(timeout=1)
            except queue.Empty:
                continue
            for attempt in range(4):
                try:
                    fn(*args)
                    break
                except TelegramError as e:
                    wait = getattr(e, "retry_after", None) or 2 ** attempt * 3
                    log.warning("Telegram send failed (%s), retry in %ss", e, wait)
                    self.last_error = str(e)
                    if "chat not found" in str(e) or "bot was blocked" in str(e):
                        break
                    time.sleep(wait)
                except requests.RequestException as e:
                    log.warning("Telegram network error (%s), retrying", e)
                    self.last_error = str(e)
                    time.sleep(2 ** attempt * 3)
                except Exception:
                    log.exception("Telegram send error")
                    break

    def send_text(self, text: str, chat_id: int | None = None) -> None:
        for cid in [chat_id] if chat_id else self.cfg.telegram.chat_ids:
            self._enqueue(lambda c=cid: self.call("sendMessage", chat_id=c, text=text, parse_mode="HTML"))

    def send_alert(self, result: EventResult, clip: Path | None) -> None:
        when = datetime.fromtimestamp(result.wall_time)
        lines = [f"🚨 <b>{_esc(result.camera)}</b> — {clock(when)} ({when:%a %d %b})"]
        for p in result.people:
            icon = "🟢" if p.trusted else ("🟠" if p.name else "🔴")
            lines.append(f"{icon} {_esc(p.label)}")
        unknowns = [p for p in result.people if p.unknown_id and p.face is not None]
        if unknowns and self.cfg.telegram.send_unknown_faces:
            lines.append("<i>Reply to a face below with /name &lt;Name&gt; or /trust &lt;Name&gt;</i>")
        caption = "\n".join(lines)
        h, w = result.frames[0].image.shape[:2] if result.frames else (0, 0)
        duration = int(round(result.end - result.start))
        for cid in self.cfg.telegram.chat_ids:
            self._enqueue(self._send_video, cid, clip, caption, w, h, duration)
            if self.cfg.telegram.send_unknown_faces:
                for p in unknowns:
                    self._enqueue(self._send_face, cid, p.unknown_id, p.face.crop, None)

    def _send_video(self, chat_id: int, clip: Path | None, caption: str, w: int, h: int, duration: int) -> None:
        if clip is None or not clip.exists():
            self.call("sendMessage", chat_id=chat_id, text=caption + "\n(clip unavailable)", parse_mode="HTML")
            return
        with open(clip, "rb") as f:
            self.call("sendVideo", files={"video": (clip.name, f, "video/mp4")}, http_timeout=120, chat_id=chat_id,
                      caption=caption, parse_mode="HTML", supports_streaming="true",
                      duration=duration, width=w or None, height=h or None)

    def _send_face(self, chat_id: int, uid: int, crop, reply_to: int | None) -> None:
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        people = self.engine.db.list_persons()[:MAX_NAME_BUTTONS]
        rows = [[{"text": f"{'🛡 ' if p['trusted'] else ''}{p['name']}", "callback_data": f"as:{uid}:{p['id']}"}
                 for p in people[i:i + 2]] for i in range(0, len(people), 2)]
        rows.append([{"text": "🗑 Ignore", "callback_data": f"ig:{uid}"}])
        self.call("sendPhoto", files={"photo": (f"unknown_{uid}.jpg", jpg.tobytes(), "image/jpeg")}, chat_id=chat_id,
                  caption=f"Unknown #{uid} — reply /name <Name> or /trust <Name>, or tap who this is:",
                  reply_markup={"inline_keyboard": rows}, reply_to_message_id=reply_to)

    def _send_photo(self, chat_id: int, jpg: bytes, caption: str) -> None:
        self.call("sendPhoto", files={"photo": ("snapshot.jpg", jpg, "image/jpeg")}, chat_id=chat_id, caption=caption)

    # ---- incoming -------------------------------------------------------------
    def _poll_loop(self) -> None:
        while not self._halt.is_set():
            try:
                if not self.bot_name:
                    me = self.call("getMe", http_timeout=15)
                    self.bot_name = me.get("username", "")
                    self.call("setMyCommands", commands=[
                        {"command": c, "description": d} for c, d in [
                            ("status", "Camera status"), ("snapshot", "Live picture"), ("disk", "Disk space"),
                            ("arm", "Enable alerts"),
                            ("disarm", "Pause alerts"), ("people", "Known people"), ("unknowns", "Unknown faces"),
                            ("help", "Help")]])
                updates = self.call("getUpdates", offset=self._offset, timeout=25, http_timeout=40,
                                    allowed_updates=["message", "callback_query"])
                self.connected, self.last_error = True, ""
                for upd in updates:
                    self._offset = upd["update_id"] + 1
                    try:
                        self._handle_update(upd)
                    except Exception:
                        log.exception("Error handling Telegram update")
            except requests.RequestException as e:
                self.connected, self.last_error = False, f"network: {e.__class__.__name__}"
                self._halt.wait(5)
            except TelegramError as e:
                self.connected, self.last_error = False, str(e)
                log.error("Telegram: %s", e)
                self._halt.wait(30 if "Conflict" not in str(e) else 60)
            except Exception:
                log.exception("Telegram poll loop error")
                self._halt.wait(10)

    def _authorized(self, chat_id: int, user_id: int | None) -> bool:
        tg = self.cfg.telegram
        return chat_id in tg.chat_ids or (user_id is not None and user_id in tg.allowed_user_ids)

    def _handle_update(self, upd: dict) -> None:
        if cb := upd.get("callback_query"):
            msg = cb.get("message") or {}
            chat_id = msg.get("chat", {}).get("id")
            if not self._authorized(chat_id, cb.get("from", {}).get("id")):
                log.warning("Unauthorized Telegram callback from user %s", cb.get("from", {}).get("id"))
                self.call("answerCallbackQuery", callback_query_id=cb["id"], text="Not authorized")
                return
            text = self._handle_callback(cb.get("data", ""))
            self.call("answerCallbackQuery", callback_query_id=cb["id"], text=text[:190])
            if msg:
                self.call("editMessageCaption", chat_id=chat_id, message_id=msg["message_id"], caption=text)
            return

        msg = upd.get("message")
        if not msg or not msg.get("text", "").startswith("/"):
            return
        chat_id, user_id = msg["chat"]["id"], msg.get("from", {}).get("id")
        if time.time() - msg.get("date", 0) > STALE_UPDATE_SECONDS:
            return  # don't act on commands queued while we were offline
        if not self._authorized(chat_id, user_id):
            log.warning("Ignoring Telegram command from unauthorized chat %s / user %s: %s", chat_id, user_id, msg["text"][:40])
            return
        cmd, _, arg = msg["text"].partition(" ")
        cmd = cmd.split("@")[0].lower()
        arg = arg.strip()
        reply = self._handle_command(cmd, arg, msg)
        if reply:
            self.send_text(reply, chat_id)

    def _disk_reply(self, chat_id: int) -> None:
        try:
            self.send_text(disk_report(self.cfg, _esc), chat_id)
        except Exception as e:
            log.exception("Disk report failed")
            self.send_text(f"Disk report failed: {_esc(str(e))}", chat_id)

    def _handle_callback(self, data: str) -> str:
        parts = data.split(":")
        db = self.engine.db
        if parts[0] == "as" and len(parts) == 3:
            person = next((p for p in db.list_persons() if p["id"] == int(parts[2])), None)
            if not person:
                return "That person no longer exists"
            return self._assign(int(parts[1]), person["name"], None)
        if parts[0] == "ig" and len(parts) == 2:
            db.delete_unknown(int(parts[1]))
            return f"Unknown #{parts[1]} discarded"
        return "Unknown action"

    def _assign(self, uid: int, name: str, trusted: bool | None) -> str:
        try:
            p = self.engine.db.assign_unknown(uid, name, trusted)
        except KeyError:
            return f"Unknown #{uid} not found (already labeled?)"
        return f"✅ Saved as {p['name']}{' (trusted — no alerts)' if p['trusted'] else ''} · {p['samples']} samples"

    @staticmethod
    def _replied_unknown(msg: dict) -> tuple[int | None, str]:
        rep = msg.get("reply_to_message") or {}
        ids = re.findall(r"Unknown #(\d+)", rep.get("caption", "") or rep.get("text", ""))
        if len(set(ids)) == 1:
            return int(ids[0]), ""
        if len(set(ids)) > 1:
            return None, "That alert has several unknown faces — reply to the face photo instead, or use /name <number> <Name>."
        return None, ""

    def _handle_command(self, cmd: str, arg: str, msg: dict) -> str | None:
        eng, db = self.engine, self.engine.db
        chat_id = msg["chat"]["id"]
        if cmd in ("/start", "/help"):
            return HELP + f"\n\nThis chat id: <code>{chat_id}</code>"
        if cmd == "/status":
            return eng.status_text()
        if cmd in ("/arm", "/disarm"):
            try:
                eng.set_armed(arg or None, cmd == "/arm")
            except KeyError as e:
                return f"No camera named {_esc(str(e))}"
            return eng.arm_text()
        if cmd == "/snapshot":
            names = [arg] if arg else [s.cfg.name for s in eng.streams.values()]
            for name in names:
                try:
                    jpg = eng.snapshot(name)
                except KeyError:
                    return f"No camera named {_esc(name)}"
                if jpg is None:
                    self.send_text(f"{_esc(name)}: no frame available", chat_id)
                else:
                    self._enqueue(self._send_photo, chat_id, jpg, f"{name} — {clock(datetime.now())}")
            return None
        if cmd == "/disk":
            self._enqueue(lambda: self.call("sendChatAction", chat_id=chat_id, action="typing"))
            # walking the folder (incl. the Python env) can take seconds; don't block other commands
            threading.Thread(target=self._disk_reply, args=(chat_id,), name="tg-disk", daemon=True).start()
            return None
        if cmd == "/people":
            people = db.list_persons()
            if not people:
                return "No known people yet."
            return "\n".join(f"{'🛡' if p['trusted'] else '👤'} {_esc(p['name'])} ({p['samples']} samples)" for p in people)
        if cmd == "/unknowns":
            unknowns = db.list_unknowns()[:5]
            if not unknowns:
                return "No unknown faces saved."
            for u in unknowns:
                imgs = db.unknown_images(u["id"])
                crop = cv2.imread(str(imgs[0])) if imgs else None
                if crop is not None:
                    self._enqueue(self._send_face, chat_id, u["id"], crop, None)
            return None
        if cmd in ("/name", "/trust", "/ignore"):
            m = re.match(r"^#?(\d+)\s*(.*)$", arg)
            uid, name = (int(m.group(1)), m.group(2).strip()) if m else (None, arg)
            err = ""
            if uid is None:
                uid, err = self._replied_unknown(msg)
            if cmd == "/ignore":
                if uid is None:
                    return err or "Reply to a face photo with /ignore, or use /ignore <number>."
                db.delete_unknown(uid)
                return f"Unknown #{uid} discarded."
            if uid is None:
                if cmd == "/trust" and name and (p := db.get_person(name)):
                    db.set_trusted(p["id"], True)
                    return f"🛡 {_esc(p['name'])} is now trusted (no alerts)."
                return err or f"Reply to a face photo with {cmd} &lt;Name&gt;, or use {cmd} &lt;number&gt; &lt;Name&gt;."
            if not name:
                return f"Usage: {cmd} &lt;Name&gt;"
            return _esc(self._assign(uid, name, True if cmd == "/trust" else None))
        if cmd == "/untrust":
            p = db.get_person(arg)
            if not p:
                return f"No person named {_esc(arg)}"
            db.set_trusted(p["id"], False)
            return f"{_esc(p['name'])} is no longer trusted — alerts will be sent."
        return "Unknown command. /help"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
