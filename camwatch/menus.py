"""Interactive menus (questionary) shown on top of the dashboard; monitoring keeps running meanwhile."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import questionary
from rich.console import Console
from rich.table import Table

from .camera import CameraStream, describe_source
from .config import AppConfig, CameraConfig
from .engine import Engine

log = logging.getLogger(__name__)
BACK = "↩  Back"


# ---- helpers ------------------------------------------------------------------------
def _select(msg: str, choices: list, **kw):
    return questionary.select(msg, choices=choices, **kw).ask()


def _text(msg: str, default: str = "", **kw) -> str | None:
    return questionary.text(msg, default=default, **kw).ask()


def _confirm(msg: str, default: bool = True) -> bool:
    return bool(questionary.confirm(msg, default=default).ask())


def _pause() -> None:
    questionary.press_any_key_to_continue().ask()


def open_image(path: Path) -> None:
    """Open an image in the OS default viewer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Could not open viewer ({e}). Image saved at: {path}")


def montage(images: list[np.ndarray], tile_h: int = 200, cols: int = 5) -> np.ndarray:
    tiles = [cv2.resize(im, (max(1, int(im.shape[1] * tile_h / im.shape[0])), tile_h)) for im in images]
    tile_w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2 + tile_w - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.full_like(tiles[0], 30)] * (cols - len(row)) if len(tiles) > cols else []
        rows.append(np.hstack(row))
    return np.vstack(rows)


def show_images(engine: Engine, paths: list[Path], name: str) -> None:
    imgs = [im for im in (cv2.imread(str(p)) for p in paths[:15]) if im is not None]
    if not imgs:
        print("No images available.")
        return
    out = engine.data / "tmp" / f"{name}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage(imgs))
    open_image(out)


def save_snapshots(engine: Engine, quiet: bool = False, cameras: list[str] | None = None) -> list[Path]:
    out = []
    for name in cameras or list(engine.streams):
        try:
            jpg = engine.snapshot(name)
        except KeyError:
            continue
        if jpg is None:
            continue
        path = engine.data / "snapshots" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpg)
        out.append(path)
    log.info("Saved %d snapshot(s) to %s", len(out), engine.data / "snapshots")
    if not quiet:
        for p in out:
            open_image(p)
    return out


def probe_camera(cam: CameraConfig, cfg: AppConfig, timeout: float = 15.0) -> tuple[np.ndarray | None, str]:
    """Open a source briefly and grab one frame (uses the same code path as monitoring)."""
    stream = CameraStream(cam, cfg.clip, 0.1)
    stream.start()
    end = time.monotonic() + timeout
    frame = None
    while time.monotonic() < end:
        _, _, frame = stream.latest()
        if frame is not None:
            break
        time.sleep(0.2)
    stream.stop()
    return frame, stream.error


# ---- main menu ---------------------------------------------------------------------------
def main_menu(engine: Engine, console: Console) -> str | None:
    while True:
        console.clear()
        console.rule("[bold cyan]camwatch menu[/] — monitoring continues in the background")
        n_unknown = len(engine.db.list_unknowns())
        choice = _select("What do you want to do?", [
            "📷  Cameras",
            "🙂  People & faces",
            f"❓  Review unknown faces ({n_unknown})",
            "🔔  Arm / disarm alerts",
            "✉️   Telegram",
            "⚙️   Settings",
            "📸  Save snapshots of all cameras",
            BACK,
            "⏻  Quit camwatch",
        ])
        if choice is None or choice == BACK:
            return None
        if choice.startswith("⏻"):
            if _confirm("Stop monitoring and quit?", default=False):
                return "quit"
            continue
        try:
            if "Cameras" in choice:
                cameras_menu(engine, console)
            elif "People" in choice:
                people_menu(engine, console)
            elif "unknown" in choice:
                unknowns_menu(engine, console)
            elif "Arm" in choice:
                arm_menu(engine)
            elif "Telegram" in choice:
                telegram_menu(engine.cfg, console, engine)
            elif "Settings" in choice:
                settings_menu(engine, console)
            elif "snapshots" in choice:
                save_snapshots(engine)
        except KeyboardInterrupt:
            continue
        except Exception as e:
            log.exception("Menu action failed")
            console.print(f"[red]Error:[/] {e}")
            _pause()


# ---- cameras ---------------------------------------------------------------------------------
SOURCE_HELP = ("USB index (0, 1, …), rtsp://user:pass@host:554/stream, http(s)://…/video.mjpg, "
               "http(s)://…/snapshot.jpg, or a video file path")


def cameras_menu(engine: Engine, console: Console) -> None:
    cfg = engine.cfg
    while True:
        console.clear()
        t = Table(title="Cameras")
        for c in ("Name", "Source", "Enabled", "Status", "Resolution"):
            t.add_column(c)
        rows = {r["name"]: r for r in engine.camera_rows()}
        for cam in cfg.cameras:
            r = rows.get(cam.name, {})
            t.add_row(cam.name, describe_source(cam.source), "yes" if cam.enabled else "no", r.get("status", ""),
                      _resolution_text(cam, r.get("resolution", (0, 0))))
        console.print(t)
        choice = _select("Cameras:", [*(questionary.Choice(c.name, c) for c in cfg.cameras), "➕  Add camera",
                                      "🔍  Scan for USB cameras", BACK])
        if choice is None or choice == BACK:
            return
        if choice == "➕  Add camera":
            cam = add_camera_flow(cfg, console)
            if cam:
                engine.apply_camera(cam)
        elif choice == "🔍  Scan for USB cameras":
            cam = scan_usb(cfg, console)
            if cam:
                engine.apply_camera(cam)
        else:
            camera_actions(engine, console, choice)


RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
FRAME_RATES = [15, 24, 25, 30, 60]


def is_usb(cam: CameraConfig) -> bool:
    return cam.source.strip().isdigit()


def _resolution_text(cam: CameraConfig, actual: tuple[int, int]) -> str:
    got = f"{actual[0]}x{actual[1]}" if actual and actual[0] else "—"
    if is_usb(cam) and cam.width and cam.height and actual and actual[0] and actual != (cam.width, cam.height):
        return f"[yellow]{got} (asked {cam.width}x{cam.height})[/]"
    return got


def pick_resolution(cam: CameraConfig, mark_current: bool = True) -> bool | None:
    """Ask for a USB capture resolution. None = cancelled, else whether it changed."""
    current = (cam.width, cam.height)
    mark = lambda wh: "  ← current" if mark_current and wh == current else ""  # noqa: E731
    choices = [questionary.Choice(f"Camera default{mark((0, 0))}", (0, 0))]
    choices += [questionary.Choice(f"{w}x{h}{mark((w, h))}", (w, h)) for w, h in RESOLUTIONS]
    choices.append(questionary.Choice("Custom…", "custom"))
    default = current if current == (0, 0) or current in RESOLUTIONS else "custom"
    pick = _select("Capture resolution (USB cameras only; network cameras are configured in their own web UI):",
                   choices, default=default)
    if pick is None:
        return None
    if pick == "custom":
        raw = _text("Resolution (WIDTHxHEIGHT):", default=f"{cam.width or 1920}x{cam.height or 1080}",
                    validate=lambda s: bool(re.fullmatch(r"\s*\d{2,5}\s*[xX]\s*\d{2,5}\s*", s)) or "e.g. 1920x1080")
        if not raw:
            return None
        pick = tuple(int(v) for v in re.split(r"[xX]", raw.replace(" ", "")))
    changed = (cam.width, cam.height) != pick
    cam.width, cam.height = pick
    return changed


def pick_fps(cam: CameraConfig) -> bool | None:
    """Ask for a USB capture frame rate. None = cancelled, else whether it changed."""
    mark = lambda f: "  ← current" if f == cam.fps else ""  # noqa: E731
    choices = [questionary.Choice(f"Camera default{mark(0)}", 0.0)]
    choices += [questionary.Choice(f"{f} fps{mark(f)}", float(f)) for f in FRAME_RATES]
    choices.append(questionary.Choice("Custom…", "custom"))
    default = cam.fps if cam.fps == 0 or cam.fps in FRAME_RATES else "custom"
    pick = _select("Capture frame rate (USB cameras only):", choices, default=default)
    if pick is None:
        return None
    if pick == "custom":
        raw = _text("Frames per second:", default=f"{cam.fps or 30:g}",
                    validate=lambda s: bool(re.fullmatch(r"\s*\d{1,3}(\.\d+)?\s*", s)) and 0 < float(s) <= 240 or "1–240")
        if not raw:
            return None
        pick = float(raw)
    changed = cam.fps != pick
    cam.fps = pick
    return changed


def pick_usb_format(cam: CameraConfig) -> bool | None:
    """Ask for USB pixel format and capture backend. None = cancelled, else whether it changed."""
    fourcc = _select("Pixel format (MJPG is needed for HD on most USB 2.0 webcams):", [
        questionary.Choice("auto (MJPG on Windows)", "auto"), questionary.Choice("MJPG", "MJPG"),
        questionary.Choice("YUY2 (uncompressed, low res/fps)", "YUY2"), questionary.Choice("none (driver default)", "none")],
        default=cam.fourcc if cam.fourcc in ("auto", "MJPG", "YUY2", "none") else "auto")
    if fourcc is None:
        return None
    backend = _select("Capture backend:", [
        questionary.Choice("auto (DirectShow on Windows)", "auto"), questionary.Choice("dshow (DirectShow)", "dshow"),
        questionary.Choice("msmf (Media Foundation)", "msmf"), questionary.Choice("avfoundation (macOS)", "avfoundation"),
        questionary.Choice("v4l2 (Linux)", "v4l2")],
        default=cam.backend if cam.backend in ("auto", "dshow", "msmf", "avfoundation", "v4l2") else "auto")
    if backend is None:
        return None
    changed = (cam.fourcc, cam.backend) != (fourcc, backend)
    cam.fourcc, cam.backend = fourcc, backend
    return changed


def _wait_resolution(engine: Engine, name: str, timeout: float = 12.0) -> tuple[int, int] | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = engine.streams.get(name)
        if s is not None and s.resolution[0]:
            return s.resolution
        time.sleep(0.2)
    return None


def _report_resolution(console: Console, cam: CameraConfig, got: tuple[int, int]) -> None:
    if is_usb(cam) and cam.width and cam.height and got != (cam.width, cam.height):
        console.print(f"[yellow]Camera delivers {got[0]}x{got[1]}, not the requested {cam.width}x{cam.height}.[/] "
                      "Try another resolution, or Pixel format → MJPG / backend → msmf.")
    else:
        console.print(f"[green]OK[/] — {got[0]}x{got[1]}")


def _report_fps(console: Console, cam: CameraConfig, measured: float) -> None:
    if cam.fps and measured < 0.6 * cam.fps:
        console.print(f"[yellow]Receiving {measured:.1f} fps, not the requested {cam.fps:g}.[/] "
                      "Try Pixel format → MJPG, or add light (webcams slow down in dim rooms).")
    else:
        console.print(f"Frame rate: {measured:.1f} fps")


def add_camera_flow(cfg: AppConfig, console: Console, source: str | None = None) -> CameraConfig | None:
    name = _text("Camera name:", validate=lambda s: bool(s.strip()) and cfg.camera(s.strip()) is None or "Name empty or already used")
    if not name:
        return None
    source = source or _text("Source:", instruction=f"\n  {SOURCE_HELP}\n  >")
    if not source:
        return None
    cam = CameraConfig(name=name.strip(), source=source.strip())
    if is_usb(cam):
        cam.width, cam.height = 1920, 1080  # preselected; most webcams sold today are 1080p
        if pick_resolution(cam, mark_current=False) is None:
            return None
    with console.status(f"Connecting to {describe_source(cam.source)} …"):
        frame, err = probe_camera(cam, cfg)
    if frame is None:
        console.print(f"[red]Could not get a frame[/] ({err or 'timeout'}).")
        if not _confirm("Save it anyway (it will keep retrying)?", default=False):
            return None
    else:
        _report_resolution(console, cam, (frame.shape[1], frame.shape[0]))
    cfg.cameras.append(cam)
    cfg.save()
    console.print(f"Saved camera [b]{cam.name}[/].")
    return cam


def scan_usb(cfg: AppConfig, console: Console) -> CameraConfig | None:
    found = []
    with console.status("Probing USB camera indexes 0–5 …"):
        for i in range(6):
            frame, _ = probe_camera(CameraConfig(name=f"usb{i}", source=str(i)), cfg, timeout=5)
            if frame is not None:
                found.append((i, frame.shape))
    if not found:
        console.print("[yellow]No USB cameras found.[/]")
        _pause()
        return None
    used = {c.source for c in cfg.cameras}
    for i, shape in found:
        console.print(f"  index {i}: {shape[1]}x{shape[0]}{'  (already configured)' if str(i) in used else ''}")
    pick = _select("Add one?", [*(questionary.Choice(f"Index {i}", str(i)) for i, _ in found if str(i) not in used), BACK])
    if pick and pick != BACK:
        return add_camera_flow(cfg, console, source=pick)
    return None


def camera_actions(engine: Engine, console: Console, cam: CameraConfig) -> None:
    cfg = engine.cfg
    usb_actions = ["📐  Resolution", "🎞   Frame rate", "🎛   Pixel format / capture backend"] if is_usb(cam) else []
    action = _select(f"{cam.name}:", ["📸  Snapshot", "✏️   Edit source", *usb_actions,
                                      "⏯   Disable" if cam.enabled else "⏯   Enable", "🗑   Remove", BACK])
    if action is None or action == BACK:
        return
    if "Snapshot" in action:
        if not save_snapshots(engine, cameras=[cam.name]):
            console.print("[yellow]No frame available.[/]")
            _pause()
    elif "Resolution" in action or "Frame rate" in action or "Pixel format" in action:
        picker = pick_resolution if "Resolution" in action else pick_fps if "Frame rate" in action else pick_usb_format
        changed = picker(cam)
        if changed:
            cfg.save()
            engine.apply_camera(cam)
            if cam.enabled:
                with console.status("Reopening camera …"):
                    got = _wait_resolution(engine, cam.name)
                    if got:
                        time.sleep(3)  # let the frame-rate average settle
                if got:
                    _report_resolution(console, cam, got)
                    if stream := engine.streams.get(cam.name):
                        _report_fps(console, cam, stream.fps)
                else:
                    console.print("[yellow]No frame yet — check the dashboard/log.[/]")
                _pause()
    elif "Edit" in action:
        src = _text("New source:", default=cam.source)
        if src and src != cam.source:
            cam.source = src.strip()
            cfg.save()
            engine.apply_camera(cam)
    elif "able" in action:
        cam.enabled = not cam.enabled
        cfg.save()
        engine.apply_camera(cam)
    elif "Remove" in action and _confirm(f"Remove camera {cam.name}?", default=False):
        engine.stop_camera(cam.name)
        cfg.cameras.remove(cam)
        cfg.save()


# ---- people ---------------------------------------------------------------------------------
def people_menu(engine: Engine, console: Console) -> None:
    db = engine.db
    while True:
        console.clear()
        people = db.list_persons()
        t = Table(title="Known people")
        for c in ("Name", "Trusted (no alerts)", "Face samples"):
            t.add_column(c)
        for p in people:
            t.add_row(p["name"], "🛡 yes" if p["trusted"] else "no", str(p["samples"]))
        console.print(t)
        choice = _select("People:", [*(questionary.Choice(p["name"], p) for p in people),
                                     "➕  Enroll a person from a live camera", BACK])
        if choice is None or choice == BACK:
            return
        if isinstance(choice, str):
            enroll_flow(engine, console)
            continue
        p = choice
        action = _select(f"{p['name']}:", [
            "🛡  Mark as NOT trusted (send alerts)" if p["trusted"] else "🛡  Mark as trusted (no alerts)",
            "➕  Add more face samples from a live camera", "🖼   View face samples", "✏️   Rename", "🗑   Delete", BACK])
        if action is None or action == BACK:
            continue
        if "trusted" in action:
            db.set_trusted(p["id"], not p["trusted"])
        elif "Add more" in action:
            enroll_flow(engine, console, name=p["name"])
        elif "View" in action:
            folder = engine.data / "faces" / str(p["id"])
            show_images(engine, sorted(folder.glob("*.jpg"), reverse=True), f"person_{p['id']}")
        elif "Rename" in action:
            new = _text("New name:", default=p["name"])
            if new and new.strip() != p["name"]:
                db.rename(p["id"], new)
        elif "Delete" in action and _confirm(f"Delete {p['name']} and all face samples?", default=False):
            db.delete_person(p["id"])


def enroll_flow(engine: Engine, console: Console, name: str | None = None, samples: int = 12, timeout: float = 60) -> None:
    db = engine.db
    live = [s for s in engine.streams.values() if s.is_live()]
    if not live:
        console.print("[red]No live cameras.[/]")
        _pause()
        return
    stream = live[0] if len(live) == 1 else _select("Which camera?", [questionary.Choice(s.cfg.name, s) for s in live])
    if stream is None:
        return
    if name is None:
        name = _text("Person's name:", validate=lambda s: bool(s.strip()) or "Enter a name")
        if not name:
            return
        existing = db.get_person(name)
        trusted = existing["trusted"] if existing else _confirm(f"Is {name} trusted (no alerts when seen)?", default=True)
    else:
        trusted = db.get_person(name)["trusted"]
    preview = _confirm("Show a live preview window?", default=True)
    console.print(f"\n[b]{name}[/] should stand 1–2 m from [b]{stream.cfg.name}[/], alone, facing the camera,\n"
                  "then slowly turn their head left/right and up/down. Press Ctrl+C to stop early.\n")
    _pause()

    got: list = []
    conflicts = 0
    last_fid = -1
    end = time.monotonic() + timeout
    status = "waiting for a face"
    try:
        with console.status("") as st:
            while len(got) < samples and time.monotonic() < end:
                fid, _, frame = stream.latest()
                if frame is None or fid == last_fid:
                    time.sleep(0.03)
                    continue
                last_fid = fid
                faces = engine.faces.analyze(frame, None, max_faces=2)
                if len(faces) > 1:
                    status = "[yellow]more than one face in view[/]"
                elif faces:
                    obs = faces[0]
                    other = db.match(obs.embedding)
                    if other and other.name.lower() != name.lower() and other.score > 0.5:
                        conflicts += 1
                        status = f"[yellow]face looks like {other.name} ({other.score:.2f}) — skipped[/]"
                    elif got and max(float(obs.embedding @ g.embedding) for g in got) > 0.93:
                        status = "turn your head a little / move"
                    else:
                        got.append(obs)
                        status = "[green]sample captured[/]"
                        time.sleep(0.25)
                else:
                    status = "no clear face — come closer / face the camera"
                st.update(f"Captured {len(got)}/{samples} · {status}")
                if preview:
                    img = frame.copy()
                    for f in faces:
                        cv2.rectangle(img, f.box[:2], f.box[2:], (0, 255, 0), 2)
                    cv2.putText(img, f"{name}: {len(got)}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    scale = min(1.0, 960 / img.shape[1])
                    cv2.imshow("camwatch enroll", cv2.resize(img, None, fx=scale, fy=scale))
                    cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        if preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    if len(got) < 3:
        console.print(f"[red]Only {len(got)} usable samples — nothing saved.[/] Try better lighting or move closer.")
        _pause()
        return
    pid = db.add_person(name, trusted)
    for obs in got:
        db.add_embedding(pid, obs.embedding, obs.crop)
    console.print(f"[green]Saved {len(got)} samples for {name}[/]{' (trusted)' if trusted else ''}."
                  + (f" Skipped {conflicts} frames resembling someone else." if conflicts else ""))
    _pause()


# ---- unknown faces ------------------------------------------------------------------------
def _ago(ts: float) -> str:
    s = int(time.time() - ts)
    return f"{s // 60}m ago" if s < 3600 else (f"{s // 3600}h ago" if s < 86400 else f"{s // 86400}d ago")


def unknowns_menu(engine: Engine, console: Console) -> None:
    db = engine.db
    while True:
        console.clear()
        unknowns = db.list_unknowns()
        if not unknowns:
            console.print("No unknown faces saved. They are collected automatically when someone unrecognized is seen.")
            _pause()
            return
        choice = _select(f"{len(unknowns)} unknown face cluster(s) — pick one to review:", [
            *(questionary.Choice(f"#{u['id']:<5} seen {u['sightings']}× · {u['camera']} · {_ago(u['last_seen'])}", u)
              for u in unknowns[:40]),
            "🗑  Delete ALL unknown faces", BACK])
        if choice is None or choice == BACK:
            return
        if isinstance(choice, str):
            if _confirm(f"Delete all {len(unknowns)} unknown faces?", default=False):
                for u in unknowns:
                    db.delete_unknown(u["id"])
            continue
        review_unknown(engine, console, choice)


def review_unknown(engine: Engine, console: Console, u: dict) -> None:
    db = engine.db
    show_images(engine, db.unknown_images(u["id"]), f"unknown_{u['id']}")
    people = db.list_persons()
    action = _select(f"Unknown #{u['id']} (opened in image viewer) — who is this?", [
        "➕  New person", "🛡  New trusted person (no alerts)",
        *(questionary.Choice(f"= {p['name']}{' (trusted)' if p['trusted'] else ''}", p) for p in people),
        "🗑  Delete (not useful)", "⏭  Skip"])
    if action is None or action == "⏭  Skip":
        return
    if isinstance(action, dict):
        p = db.assign_unknown(u["id"], action["name"])
    elif "Delete" in action:
        db.delete_unknown(u["id"])
        return
    else:
        name = _text("Name:", validate=lambda s: bool(s.strip()) or "Enter a name")
        if not name:
            return
        p = db.assign_unknown(u["id"], name, trusted="trusted" in action)
    console.print(f"[green]Saved as {p['name']}[/] ({p['samples']} samples{', trusted' if p['trusted'] else ''}).")
    time.sleep(1)


# ---- arming ---------------------------------------------------------------------------------
def arm_menu(engine: Engine) -> None:
    choice = _select("Alerts:", ["🔔  Arm all cameras", "🔕  Disarm all cameras", "🎛   Choose per camera", BACK])
    if choice is None or choice == BACK:
        return
    if "Arm all" in choice:
        engine.set_armed(None, True)
    elif "Disarm all" in choice:
        engine.set_armed(None, False)
    else:
        names = [c.name for c in engine.cfg.cameras]
        armed = questionary.checkbox("Armed cameras (space to toggle):", choices=[
            questionary.Choice(n, n, checked=engine.is_armed(n)) for n in names]).ask()
        if armed is None:
            return
        engine.armed = True
        for n in names:
            engine.set_armed(n, n in armed)


# ---- telegram -------------------------------------------------------------------------------
def telegram_menu(cfg: AppConfig, console: Console, engine: Engine | None = None) -> None:
    from .telegram import TelegramError, api_call, discover_chats

    tg = cfg.telegram
    while True:
        console.clear()
        token = cfg.telegram_token()
        console.print(f"Enabled: [b]{tg.enabled}[/]   Token: {'set' if token else '[red]not set[/]'}"
                      f"{' (from env CAMWATCH_TELEGRAM_TOKEN)' if os.environ.get('CAMWATCH_TELEGRAM_TOKEN') else ''}\n"
                      f"Alert chats: {tg.chat_ids or '[red]none[/]'}   Command users: {tg.allowed_user_ids or 'chat members only'}")
        choice = _select("Telegram:", ["🔑  Set bot token", "🔎  Find chats & users (after sending /start to the bot)",
                                       "✏️   Edit chat / user IDs manually", "📨  Send test message",
                                       "⏯   Disable" if tg.enabled else "⏯   Enable", BACK])
        if choice is None or choice == BACK:
            return
        try:
            if "token" in choice:
                console.print("Create a bot with [b]@BotFather[/] in Telegram (/newbot) and paste the token.")
                new = questionary.password("Bot token:").ask()
                if new:
                    me = api_call(new.strip(), "getMe", http_timeout=15)
                    tg.bot_token = new.strip()
                    tg.enabled = True
                    console.print(f"[green]Token OK[/] — bot @{me['username']}. Now open Telegram, send /start to "
                                  f"@{me['username']} (or add it to a group), then use 'Find chats'.")
                    _pause()
            elif "Find" in choice:
                if not token:
                    console.print("[red]Set the token first.[/]")
                    _pause()
                    continue
                found = discover_chats(token)
                if not found:
                    console.print("[yellow]No messages found.[/] Send /start to the bot and try again "
                                  "(if the bot is already running elsewhere, stop it first).")
                    _pause()
                    continue
                chats = questionary.checkbox("Which chats should receive alerts?", choices=[
                    questionary.Choice(f"{f['title']} ({f['type']}, id {f['chat_id']})", f["chat_id"],
                                       checked=True) for f in found]).ask() or []
                users = {f["user_id"]: f["user"] for f in found if f["user_id"]}
                allowed = questionary.checkbox("Which users may send commands (/disarm, /name, …)?", choices=[
                    questionary.Choice(f"{n} (id {uid})", uid, checked=True) for uid, n in users.items()]).ask() or []
                tg.chat_ids = sorted(set(tg.chat_ids) | set(chats))
                tg.allowed_user_ids = sorted(set(tg.allowed_user_ids) | set(allowed))
            elif "manually" in choice:
                c = _text("Alert chat IDs (comma separated):", default=",".join(map(str, tg.chat_ids)))
                u = _text("Allowed user IDs (comma separated):", default=",".join(map(str, tg.allowed_user_ids)))
                if c is not None:
                    tg.chat_ids = [int(x) for x in c.replace(" ", "").split(",") if x]
                if u is not None:
                    tg.allowed_user_ids = [int(x) for x in u.replace(" ", "").split(",") if x]
            elif "test" in choice:
                for cid in tg.chat_ids:
                    api_call(token, "sendMessage", chat_id=cid, text="✅ camwatch test message")
                console.print(f"[green]Sent to {len(tg.chat_ids)} chat(s).[/]")
                _pause()
            else:
                tg.enabled = not tg.enabled
            cfg.save()
            if engine:
                engine.apply_telegram()
        except (TelegramError, ValueError) as e:
            console.print(f"[red]{e}[/]")
            _pause()


# ---- settings -----------------------------------------------------------------------------
SETTINGS = [
    ("detection", "confidence", "YOLO person confidence (0–1)"),
    ("detection", "detect_fps", "Inference rate per camera"),
    ("detection", "min_hits", "Detections before a person counts"),
    ("detection", "min_box_height", "Ignore people smaller than this fraction of frame height"),
    ("detection", "overlap_confirm_seconds", "Box on top of a tracked person must last this long (s)"),
    ("face", "match_threshold", "Face match threshold (higher = stricter)"),
    ("face", "trusted_min_matches", "Face matches needed before a trusted person suppresses alerts"),
    ("face", "detector_score", "Face detector confidence"),
    ("face", "min_face_px", "Minimum face size in pixels"),
    ("face", "save_unknowns", "Save unknown faces for labeling"),
    ("events", "cooldown_seconds", "No re-alert if the same person leaves and returns within (s)"),
    ("events", "lost_memory_seconds", "Still person hidden up to this long isn't a new arrival (s)"),
    ("events", "post_seconds", "Clip seconds after detection (also the identification window)"),
    ("clip", "annotate", "Draw boxes and names on clips"),
    ("clip", "retention_days", "Delete saved clips after N days (0 = keep)"),
    ("telegram", "send_unknown_faces", "Send unknown face photos for labeling"),
    ("detection", "model", "YOLO model (restart needed)"),
    ("detection", "device", "Device auto/cuda:0/cpu (restart needed)"),
    ("events", "pre_seconds", "Clip seconds before detection (restart needed to increase)"),
    ("clip", "fps", "Clip FPS (restart needed)"),
    ("clip", "max_width", "Clip max width (restart needed)"),
]


def settings_menu(engine: Engine, console: Console) -> None:
    cfg = engine.cfg
    while True:
        console.clear()
        choices = []
        for section, key, desc in SETTINGS:
            val = getattr(getattr(cfg, section), key)
            choices.append(questionary.Choice(f"{desc:<62} {val}", (section, key)))
        choice = _select("Settings (saved to config.yaml):", [*choices, BACK])
        if choice is None or choice == BACK:
            return
        section, key = choice
        obj = getattr(cfg, section)
        ftype = {f.name: f.type for f in fields(obj)}[key]
        cur = getattr(obj, key)
        try:
            if ftype in ("bool", bool):
                new = not cur
            else:
                raw = _text(f"{section}.{key}:", default=str(cur))
                if raw is None:
                    continue
                new = {"int": int, "float": float}.get(str(ftype), str)(raw)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            _pause()
            continue
        setattr(obj, key, new)
        if (section, key) == ("face", "match_threshold"):
            engine.db.threshold = new
        cfg.save()
