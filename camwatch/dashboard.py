"""Rich live dashboard with single-key shortcuts; menus open on top while monitoring continues."""

from __future__ import annotations

import logging
import sys
import time
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import Engine
from .timefmt import clock

DECISION_STYLE = {"alert": "bold red", "trusted": "green", "cooldown": "yellow", "disarmed": "dim",
                  "unchanged": "dim"}


class LogBuffer(logging.Handler):
    """Keeps the latest log lines for the dashboard."""

    def __init__(self, size: int = 200):
        super().__init__()
        self.lines: deque[tuple[str, str]] = deque(maxlen=size)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append((record.levelname, f"{clock(record.created)} {record.getMessage()}"))
        except Exception:
            pass


class KeyReader:
    """Non-blocking single key reads (Windows: msvcrt, POSIX: cbreak mode)."""

    def __enter__(self):
        if sys.platform != "win32" and sys.stdin.isatty():
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32" and getattr(self, "_old", None) is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None

    def get(self, timeout: float) -> str | None:
        if sys.platform == "win32":
            import msvcrt
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):  # arrow/function keys: swallow second byte
                        msvcrt.getwch()
                        return None
                    return ch
                time.sleep(0.05)
            return None
        import select
        if not sys.stdin.isatty():
            time.sleep(timeout)
            return None
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


def _age(ts: float) -> str:
    s = int(time.time() - ts)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def render(engine: Engine, logs: LogBuffer, height: int) -> Layout:
    up = int(time.time() - engine.started)
    armed = engine.armed and not engine.disarmed_cameras
    arm = Text(" ARMED ", style="bold white on red") if armed else Text(
        " DISARMED " if not engine.armed else " PARTIAL ", style="bold black on yellow")
    tg = "off"
    if engine.bot:
        tg = f"@{engine.bot.bot_name}" if engine.bot.connected else f"[red]error: {engine.bot.last_error[:40]}[/]"
    header = Text.assemble(("camwatch ", "bold cyan"), arm,
                           f"  up {up // 86400}d {up % 86400 // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}"
                           f"  ·  detector {engine.detector.device}  ·  faces {len(engine.db.list_persons())} known, "
                           f"{len(engine.db.list_unknowns())} unknown  ·  telegram ")
    header.append_text(Text.from_markup(tg))

    cams = Table(expand=True, box=None, header_style="bold", pad_edge=False)
    for col, kw in [("Camera", {}), ("Status", {}), ("Res", {}), ("FPS", {"justify": "right"}),
                    ("Det/s", {"justify": "right"}), ("Alerts", {}), ("In view", {"ratio": 2}), ("Last event", {"ratio": 3})]:
        cams.add_column(col, **kw)
    for r in engine.camera_rows():
        st = r["status"]
        st_text = Text(st, style="green" if st == "live" else ("dim" if st == "disabled" else "red"))
        if r["recording"]:
            st_text.append(" ●REC", style="bold red")
        if r["error"] and st != "live":
            st_text.append(f" {r['error'][:30]}", style="dim red")
        le = r["last_event"]
        last = Text("—", style="dim")
        if le:
            last = Text.assemble((f"{le.decision} ", DECISION_STYLE.get(le.decision, "")), f"{le.summary()} ",
                                 (_age(le.wall_time), "dim"))
        res = f"{r['resolution'][0]}x{r['resolution'][1]}" if r["resolution"][0] else "—"
        cams.add_row(r["name"], st_text, res, f"{r['fps']:.0f}", f"{r['infer_fps']:.1f}",
                     Text("on", style="green") if r["armed"] else Text("off", style="yellow"),
                     ", ".join(r["visible"]) or Text("—", style="dim"), last)
    if not engine.cfg.cameras:
        cams.add_row(Text("No cameras configured — press [m] → Cameras → Add", style="yellow"))

    ev = Table(expand=True, box=None, show_header=False, pad_edge=False)
    ev.add_column(width=11, no_wrap=True, justify="right")
    ev.add_column(width=14, no_wrap=True)
    ev.add_column(width=9, no_wrap=True)
    ev.add_column(ratio=1)
    for e in list(engine.events)[:12]:
        ev.add_row(clock(e.wall_time), e.camera,
                   Text(e.decision, style=DECISION_STYLE.get(e.decision, "")), e.summary)

    n_log = max(3, height - 14 - len(engine.cfg.cameras) - min(12, len(engine.events)))
    log_text = Text()
    for level, line in list(logs.lines)[-n_log:]:
        style = {"WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold red"}.get(level, "dim")
        log_text.append(line[:300] + "\n", style=style)

    keys = Text.from_markup("[b]m[/] menu   [b]a[/] arm/disarm all   [b]s[/] save snapshots   [b]q[/] quit")
    layout = Layout()
    layout.split_column(
        Layout(Panel(header, border_style="cyan"), size=3),
        Layout(Panel(cams, title="Cameras", border_style="blue"), size=len(engine.cfg.cameras) + 3 if engine.cfg.cameras else 4),
        Layout(Panel(ev if engine.events else Text("No events yet", style="dim"), title="Events", border_style="blue"),
               size=min(12, len(engine.events)) + 2 if engine.events else 3),
        Layout(Panel(log_text, title="Log", border_style="dim")),
        Layout(keys, size=1),
    )
    return layout


def run_dashboard(engine: Engine, logs: LogBuffer, console: Console) -> None:
    from .menus import main_menu, save_snapshots

    while True:
        quit_requested = False
        with KeyReader() as keys, Live(render(engine, logs, console.height), console=console, screen=True,
                                       auto_refresh=False) as live:
            while True:
                key = keys.get(0.5)
                if key in ("q", "Q", "\x03"):
                    quit_requested = True
                    break
                if key in ("m", "M"):
                    break
                if key in ("a", "A"):
                    engine.set_armed(None, not (engine.armed and not engine.disarmed_cameras))
                if key in ("s", "S"):
                    save_snapshots(engine, quiet=True)
                live.update(render(engine, logs, console.height), refresh=True)
        if quit_requested:
            return
        try:
            if main_menu(engine, console) == "quit":
                return
        except KeyboardInterrupt:
            pass
