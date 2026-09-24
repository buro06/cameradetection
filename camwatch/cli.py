"""camwatch command line: interactive dashboard, headless service mode, setup wizard and diagnostics."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import platform
import signal
import sys
import threading
from pathlib import Path

from rich.console import Console

from . import __version__
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config

log = logging.getLogger("camwatch")


def setup_logging(cfg: AppConfig, console_logging: bool, verbose: bool):
    from .dashboard import LogBuffer

    logs_dir = cfg.data_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-12s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(logs_dir / "camwatch.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    buf = LogBuffer()
    buf.setLevel(logging.INFO)
    root.addHandler(buf)
    if console_logging:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for noisy in ("urllib3", "ultralytics", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return buf


def cmd_run(args, console: Console) -> int:
    cfg = load_config(args.config)
    if not Path(args.config).exists():
        if args.headless:
            console.print(f"[red]{args.config} not found.[/] Run `camwatch setup` first.")
            return 1
        console.print(f"[yellow]No config at {args.config} — starting setup.[/]")
        if cmd_setup(args, console) != 0:
            return 1
        cfg = load_config(args.config)
    logs = setup_logging(cfg, console_logging=args.headless, verbose=args.verbose)

    from .engine import Engine

    with console.status("Loading models (first run downloads YOLO + face models) …"):
        engine = Engine(cfg)
    engine.start()

    try:
        if args.headless:
            stop = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
                if sig is not None:
                    signal.signal(sig, lambda *_: stop.set())
            while not stop.wait(1):
                pass
        else:
            from .dashboard import run_dashboard
            run_dashboard(engine, logs, console)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down")
        engine.stop()
    return 0


def cmd_setup(args, console: Console) -> int:
    import questionary

    from .menus import add_camera_flow, telegram_menu

    path = Path(args.config)
    cfg = load_config(path)
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    cfg.save(path)
    console.rule("[bold cyan]camwatch setup")
    console.print(f"Config file: [b]{path.resolve()}[/]\n")
    if questionary.confirm("Configure Telegram notifications now?", default=not cfg.telegram_token()).ask():
        telegram_menu(cfg, console)
    while questionary.confirm("Add a camera?", default=not cfg.cameras).ask():
        add_camera_flow(cfg, console)
    cfg.save(path)
    console.print("\n[green]Setup saved.[/] Start monitoring with [b]camwatch[/] (interactive) or "
                  "[b]camwatch run --headless[/] (service). Enroll faces from the dashboard menu (press m).")
    return 0


def cmd_doctor(args, console: Console) -> int:
    ok_all = True

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= ok
        console.print(f"{'[green]✔[/]' if ok else '[red]✘[/]'} {name}{f' — {detail}' if detail else ''}")

    cfg = load_config(args.config)
    logging.getLogger("camwatch").setLevel(logging.ERROR)  # results are printed below instead
    check("Python", sys.version_info >= (3, 11), f"{platform.python_version()} on {platform.platform()}")
    try:
        import cv2
        check("OpenCV", hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"), cv2.__version__)
    except ImportError as e:
        check("OpenCV", False, str(e))
    try:
        import torch
        import ultralytics
        check("PyTorch / Ultralytics", True, f"torch {torch.__version__}, ultralytics {ultralytics.__version__}")
        from .detector import cuda_check, resolve_device
        cuda_ok, msg = cuda_check()
        console.print(f"{'[green]✔[/]' if cuda_ok else '[yellow]![/]'} CUDA — {msg}")
        console.print(f"  → detector device: [b]{resolve_device(cfg.detection.device)}[/]")
    except ImportError as e:
        check("PyTorch / Ultralytics", False, str(e))
    from .recorder import ffmpeg_exe
    exe = ffmpeg_exe()
    check("ffmpeg (H.264 clips)", exe is not None, exe or "pip install imageio-ffmpeg")
    try:
        from .faces import ensure_models
        with console.status("Checking face models …"):
            ensure_models(cfg.data_path / "models")
        check("Face models (YuNet + SFace)", True, str(cfg.data_path / "models"))
    except Exception as e:
        check("Face models (YuNet + SFace)", False, str(e))
    token = cfg.telegram_token()
    if cfg.telegram.enabled and token:
        from .telegram import api_call
        try:
            me = api_call(token, "getMe", http_timeout=15)
            check("Telegram bot", True, f"@{me['username']}")
            check("Telegram alert chats", bool(cfg.telegram.chat_ids), str(cfg.telegram.chat_ids or "none configured"))
        except Exception as e:
            check("Telegram bot", False, str(e))
    else:
        console.print("[yellow]![/] Telegram disabled or no token")
    if cfg.cameras:
        from .menus import probe_camera
        from .camera import describe_source, set_rtsp_transport
        set_rtsp_transport(cfg.rtsp_transport)
        for cam in cfg.cameras:
            if not cam.enabled:
                console.print(f"  - camera {cam.name}: disabled")
                continue
            with console.status(f"Connecting to {cam.name} …"):
                frame, err = probe_camera(cam, cfg)
            check(f"Camera {cam.name}", frame is not None,
                  f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else f"{describe_source(cam.source)}: {err or 'timeout'}")
    else:
        console.print("[yellow]![/] No cameras configured")
    return 0 if ok_all else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="camwatch", description="24/7 person & face detection with Telegram alerts")
    p.add_argument("--config", "-c", default=str(DEFAULT_CONFIG_PATH), help="config file (default: config.yaml)")
    p.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"camwatch {__version__}")
    sub = p.add_subparsers(dest="command")
    run = sub.add_parser("run", help="start monitoring (default)")
    run.add_argument("--headless", action="store_true", help="no dashboard; log to stdout (for services)")
    sub.add_parser("setup", help="interactive setup wizard (Telegram + cameras)")
    sub.add_parser("doctor", help="check GPU, models, Telegram and cameras")
    args = p.parse_args(argv)

    console = Console()
    if args.command in (None, "run"):
        args.headless = getattr(args, "headless", False)
        return cmd_run(args, console)
    if args.command == "setup":
        return cmd_setup(args, console)
    if args.command == "doctor":
        return cmd_doctor(args, console)
    return 2
