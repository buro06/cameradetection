"""Ultralytics YOLO person detector (batched across cameras)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import DetectionConfig

log = logging.getLogger(__name__)
PERSON_CLASS = 0
LOW_CONFIDENCE = 0.1  # weaker boxes are still returned: the tracker uses them to keep following known people


def cuda_check() -> tuple[bool, str]:
    """True if CUDA works with this GPU. Catches 'no kernel image' errors on old GPUs (e.g. Maxwell sm_52)."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} reports no CUDA device"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    try:
        x = torch.ones(64, 64, device="cuda:0")
        (x @ x).sum().item()
    except Exception as e:
        return False, (f"{name} ({arch}) is not supported by torch {torch.__version__} "
                       f"(built for {', '.join(torch.cuda.get_arch_list())}): {e}. "
                       "Install a cu126 build: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    return True, f"{name} ({arch}), torch {torch.__version__}, CUDA {torch.version.cuda}"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    ok, msg = cuda_check()
    if ok:
        return "cuda:0"
    log.warning("CUDA unavailable: %s", msg)
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class PersonDetector:
    def __init__(self, cfg: DetectionConfig, models_dir: Path):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        models_dir.mkdir(parents=True, exist_ok=True)
        weights = Path(cfg.model)
        if not weights.is_absolute() and weights.parent == Path("."):
            weights = models_dir / weights  # official names (yolo11s.pt, ...) auto-download here
        self.model = YOLO(str(weights))
        self.half = cfg.half and self.device.startswith("cuda")
        log.info("YOLO %s on %s (half=%s)", weights.name, self.device, self.half)
        self.detect([np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8)])  # warm-up

    def detect(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Returns per frame an (N, 5) array: x1, y1, x2, y2, confidence. Includes boxes down to LOW_CONFIDENCE;
        only those above `confidence` can start a new person."""
        extra = {"half": True} if self.half else {}  # `half` is deprecated in newer ultralytics; only pass when used
        results = self.model.predict(
            frames, classes=[PERSON_CLASS], conf=min(self.cfg.confidence, LOW_CONFIDENCE), imgsz=self.cfg.imgsz,
            device=self.device, verbose=False, **extra,
        )
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 5), np.float32))
            else:
                out.append(np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32))
        return out
