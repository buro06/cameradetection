"""Pop-out windows with a camera's live picture and the current detections drawn on it."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_FPS = 15.0
START_WIDTH = 960  # initial window width; the window can be resized freely
CLOSE_KEYS = (27, ord("q"), ord("Q"))  # Esc / q with a live window focused closes all of them


def window_title(camera: str) -> str:
    return f"camwatch - {camera}"


class LiveView:
    """OpenCV windows, one per camera, drawn from the dashboard's main thread (macOS requires that)
    by calling `pump()` often. They are hidden while a menu is open, since nothing redraws them then,
    and come back afterwards."""

    def __init__(self, engine):
        self.engine = engine
        self.cameras: list[str] = []  # cameras the user wants a window for
        self._shown: set[str] = set()
        self._last = 0.0

    @property
    def active(self) -> bool:
        return bool(self.cameras)

    def toggle(self, cameras: list[str]) -> None:
        """Open a window for each camera, or close them all if any are open."""
        if self.cameras:
            self.close()
        elif cameras:
            self.cameras = list(cameras)
            log.info("Live view opened: %s (Esc or q in the window closes it)", ", ".join(cameras))
        else:
            log.warning("Live view: no cameras are running")

    def pump(self) -> None:
        if not self.cameras:
            return
        now = time.monotonic()
        if now - self._last < 1.0 / MAX_FPS:
            return
        self._last = now
        try:
            for name in list(self.cameras):
                title = window_title(name)
                if name in self._shown and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    self._drop(name)  # closed with the window's close button
                    continue
                try:
                    img = self._picture(name)
                except KeyError:  # camera removed or disabled
                    self._drop(name)
                    continue
                if name not in self._shown:
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    w = min(START_WIDTH, img.shape[1])
                    cv2.resizeWindow(title, w, int(img.shape[0] * w / img.shape[1]))
                    self._shown.add(name)
                cv2.imshow(title, img)
            if self._shown and (cv2.waitKey(1) & 0xFF) in CLOSE_KEYS:
                self.close()
        except cv2.error as e:
            log.error("Live view unavailable: this OpenCV build can't open windows (%s). If "
                      "opencv-python-headless is installed, uninstall it and reinstall opencv-python.",
                      str(e).strip().splitlines()[-1])
            self.cameras, self._shown = [], set()

    def hide(self) -> None:
        """Close the windows but remember them, e.g. while a menu is open."""
        for name in list(self._shown):
            self._destroy(name)

    def close(self) -> None:
        self.hide()
        self.cameras = []

    # ---- internals ------------------------------------------------------------------------
    def _picture(self, name: str) -> np.ndarray:
        img = self.engine.annotated_frame(name)
        stream = self.engine.streams.get(name)
        if img is None:
            img = np.zeros((360, 640, 3), np.uint8)
        if stream is None or not stream.is_live():
            text = "connecting" if stream is not None and stream.status == "starting" else "no signal"
            cv2.putText(img, text, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 230), 2, cv2.LINE_AA)
        return img

    def _drop(self, name: str) -> None:
        self._destroy(name)
        if name in self.cameras:
            self.cameras.remove(name)

    def _destroy(self, name: str) -> None:
        if name in self._shown:
            self._shown.discard(name)
            try:
                cv2.destroyWindow(window_title(name))
                cv2.waitKey(1)  # let the window system actually remove it
            except cv2.error:
                pass
