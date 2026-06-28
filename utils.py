# utils.py — Drawing utilities: bounding boxes, labels, FPS counter.

import time
import cv2
import numpy as np

COLOR_KNOWN     = (0, 220, 0)       # Green  — recognised person
COLOR_UNKNOWN   = (0, 0, 220)       # Red    — unknown face
COLOR_SPOOF     = (0, 0, 255)       # Red    — spoof attack
COLOR_LABEL_BG  = (30, 30, 30)      # Dark label background
COLOR_FPS       = (255, 200, 0)     # Cyan-ish FPS text
BOX_THICKNESS   = 2


def draw_face(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    similarity: float = 0.0,
) -> None:
    """Draw bounding box + name label on *frame* in-place."""
    x, y, w, h = box
    if label == "SPOOF":
        color = COLOR_SPOOF
    else:
        color = COLOR_KNOWN if label != "Unknown" else COLOR_UNKNOWN

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

    # Label text
    if label == "SPOOF":
        text = f"SPOOF ({similarity:.2f})"
    elif label != "Unknown":
        text = f"{label} ({similarity:.2f})"
    else:
        text = label
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)

    ly = max(y - th - bl - 4, 0)
    cv2.rectangle(frame, (x, ly), (x + tw + 4, ly + th + bl + 4),
                  COLOR_LABEL_BG, cv2.FILLED)
    cv2.putText(frame, text, (x + 2, ly + th + 2),
                font, scale, color, thick, cv2.LINE_AA)


class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, smoothing: int = 30):
        self._times: list[float] = []
        self._smoothing = smoothing

    def tick(self) -> float:
        self._times.append(time.monotonic())
        if len(self._times) > self._smoothing:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS in the top-left corner of *frame* in-place."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2, cv2.LINE_AA)
