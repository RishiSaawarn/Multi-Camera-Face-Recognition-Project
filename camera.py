# camera.py — Camera input abstraction (identical interface to project1).
#
# Switching from webcam to IP stream (Raspberry Pi) only requires changing
# CAMERA_SOURCE in config.py — no other code changes needed.

import cv2


def get_camera(source=0) -> cv2.VideoCapture:
    """
    Open and return an OpenCV VideoCapture object.

    Parameters
    ----------
    source : int | str
        int → local webcam index (0 = system default)
        str → URL: "rtsp://...", "http://...", "/dev/video0"

    Raises
    ------
    RuntimeError if the source cannot be opened.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera source: {source!r}. "
            "Check the device index / URL and that the device is connected."
        )

    # Reduce buffer size to 1 frame — minimises latency for live streams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[camera] Opened source: {source!r}")
    return cap


def release_camera(cap: cv2.VideoCapture) -> None:
    """Safely release the VideoCapture resource."""
    if cap and cap.isOpened():
        cap.release()
        print("[camera] Camera released.")
