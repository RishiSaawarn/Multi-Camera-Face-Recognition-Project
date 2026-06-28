# streamer.py — MJPEG-over-HTTP streaming server.
#
# Runs in a background thread; main.py pushes annotated frames into it.
# View on any browser on the same network:
#   http://<raspberry-pi-ip>:8080/video
#
# No extra dependencies — uses only Python stdlib + OpenCV (already installed).

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared frame buffer — thread-safe
# ---------------------------------------------------------------------------

class FrameBuffer:
    """
    Thread-safe single-frame buffer.

    The HTTP handler reads from it; main.py writes into it.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None   # latest JPEG-encoded frame
        self._event = threading.Event()        # signals a new frame is ready

    def push(self, frame: np.ndarray, jpeg_quality: int = 80) -> None:
        """
        Encode *frame* as JPEG and store it.

        Parameters
        ----------
        frame : np.ndarray  BGR image from OpenCV.
        jpeg_quality : int  JPEG compression quality (1–100).
        """
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frame = buf.tobytes()
        self._event.set()   # wake any waiting HTTP handler

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Block until a new frame is available (or *timeout* seconds elapse).

        Returns JPEG bytes or None.
        """
        self._event.wait(timeout)
        self._event.clear()
        with self._lock:
            return self._frame


# Module-level singleton shared between main.py and the HTTP handler
_buffer = FrameBuffer()


# ---------------------------------------------------------------------------
# MJPEG HTTP handler
# ---------------------------------------------------------------------------

_BOUNDARY = b"--mjpegframe"

class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves a single MJPEG stream at /video and a simple HTML page at /."""

    # Suppress default request logging to keep console clean
    def log_message(self, fmt, *args):  # noqa: N802
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/video":
            self._serve_stream()
        elif self.path in ("/", "/index.html"):
            self._serve_index()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                jpeg = _buffer.get(timeout=2.0)
                if jpeg is None:
                    continue

                # MJPEG boundary + headers
                header = (
                    _BOUNDARY
                    + b"\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                try:
                    self.wfile.write(header + jpeg + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break   # client disconnected
        except Exception:
            pass

    def _serve_index(self):
        html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Face Recognition &mdash; Live Stream</title>
  <style>
    body { margin:0; background:#111; display:flex;
           flex-direction:column; align-items:center;
           justify-content:center; height:100vh; }
    h1   { color:#eee; font-family:sans-serif; margin-bottom:12px; font-size:1.2rem; }
    img  { max-width:95vw; border-radius:8px; box-shadow:0 0 24px #000; }
  </style>
</head>
<body>
  <h1>&#x1F4F7; Face Recognition &mdash; Live Stream</h1>
  <img src="/video" alt="Live stream" />
</body>
</html>
""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MJPEGStreamer:
    """
    MJPEG streaming server that runs in a background daemon thread.

    Usage
    -----
    streamer = MJPEGStreamer(port=8080)
    streamer.start()

    # inside loop:
    streamer.push(annotated_frame)

    # on exit:
    streamer.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 80):
        self._host    = host
        self._port    = port
        self._quality = jpeg_quality
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start HTTP server in a background daemon thread."""
        self._server = HTTPServer((self._host, self._port), _MJPEGHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,    # dies automatically when main thread exits
            name="mjpeg-streamer",
        )
        self._thread.start()
        print(
            f"[streamer] MJPEG stream started — "
            f"open http://<this-device-ip>:{self._port}/ in a browser"
        )

    def push(self, frame: np.ndarray) -> None:
        """Push an annotated BGR frame into the stream."""
        _buffer.push(frame, jpeg_quality=self._quality)

    def stop(self) -> None:
        """Shut down the HTTP server gracefully."""
        if self._server:
            self._server.shutdown()
            print("[streamer] MJPEG server stopped.")
