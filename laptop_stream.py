# laptop_stream.py — Stream the laptop's built-in camera over HTTP (MJPEG).
#
# Run this on your LAPTOP (Windows):
#   python laptop_stream.py
#
# The RPi then reads the stream URL in config.py:
#   CAMERA_SOURCE = "http://[laptop-ipv6]:8081/video"   ← IPv6 (bracket notation)
CAMERA_SOURCE = "http://192.168.0.104:8081/video" #    ← IPv4 alternative
#
# Requirements (already installed):
#   opencv-python, numpy  — no extra packages needed

import sys
import time
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

# ---------------------------------------------------------------------------
# Settings — change PORT if 8081 is already in use
# ---------------------------------------------------------------------------
PORT           = 8081
CAMERA_INDEX   = 0       # 0 = default built-in laptop webcam
JPEG_QUALITY   = 80      # 1–100 (lower = less bandwidth, more artefacts)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class _CameraThread(threading.Thread):
    """Reads frames from the webcam continuously in a background thread."""

    def __init__(self):
        super().__init__(daemon=True, name="camera-reader")
        self._cap   = cv2.VideoCapture(CAMERA_INDEX)
        self._lock  = threading.Lock()
        self._frame = None
        self._ok    = self._cap.isOpened()

        if not self._ok:
            print(f"[stream] ERROR: Cannot open camera index {CAMERA_INDEX}.")
            sys.exit(1)

        # Reduce OS buffer to 1 frame for minimum latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[stream] Camera {CAMERA_INDEX} opened.")

    def run(self):
        while True:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        """Return the latest frame as JPEG bytes, or None."""
        with self._lock:
            if self._frame is None:
                return None
            ok, buf = cv2.imencode(
                ".jpg", self._frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            return buf.tobytes() if ok else None


# Module-level camera thread (starts once)
_cam = _CameraThread()
_cam.start()

_BOUNDARY = b"--frame"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _StreamHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):   # suppress per-request logs
        pass

    def do_GET(self):   # noqa: N802
        if self.path == "/video":
            self._serve_mjpeg()
        elif self.path in ("/", "/index.html"):
            self._serve_index()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_mjpeg(self):
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
                jpeg = _cam.get_frame()
                if jpeg is None:
                    time.sleep(0.01)
                    continue

                header = (
                    _BOUNDARY
                    + b"\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                try:
                    self.wfile.write(header + jpeg + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break   # RPi disconnected
        except Exception:
            pass

    def _serve_index(self):
        html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Laptop Camera Stream</title>
  <style>
    body { margin:0; background:#111; display:flex; flex-direction:column;
           align-items:center; justify-content:center; height:100vh; }
    h1   { color:#eee; font-family:sans-serif; margin-bottom:12px; }
    img  { max-width:95vw; border-radius:8px; }
  </style>
</head>
<body>
  <h1>Laptop Camera &mdash; Raw Feed</h1>
  <img src="/video" alt="stream" />
</body>
</html>""".encode("utf-8")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        except Exception:
            pass



# ---------------------------------------------------------------------------
# Helpers — print local IP addresses for reference
# ---------------------------------------------------------------------------

def _print_addresses(port: int) -> None:
    print("\n[stream] Laptop camera stream running. Use one of these URLs on RPi:\n")

    hostname = socket.gethostname()

    # IPv4 addresses
    try:
        ipv4 = socket.gethostbyname(hostname)
        if not ipv4.startswith("127."):
            print(f"  IPv4  →  http://{ipv4}:{port}/video")
    except Exception:
        pass

    # IPv6 addresses (all interfaces)
    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET6):
            addr = info[4][0]
            # Skip loopback (::1)
            if addr == "::1":
                continue
            # For link-local addresses, strip the zone ID for display
            display = addr.split("%")[0]
            print(f"  IPv6  →  http://[{display}]:{port}/video")
    except Exception:
        pass

    print(f"\n  Also verify in browser: http://localhost:{port}/")
    print("\n  In RPi config.py set:")
    print(f'  CAMERA_SOURCE = "http://[<laptop-ipv6>]:{port}/video"\n')
    print("[stream] Press Ctrl+C to stop.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Listen on all interfaces (IPv4 + IPv6)
    server = HTTPServer(("", PORT), _StreamHandler)
    _print_addresses(PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stream] Stopped.")
    finally:
        server.server_close()
