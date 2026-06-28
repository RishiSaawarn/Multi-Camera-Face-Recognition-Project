# config.py — Central configuration for face_recognition_project2.
# All tunable parameters here; no hardcoded values in other modules.

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# List of camera sources. 
#   int  → local webcam index (0 = default laptop cam)
#   str  → IP/RTSP URL  e.g. "http://192.168.0.100:8081/video"
#
# Examples: 
#   Test on laptop: CAMERA_SOURCES = [0, 0]  (Duplicates webcam side-by-side)
#   Deploy on RPi:  CAMERA_SOURCES = ["http://<LAPTOP1_IP>:8081/video", "http://<LAPTOP2_IP>:8081/video"]
CAMERA_SOURCES = [
    "http://192.168.0.100:8081/video",
    "http://192.168.0.105:8081/video"
]
# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------
# Scale factor applied before detection  (0.5 = half resolution = faster)
FRAME_RESIZE = 1.0   # Keep full resolution — YOLOv8 handles internal resizing

# ---------------------------------------------------------------------------
# Face detection — YOLOv8n-face ONNX
# ---------------------------------------------------------------------------
# Path to the downloaded ONNX model (populated by download_models.py)
YOLO_MODEL_PATH = "models/yolov8n-face-lindevs.onnx"

# YOLOv8 input dimensions (must match the exported model)
YOLO_INPUT_SIZE = 640          # model expects 640×640
YOLO_CONF_THRESHOLD = 0.45     # min confidence to consider it a face
YOLO_IOU_THRESHOLD = 0.45      # NMS overlap threshold

# ---------------------------------------------------------------------------
# Face Anti-Spoofing (Liveness Detection) — MiniFASNet ONNX
# ---------------------------------------------------------------------------
# Set True to enable checking if faces are real humans vs photos/screens
ENABLE_ANTI_SPOOFING = True

# Path to the FAS model
FAS_MODEL_PATH = "models/MiniFASNetV2.onnx"

# The confidence threshold to accept a face as "Real".
# Below this, it's flagged as SPOOF. Typical values: 0.8 to 0.9.
FAS_REAL_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Face recognition
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset"

# Cosine-similarity threshold for a positive match  (0.0–1.0)
SIMILARITY_THRESHOLD = 0.60

# Face ROI size before computing embedding
EMBEDDING_SIZE = (128, 128)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_NAME = "Face Recognition v2 — YOLOv8n"
SHOW_FPS = True

# ---------------------------------------------------------------------------
# MJPEG streaming (for headless Raspberry Pi deployment)
# ---------------------------------------------------------------------------
# Set True to start an HTTP server that streams the annotated video.
# View on your laptop: http://<rpi-ip>:<STREAM_PORT>/
STREAM_ENABLED = True

# Port the HTTP server listens on (open this in your firewall if needed)
STREAM_PORT = 8080

# JPEG quality for the stream (1-100).  Lower = smaller bandwidth, more artefacts.
STREAM_JPEG_QUALITY = 75

# Set True to also show a local cv2 window (requires a display / monitor).
# Set False when running headless on RPi.
SHOW_LOCAL_WINDOW = False   # change to False on headless RPi
