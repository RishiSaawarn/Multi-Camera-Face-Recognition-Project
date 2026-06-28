# main.py — Entry point for face_recognition_project2.
#
# Detection backend: YOLOv8n-face via ONNX Runtime (CPU)
#
# Run:
#   python main.py
#
# Exit: press ESC.
# Prerequisites: run  python download_models.py  first.

import cv2
import sys
import os
import numpy as np

from config import (
    CAMERA_SOURCES,
    FRAME_RESIZE,
    DATASET_PATH,
    SIMILARITY_THRESHOLD,
    WINDOW_NAME,
    SHOW_FPS,
    YOLO_MODEL_PATH,
    STREAM_ENABLED,
    STREAM_PORT,
    STREAM_JPEG_QUALITY,
    SHOW_LOCAL_WINDOW,
    ENABLE_ANTI_SPOOFING,
    FAS_MODEL_PATH,
)
from camera import get_camera, release_camera
from detection import FaceDetector
from antispoofing import AntiSpoofingEngine
from recognition import load_known_faces, recognise_face, extract_face_roi
from utils import draw_face, draw_fps, FPSCounter
from streamer import MJPEGStreamer

import os


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Preflight — make sure the ONNX model exists
    # ------------------------------------------------------------------
    if not os.path.exists(YOLO_MODEL_PATH):
        print(
            f"[main] ERROR: ONNX model not found at '{YOLO_MODEL_PATH}'.\n"
            "       Run:  python download_models.py"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Cameras
    # ------------------------------------------------------------------
    caps = []
    for src in CAMERA_SOURCES:
        try:
            cap = get_camera(src)
            caps.append(cap)
        except RuntimeError as exc:
            print(f"[main] ERROR: {exc}")
    
    if not caps:
        print("[main] FATAL: No camera sources could be opened.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. YOLOv8n-face ONNX detector (initialised ONCE)
    # ------------------------------------------------------------------
    detector = FaceDetector(model_path=YOLO_MODEL_PATH)

    # ------------------------------------------------------------------
    # 3. Known face embeddings
    # ------------------------------------------------------------------
    known_faces = load_known_faces(DATASET_PATH)

    # ------------------------------------------------------------------
    # 4. Anti-Spoofing (Liveness Engine)
    # ------------------------------------------------------------------
    fas_engine = None
    if ENABLE_ANTI_SPOOFING:
        if not os.path.exists(FAS_MODEL_PATH):
            print(
                f"[main] ERROR: FAS model not found at '{FAS_MODEL_PATH}'.\n"
                "       Run:  python download_models.py"
            )
            sys.exit(1)
        fas_engine = AntiSpoofingEngine(model_path=FAS_MODEL_PATH)

    # ------------------------------------------------------------------
    # 5. MJPEG Streamer (optional — for headless / remote viewing)
    # ------------------------------------------------------------------
    streamer = None
    if STREAM_ENABLED:
        streamer = MJPEGStreamer(port=STREAM_PORT, jpeg_quality=STREAM_JPEG_QUALITY)
        streamer.start()

    # ------------------------------------------------------------------
    # 5. Display setup
    # ------------------------------------------------------------------
    fps_counter = FPSCounter(smoothing=30)
    if SHOW_LOCAL_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[main] Starting capture loop. Press ESC to exit.")

    # ------------------------------------------------------------------
    # 5. Main capture loop
    # ------------------------------------------------------------------
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        if not frames:
            print("[main] WARNING: Failed to read from any camera source.")
            break

        # ---- 5a. Side-by-Side Image Stitching ---------------------------
        if len(frames) == 1:
            stitched_frame = frames[0]
        else:
            # Resize all frames to match the height of the first frame
            target_h = frames[0].shape[0]
            resized_frames = []
            for f in frames:
                h, w = f.shape[:2]
                if h != target_h:
                    scale = target_h / h
                    resized_w = int(w * scale)
                    f = cv2.resize(f, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(f)
            # Glue horizontally
            stitched_frame = np.hstack(resized_frames)

        # Optional extra global downscale
        if FRAME_RESIZE != 1.0:
            display_frame = stitched_frame.copy()
            proc_frame = cv2.resize(
                stitched_frame, None,
                fx=FRAME_RESIZE, fy=FRAME_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            display_frame = stitched_frame
            proc_frame = stitched_frame

        # ---- 5b. YOLOv8n-face detection --------------------------------
        # detect_faces() handles its own internal resize to 640×640
        boxes = detector.detect_faces(proc_frame)

        # ---- 5c. Scale boxes back if frame was downscaled --------------
        inv = 1.0 / FRAME_RESIZE
        if FRAME_RESIZE != 1.0:
            boxes = [
                (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                for (x, y, w, h) in boxes
            ]

        # ---- 5d. Recognise each face -----------------------------------
        for box in boxes:
            roi = extract_face_roi(display_frame, box)
            if roi is None or roi.size == 0:
                continue

            # Anti-Spoofing Check
            if fas_engine is not None:
                is_real, real_prob = fas_engine.check(display_frame, box)
                if not is_real:
                    # Fake face! Draw SPOOF box and skip recognition
                    draw_face(display_frame, box, "SPOOF", real_prob)
                    continue

            # If real (or if FAS disabled), proceed to recognize identity
            label, similarity = recognise_face(
                roi, known_faces, threshold=SIMILARITY_THRESHOLD
            )
            draw_face(display_frame, box, label, similarity)

        # ---- 5e. FPS overlay -------------------------------------------
        if SHOW_FPS:
            draw_fps(display_frame, fps_counter.tick())

        # ---- 5f. Push to MJPEG stream ----------------------------------
        if streamer:
            streamer.push(display_frame)

        # ---- 5g. Optional local window ---------------------------------
        if SHOW_LOCAL_WINDOW:
            cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            print("[main] ESC pressed — exiting.")
            break

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    if streamer:
        streamer.stop()
    detector.close()
    for cap in caps:
        release_camera(cap)
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    main()
