# download_models.py — Download required ONNX model(s) on first run.
#
# Run ONCE before main.py:
#   python download_models.py
#
# Downloads:
#   models/yolov8n-face.onnx  (~6 MB)
#       Source: Hugging Face — arnabdhar/YOLOv8-Face-Detection

import os
import sys
import urllib.request


MODELS_DIR = "models"

DOWNLOADS = {
    # Source: lindevs/yolov8-face — GitHub release v1.0.1
    # YOLOv8n trained on WiderFace, ONNX opset 19, ~12 MB
    "yolov8n-face-lindevs.onnx": (
        "https://github.com/lindevs/yolov8-face/releases/download/1.0.1"
        "/yolov8n-face-lindevs.onnx"
    ),
    # Source: yakhyo/face-anti-spoofing — ONNX weights
    # MiniFASNetV2 for Liveness Detection, ~1.7 MB
    "MiniFASNetV2.onnx": (
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
        "MiniFASNetV2.onnx"
    ),
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        sys.stdout.write(
            f"\r  [{'█' * bar}{' ' * (50 - bar)}] {pct:5.1f}%  "
            f"({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)"
        )
        sys.stdout.flush()


def download_all() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in DOWNLOADS.items():
        dest = os.path.join(MODELS_DIR, filename)

        if os.path.exists(dest):
            print(f"[download] '{dest}' already exists — skipping.")
            continue

        print(f"[download] Downloading '{filename}' from:\n  {url}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            print(f"\n[download] Saved → '{dest}'")
        except Exception as exc:
            print(f"\n[download] ERROR: {exc}")
            # Remove partially downloaded file
            if os.path.exists(dest):
                os.remove(dest)
            sys.exit(1)

    print("\n[download] All models ready.")


if __name__ == "__main__":
    download_all()
