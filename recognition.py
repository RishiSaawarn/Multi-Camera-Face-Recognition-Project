# recognition.py — Face embedding and cosine-similarity matching.
#
# Approach: CLAHE + L2-normalised pixel embedding (no external model needed).
# The module is independent of the detection backend — swap detection freely.
#
# To upgrade to deep-learning embeddings (ArcFace/MobileFaceNet via ONNX),
# only replace _compute_embedding() — everything else stays the same.

import os
import cv2
import numpy as np
from typing import Optional

from config import DATASET_PATH, EMBEDDING_SIZE, SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _compute_embedding(face_roi: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 1-D embedding vector from a BGR face crop.

    Steps:
        1. Resize to EMBEDDING_SIZE
        2. Convert to grayscale
        3. CLAHE — robust to varied lighting
        4. Flatten → L2-normalise → float32

    Replace this function with an ONNX model call (MobileFaceNet / ArcFace)
    for production-grade accuracy.
    """
    resized   = cv2.resize(face_roi, EMBEDDING_SIZE, interpolation=cv2.INTER_AREA)
    gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(gray)

    flat = equalised.flatten().astype(np.float32)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat /= norm
    return flat


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two L2-normalised vectors == cosine similarity."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_known_faces(dataset_path: str = DATASET_PATH) -> dict[str, np.ndarray]:
    """
    Load face images from *dataset_path* and compute their embeddings.

    Layout:
        dataset/
            rishi.jpg       ← label becomes 'rishi'
            alice.png

    Returns
    -------
    dict:  label → embedding vector
    """
    known: dict[str, np.ndarray] = {}

    if not os.path.isdir(dataset_path):
        print(
            f"[recognition] WARNING: '{dataset_path}' not found. "
            "All faces will be labelled 'Unknown'."
        )
        return known

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for fname in os.listdir(dataset_path):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in supported:
            continue

        img = cv2.imread(os.path.join(dataset_path, fname))
        if img is None:
            print(f"[recognition] WARNING: Could not read '{fname}', skipping.")
            continue

        known[name] = _compute_embedding(img)
        print(f"[recognition] Loaded: '{name}'")

    print(f"[recognition] Total known faces: {len(known)}")
    return known


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognise_face(
    face_roi: np.ndarray,
    known_faces: dict[str, np.ndarray],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str, float]:
    """
    Match a face ROI against all known embeddings.

    Returns
    -------
    (label, similarity)
        label      → person name or "Unknown"
        similarity → best cosine similarity score
    """
    if not known_faces or face_roi.size == 0:
        return "Unknown", 0.0

    query = _compute_embedding(face_roi)
    best_label, best_score = "Unknown", -1.0

    for label, emb in known_faces.items():
        score = _cosine_similarity(query, emb)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score < threshold:
        return "Unknown", best_score

    return best_label, best_score


def extract_face_roi(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    padding: float = 0.10,
) -> Optional[np.ndarray]:
    """
    Crop a face ROI from *frame* with optional padding.

    Parameters  (x, y, w, h) pixel bounding box.
    Returns     Cropped BGR region or None.
    """
    x, y, w, h = box
    fh, fw = frame.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x);  y2 = min(fh, y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]
