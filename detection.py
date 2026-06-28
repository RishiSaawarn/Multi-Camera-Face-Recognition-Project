# detection.py — Face detection using YOLOv8n-face ONNX (CPU).
#
# Pipeline per frame:
#   1. Letterbox-resize frame to 640×640 (preserving aspect ratio)
#   2. Normalise pixels to [0, 1], convert BGR→RGB, HWC→NCHW
#   3. Run ONNX inference
#   4. Parse raw output → bounding boxes + scores
#   5. Apply Non-Maximum Suppression (NMS)
#   6. Scale boxes back to original frame coordinates
#
# No PyTorch / ultralytics required at runtime — only onnxruntime + numpy.

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    YOLO_MODEL_PATH,
    YOLO_INPUT_SIZE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)


class FaceDetector:
    """
    YOLOv8n-face detector running on CPU via ONNX Runtime.

    Initialise once at startup — the session is reused for every frame.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        input_size: int = YOLO_INPUT_SIZE,
        min_confidence: float = YOLO_CONF_THRESHOLD,
        nms_iou: float = YOLO_IOU_THRESHOLD,
    ):
        # Force CPU execution (works identically on Windows and RPi)
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self._input_name  = self._session.get_inputs()[0].name
        self._input_size  = input_size
        self._min_conf    = min_confidence
        self._nms_iou     = nms_iou

        # Inspect output shape to handle both (1,5,8400) and (1,8400,5)
        out_shape = self._session.get_outputs()[0].shape
        self._transposed = (len(out_shape) == 3 and out_shape[2] == 5)

        print(
            f"[detection] YOLOv8n-face ONNX loaded from '{model_path}' "
            f"| input={input_size}×{input_size} | conf≥{min_confidence}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect all faces in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR image from cv2.VideoCapture.

        Returns
        -------
        list of (x, y, w, h) in pixel coordinates of the ORIGINAL frame.
        """
        orig_h, orig_w = frame.shape[:2]

        # Step 1: letterbox + preprocess
        blob, (pad_x, pad_y), scale = self._preprocess(frame)

        # Step 2: inference
        raw = self._session.run(None, {self._input_name: blob})[0]

        # Step 3: parse + NMS
        boxes = self._postprocess(raw, orig_w, orig_h, pad_x, pad_y, scale)
        return boxes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        """
        Letterbox-resize to (input_size × input_size) and build ONNX blob.

        Returns
        -------
        blob : np.ndarray  shape (1, 3, H, W)  float32 in [0, 1]
        (pad_x, pad_y) : pixel padding added on each side
        scale : float — resize scale factor
        """
        h, w = frame.shape[:2]
        s = self._input_size

        # Scale so the longer side fits exactly into s
        scale = min(s / w, s / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to s×s with grey (114)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_x = (s - new_w) // 2
        pad_y = (s - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR → RGB, HWC → NCHW, [0,255] → [0,1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)          # add batch dim

        return blob, (pad_x, pad_y), scale

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        pad_x: int,
        pad_y: int,
        scale: float,
    ) -> list[tuple[int, int, int, int]]:
        """
        Parse ONNX output, apply NMS, map boxes back to original coordinates.

        YOLOv8 output layout (standard export):
            shape (1, 4+num_classes, num_anchors)  →  (1, 5, 8400) for 1 class
        Some exports are transposed:
            shape (1, num_anchors, 5)               →  (1, 8400, 5)

        Values: [cx, cy, w, h, face_confidence]  (cx/cy/w/h in input-image px)
        """
        # Remove batch dim  → (5, 8400) or (8400, 5)
        pred = raw[0]

        if self._transposed:
            # shape (8400, 5) → transpose to (5, 8400) for uniform handling
            pred = pred.T

        # pred is now (5, 8400): rows = [cx, cy, w, h, conf]
        # Filter by confidence first (fast, reduces NMS candidates)
        scores = pred[4]
        mask = scores >= self._min_conf
        if not mask.any():
            return []

        # Filtered arrays  (N, )
        cx, cy, bw, bh = pred[0][mask], pred[1][mask], pred[2][mask], pred[3][mask]
        confs = scores[mask]

        # Convert centre-format → corner-format in letterboxed coords
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Map back: remove padding, undo scale
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clamp to original frame bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Build arrays for cv2.dnn.NMSBoxes
        boxes_xyxy  = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        confs_list  = confs.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy,
            confs_list,
            self._min_conf,
            self._nms_iou,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                xi, yi, wi, hi = boxes_xyxy[i]
                results.append((
                    int(xi), int(yi),
                    int(wi), int(hi),
                ))

        return results

    def close(self):
        """No explicit cleanup needed for ONNX Runtime."""
        print("[detection] FaceDetector closed.")
