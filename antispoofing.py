import cv2
import numpy as np
import onnxruntime as ort

from config import FAS_MODEL_PATH, FAS_REAL_THRESHOLD

class AntiSpoofingEngine:
    """
    MiniFASNet Liveness Detection running on ONNX Runtime.
    """

    def __init__(self, model_path: str = FAS_MODEL_PATH, threshold: float = FAS_REAL_THRESHOLD):
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._threshold = threshold
        
        print(f"[antispoofing] MiniFASNet loaded from '{model_path}' | real≥{self._threshold}")

    def check(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[bool, float]:
        """
        Check if the face inside the box is real.

        Returns:
            (is_real, confidence_score)
        """
        x, y, w, h = box
        img_h, img_w = frame.shape[:2]

        # 1. FAS models require context around the face (scale factor typically 2.7)
        scale = 2.7
        cx, cy = x + w / 2, y + h / 2
        
        # New width and height
        new_w = w * scale
        new_h = h * scale
        
        # Ensure it's a square
        side = max(new_w, new_h)
        
        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(img_w, int(cx + side / 2))
        y2 = min(img_h, int(cy + side / 2))

        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return False, 0.0

        # 2. Resize and preprocess
        # MiniFASNet expects 80x80 input
        resized = cv2.resize(crop, (80, 80))
        
        # BGR object
        # The model usually expects BGR -> RGB? Yakhyo's FAS trained on PyTorch typically uses RGB, 
        # but Silent-Face-AntiSpoofing uses BGR. We'll leave as BGR since cv2 reads BGR and many FAS run on raw cv2 frames.
        blob = cv2.dnn.blobFromImage(resized, 1.0, (80, 80), (0, 0, 0), swapRB=False, crop=False)

        # 3. Inference
        out = self._session.run(None, {self._input_name: blob})[0]
        
        # Output is typically logits for classes: [fake_1, fake_2, real]
        # In yakhyo's exporter, index 0 is Fake, index 1 is Fake (or spoof), index 2 is Real.
        # Alternatively, a Softmax over the logits
        logits = out[0]
        
        # Softmax
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        # Probability of being real is ALWAYS index 1 in Silent-Face-Anti-Spoofing MiniFASNet
        if len(probs) == 3:
            real_prob = float(probs[1])
        elif len(probs) == 2:
            real_prob = float(probs[1]) 
        else:
            real_prob = float(probs[0])
            
        is_real = real_prob >= self._threshold
        return is_real, real_prob
