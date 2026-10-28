"""
Object detector module for distraction detection (phone, cigarette, drink).
Uses YOLO with spatial proximity constraints (near face or hands) and persistence time tracking.
"""

import os
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("ObjectDetector")


class DistractionTracker:
    """Tracks duration of detected distraction objects near the driver to prevent false positives."""
    def __init__(self):
        self.start_times: Dict[str, Optional[float]] = {
            "phone": None,
            "smoke": None,
            "drink": None
        }
        self.limits: Dict[str, float] = {
            "phone": config.DISTRACTION_PHONE_LIMIT,
            "smoke": config.DISTRACTION_SMOKE_LIMIT,
            "drink": config.DISTRACTION_DRINK_LIMIT
        }

    def update(self, detections: List[Dict], now: Optional[float] = None) -> Dict:
        """
        Update distraction states based on current detections.
        Returns dict with alert flags for each distraction class.
        """
        if now is None:
            now = time.time()

        # Map synonym keywords to canonical targets
        synonyms = {
            "phone": ["phone", "cell", "mobile"],
            "smoke": ["smoke", "cigar", "cigarette", "tobacco", "vape"],
            "drink": ["drink", "bottle", "cup", "beverage", "can"]
        }

        # Find classes currently near driver
        active_classes = set()
        for det in detections:
            if det.get("is_near_driver", False):
                raw_label = str(det.get("label", "")).lower()
                for target_key, syn_list in synonyms.items():
                    if any(s in raw_label for s in syn_list):
                        active_classes.add(target_key)

        alerts = {}
        durations = {}
        for target_key, limit in self.limits.items():
            if target_key in active_classes:
                if self.start_times[target_key] is None:
                    self.start_times[target_key] = now
                duration = now - self.start_times[target_key]
                durations[target_key] = duration
                alerts[target_key] = duration >= limit
            else:
                self.start_times[target_key] = None
                durations[target_key] = 0.0
                alerts[target_key] = False

        return {
            "is_phone_alert": alerts["phone"],
            "is_smoke_alert": alerts["smoke"],
            "is_drink_alert": alerts["drink"],
            "durations": durations,
            "active_classes": list(active_classes)
        }

    def reset(self):
        """Reset all distraction timers."""
        for k in self.start_times:
            self.start_times[k] = None


class ObjectDetector:
    """YOLO-based object detector for distraction recognition with spatial constraints."""
    def __init__(self, model_path=None, enabled: bool = config.ENABLE_OBJECT_DETECTION):
        self.enabled = enabled
        self.model_path = model_path or str(config.YOLO_MODEL_PATH)
        self.model = None
        self.is_ready = False
        self.tracker = DistractionTracker()

        if self.enabled:
            self._load_model()

    def _load_model(self):
        """Attempt to load YOLO model using ultralytics with automatic fallbacks."""
        candidates = [
            self.model_path,
            "model/distraction.pt",
            "distraction.pt",
            "model/yolo11n.pt",
            "yolo11n.pt",
            "model/yolov8n.pt",
            "yolov8n.pt"
        ]

        found_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                found_path = candidate
                break

        if not found_path:
            self.is_ready = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(found_path)
            self.model_path = found_path
            self.is_ready = True
            logger.info(f"Đã nạp thành công mô hình YOLO: {found_path}")
        except Exception as e:
            logger.warning(f"Không thể tải model YOLO '{found_path}': {e}. Vô hiệu hóa nhận diện vật thể.")
            self.is_ready = False

    def detect(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None,
               wrist_coords: Optional[List[Tuple[int, int]]] = None) -> List[Dict]:
        """
        Run inference on frame and return list of detections with spatial constraint check.
        Detection format: {"label": str, "conf": float, "bbox": (x1, y1, x2, y2), "is_near_driver": bool}
        """
        if not self.enabled or not self.is_ready or self.model is None or frame is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=config.YOLO_CONF_THRESHOLD,
                iou=config.YOLO_IOU_THRESHOLD,
                verbose=False
            )
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = self.model.names.get(cls_id, str(cls_id))

                    # Spatial constraint check
                    near = self._check_spatial_proximity((x1, y1, x2, y2), face_bbox, wrist_coords)
                    detections.append({
                        "label": label,
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2),
                        "is_near_driver": near
                    })
            return detections
        except Exception:
            return []

    def process(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None,
                wrist_coords: Optional[List[Tuple[int, int]]] = None,
                now: Optional[float] = None) -> Dict:
        """
        Process frame: detect objects and track distraction duration.
        Returns dict:
        - 'detections': List[Dict]
        - 'is_phone_alert': bool
        - 'is_smoke_alert': bool
        - 'is_drink_alert': bool
        - 'active_classes': List[str]
        """
        if not self.enabled or not self.is_ready:
            return {
                "detections": [],
                "is_phone_alert": False,
                "is_smoke_alert": False,
                "is_drink_alert": False,
                "active_classes": []
            }

        detections = self.detect(frame, face_bbox=face_bbox, wrist_coords=wrist_coords)
        tracker_result = self.tracker.update(detections, now=now)
        tracker_result["detections"] = detections
        return tracker_result

    def _check_spatial_proximity(self, obj_bbox: Tuple[int, int, int, int],
                                 face_bbox: Optional[Tuple[int, int, int, int]],
                                 wrist_coords: Optional[List[Tuple[int, int]]]) -> bool:
        """Check if detected object bbox is close to face or hands."""
        ox1, oy1, ox2, oy2 = obj_bbox
        o_cx, o_cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2

        # Check proximity to face
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            # Expanded face region (e.g. ear/cheek/mouth area)
            pad_w = (fx2 - fx1) * 0.8
            pad_h = (fy2 - fy1) * 0.8
            if (fx1 - pad_w <= o_cx <= fx2 + pad_w) and (fy1 - pad_h <= o_cy <= fy2 + pad_h):
                return True

        # Check proximity to wrists
        if wrist_coords:
            for (wx, wy) in wrist_coords:
                dist = np.hypot(o_cx - wx, o_cy - wy)
                if dist < 120:  # Within 120 pixels of wrist
                    return True

        return False
