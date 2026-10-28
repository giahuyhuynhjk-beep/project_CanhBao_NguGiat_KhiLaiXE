"""
Hand analyzer module using MediaPipe HandLandmarker.
Monitors hands on steering wheel, checks wrist positions against Steering Wheel ROI,
and provides an interactive calibration tool for setting the ROI.
"""

import os
import json
import time
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("HandAnalyzer")


class HandAnalyzer:
    def __init__(self, model_path=None, roi_config_path=None):
        if model_path is None:
            model_path = str(config.HAND_LANDMARKER_MODEL)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy file model HandLandmarker: {model_path}")

        base_options_hand = python.BaseOptions(model_asset_path=model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options_hand)

        self.roi = config.DEFAULT_STEERING_ROI.copy()
        self.roi_path = str(roi_config_path or config.STEERING_ROI_PATH)
        self.load_roi()

        # Timers
        self.no_hands_start_time: Optional[float] = None
        self.out_of_roi_start_time: Optional[float] = None

    def load_roi(self):
        """Load steering wheel ROI coordinates from JSON file if available."""
        if os.path.exists(self.roi_path):
            try:
                with open(self.roi_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if all(k in data for k in ("x_min", "y_min", "x_max", "y_max")):
                        self.roi = data
                        print(f"[INFO] Đã nạp Steering Wheel ROI từ: {self.roi_path}")
            except Exception as e:
                print(f"[WARNING] Không đọc được file ROI, sử dụng mặc định: {e}")

    def save_roi(self, roi_dict: dict):
        """Save calibrated ROI to file."""
        self.roi = roi_dict
        os.makedirs(os.path.dirname(self.roi_path), exist_ok=True)
        with open(self.roi_path, "w", encoding="utf-8") as f:
            json.dump(roi_dict, f, indent=4)
        print(f"[INFO] Đã lưu Steering Wheel ROI mới vào: {self.roi_path}")

    def detect(self, mp_image: mp.Image):
        """Detect hands from mp.Image."""
        return self.detector.detect(mp_image)

    def is_legacy_hands_off(self, hand_landmarks) -> bool:
        """Legacy logic: hands-off if fewer than 2 hands are detected in the frame."""
        return not hand_landmarks or len(hand_landmarks) < 2

    def get_wrist_coordinates(self, hand_landmarks, img_w: int, img_h: int) -> List[Tuple[int, int]]:
        """Extract pixel (x, y) coordinates of wrists (landmark index 0) from detected hands."""
        wrists = []
        if not hand_landmarks:
            return wrists
        for hand in hand_landmarks:
            if len(hand) > 0:
                wx = int(hand[0].x * img_w)
                wy = int(hand[0].y * img_h)
                wrists.append((wx, wy))
        return wrists

    def check_steering_status(
        self,
        hand_landmarks,
        img_w: int,
        img_h: int,
        now: Optional[float] = None,
        fallback_wrists: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze hands relative to steering wheel ROI.
        Distinguishes between:
        1. 'NO_HANDS': No hands detected (possible camera occlusion or hands low).
        2. 'OUT_OF_ROI': Hands detected, but both wrists are outside the Steering Wheel ROI.
        3. 'OK': At least one wrist is safely inside the Steering Wheel ROI.

        If hand landmarks are occluded or missing, fallback_wrists (e.g. from PoseLandmarker)
        are checked before declaring NO_HANDS, preventing false alarms when hands are partially covered.

        Returns dict:
        - 'status': 'OK' | 'NO_HANDS' | 'OUT_OF_ROI'
        - 'hand_count': int
        - 'wrists': List[Tuple[int, int]]
        - 'is_hands_off_alert': bool (NO_HANDS > HANDS_OFF_TIME_LIMIT)
        - 'is_out_of_roi_alert': bool (OUT_OF_ROI > HANDS_OUT_OF_ROI_LIMIT)
        - 'duration': float
        """
        if now is None:
            now = time.time()

        wrists = self.get_wrist_coordinates(hand_landmarks, img_w, img_h)
        # Use fallback wrists (e.g. from Pose analyzer) if hand landmarker missed wrists due to partial occlusion
        if len(wrists) == 0 and fallback_wrists:
            wrists = list(fallback_wrists)

        hand_count = len(wrists)

        # Steering ROI boundary in pixels
        rx1 = self.roi["x_min"] * img_w
        ry1 = self.roi["y_min"] * img_h
        rx2 = self.roi["x_max"] * img_w
        ry2 = self.roi["y_max"] * img_h

        # Count wrists inside and outside ROI
        wrists_inside = 0
        wrists_outside = 0
        for (wx, wy) in wrists:
            if rx1 <= wx <= rx2 and ry1 <= wy <= ry2:
                wrists_inside += 1
            else:
                wrists_outside += 1

        is_hands_off_alert = False
        is_out_of_roi_alert = False
        status = "OK"
        duration = 0.0

        if wrists_inside > 0:
            # At least one hand/wrist is safely in steering ROI (safe even if other hand is occluded)
            status = "OK"
            self.no_hands_start_time = None
            self.out_of_roi_start_time = None

        elif wrists_outside > 0:
            # Wrists detected but all are outside steering ROI
            status = "OUT_OF_ROI"
            self.no_hands_start_time = None
            if self.out_of_roi_start_time is None:
                self.out_of_roi_start_time = now
            duration = now - self.out_of_roi_start_time
            if duration >= config.HANDS_OUT_OF_ROI_LIMIT:
                is_out_of_roi_alert = True

        else:
            # 0 wrists detected in frame (camera blocked or hands occluded)
            status = "NO_HANDS"
            self.out_of_roi_start_time = None
            if self.no_hands_start_time is None:
                self.no_hands_start_time = now
            duration = now - self.no_hands_start_time
            if duration >= config.HANDS_OFF_TIME_LIMIT:
                is_hands_off_alert = True

        return {
            "status": status,
            "hand_count": hand_count,
            "wrists": wrists,
            "is_hands_off_alert": is_hands_off_alert,
            "is_out_of_roi_alert": is_out_of_roi_alert,
            "duration": duration
        }


def calibrate_steering_roi(camera_id: int = 0):
    """
    Interactive GUI tool to calibrate and save Steering Wheel ROI.
    Allows user to drag a rectangle on the video feed to define the steering wheel area.
    """
    print("\n[INFO] Đang mở công cụ hiệu chuẩn vùng vô-lăng (ROI)...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam.")
        return

    analyzer = HandAnalyzer()
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    temp_rect = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, fx, fy, temp_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            fx, fy = x, y
            temp_rect = (min(ix, fx), min(iy, fy), max(ix, fx), max(iy, fy))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            temp_rect = (min(ix, fx), min(iy, fy), max(ix, fx), max(iy, fy))

    cv2.namedWindow("Hieu Chuan Vung Vo-lang (Steering ROI)")
    cv2.setMouseCallback("Hieu Chuan Vung Vo-lang (Steering ROI)", mouse_callback)

    saved_msg_time = 0
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        display = frame.copy()

        # Draw current or drawing ROI
        if temp_rect is not None:
            x1, y1, x2, y2 = temp_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "Vung duoc chon", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Draw existing ROI from config
            rx1 = int(analyzer.roi["x_min"] * w)
            ry1 = int(analyzer.roi["y_min"] * h)
            rx2 = int(analyzer.roi["x_max"] * w)
            ry2 = int(analyzer.roi["y_max"] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            cv2.putText(display, "Vung hien tai", (rx1, max(20, ry1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Instructions banner
        cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(display, "Keo chuot trai de chon vung vo-lang.", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Nhan 'S' de LUU - 'R' de RESET - 'Q' de THOAT", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if saved and (time.time() - saved_msg_time < 2.0):
            cv2.putText(display, "DA LUU THANH CONG!", (w // 2 - 120, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Hieu Chuan Vung Vo-lang (Steering ROI)", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            temp_rect = None
            saved = False
        elif key == ord('s') or key == ord('S'):
            if temp_rect is not None:
                x1, y1, x2, y2 = temp_rect
                if (x2 - x1 > 20) and (y2 - y1 > 20):
                    new_roi = {
                        "x_min": round(x1 / w, 3),
                        "y_min": round(y1 / h, 3),
                        "x_max": round(x2 / w, 3),
                        "y_max": round(y2 / h, 3)
                    }
                    analyzer.save_roi(new_roi)
                    saved = True
                    saved_msg_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng công cụ hiệu chuẩn vô-lăng.")


if __name__ == "__main__":
    calibrate_steering_roi()
