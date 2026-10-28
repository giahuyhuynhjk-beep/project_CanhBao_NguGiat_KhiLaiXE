"""
Pose analyzer module using MediaPipe PoseLandmarker.
Monitors driver posture, shoulder tilt, slumping, and extracts wrist positions.
"""

import os
import math
import time
from typing import Tuple, List, Optional, Dict
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("PoseAnalyzer")


class PoseAnalyzer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(config.POSE_LANDMARKER_MODEL)

        self.is_available = False
        self.detector = None
        self.model_path = model_path

        self._init_detector()

        # State tracking
        self.last_tilt_angle = 0.0
        self.tilt_start_time: Optional[float] = None
        self.tilt_duration = 0.0

    @property
    def slump_start_time(self) -> Optional[float]:
        return self.tilt_start_time

    @slump_start_time.setter
    def slump_start_time(self, val: Optional[float]):
        self.tilt_start_time = val

    def _init_detector(self):
        """Initialize PoseLandmarker safely with graceful degradation."""
        if os.path.exists(self.model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    num_poses=1
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.is_available = True
                logger.info(f"Đã nạp thành công mô hình Pose: {self.model_path}")
            except Exception as e:
                logger.warning(f"Không thể tải PoseLandmarker: {e}. Vô hiệu hóa module Pose.")
                self.is_available = False
        else:
            logger.info(f"Chưa có file '{self.model_path}'. Chế độ giám sát vai sẽ tạm tắt.")
            self.is_available = False

    def detect(self, mp_image: mp.Image):
        """Run pose landmark detection if model is available."""
        if not self.is_available or self.detector is None:
            return None
        try:
            return self.detector.detect(mp_image)
        except Exception:
            return None

    def calculate_shoulder_tilt(self, pose_landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
        """
        Calculate shoulder tilt angle relative to horizontal axis.
        Left shoulder: index 11, Right shoulder: index 12.
        Returns: (signed_angle_deg, absolute_tilt_deg)
        """
        if not pose_landmarks or len(pose_landmarks) == 0:
            return 0.0, 0.0

        landmarks = pose_landmarks[0]
        try:
            left_s = landmarks[11]
            right_s = landmarks[12]

            dx = (right_s.x - left_s.x) * img_w
            dy = (right_s.y - left_s.y) * img_h

            if dx == 0:
                return 90.0, 90.0

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            # Normalize so 0 degrees represents level shoulders
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            return float(angle_deg), float(abs(angle_deg))
        except Exception:
            return 0.0, 0.0

    def update_posture(self, pose_landmarks, img_w: int, img_h: int, now: Optional[float] = None) -> Dict:
        """
        Update driver posture and check shoulder slump condition.
        Returns dict:
        - 'tilt_angle': float (signed)
        - 'abs_tilt': float
        - 'is_slump_alert': bool (tilt > threshold for > SHOULDER_TILT_TIME_LIMIT)
        - 'tilt_duration': float
        - 'wrists': List of (x, y) coordinates from pose landmarks
        """
        if now is None:
            now = time.time()

        if not pose_landmarks or len(pose_landmarks) == 0:
            self.tilt_start_time = None
            self.tilt_duration = 0.0
            return {
                "tilt_angle": self.last_tilt_angle,
                "abs_tilt": abs(self.last_tilt_angle),
                "is_slump_alert": False,
                "tilt_duration": 0.0,
                "wrists": []
            }

        signed_angle, abs_tilt = self.calculate_shoulder_tilt(pose_landmarks, img_w, img_h)
        self.last_tilt_angle = signed_angle

        # Check posture slump duration
        is_slump_alert = False
        if abs_tilt >= config.SHOULDER_TILT_THRESHOLD_DEGREE:
            if self.tilt_start_time is None:
                self.tilt_start_time = now
            self.tilt_duration = now - self.tilt_start_time
            if self.tilt_duration >= config.SHOULDER_TILT_TIME_LIMIT:
                is_slump_alert = True
        else:
            self.tilt_start_time = None
            self.tilt_duration = 0.0

        # Extract wrists from pose (landmarks 15: left wrist, 16: right wrist)
        wrists = []
        try:
            lms = pose_landmarks[0]
            if len(lms) > 16:
                wrists.append((int(lms[15].x * img_w), int(lms[15].y * img_h)))
                wrists.append((int(lms[16].x * img_w), int(lms[16].y * img_h)))
        except Exception:
            pass

        return {
            "tilt_angle": signed_angle,
            "abs_tilt": abs_tilt,
            "is_slump_alert": is_slump_alert,
            "tilt_duration": self.tilt_duration,
            "wrists": wrists
        }
