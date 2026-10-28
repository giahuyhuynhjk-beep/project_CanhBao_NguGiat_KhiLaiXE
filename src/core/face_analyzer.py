"""
Face analyzer module using MediaPipe FaceLandmarker.
Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose (Yaw/Pitch/Roll).
Implements per-driver EAR calibration, PERCLOS, and yawn frequency tracking.
"""

import os
import math
import time
from collections import deque
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("FaceAnalyzer")


class HeadPoseEstimator:
    """Estimates head pose (Yaw, Pitch, Roll) using 3D canonical facial model and cv2.solvePnP."""
    def __init__(self):
        # 3D canonical facial model points in mm
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, 110.0, -65.0),         # Chin (index 152)
            (-75.0, -50.0, -60.0),       # Left eye outer corner (index 33)
            (75.0, -50.0, -60.0),        # Right eye outer corner (index 263)
            (-50.0, 70.0, -40.0),        # Left mouth corner (index 61)
            (50.0, 70.0, -40.0)          # Right mouth corner (index 291)
        ], dtype=np.float64)

    def estimate_from_landmarks(self, landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
        """
        Estimate (pitch, yaw, roll) in degrees from landmarks.
        Returns: (pitch, yaw, roll)
        - Pitch: (+) looking up, (-) looking down / slumping
        - Yaw: (+) looking right, (-) looking left
        - Roll: (+) tilting head right, (-) tilting head left
        """
        try:
            image_points = np.array([
                (landmarks[1].x * img_w, landmarks[1].y * img_h),
                (landmarks[152].x * img_w, landmarks[152].y * img_h),
                (landmarks[33].x * img_w, landmarks[33].y * img_h),
                (landmarks[263].x * img_w, landmarks[263].y * img_h),
                (landmarks[61].x * img_w, landmarks[61].y * img_h),
                (landmarks[291].x * img_w, landmarks[291].y * img_h)
            ], dtype=np.float64)

            focal_length = img_w
            center = (img_w / 2.0, img_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return 0.0, 0.0, 0.0

            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

            # Normalize roll if decomposed near +/- 180
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180

            return pitch, yaw, roll
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def estimate_from_transformation_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
        """Extract Euler angles (pitch, yaw, roll) from MediaPipe 4x4 transformation matrix."""
        try:
            R = matrix[:3, :3]
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                pitch = math.atan2(R[2, 1], R[2, 2])
                yaw = math.atan2(-R[2, 0], sy)
                roll = math.atan2(R[1, 0], R[0, 0])
            else:
                pitch = math.atan2(-R[1, 2], R[1, 1])
                yaw = math.atan2(-R[2, 0], sy)
                roll = 0.0
            return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
        except Exception:
            return 0.0, 0.0, 0.0


class DrowsinessDetector:
    """Handles EAR calculation, moving average smoothing, per-driver calibration, and PERCLOS."""
    def __init__(self):
        self.default_threshold = config.EAR_THRESHOLD_DEFAULT
        self.current_threshold = self.default_threshold
        self.calibration_ratio = config.EAR_CALIBRATION_RATIO
        self.calibration_duration = config.EAR_CALIBRATION_SECONDS
        self.smoothing_frames = config.EAR_SMOOTHING_FRAMES

        # Rolling buffers
        self.ear_history = deque(maxlen=self.smoothing_frames)
        self.perclos_history = deque()  # Tuples of (timestamp, is_closed)

        # Calibration state
        self.is_calibrated: bool = False
        self.is_calibrating: bool = False
        self.calibration_start_time: Optional[float] = None
        self.calibration_samples: List[float] = []

        # Timers & Occlusion
        self.eye_closed_start_time: Optional[float] = None
        self.is_sunglasses_mode = False

    def reset_calibration(self):
        """Reset calibration state to default."""
        self.is_calibrated = False
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_samples.clear()
        self.current_threshold = self.default_threshold
        self.ear_history.clear()
        self.perclos_history.clear()
        self.eye_closed_start_time = None
        self.is_sunglasses_mode = False

    # Backward-compatible alias
    reset_driver = reset_calibration

    def handle_face_lost(self):
        """Safely reset timers when face is temporarily not detected in the frame."""
        self.eye_closed_start_time = None

    def start_calibration(self, driver_name: Optional[str] = None, now: Optional[float] = None):
        """Begin EAR calibration session when face is detected."""
        if now is None:
            now = time.time()

        self.is_calibrating = True
        self.calibration_start_time = now
        self.calibration_samples.clear()
        logger.info("Bắt đầu phiên hiệu chuẩn EAR cá nhân hóa...")

    def update(self, raw_ear: float, yaw: float, pitch: float, now: Optional[float] = None) -> Dict:
        """
        Update with new EAR measurement.
        Returns dict containing:
        - 'smoothed_ear': float
        - 'threshold': float
        - 'is_calibrating': bool
        - 'calibration_progress': float (0.0 to 1.0)
        - 'eye_closed': bool
        - 'drowsy_closed_alert': bool (continuous closure > DROWSY_TIME_LIMIT)
        - 'perclos': float
        - 'drowsy_perclos_alert': bool (perclos >= PERCLOS_THRESHOLD)
        """
        if now is None:
            now = time.time()

        # Smooth EAR
        self.ear_history.append(raw_ear)
        smoothed_ear = float(np.mean(self.ear_history))

        # Handle calibration
        calibration_progress = 0.0
        if self.is_calibrating and self.calibration_start_time is not None:
            elapsed = now - self.calibration_start_time
            calibration_progress = min(1.0, elapsed / self.calibration_duration)

            # Sample only if driver is looking straight
            if abs(yaw) <= 15.0 and abs(pitch) <= 15.0:
                self.calibration_samples.append(smoothed_ear)

            if elapsed >= self.calibration_duration:
                self.is_calibrating = False
                if len(self.calibration_samples) >= 15:
                    baseline_open_ear = float(np.mean(self.calibration_samples))
                    # Clamp between 0.18 and 0.26 to avoid false alarms when squinting or opening eyes normally
                    self.current_threshold = max(0.18, min(0.26, baseline_open_ear * self.calibration_ratio))
                    self.is_calibrated = True
                    logger.info(f"Hiệu chuẩn EAR thành công: Baseline={baseline_open_ear:.3f}, Ngưỡng={self.current_threshold:.3f}")
                else:
                    self.current_threshold = self.default_threshold
                    logger.warning(f"Hiệu chuẩn EAR thất bại (ít mẫu hợp lệ). Sử dụng ngưỡng mặc định {self.default_threshold:.2f}")

        # Check sunglasses / eye occlusion
        is_sunglasses_mode = self.is_sunglasses_mode
        is_closed = False
        drowsy_closed_alert = False
        drowsy_perclos_alert = False
        perclos = 0.0

        if is_sunglasses_mode:
            # Under sunglasses mode, disable EAR-based drowsiness alert to prevent false alarms
            pass
        else:
            # Check eye closure
            is_closed = smoothed_ear < self.current_threshold

            if is_closed:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = now
                duration = now - self.eye_closed_start_time
                if duration >= config.DROWSY_TIME_LIMIT:
                    drowsy_closed_alert = True
            else:
                self.eye_closed_start_time = None

            # Update PERCLOS history
            self.perclos_history.append((now, is_closed))
            # Remove items older than PERCLOS_WINDOW_SECONDS
            cutoff = now - config.PERCLOS_WINDOW_SECONDS
            while self.perclos_history and self.perclos_history[0][0] < cutoff:
                self.perclos_history.popleft()

            perclos = 0.0
            drowsy_perclos_alert = False
            if len(self.perclos_history) >= 30:  # Need sufficient samples
                closed_count = sum(1 for _, closed in self.perclos_history if closed)
                perclos = closed_count / len(self.perclos_history)
                if perclos >= config.PERCLOS_THRESHOLD:
                    drowsy_perclos_alert = True

        return {
            "smoothed_ear": smoothed_ear,
            "threshold": self.current_threshold,
            "is_calibrating": self.is_calibrating,
            "calibration_progress": calibration_progress,
            "eye_closed": is_closed if not is_sunglasses_mode else False,
            "drowsy_closed_alert": drowsy_closed_alert,
            "perclos": perclos,
            "drowsy_perclos_alert": drowsy_perclos_alert,
            "is_sunglasses_mode": is_sunglasses_mode
        }

    def check_sunglasses_occlusion(self, frame: np.ndarray, landmarks, img_w: int, img_h: int, smoothed_ear: float) -> bool:
        """
        Detect if eyes are occluded by dark sunglasses or masks.
        Compares brightness in the eye socket region vs forehead region.
        """
        try:
            # Left eye center ~ landmark 159, Right eye center ~ landmark 386
            lx, ly = int(landmarks[159].x * img_w), int(landmarks[159].y * img_h)
            rx, ry = int(landmarks[386].x * img_w), int(landmarks[386].y * img_h)
            # Forehead center ~ landmark 10
            fx, fy = int(landmarks[10].x * img_w), int(landmarks[10].y * img_h)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Sample eye patches (15x15)
            rad = 8
            l_patch = gray[max(0, ly - rad):min(img_h, ly + rad), max(0, lx - rad):min(img_w, lx + rad)]
            r_patch = gray[max(0, ry - rad):min(img_h, ry + rad), max(0, rx - rad):min(img_w, rx + rad)]
            f_patch = gray[max(0, fy - rad):min(img_h, fy + rad), max(0, fx - rad):min(img_w, fx + rad)]

            if l_patch.size > 0 and r_patch.size > 0 and f_patch.size > 0:
                eye_bright = (float(np.mean(l_patch)) + float(np.mean(r_patch))) / 2.0
                face_bright = float(np.mean(f_patch))
                # If eyes are heavily shaded or abnormally dark relative to face
                if (eye_bright < 40 and face_bright > 80) or (eye_bright < 25):
                    self.is_sunglasses_mode = True
                    return True

            # If EAR is near zero constantly (e.g. < 0.08) for consecutive frames
            if smoothed_ear < 0.08 and self.eye_closed_start_time and (time.time() - self.eye_closed_start_time > 4.0):
                self.is_sunglasses_mode = True
                return True

            self.is_sunglasses_mode = False
            return False
        except Exception:
            return False


class YawnDetector:
    """Tracks MAR, individual yawns (duration >= 1.0s), and yawn frequency per minute."""
    def __init__(self):
        self.mar_threshold = config.MAR_THRESHOLD
        self.min_duration = config.MAR_MIN_DURATION
        self.window_seconds = config.YAWN_WINDOW_SECONDS
        self.limit_per_minute = config.YAWN_PER_MINUTE_LIMIT

        self.yawn_start_time: Optional[float] = None
        self.recent_yawns = deque()  # Timestamps of completed yawns

    def update(self, mar: float, now: Optional[float] = None) -> Tuple[bool, int, bool]:
        """
        Update mouth aspect ratio.
        Returns: (is_currently_yawning, yawns_in_last_minute, is_yawn_alert)
        """
        if now is None:
            now = time.time()

        is_currently_yawning = False

        if mar >= self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            duration = now - self.yawn_start_time
            if duration >= self.min_duration:
                is_currently_yawning = True
        else:
            if self.yawn_start_time is not None:
                duration = now - self.yawn_start_time
                if duration >= self.min_duration:
                    self.recent_yawns.append(now)
                self.yawn_start_time = None

        # Clean yawns older than window_seconds
        cutoff = now - self.window_seconds
        while self.recent_yawns and self.recent_yawns[0] < cutoff:
            self.recent_yawns.popleft()

        yawn_count = len(self.recent_yawns)
        is_yawn_alert = yawn_count >= self.limit_per_minute

        return is_currently_yawning, yawn_count, is_yawn_alert


class FaceAnalyzer:
    """High-level face analysis facade coordinating MediaPipe, EAR, MAR, and Head Pose."""
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(config.FACE_LANDMARKER_MODEL)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy file model FaceLandmarker: {model_path}")

        base_options_face = python.BaseOptions(model_asset_path=model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options_face)

        self.left_eye_indices = config.LEFT_EYE_LANDMARKS
        self.right_eye_indices = config.RIGHT_EYE_LANDMARKS
        self.mouth_indices = config.MOUTH_LANDMARKS

        # Sub-detectors
        self.head_pose_estimator = HeadPoseEstimator()
        self.drowsiness_detector = DrowsinessDetector()
        self.yawn_detector = YawnDetector()

    def detect(self, mp_image: mp.Image):
        """Run landmark detection on mp.Image."""
        return self.detector.detect(mp_image)

    @staticmethod
    def calculate_ear(landmarks, eye_indices, img_w: int, img_h: int) -> float:
        """Calculate Eye Aspect Ratio (EAR) given landmark coordinates."""
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append((lm.x * img_w, lm.y * img_h))
        pts = np.array(pts)
        v1 = dist.euclidean(pts[1], pts[5])
        v2 = dist.euclidean(pts[2], pts[4])
        h1 = dist.euclidean(pts[0], pts[3])
        if h1 == 0:
            return 0.0
        return float((v1 + v2) / (2.0 * h1))

    @staticmethod
    def calculate_mar(landmarks, img_w: int, img_h: int) -> float:
        """Calculate Mouth Aspect Ratio (MAR)."""
        try:
            # Lip inner landmarks
            p_top = np.array([landmarks[13].x * img_w, landmarks[13].y * img_h])
            p_bottom = np.array([landmarks[14].x * img_w, landmarks[14].y * img_h])
            p_top_l = np.array([landmarks[81].x * img_w, landmarks[81].y * img_h])
            p_bot_l = np.array([landmarks[178].x * img_w, landmarks[178].y * img_h])
            p_top_r = np.array([landmarks[311].x * img_w, landmarks[311].y * img_h])
            p_bot_r = np.array([landmarks[402].x * img_w, landmarks[402].y * img_h])
            p_left = np.array([landmarks[61].x * img_w, landmarks[61].y * img_h])
            p_right = np.array([landmarks[291].x * img_w, landmarks[291].y * img_h])

            v1 = dist.euclidean(p_top, p_bottom)
            v2 = dist.euclidean(p_top_l, p_bot_l)
            v3 = dist.euclidean(p_top_r, p_bot_r)
            horizontal = dist.euclidean(p_left, p_right)

            if horizontal == 0:
                return 0.0
            return float((v1 + v2 + v3) / (3.0 * horizontal))
        except Exception:
            return 0.0

    def estimate_head_pose(self, landmarks, img_w: int, img_h: int, transformation_matrix=None) -> Tuple[float, float, float]:
        """Estimate head pose (Pitch, Yaw, Roll) in degrees."""
        if transformation_matrix is not None:
            return self.head_pose_estimator.estimate_from_transformation_matrix(transformation_matrix)
        return self.head_pose_estimator.estimate_from_landmarks(landmarks, img_w, img_h)

    @staticmethod
    def check_head_pose_legacy(landmarks) -> bool:
        """Legacy head pose check for backward compatibility."""
        nose = landmarks[1]
        return not (nose.y > 0.7 or nose.x < 0.35 or nose.x > 0.65)

    def is_distracted(self, pitch: float, yaw: float) -> bool:
        """Check if head pose exceeds distraction thresholds."""
        if abs(yaw) > config.YAW_THRESHOLD_DEGREE:
            return True
        if pitch > config.PITCH_UP_THRESHOLD_DEGREE or pitch < -config.PITCH_DOWN_THRESHOLD_DEGREE:
            return True
        return False

    @staticmethod
    def get_face_bbox(landmarks, img_w: int, img_h: int, padding: int = 20):
        """Extract bounding box (l, t, r, b) from face landmarks."""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        l = max(0, int(min(x_coords) * img_w) - padding)
        t = max(0, int(min(y_coords) * img_h) - padding)
        r = min(img_w, int(max(x_coords) * img_w) + padding)
        b = min(img_h, int(max(y_coords) * img_h) + padding)
        return (l, t, r, b)
