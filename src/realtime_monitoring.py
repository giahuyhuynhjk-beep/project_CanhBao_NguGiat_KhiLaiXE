"""
Real-time Driver Monitoring System (DOAN_CNPM_NHANDIENTAIXE).
Anonymous Session-based Driver Safety Monitoring:
- MediaPipe FaceLandmarker: Personalized EAR session calibration, PERCLOS, MAR, Head Pose (solvePnP)
- MediaPipe Hand & Pose Landmarker: Steering Wheel ROI monitoring with fallback wrists
- YOLO Object Detection: Distraction detection (phone, smoke, drink) with spatial constraints
- State Machine & Composite Risk Scoring
- Session Logging to events.csv and sessions.csv without personal driver identity
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.config as config
from src.core.preprocessor import ImagePreprocessor
from src.core.face_analyzer import FaceAnalyzer
from src.core.hand_analyzer import HandAnalyzer
from src.core.pose_analyzer import PoseAnalyzer
from src.core.object_detector import ObjectDetector
from src.core.state_machine import DriverStateMachine
from src.utils.alert_system import AlertSystem
from src.utils.event_logger import EventLogger
from src.utils.visualizer import Visualizer
from src.utils.logger import get_logger

logger = get_logger("RealtimeMonitor")


class RealtimeMonitor:
    def __init__(
        self,
        session_id: Optional[str] = None,
        headless: bool = False,
        enable_visualizer: bool = True,
        enable_audio: bool = config.ENABLE_AUDIO
    ):
        self.headless = headless
        self.enable_visualizer = enable_visualizer and not headless
        self.enable_audio = enable_audio

        # Session ID generation
        now_dt = datetime.now()
        base_sid = session_id or now_dt.strftime("SES_%Y%m%d_%H%M%S")
        if getattr(config, "DRIVER_LABEL", ""):
            self.session_id = f"{config.DRIVER_LABEL}_{base_sid}"
        else:
            self.session_id = base_sid

        logger.info(f"Khởi động phiên giám sát an toàn tài xế: {self.session_id}")

        # Submodules
        self.preprocessor = ImagePreprocessor()
        self.face_analyzer = FaceAnalyzer()
        self.hand_analyzer = HandAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        self.object_detector = ObjectDetector(enabled=config.ENABLE_OBJECT_DETECTION)
        self.state_machine = DriverStateMachine()
        self.alert_system = AlertSystem(enable_audio=self.enable_audio)
        self.event_logger = EventLogger(session_id=self.session_id)
        self.visualizer = Visualizer(enabled=self.enable_visualizer)

        # Shortcuts to face sub-detectors
        self.drowsiness_detector = self.face_analyzer.drowsiness_detector
        self.yawn_detector = self.face_analyzer.yawn_detector

        # Absence tracking & Recalibration
        self.face_absent_start_time: Optional[float] = None
        self.distracted_start_time: Optional[float] = None

        # Threading for YOLO Distraction Detection
        self.is_running = True
        self.yolo_frame_to_process = None
        self.yolo_face_bbox = None
        self.yolo_wrists = []
        self.latest_distraction_status = {
            "detections": [],
            "is_phone_alert": False,
            "is_smoke_alert": False,
            "is_drink_alert": False,
            "active_classes": []
        }
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()

        # FPS calculation
        self.fps = 0.0
        self.prev_time = time.time()
        self.frame_counter = 0

    def _yolo_worker(self):
        """Asynchronous YOLO object detection worker thread."""
        while self.is_running:
            if self.yolo_frame_to_process is not None and self.object_detector.is_ready:
                frame = self.yolo_frame_to_process
                face_bbox = self.yolo_face_bbox
                wrists = self.yolo_wrists
                self.yolo_frame_to_process = None

                res = self.object_detector.process(frame, face_bbox=face_bbox, wrist_coords=wrists)
                self.latest_distraction_status = res
            else:
                time.sleep(0.02)

    def recalibrate(self, now: Optional[float] = None):
        """Manually trigger EAR recalibration (e.g. when user presses 'C')."""
        if now is None:
            now = time.time()
        self.drowsiness_detector.reset_calibration()
        self.drowsiness_detector.start_calibration(now=now)
        logger.info("Chủ động kích hoạt hiệu chuẩn lại EAR theo yêu cầu.")

    def calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """Proxy to FaceAnalyzer.calculate_ear for backward compatibility."""
        return self.face_analyzer.calculate_ear(landmarks, eye_indices, img_w, img_h)

    def check_head_pose(self, landmarks):
        """Proxy to FaceAnalyzer.check_head_pose_legacy for backward compatibility."""
        return self.face_analyzer.check_head_pose_legacy(landmarks)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame:
        1. Calculate FPS
        2. Preprocess (CLAHE + Gamma low-light enhancement)
        3. Landmark detection (Face, Hands, Pose via MediaPipe)
        4. Session-based EAR calibration & PERCLOS
        5. MAR & Yawn frequency calculation
        6. Head Pose (Euler Yaw/Pitch/Roll) distraction check
        7. Steering Wheel ROI wrist checking (with Pose fallback)
        8. YOLO distraction detection (phone, smoke, drink)
        9. State machine & alert matrix dispatch
        10. Render HUD, ROI, and alert banners
        """
        now = time.time()
        time_diff = now - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.prev_time = now

        # --- 0. PREPROCESSING (LOW LIGHT CLAHE + GAMMA) ---
        frame, is_low_light = self.preprocessor.process(frame)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.frame_counter += 1

        # Scale down for MediaPipe detection
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)

        results_face = self.face_analyzer.detect(mp_image)
        results_hand = self.hand_analyzer.detect(mp_image)

        # Run Pose detection periodically (every POSE_CHECK_INTERVAL frames)
        if not hasattr(self, "latest_pose_info"):
            self.latest_pose_info = {"tilt_angle": 0.0, "abs_tilt": 0.0, "is_slump_alert": False, "tilt_duration": 0.0, "wrists": []}

        if (self.frame_counter % config.POSE_CHECK_INTERVAL == 0) and self.pose_analyzer.is_available:
            results_pose = self.pose_analyzer.detect(mp_image)
            pose_lms = results_pose.pose_landmarks if results_pose else []
            self.latest_pose_info = self.pose_analyzer.update_posture(pose_lms, w, h, now=now)
            if config.ENABLE_AFFINE_NORMALIZATION and self.latest_pose_info["abs_tilt"] > 15.0:
                frame = self.preprocessor.align_face_affine(frame, self.latest_pose_info["tilt_angle"])

        # Telemetry metrics
        ear = 0.0
        ear_thresh = self.drowsiness_detector.current_threshold
        perclos = 0.0
        mar = 0.0
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        yawns_count = len(self.yawn_detector.recent_yawns)
        face_rect = None
        is_calibrating = self.drowsiness_detector.is_calibrating
        calib_progress = 0.0

        active_alerts: List[str] = []

        # --- 1. FACE ANALYSIS & ABSENCE RECALIBRATION ---
        is_drowsy_cond = False
        is_yawn_cond = False
        is_distract_cond = False

        if results_face.face_landmarks:
            landmarks = results_face.face_landmarks[0]
            face_rect = self.face_analyzer.get_face_bbox(landmarks, w, h, padding=20)

            # Check if face reappeared after being absent for > DRIVER_ABSENT_RESET_SEC
            if self.face_absent_start_time is not None:
                absent_sec = now - self.face_absent_start_time
                self.face_absent_start_time = None
                if absent_sec >= config.DRIVER_ABSENT_RESET_SEC:
                    logger.info(f"Phát hiện tài xế xuất hiện sau {absent_sec:.1f}s vắng mặt. Tự động reset và hiệu chuẩn lại...")
                    self.state_machine.reset_all()
                    self.yawn_detector.recent_yawns.clear()
                    self.recalibrate(now=now)

            # Auto-trigger calibration on startup if not yet calibrated
            if not self.drowsiness_detector.is_calibrated and not self.drowsiness_detector.is_calibrating:
                self.drowsiness_detector.start_calibration(now=now)

            # Head Pose (solvePnP)
            matrix = results_face.facial_transformation_matrixes[0] if results_face.facial_transformation_matrixes else None
            pitch, yaw, roll = self.face_analyzer.estimate_head_pose(landmarks, w, h, transformation_matrix=matrix)

            # EAR calculation
            left_ear = self.calculate_ear(landmarks, self.face_analyzer.left_eye_indices, w, h)
            right_ear = self.calculate_ear(landmarks, self.face_analyzer.right_eye_indices, w, h)
            raw_ear = (left_ear + right_ear) / 2.0

            # Sunglasses / Occlusion check
            self.drowsiness_detector.check_sunglasses_occlusion(frame, landmarks, w, h, raw_ear)

            # Drowsiness & Calibration Update
            drowsy_info = self.drowsiness_detector.update(raw_ear, yaw, pitch, now=now)
            ear = drowsy_info["smoothed_ear"]
            ear_thresh = drowsy_info["threshold"]
            is_calibrating = drowsy_info["is_calibrating"]
            calib_progress = drowsy_info["calibration_progress"]
            perclos = drowsy_info["perclos"]
            is_drowsy_cond = drowsy_info["drowsy_closed_alert"] or drowsy_info["drowsy_perclos_alert"]

            # MAR & Yawn Detection
            mar = self.face_analyzer.calculate_mar(landmarks, w, h)
            _, yawns_count, is_yawn_alert = self.yawn_detector.update(mar, now=now)
            is_yawn_cond = is_yawn_alert

            # Head Pose Distraction Check
            is_distract_cond = self.face_analyzer.is_distracted(pitch, yaw)

        else:
            # Face not detected in current frame
            if self.face_absent_start_time is None:
                self.face_absent_start_time = now
            self.drowsiness_detector.handle_face_lost()

        # --- 2. STEERING WHEEL & HAND ANALYSIS ---
        fallback_wrists = self.latest_pose_info.get("wrists", [])
        steering_info = self.hand_analyzer.check_steering_status(
            results_hand.hand_landmarks,
            w, h, now=now,
            fallback_wrists=fallback_wrists
        )

        # Dispatch frame to YOLO distraction detector every YOLO_INFERENCE_INTERVAL frames
        if (self.frame_counter % config.YOLO_INFERENCE_INTERVAL == 0) and self.object_detector.is_ready:
            if self.yolo_frame_to_process is None:
                self.yolo_frame_to_process = frame.copy()
                self.yolo_face_bbox = face_rect
                self.yolo_wrists = steering_info["wrists"]

        # --- 3. STATE MACHINE & ALERT MATRIX ---
        conditions = {
            "DROWSY": is_drowsy_cond,
            "YAWN": is_yawn_cond,
            "DISTRACTED": is_distract_cond,
            "HANDS_OFF": steering_info["is_out_of_roi_alert"],
            "HANDS_NONE": steering_info["is_hands_off_alert"],
            "PHONE": self.latest_distraction_status.get("is_phone_alert", False),
            "SMOKE": self.latest_distraction_status.get("is_smoke_alert", False),
            "DRINK": self.latest_distraction_status.get("is_drink_alert", False),
            "SLUMP": self.latest_pose_info.get("is_slump_alert", False)
        }

        state_level, risk_score, active_violations, triggered_alerts = self.state_machine.process_all_conditions(conditions, now=now)
        self.event_logger.update_risk_score(risk_score)

        # Dispatch triggered alerts to audio and event logger
        for alert in triggered_alerts:
            self.alert_system.trigger(action=alert["action"], voice_msg=alert["voice_msg"])
            self.event_logger.log_event(
                session_id=self.session_id,
                alert_type=alert["name"],
                level=alert["level"],
                duration=alert["duration"],
                frame=frame,
                risk_score=risk_score
            )

        # Generate alert text banners
        active_alerts = []
        for v_key in active_violations:
            meta = config.ALERT_MATRIX.get(v_key, {})
            name = meta.get("name", v_key).upper()
            active_alerts.append(f"CANH BAO: {name}!")

        # --- 4. VISUALIZATION (SKIPPED IN HEADLESS / DISABLED MODE) ---
        if self.enable_visualizer:
            if face_rect:
                self.visualizer.draw_face_box(
                    frame,
                    face_rect,
                    name=getattr(config, "DRIVER_LABEL", "") or None,
                    is_calibrating=is_calibrating,
                    calibration_progress=calib_progress
                )

            # Draw steering wheel ROI and wrists
            self.visualizer.draw_roi(frame, self.hand_analyzer.roi, steering_status=steering_info["status"])
            self.visualizer.draw_wrists(frame, steering_info["wrists"])

            # Draw detected distraction objects (bounding boxes)
            if self.latest_distraction_status.get("detections"):
                self.visualizer.draw_objects(frame, self.latest_distraction_status["detections"])

            # Draw HUD dashboard
            self.visualizer.draw_hud(
                frame, fps=self.fps, ear=ear, ear_thresh=ear_thresh,
                mar=mar, yaw=yaw, pitch=pitch, perclos=perclos,
                yawns_count=yawns_count,
                state_level=state_level.name,
                risk_score=risk_score,
                shoulder_tilt=self.latest_pose_info.get("abs_tilt", 0.0),
                is_sunglasses=self.drowsiness_detector.is_sunglasses_mode,
                is_low_light=is_low_light
            )

            # Draw active alerts
            self.visualizer.draw_alerts(frame, active_alerts)

        return frame

    def stop(self):
        """Safely terminate all worker threads, end session and write summaries."""
        self.is_running = False
        if hasattr(self, "yolo_thread") and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=1.0)
        self.alert_system.stop()
        self.event_logger.stop()


def main():
    ap = argparse.ArgumentParser(description="Realtime Anonymous Driver Monitoring System (DMS)")
    ap.add_argument("-c", "--camera", default=0, help="Camera index (0, 1, ...) hoặc đường dẫn video file")
    ap.add_argument("-s", "--session", default=None, help="Tùy chọn mã định danh phiên làm việc (Session ID)")
    ap.add_argument("--headless", action="store_true", help="Chạy chế độ headless (không mở cửa sổ GUI hiển thị)")
    ap.add_argument("--no-visualizer", action="store_true", help="Tắt visualizer drawing để tối ưu FPS tối đa")
    ap.add_argument("--no-audio", action="store_true", help="Tắt âm thanh cảnh báo")
    args = vars(ap.parse_args())

    cam_source = args["camera"]
    try:
        cam_source = int(cam_source)
    except ValueError:
        pass

    is_headless = args["headless"]
    enable_vis = not args["no_visualizer"] and not is_headless
    enable_aud = not args["no_audio"]

    monitor = RealtimeMonitor(
        session_id=args["session"],
        headless=is_headless,
        enable_visualizer=enable_vis,
        enable_audio=enable_aud
    )

    vs = cv2.VideoCapture(cam_source)
    if not vs.isOpened():
        logger.error(f"Không thể mở nguồn video/camera: {cam_source}")
        return

    logger.info(f"Bắt đầu giám sát phiên {monitor.session_id} (Headless={is_headless}, Visualizer={enable_vis}, Audio={enable_aud}).")
    logger.info("Nhấn 'q' để thoát, nhấn 'c' để hiệu chuẩn lại EAR.")

    frame_idx = 0
    try:
        while True:
            ret, frame = vs.read()
            if not ret:
                break

            processed_frame = monitor.process_frame(frame)
            frame_idx += 1

            if not is_headless:
                cv2.imshow("Driver Monitoring System (DMS)", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("c"), ord("C")):
                    monitor.recalibrate()
            else:
                # In headless mode, log status periodically
                if frame_idx % 30 == 0:
                    logger.info(f"[HEADLESS] Frame={frame_idx} | FPS={monitor.fps:.1f} | State={monitor.state_machine.current_level.name} | Risk={monitor.state_machine.total_risk_score} | EAR_Thresh={monitor.drowsiness_detector.current_threshold:.2f}")

    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng (Ctrl+C)...")
    finally:
        logger.info("Đang dừng hệ thống giám sát...")
        monitor.stop()
        vs.release()
        if not is_headless:
            cv2.destroyAllWindows()
        logger.info("Đã kết thúc phiên giám sát an toàn.")


if __name__ == "__main__":
    main()
