"""
Configuration module for Driver Monitoring System (DOAN_CNPM_NHANDIENTAIXE).
Contains all constants, model paths, thresholds, time limits, and hyperparameters.
"""

import os
import sys
from pathlib import Path
import types
import pickle
import numpy as np


# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models
MODEL_DIR = PROJECT_ROOT / "model"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_LANDMARKER_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_LANDMARKER_MODEL = MODEL_DIR / "pose_landmarker.task"
YOLO_MODEL_PATH = MODEL_DIR / "yolo26n.pt"
STEERING_ROI_PATH = MODEL_DIR / "steering_roi.json"

# Fallback for models in project root if not in model/
if not FACE_LANDMARKER_MODEL.exists() and (PROJECT_ROOT / "face_landmarker.task").exists():
    FACE_LANDMARKER_MODEL = PROJECT_ROOT / "face_landmarker.task"
if not HAND_LANDMARKER_MODEL.exists() and (PROJECT_ROOT / "hand_landmarker.task").exists():
    HAND_LANDMARKER_MODEL = PROJECT_ROOT / "hand_landmarker.task"

# Reports & Sessions
REPORT_DIR = PROJECT_ROOT / "report"
EVENTS_PATH = REPORT_DIR / "events.csv"
SESSIONS_PATH = REPORT_DIR / "sessions.csv"
SNAPSHOTS_DIR = REPORT_DIR / "snapshots"

# Session & Driver state tracking
DRIVER_ABSENT_RESET_SEC = 10.0  # Reset session and recalibrate if face is absent for > 10s
DRIVER_LABEL = ""  # Optional static driver label (default empty)

# Assets
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
DEFAULT_FONT_PATH = FONTS_DIR / "arial.ttf"

# --- DROWSINESS (EAR & PERCLOS) ---
# Phase 1: Default EAR threshold is 0.22 (reduced false alarms from 0.30)
EAR_THRESHOLD_DEFAULT = 0.22
EAR_CALIBRATION_RATIO = 0.72  # Personalized threshold = 72% of baseline open eye EAR
EAR_CALIBRATION_SECONDS = 4.0  # Duration in seconds to calibrate EAR
EAR_SMOOTHING_FRAMES = 5  # Moving average window
DROWSY_TIME_LIMIT = 1.5  # Seconds of continuous eye closure before 🔴 Very High alert
PERCLOS_WINDOW_SECONDS = 60  # Sliding window size for PERCLOS calculation
PERCLOS_THRESHOLD = 0.35  # Trigger alert if eyes are closed >= 35% of the window

# Eye landmarks indices (MediaPipe 468/478 landmarks)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# --- YAWNING (MAR) ---
MAR_THRESHOLD = 0.60
MAR_MIN_DURATION = 1.0  # Continuous seconds exceeding threshold to count as 1 valid yawn
MAR_TIME_LIMIT = 1.0  # Backward-compatible alias
YAWN_WINDOW_SECONDS = 60  # Time window to count yawn frequency (1 minute)
YAWN_PER_MINUTE_LIMIT = 3  # Alert if > 3 yawns per minute (🟡 Medium alert)
MOUTH_LANDMARKS = [61, 291, 13, 14, 78, 308, 81, 178, 311, 402]

# --- HEAD POSE / DISTRACTION ---
YAW_THRESHOLD_DEGREE = 30.0  # Alert if |Yaw| > 30 degrees (🟠 High alert)
PITCH_UP_THRESHOLD_DEGREE = 25.0  # Head looking up
PITCH_DOWN_THRESHOLD_DEGREE = 20.0  # Head slumping / looking down
DISTRACTED_TIME_LIMIT = 2.5  # Seconds looking away before alert

# --- HANDS & STEERING WHEEL ROI ---
HANDS_OFF_TIME_LIMIT = 5.0  # Seconds without detecting hands (e.g. occlusion) before alert
HANDS_OUT_OF_ROI_LIMIT = 3.0  # Seconds both wrists outside steering wheel ROI before 🔴 Very High alert
WRIST_DETECTION_SOURCE = "hand"  # "hand" (HandLandmarker) or "pose" (PoseLandmarker)
DEFAULT_STEERING_ROI = {
    "x_min": 0.20,
    "y_min": 0.50,
    "x_max": 0.80,
    "y_max": 1.00
}

# --- BODY POSE & SHOULDER TILT ---
ENABLE_POSE_DETECTION = True
POSE_CHECK_INTERVAL = 3  # Check pose every N frames to save CPU
SHOULDER_TILT_THRESHOLD_DEGREE = 15.0
SHOULDER_TILT_TIME_LIMIT = 3.0
SLUMP_TIME_LIMIT = SHOULDER_TILT_TIME_LIMIT
ENABLE_AFFINE_NORMALIZATION = False

# --- PREPROCESSING & LIGHTING ---
LOW_LIGHT_THRESHOLD = 60.0  # Average frame brightness below which night mode activates
GAMMA_CORRECTION_VALUE = 1.5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- YOLO DISTRACTION OBJECT DETECTION ---
ENABLE_OBJECT_DETECTION = False  # Controlled flag; enabled when model is present
YOLO_CONF_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD = 0.45
YOLO_INFERENCE_INTERVAL = 5  # Inference every N frames
YOLO_CLASSES = {
    "phone": 0,
    "smoke": 1,
    "drink": 2
}
DISTRACTION_PHONE_LIMIT = 1.5
DISTRACTION_SMOKE_LIMIT = 1.5
DISTRACTION_DRINK_LIMIT = 3.0

# --- STATE MACHINE & RISK SCORING ---
COOLDOWN_SECONDS = 5.0
RISK_THRESHOLD_WARNING = 30
RISK_THRESHOLD_ALARM = 60

# --- ALERT MATRIX SPECIFICATION ---
ALERT_MATRIX = {
    "DROWSY": {
        "name": "Ngủ gật",
        "condition": "EAR < ngưỡng trong > 1.5s hoặc PERCLOS >= 35%",
        "level": "VERY_HIGH",  # 🔴 Rất cao
        "risk_weight": 40,
        "cooldown": 4.0,
        "action": "SIREN_LOUD",
        "voice_msg": "Cảnh báo, tài xế ngủ gật!"
    },
    "YAWN": {
        "name": "Ngáp liên tục",
        "condition": "MAR > 0.60, > 3 lần/phút",
        "level": "MEDIUM",     # 🟡 Trung bình
        "risk_weight": 15,
        "cooldown": 8.0,
        "action": "CHIME",
        "voice_msg": "Bạn đang có dấu hiệu mệt mỏi, hãy nghỉ ngơi."
    },
    "DISTRACTED": {
        "name": "Quay đầu / lệch hướng",
        "condition": "|Yaw| > 30° hoặc Pitch lệch trong > 2.5s",
        "level": "HIGH",       # 🟠 Cao
        "risk_weight": 30,
        "cooldown": 4.0,
        "action": "VOICE",
        "voice_msg": "Chú ý quan sát đường!"
    },
    "PHONE": {
        "name": "Sử dụng điện thoại",
        "condition": "YOLO phát hiện > 1.5s",
        "level": "HIGH",       # 🟠 Cao
        "risk_weight": 30,
        "cooldown": 5.0,
        "action": "VOICE",
        "voice_msg": "Không sử dụng điện thoại khi lái xe!"
    },
    "SMOKE": {
        "name": "Hút thuốc",
        "condition": "YOLO phát hiện > 1.5s",
        "level": "HIGH",       # 🟠 Cao
        "risk_weight": 25,
        "cooldown": 5.0,
        "action": "VOICE",
        "voice_msg": "Không hút thuốc khi lái xe!"
    },
    "DRINK": {
        "name": "Uống nước",
        "condition": "YOLO phát hiện > 3.0s",
        "level": "MEDIUM",     # 🟡 Trung bình
        "risk_weight": 15,
        "cooldown": 8.0,
        "action": "VOICE",
        "voice_msg": "Hãy tập trung lái xe."
    },
    "SLUMP": {
        "name": "Lệch vai / tư thế xấu",
        "condition": "Góc vai > 15° trong > 3.0s",
        "level": "MEDIUM",     # 🟡 Trung bình
        "risk_weight": 15,
        "cooldown": 8.0,
        "action": "VOICE",
        "voice_msg": "Xin hãy chỉnh lại tư thế ngồi lái."
    },
    "HANDS_OFF": {
        "name": "Rời tay khỏi vô-lăng",
        "condition": "Cả hai cổ tay ngoài ROI > 3.0s",
        "level": "VERY_HIGH",  # 🔴 Rất cao
        "risk_weight": 35,
        "cooldown": 3.0,
        "action": "SIREN_RAPID",
        "voice_msg": "Nguy hiểm, hãy đặt tay lên vô lăng ngay!"
    },
    "HANDS_NONE": {
        "name": "Không thấy tay lái",
        "condition": "Không phát hiện thấy tay > 5.0s",
        "level": "MEDIUM",     # 🟡 Trung bình
        "risk_weight": 20,
        "cooldown": 6.0,
        "action": "VOICE",
        "voice_msg": "Không thấy tay lái, xin chú ý vô lăng."
    }
}

RISK_WEIGHTS = {k: v["risk_weight"] for k, v in ALERT_MATRIX.items()}

# --- AUDIO & ALERTS ---
ENABLE_AUDIO = True
BEEP_SOUND_FREQ = 2500
BEEP_SOUND_DURATION_MS = 300
VOICE_RATE = 150
VOICE_VOLUME = 1.0

# --- GENERAL UI & VISUALIZATION ---
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SHOW_FPS = True
SHOW_LANDMARKS = False
SHOW_CHARTS = True
