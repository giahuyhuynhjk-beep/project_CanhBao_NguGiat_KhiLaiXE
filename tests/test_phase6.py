"""
Comprehensive unit tests for Phase 6 (GĐ6):
Quality, Robustness, Documentation, and Edge Case Handling.

Tests cover:
- EAR calculation (normal open, squinting/nheo nhẹ, closed)
- MAR calculation (closed lips, normal talking, wide open yawn)
- Yaw / Pitch head pose distraction
- Shoulder tilt angle and posture slump
- State machine transitions (NORMAL -> WARNING -> ALARM)
- Behavior timers, debounce, and cooldown
- PERCLOS sliding window calculation
- Steering Wheel ROI and partial hand occlusion tolerance
- Yawn frequency per minute tracking
- YOLO spatial proximity constraints (face/wrists proximity)
- Safe degradation when models (YOLO / Pose) or landmarks are missing
- Vietnamese Pillow text caching and headless visualizer toggle
- EventLogger events.csv and snapshot persistence
"""

import os
import sys
import time
import math
import tempfile
import cv2
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.core.face_analyzer import FaceAnalyzer, DrowsinessDetector, YawnDetector, HeadPoseEstimator
from src.core.hand_analyzer import HandAnalyzer
from src.core.pose_analyzer import PoseAnalyzer
from src.core.object_detector import ObjectDetector, DistractionTracker
from src.core.state_machine import DriverStateMachine, AlertLevel, BehaviorTracker
from src.utils.visualizer import Visualizer
from src.utils.event_logger import EventLogger


class DummyLandmark:
    """Mock landmark point with normalized x, y, z coordinates."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


# ---------------------------------------------------------------------------
# 1. EAR: NORMAL, SQUINTING (NHEO NHẸ), AND CLOSED
# ---------------------------------------------------------------------------
def test_ear_normal_squint_and_closed():
    """
    Ensure EAR calculation correctly distinguishes between:
    - Normal open eye (EAR > 0.30)
    - Squinting eye / nheo mắt nhẹ (EAR ~ 0.24-0.27, MUST NOT trigger drowsiness alert)
    - Closed eye (EAR < 0.10, triggers drowsiness alert)
    """
    indices = [0, 1, 2, 3, 4, 5]

    # Open eye: wide vertical aperture
    open_eye = [
        DummyLandmark(0.10, 0.20),
        DummyLandmark(0.15, 0.16),
        DummyLandmark(0.20, 0.16),
        DummyLandmark(0.25, 0.20),
        DummyLandmark(0.20, 0.24),
        DummyLandmark(0.15, 0.24),
    ]
    # Squinting eye (nheo nhẹ): slightly narrowed vertical aperture (dy = 0.05 * 480 = 24, 48/192 = 0.250)
    squint_eye = [
        DummyLandmark(0.10, 0.20),
        DummyLandmark(0.15, 0.175),
        DummyLandmark(0.20, 0.175),
        DummyLandmark(0.25, 0.20),
        DummyLandmark(0.20, 0.225),
        DummyLandmark(0.15, 0.225),
    ]
    # Fully closed eye: vertical aperture near zero
    closed_eye = [
        DummyLandmark(0.10, 0.20),
        DummyLandmark(0.15, 0.199),
        DummyLandmark(0.20, 0.199),
        DummyLandmark(0.25, 0.20),
        DummyLandmark(0.20, 0.201),
        DummyLandmark(0.15, 0.201),
    ]

    ear_open = FaceAnalyzer.calculate_ear(open_eye, indices, 640, 480)
    ear_squint = FaceAnalyzer.calculate_ear(squint_eye, indices, 640, 480)
    ear_closed = FaceAnalyzer.calculate_ear(closed_eye, indices, 640, 480)

    assert ear_open > 0.30, f"Open EAR should be > 0.30, got {ear_open:.3f}"
    assert 0.22 <= ear_squint <= 0.30, f"Squint EAR should be between 0.22 and 0.30, got {ear_squint:.3f}"
    assert ear_closed < 0.10, f"Closed EAR should be < 0.10, got {ear_closed:.3f}"

    # Verify DrowsinessDetector with squinting eye does NOT trigger false sleep alert
    detector = DrowsinessDetector()
    detector.current_threshold = 0.22
    info = detector.update(raw_ear=ear_squint, yaw=0.0, pitch=0.0, now=100.0)
    assert not info["eye_closed"], "Squinting eye should NOT be marked as closed!"
    assert not info["drowsy_closed_alert"], "Squinting eye should NOT trigger drowsiness alert!"


# ---------------------------------------------------------------------------
# 2. MAR: CLOSED, TALKING, AND WIDE OPEN YAWN
# ---------------------------------------------------------------------------
def test_mar_closed_talking_and_yawn():
    """
    Verify MAR distinguishes between closed mouth, normal talking (MAR ~ 0.35-0.45),
    and wide open yawn (MAR >= 0.60).
    """
    def make_mouth_landmarks(vertical_gap: float):
        lms = [DummyLandmark(0.5, 0.5) for _ in range(468)]
        lms[61] = DummyLandmark(0.40, 0.50)   # left corner
        lms[291] = DummyLandmark(0.60, 0.50)  # right corner
        lms[13] = DummyLandmark(0.50, 0.50 - vertical_gap / 2)  # top lip
        lms[14] = DummyLandmark(0.50, 0.50 + vertical_gap / 2)  # bottom lip
        lms[81] = DummyLandmark(0.45, 0.50 - vertical_gap / 2)
        lms[178] = DummyLandmark(0.45, 0.50 + vertical_gap / 2)
        lms[311] = DummyLandmark(0.55, 0.50 - vertical_gap / 2)
        lms[402] = DummyLandmark(0.55, 0.50 + vertical_gap / 2)
        return lms

    mar_closed = FaceAnalyzer.calculate_mar(make_mouth_landmarks(0.01), 640, 480)
    mar_talking = FaceAnalyzer.calculate_mar(make_mouth_landmarks(0.07), 640, 480)
    mar_yawn = FaceAnalyzer.calculate_mar(make_mouth_landmarks(0.20), 640, 480)

    assert mar_closed < 0.20, f"Closed mouth MAR should be < 0.20, got {mar_closed:.3f}"
    assert mar_talking < config.MAR_THRESHOLD, f"Talking MAR ({mar_talking:.3f}) should be < threshold ({config.MAR_THRESHOLD})"
    assert mar_yawn >= config.MAR_THRESHOLD, f"Yawn MAR ({mar_yawn:.3f}) should be >= threshold ({config.MAR_THRESHOLD})"


# ---------------------------------------------------------------------------
# 3. YAW / PITCH HEAD POSE DISTRACTION
# ---------------------------------------------------------------------------
def test_head_pose_yaw_pitch_distraction():
    """Verify head pose distraction detection for Yaw (left/right) and Pitch (up/down)."""
    pose_estimator = HeadPoseEstimator()
    assert not pose_estimator.estimate_from_transformation_matrix(np.eye(4))[0] != 0.0

    # FaceAnalyzer.is_distracted logic check
    class DummyAnalyzer:
        is_distracted = FaceAnalyzer.is_distracted

    analyzer = DummyAnalyzer()

    # Normal looking forward
    assert not analyzer.is_distracted(pitch=0.0, yaw=0.0)
    assert not analyzer.is_distracted(pitch=-5.0, yaw=10.0)

    # Distracted: turning head left or right (> 30 deg)
    assert analyzer.is_distracted(pitch=0.0, yaw=35.0)
    assert analyzer.is_distracted(pitch=0.0, yaw=-35.0)

    # Distracted: head tilted up (> 25 deg) or looking down / slumping (< -20 deg)
    assert analyzer.is_distracted(pitch=28.0, yaw=0.0)
    assert analyzer.is_distracted(pitch=-25.0, yaw=0.0)


# ---------------------------------------------------------------------------
# 4. GÓC VAI & POSTURE SLUMP
# ---------------------------------------------------------------------------
def test_shoulder_tilt_and_slump_posture():
    """Verify shoulder angle calculation and persistence timer for posture slumping."""
    pose_analyzer = PoseAnalyzer()

    # Level shoulders: Left shoulder (11), Right shoulder (12)
    level_pose = [[DummyLandmark(0.40, 0.60), DummyLandmark(0.60, 0.60)]]
    level_landmarks = [DummyLandmark(0.0, 0.0) for _ in range(33)]
    level_landmarks[11] = DummyLandmark(0.40, 0.60)
    level_landmarks[12] = DummyLandmark(0.60, 0.60)

    signed_angle, abs_tilt = pose_analyzer.calculate_shoulder_tilt([level_landmarks], 640, 480)
    assert abs_tilt < 1.0, f"Level shoulders should have tilt ~0 deg, got {abs_tilt}"

    # Slanted shoulders: Left shoulder higher than right by 100 pixels
    slanted_landmarks = [DummyLandmark(0.0, 0.0) for _ in range(33)]
    slanted_landmarks[11] = DummyLandmark(0.35, 0.50)  # y = 240
    slanted_landmarks[12] = DummyLandmark(0.65, 0.70)  # y = 336 (dx=192, dy=96 -> ~26.5 deg)

    signed_angle, abs_tilt = pose_analyzer.calculate_shoulder_tilt([slanted_landmarks], 640, 480)
    assert abs_tilt > config.SHOULDER_TILT_THRESHOLD_DEGREE, f"Expected tilt > 15 deg, got {abs_tilt:.1f}"

    # Test slump duration tracking
    res1 = pose_analyzer.update_posture([slanted_landmarks], 640, 480, now=100.0)
    assert not res1["is_slump_alert"], "Slump should not trigger immediately"

    # Advance time beyond SHOULDER_TILT_TIME_LIMIT (3.0s)
    res2 = pose_analyzer.update_posture([slanted_landmarks], 640, 480, now=103.5)
    assert res2["is_slump_alert"], "Slump SHOULD trigger after 3.5s of continuous slant!"


# ---------------------------------------------------------------------------
# 5. STATE MACHINE TRANSITIONS (NORMAL -> WARNING -> ALARM)
# ---------------------------------------------------------------------------
def test_state_machine_transitions():
    """Verify state transitions and composite risk scoring."""
    sm = DriverStateMachine()

    # Initial state: NORMAL
    level, score, active, alerts = sm.process_all_conditions({}, now=100.0)
    assert level == AlertLevel.NORMAL
    assert score == 0

    # Trigger a MEDIUM violation (e.g. YAWN for > 1.0s)
    sm.process_all_conditions({"YAWN": True}, now=100.0)
    level, score, active, alerts = sm.process_all_conditions({"YAWN": True}, now=101.5)
    assert "YAWN" in active
    assert level == AlertLevel.WARNING

    # Trigger a VERY_HIGH violation (e.g. DROWSY for > 1.5s)
    sm.process_all_conditions({"DROWSY": True}, now=100.0)
    level, score, active, alerts = sm.process_all_conditions({"DROWSY": True}, now=102.0)
    assert "DROWSY" in active
    assert level == AlertLevel.ALARM, "VERY_HIGH violation MUST immediately elevate to ALARM!"

    # Reset all
    sm.reset_all()
    assert sm.current_level == AlertLevel.NORMAL
    assert sm.total_risk_score == 0


# ---------------------------------------------------------------------------
# 6. BEHAVIOR TIMERS, DEBOUNCE, AND COOLDOWN
# ---------------------------------------------------------------------------
def test_behavior_timer_and_cooldown():
    """Test BehaviorTracker time duration threshold and alert cooldown mechanism."""
    tracker = BehaviorTracker(key="TEST", time_limit=2.0, risk_weight=20, cooldown=5.0)

    # Condition active for 1.0s (< limit 2.0s)
    assert not tracker.update(True, now=100.0)
    assert not tracker.update(True, now=101.0)
    assert not tracker.is_active

    # Condition reaches 2.5s (>= limit 2.0s)
    assert tracker.update(True, now=102.5)
    assert tracker.is_active
    assert tracker.duration >= 2.5

    # Can trigger alert on first occurrence
    assert tracker.can_trigger_alert(now=102.5)

    # Immediately checking again should be blocked by cooldown (5.0s)
    assert not tracker.can_trigger_alert(now=103.5)
    assert not tracker.can_trigger_alert(now=107.0)

    # After cooldown expires (> 5.0s)
    assert tracker.can_trigger_alert(now=108.0)


# ---------------------------------------------------------------------------
# 7. PERCLOS SLIDING WINDOW CALCULATION
# ---------------------------------------------------------------------------
def test_perclos_sliding_window():
    """Verify PERCLOS ratio calculation over a sliding window."""
    detector = DrowsinessDetector()
    t = 100.0

    # 40 frames: 20 frames open, 20 frames closed (accounting for 2-3 frame smoothing window)
    for i in range(20):
        detector.update(raw_ear=0.35, yaw=0.0, pitch=0.0, now=t)
        t += 1.0

    for i in range(20):
        info = detector.update(raw_ear=0.05, yaw=0.0, pitch=0.0, now=t)
        t += 1.0

    assert info["perclos"] >= config.PERCLOS_THRESHOLD, f"Expected PERCLOS >= 0.35, got {info['perclos']}"
    assert info["drowsy_perclos_alert"], "PERCLOS >= 35% should trigger alert!"


# ---------------------------------------------------------------------------
# 8. STEERING ROI AND PARTIAL HAND OCCLUSION TOLERANCE
# ---------------------------------------------------------------------------
def test_steering_roi_and_partial_hand_occlusion():
    """
    Ensure:
    1. Both wrists in ROI -> OK
    2. Partial occlusion: ONE hand in ROI, other hand not seen/blocked -> OK (NO FALSE ALARM)
    3. Both wrists outside ROI -> OUT_OF_ROI
    4. Fallback wrists from Pose when hand landmarks are occluded -> OK
    """
    hand_analyzer = HandAnalyzer()
    img_w, img_h = 640, 480

    # 1. Both wrists inside ROI (ROI x: 0.20-0.80, y: 0.50-1.00 -> x: 128-512, y: 240-480)
    hand1 = [DummyLandmark(0.35, 0.70)]
    hand2 = [DummyLandmark(0.65, 0.70)]
    res_both = hand_analyzer.check_steering_status([hand1, hand2], img_w, img_h, now=100.0)
    assert res_both["status"] == "OK"

    # 2. PARTIAL OCCLUSION: Only ONE hand detected, and it is on steering wheel
    res_one = hand_analyzer.check_steering_status([hand1], img_w, img_h, now=100.0)
    assert res_one["status"] == "OK", "Single hand in ROI should be status OK (no false alarm on partial occlusion)!"
    assert not res_one["is_hands_off_alert"]

    # 3. Both wrists outside ROI (e.g. up near head or down off wheel)
    hand_out1 = [DummyLandmark(0.10, 0.20)]
    hand_out2 = [DummyLandmark(0.90, 0.20)]
    res_out = hand_analyzer.check_steering_status([hand_out1, hand_out2], img_w, img_h, now=100.0)
    assert res_out["status"] == "OUT_OF_ROI"

    # 4. Fallback wrists from PoseLandmarker when HandLandmarker detects 0 hands
    pose_wrists = [(300, 350)]  # Pixel coordinate inside ROI
    res_fallback = hand_analyzer.check_steering_status([], img_w, img_h, now=100.0, fallback_wrists=pose_wrists)
    assert res_fallback["status"] == "OK", "Fallback wrists from Pose must keep status OK when hands are occluded!"


# ---------------------------------------------------------------------------
# 9. YAWN FREQUENCY PER MINUTE (>= 3 YAWNS/MIN ALERT)
# ---------------------------------------------------------------------------
def test_yawn_frequency_counter_per_minute():
    """Verify yawn frequency counter registers completed yawns and alerts on >= 3 per min."""
    detector = YawnDetector()
    t = 100.0

    def simulate_yawn(start_t: float, duration: float = 1.2):
        # Yawn start
        detector.update(mar=0.70, now=start_t)
        # Yawn end
        return detector.update(mar=0.15, now=start_t + duration)

    # Yawn 1
    _, count1, alert1 = simulate_yawn(t)
    assert count1 == 1
    assert not alert1

    # Yawn 2 (15 seconds later)
    t += 15.0
    _, count2, alert2 = simulate_yawn(t)
    assert count2 == 2
    assert not alert2

    # Yawn 3 (another 15 seconds later) -> Total 3 yawns within 60s
    t += 15.0
    _, count3, alert3 = simulate_yawn(t)
    assert count3 == 3
    assert alert3, "3 yawns in 1 minute MUST trigger yawn alert!"

    # Move time forward by 70s -> Older yawns should expire
    t += 70.0
    _, count_expired, alert_expired = detector.update(mar=0.15, now=t)
    assert count_expired == 0
    assert not alert_expired


# ---------------------------------------------------------------------------
# 10. YOLO SPATIAL PROXIMITY CONSTRAINTS
# ---------------------------------------------------------------------------
def test_yolo_spatial_proximity_constraints():
    """
    Test spatial constraint check:
    - Object close to driver's face (ear/mouth) -> is_near_driver = True
    - Object far away (e.g. on passenger seat) -> is_near_driver = False
    - Distraction duration tracker triggers only after persistence threshold.
    """
    detector = ObjectDetector(enabled=False)

    face_bbox = (200, 150, 350, 320)  # Center ~ (275, 235)
    wrist_coords = [(280, 400)]

    # Phone held at ear: bbox (320, 180, 380, 280) -> center ~ (350, 230)
    phone_near_face = (320, 180, 380, 280)
    assert detector._check_spatial_proximity(phone_near_face, face_bbox, wrist_coords)

    # Phone on passenger seat / far away: bbox (550, 350, 620, 450)
    phone_far_away = (550, 350, 620, 450)
    assert not detector._check_spatial_proximity(phone_far_away, face_bbox, wrist_coords)

    # Test DistractionTracker persistence
    tracker = DistractionTracker()
    det_phone_near = [{"label": "phone", "is_near_driver": True}]

    res1 = tracker.update(det_phone_near, now=100.0)
    assert not res1["is_phone_alert"], "Phone should not alert at 0s"

    res2 = tracker.update(det_phone_near, now=101.6)  # limit is 1.5s
    assert res2["is_phone_alert"], "Phone alert MUST trigger after 1.6s of continuous holding!"


# ---------------------------------------------------------------------------
# 11. SAFE DEGRADATION (MISSING MODELS & FACE LOSS)
# ---------------------------------------------------------------------------
def test_safe_degradation_missing_models_and_face_loss():
    """Verify system runs safely without crash when models or face are missing."""
    # ObjectDetector with non-existent model
    obj_det = ObjectDetector(model_path="non_existent_yolo.pt", enabled=True)
    assert not obj_det.is_ready
    assert obj_det.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []

    # PoseAnalyzer with non-existent model
    pose = PoseAnalyzer(model_path="non_existent_pose.task")
    assert not pose.is_available
    assert pose.detect(None) is None
    res = pose.update_posture([], 640, 480)
    assert res["abs_tilt"] == 0.0
    assert not res["is_slump_alert"]

    # DrowsinessDetector handles face loss safely
    drowsy_det = DrowsinessDetector()
    drowsy_det.eye_closed_start_time = 100.0
    drowsy_det.handle_face_lost()
    assert drowsy_det.eye_closed_start_time is None, "Face loss must reset eye closure timer!"


# ---------------------------------------------------------------------------
# 12. PILLOW VIETNAMESE RENDERING CACHE AND HEADLESS TOGGLE
# ---------------------------------------------------------------------------
def test_visualizer_vietnamese_pillow_cache_and_headless():
    """
    Verify:
    1. Visualizer renders Vietnamese accented text via Pillow without errors.
    2. Text cache correctly retains and reuses pre-rendered bitmaps.
    3. Headless / disabled visualizer returns frames untouched with zero overhead.
    """
    vis = Visualizer(enabled=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Render accented text
    tw, th = vis.draw_vietnamese_text(frame, "CẢNH BÁO: TÀI XẾ BUỒN NGỦ!", (20, 30), font_size=18, color=(0, 0, 255))
    assert tw > 0 and th > 0

    # 2. Text cache check
    cache_len_before = len(vis._text_cache)
    assert cache_len_before > 0
    # Second call with same text should hit cache
    vis.draw_vietnamese_text(frame, "CẢNH BÁO: TÀI XẾ BUỒN NGỦ!", (20, 30), font_size=18, color=(0, 0, 255))
    assert len(vis._text_cache) == cache_len_before, "Cache size should not grow on cache hit!"

    # 3. Headless / disabled mode
    vis_disabled = Visualizer(enabled=False)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vis_disabled.draw_face_box(raw_frame, (10, 10, 50, 50), "Nguyen Van A")
    vis_disabled.draw_alerts(raw_frame, ["CANH BAO"])
    assert np.all(raw_frame == 0), "Disabled visualizer should leave frame completely untouched!"


# ---------------------------------------------------------------------------
# 13. EVENT LOGGER: CSV AND SNAPSHOT PERSISTENCE
# ---------------------------------------------------------------------------
def test_event_logger_csv_and_snapshot(tmp_path):
    """Verify event logging writes correct CSV records and saves snapshot images."""
    sess_path = tmp_path / "sessions.csv"
    ev_path = tmp_path / "events.csv"
    snap_dir = tmp_path / "snapshots"

    logger = EventLogger(events_path=ev_path, sessions_path=sess_path, snapshots_dir=snap_dir, session_id="SES_P6_001")

    # Log violation event with snapshot
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (100, 100), 40, (0, 255, 0), -1)

    logger.log_event(
        session_id="SES_P6_001",
        alert_type="Ngủ gật",
        level="VERY_HIGH",
        duration=2.5,
        frame=dummy_frame,
        risk_score=40
    )

    # Wait briefly for background worker thread to flush
    time.sleep(0.4)
    logger.stop()

    # Verify events.csv contains record
    assert ev_path.exists()
    with open(ev_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Session_ID" in content
        assert "SES_P6_001" in content
        assert "Ngủ gật" in content
        assert "VERY_HIGH" in content

    # Verify sessions.csv contains summary record
    assert sess_path.exists()
    with open(sess_path, "r", encoding="utf-8") as f:
        s_content = f.read()
        assert "SES_P6_001" in s_content

    # Verify snapshot image exists in directory
    snapshots = list(snap_dir.glob("*.jpg"))
    assert len(snapshots) >= 1, "Snapshot image file should have been saved to snapshots directory!"
