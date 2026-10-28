"""
Unit tests for Phase 1: High Precision Face, Eye, Yawn, Pose, and Steering Wheel ROI monitoring.
"""

import sys
import os
import time
import math
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.core.face_analyzer import FaceAnalyzer, DrowsinessDetector, YawnDetector, HeadPoseEstimator
from src.core.hand_analyzer import HandAnalyzer


class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def test_ear_calculation():
    """Verify that EAR correctly detects open vs closed eye landmarks."""
    # 6 landmarks for eye: [p0, p1, p2, p3, p4, p5]
    # Wide open eye: vertical distance (p1-p5, p2-p4) is large relative to horizontal (p0-p3)
    open_eye = [
        DummyLandmark(0.10, 0.20),  # p0: left corner
        DummyLandmark(0.15, 0.16),  # p1: top-left
        DummyLandmark(0.20, 0.16),  # p2: top-right
        DummyLandmark(0.25, 0.20),  # p3: right corner
        DummyLandmark(0.20, 0.24),  # p4: bottom-right
        DummyLandmark(0.15, 0.24),  # p5: bottom-left
    ]
    # Closed eye: vertical distance is near zero
    closed_eye = [
        DummyLandmark(0.10, 0.20),  # p0
        DummyLandmark(0.15, 0.198), # p1
        DummyLandmark(0.20, 0.198), # p2
        DummyLandmark(0.25, 0.20),  # p3
        DummyLandmark(0.20, 0.202), # p4
        DummyLandmark(0.15, 0.202), # p5
    ]

    indices = [0, 1, 2, 3, 4, 5]
    ear_open = FaceAnalyzer.calculate_ear(open_eye, indices, 640, 480)
    ear_closed = FaceAnalyzer.calculate_ear(closed_eye, indices, 640, 480)

    print(f"EAR Open: {ear_open:.3f}, EAR Closed: {ear_closed:.3f}")
    assert ear_open > 0.30, f"Expected open EAR > 0.30, got {ear_open}"
    assert ear_closed < 0.10, f"Expected closed EAR < 0.10, got {ear_closed}"


def test_drowsiness_calibration_and_perclos():
    """Test personalized EAR calibration and PERCLOS calculation."""
    detector = DrowsinessDetector()
    t0 = 1000.0
    detector.start_calibration("test_driver", now=t0)
    assert detector.is_calibrating is True
    for i in range(30):
        t = t0 + (i * 0.15)
        detector.update(raw_ear=0.32, yaw=0.0, pitch=0.0, now=t)

    # After 4.5 seconds, calibration should complete
    res = detector.update(raw_ear=0.32, yaw=0.0, pitch=0.0, now=t0 + 4.5)
    assert res["is_calibrating"] is False
    # Target threshold is around 0.32 * 0.72 = 0.2304
    expected_threshold = 0.32 * config.EAR_CALIBRATION_RATIO
    assert abs(res["threshold"] - expected_threshold) < 0.02, f"Expected {expected_threshold}, got {res['threshold']}"
    print(f"Calibrated threshold: {res['threshold']:.3f}")

    # Test continuous eye closure detection
    t_start = 2000.0
    # Feed 5 frames to fill the moving average buffer with closed eye values
    for i in range(5):
        detector.update(raw_ear=0.10, yaw=0.0, pitch=0.0, now=t_start + i * 0.05)

    t_closed_active = t_start + 0.3
    res1 = detector.update(raw_ear=0.10, yaw=0.0, pitch=0.0, now=t_closed_active)
    assert res1["drowsy_closed_alert"] is False
    # Exceed DROWSY_TIME_LIMIT (1.5s)
    res2 = detector.update(raw_ear=0.10, yaw=0.0, pitch=0.0, now=t_closed_active + config.DROWSY_TIME_LIMIT + 0.2)
    assert res2["drowsy_closed_alert"] is True
    print("Continuous eye closure test passed!")

    # Test PERCLOS over sliding window
    detector.perclos_history.clear()
    t_perclos = 3000.0
    # Simulate 40% closed eye frames
    for i in range(100):
        t = t_perclos + i * 0.2
        is_closed_sample = (i % 10 < 4)  # 40% closed
        val = 0.10 if is_closed_sample else 0.35
        out = detector.update(raw_ear=val, yaw=0.0, pitch=0.0, now=t)

    assert out["perclos"] >= 0.35
    assert out["drowsy_perclos_alert"] is True
    print(f"PERCLOS test passed! Ratio: {out['perclos'] * 100:.1f}%")


def test_yawn_detector():
    """Test MAR calculation and frequency tracking (> 3 yawns/minute)."""
    yawn_det = YawnDetector()
    t0 = 5000.0

    # Simulate 3 yawns within 40 seconds
    for y_idx in range(3):
        # Yawn start (MAR high for 1.2s)
        t_yawn_start = t0 + (y_idx * 12.0)
        yawn_det.update(mar=0.75, now=t_yawn_start)
        yawn_det.update(mar=0.75, now=t_yawn_start + 1.2)
        # Mouth closes
        _, count, alert = yawn_det.update(mar=0.20, now=t_yawn_start + 1.5)

    assert count == 3, f"Expected 3 yawns, got {count}"
    assert alert is True, "Expected yawn alert to trigger on 3rd yawn within 1 minute"
    print(f"Yawn test passed! Count: {count}, Alert: {alert}")


def test_head_pose_and_distraction():
    """Verify solvePnP head pose estimation and distraction threshold logic."""
    estimator = HeadPoseEstimator()
    w, h = 640, 480

    # Build dummy landmarks with nose at center
    def make_landmarks(nose_x, nose_y):
        lms = {}
        lms[1] = DummyLandmark(nose_x, nose_y)               # nose tip
        lms[152] = DummyLandmark(0.50, 0.75)                 # chin
        lms[33] = DummyLandmark(0.38, 0.40)                  # left eye
        lms[263] = DummyLandmark(0.62, 0.40)                 # right eye
        lms[61] = DummyLandmark(0.42, 0.65)                  # left mouth
        lms[291] = DummyLandmark(0.58, 0.65)                 # right mouth
        return lms

    # Straight head
    lms_straight = make_landmarks(0.50, 0.50)
    pitch, yaw, roll = estimator.estimate_from_landmarks(lms_straight, w, h)
    print(f"Straight Pose -> Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")
    assert abs(yaw) < 15.0
    assert abs(pitch) < 15.0

    # Turned head (Nose shifted significantly to the right)
    lms_turned = make_landmarks(0.58, 0.50)
    pitch_t, yaw_t, roll_t = estimator.estimate_from_landmarks(lms_turned, w, h)
    print(f"Turned Pose -> Pitch: {pitch_t:.1f}, Yaw: {yaw_t:.1f}")
    assert abs(yaw_t) > 20.0


def test_steering_wheel_roi():
    """Test Steering Wheel ROI: OK vs NO_HANDS vs OUT_OF_ROI."""
    # Test hand landmarks
    # ROI: x: [0.2, 0.8], y: [0.5, 1.0]
    dummy_hand_in_roi = [[DummyLandmark(0.50, 0.70)]]   # wrist inside ROI
    dummy_hand_outside = [[DummyLandmark(0.10, 0.30)]]  # wrist outside ROI (top-left)

    analyzer = HandAnalyzer()
    analyzer.roi = {"x_min": 0.20, "y_min": 0.50, "x_max": 0.80, "y_max": 1.00}

    # 1. Hand inside ROI
    res_ok = analyzer.check_steering_status(dummy_hand_in_roi, 640, 480, now=10.0)
    assert res_ok["status"] == "OK"

    # 2. Hand outside ROI
    t0 = 20.0
    res_out1 = analyzer.check_steering_status(dummy_hand_outside, 640, 480, now=t0)
    assert res_out1["status"] == "OUT_OF_ROI"
    assert res_out1["is_out_of_roi_alert"] is False
    # Exceed limit
    res_out2 = analyzer.check_steering_status(dummy_hand_outside, 640, 480, now=t0 + config.HANDS_OUT_OF_ROI_LIMIT + 0.1)
    assert res_out2["is_out_of_roi_alert"] is True

    # 3. No hands detected (occlusion)
    t1 = 40.0
    res_none1 = analyzer.check_steering_status([], 640, 480, now=t1)
    assert res_none1["status"] == "NO_HANDS"
    assert res_none1["is_hands_off_alert"] is False
    # Exceed hands off limit
    res_none2 = analyzer.check_steering_status([], 640, 480, now=t1 + config.HANDS_OFF_TIME_LIMIT + 0.1)
    assert res_none2["is_hands_off_alert"] is True

    print("Steering Wheel ROI test passed successfully!")


if __name__ == "__main__":
    test_ear_calculation()
    test_drowsiness_calibration_and_perclos()
    test_yawn_detector()
    test_head_pose_and_distraction()
    test_steering_wheel_roi()
    print("\nALL PHASE 1 UNIT TESTS PASSED!")
