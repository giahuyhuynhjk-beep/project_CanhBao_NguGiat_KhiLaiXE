import os
import sys
import time
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.core.preprocessor import Preprocessor
from src.core.pose_analyzer import PoseAnalyzer
from src.core.face_analyzer import DrowsinessDetector, FaceAnalyzer


class MockLandmark:
    def __init__(self, x, y, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_preprocessor_low_light():
    preprocessor = Preprocessor()

    # 1. Dark image (mean brightness < 60)
    dark_img = np.full((480, 640, 3), 30, dtype=np.uint8)
    enhanced_img, is_low_light = preprocessor.process(dark_img)

    assert is_low_light is True
    # Enhanced image should be brighter than original dark image
    assert np.mean(enhanced_img) > np.mean(dark_img)

    # 2. Normal image (brightness >= 60)
    normal_img = np.full((480, 640, 3), 120, dtype=np.uint8)
    out_img, is_low_light_norm = preprocessor.process(normal_img)

    assert is_low_light_norm is False
    assert np.allclose(out_img, normal_img)


def test_preprocessor_affine_alignment():
    preprocessor = Preprocessor()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    aligned = preprocessor.align_face_affine(img, angle=25.0, force=True)

    assert aligned.shape == img.shape


def test_pose_analyzer_shoulder_tilt_and_slump():
    analyzer = PoseAnalyzer()

    # Landmark 11 (left shoulder), Landmark 12 (right shoulder)
    # Case 1: Level shoulders
    landmarks_level = [MockLandmark(0.5, 0.5) for _ in range(33)]
    landmarks_level[11] = MockLandmark(0.4, 0.5)
    landmarks_level[12] = MockLandmark(0.6, 0.5)

    info_level = analyzer.update_posture([landmarks_level], img_w=640, img_h=480, now=100.0)
    assert abs(info_level["tilt_angle"]) < 2.0
    assert info_level["is_slump_alert"] is False

    # Case 2: Heavily tilted shoulders (e.g. dy = 0.15, dx = 0.2 -> tan theta = 0.75 -> ~36.8 deg)
    landmarks_tilt = [MockLandmark(0.5, 0.5) for _ in range(33)]
    landmarks_tilt[11] = MockLandmark(0.4, 0.40)
    landmarks_tilt[12] = MockLandmark(0.6, 0.55)

    # First detection
    info_tilt = analyzer.update_posture([landmarks_tilt], img_w=640, img_h=480, now=100.0)
    assert info_tilt["abs_tilt"] > config.SHOULDER_TILT_THRESHOLD_DEGREE
    assert info_tilt["is_slump_alert"] is False  # Just started, duration < LIMIT

    # Tilted for > LIMIT (3.0s)
    info_tilt_alert = analyzer.update_posture([landmarks_tilt], img_w=640, img_h=480, now=103.5)
    assert info_tilt_alert["tilt_duration"] >= config.SLUMP_TIME_LIMIT
    assert info_tilt_alert["is_slump_alert"] is True

    # Recovers back to level
    info_recovered = analyzer.update_posture([landmarks_level], img_w=640, img_h=480, now=104.0)
    assert info_recovered["is_slump_alert"] is False
    assert analyzer.slump_start_time is None


def test_sunglasses_occlusion_and_ear_bypass():
    detector = DrowsinessDetector()
    detector.current_threshold = 0.22

    # Normal mode with closed eyes
    res_normal = detector.update(raw_ear=0.15, yaw=0.0, pitch=0.0, now=100.0)
    assert res_normal["eye_closed"] is True

    # Advance time > 1.5s -> should alert
    res_alert = detector.update(raw_ear=0.15, yaw=0.0, pitch=0.0, now=102.0)
    assert res_alert["drowsy_closed_alert"] is True

    # Switch to sunglasses mode
    detector.is_sunglasses_mode = True
    res_sunglasses = detector.update(raw_ear=0.15, yaw=0.0, pitch=0.0, now=103.0)

    # In sunglasses mode, EAR closure alert is bypassed to prevent false alarms
    assert res_sunglasses["drowsy_closed_alert"] is False
    assert res_sunglasses["eye_closed"] is False
    assert res_sunglasses["is_sunglasses_mode"] is True


def test_sunglasses_region_sampling():
    detector = DrowsinessDetector()

    # Synthesize image with dark eye region and bright forehead
    frame = np.full((480, 640, 3), 150, dtype=np.uint8)  # Bright face

    # Create dummy landmarks (478 points for face mesh)
    landmarks = [MockLandmark(0.5, 0.5) for _ in range(478)]
    landmarks[159] = MockLandmark(0.4, 0.4)  # Left eye
    landmarks[386] = MockLandmark(0.6, 0.4)  # Right eye
    landmarks[10] = MockLandmark(0.5, 0.2)   # Forehead

    # Make eye patches pitch black (sunglasses)
    lx, ly = int(0.4 * 640), int(0.4 * 480)
    rx, ry = int(0.6 * 640), int(0.4 * 480)
    frame[ly-20:ly+20, lx-20:lx+20] = 10
    frame[ry-20:ry+20, rx-20:rx+20] = 10

    is_occluded = detector.check_sunglasses_occlusion(frame, landmarks, 640, 480, smoothed_ear=0.15)
    assert is_occluded is True
    assert detector.is_sunglasses_mode is True
