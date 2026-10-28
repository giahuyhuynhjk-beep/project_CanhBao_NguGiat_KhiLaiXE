import os
import sys
import yaml
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.core.object_detector import ObjectDetector, DistractionTracker


def test_distraction_tracker_phone():
    tracker = DistractionTracker()

    det_phone_near = [{"label": "phone", "conf": 0.85, "bbox": (100, 100, 200, 200), "is_near_driver": True}]
    det_none = []

    # Frame 1 at t=10.0
    res1 = tracker.update(det_phone_near, now=10.0)
    assert res1["is_phone_alert"] is False
    assert res1["durations"]["phone"] == 0.0

    # Frame 2 at t=11.0 (< 1.5s limit)
    res2 = tracker.update(det_phone_near, now=11.0)
    assert res2["is_phone_alert"] is False
    assert res2["durations"]["phone"] == 1.0

    # Frame 3 at t=11.6 (> 1.5s limit) -> Should trigger alert
    res3 = tracker.update(det_phone_near, now=11.6)
    assert res3["is_phone_alert"] is True
    assert res3["durations"]["phone"] == pytest.approx(1.6)

    # Phone put away at t=12.0 -> Alert resets immediately
    res4 = tracker.update(det_none, now=12.0)
    assert res4["is_phone_alert"] is False
    assert tracker.start_times["phone"] is None


def test_distraction_tracker_smoke_and_drink():
    tracker = DistractionTracker()

    det_smoke = [{"label": "cigarette", "conf": 0.75, "bbox": (100, 100, 150, 150), "is_near_driver": True}]
    det_drink = [{"label": "drink_bottle", "conf": 0.80, "bbox": (100, 100, 160, 220), "is_near_driver": True}]

    # Smoke: limit is 1.5s
    tracker.update(det_smoke, now=20.0)
    assert tracker.update(det_smoke, now=21.6)["is_smoke_alert"] is True

    # Drink: limit is 3.0s (swigging water is permitted briefly)
    tracker.reset()
    tracker.update(det_drink, now=30.0)
    assert tracker.update(det_drink, now=32.0)["is_drink_alert"] is False  # 2.0s < 3.0s
    assert tracker.update(det_drink, now=33.2)["is_drink_alert"] is True   # 3.2s >= 3.0s


def test_spatial_proximity_constraints():
    detector = ObjectDetector(enabled=False)

    face_bbox = (200, 150, 350, 320)  # Center ~ (275, 235)
    wrists = [(450, 400)]

    # Object 1: Right at the driver's ear/face
    obj_near_face = (180, 170, 240, 250)
    assert detector._check_spatial_proximity(obj_near_face, face_bbox, wrists) is True

    # Object 2: Right by the driver's hand/wrist
    obj_near_wrist = (430, 380, 480, 430)  # center (455, 405), dist ~ 7px
    assert detector._check_spatial_proximity(obj_near_wrist, face_bbox, wrists) is True

    # Object 3: Background object (e.g. bottom-left corner of the car)
    obj_far = (10, 400, 50, 450)
    assert detector._check_spatial_proximity(obj_far, face_bbox, wrists) is False


def test_object_detector_graceful_degradation():
    # When enabled=False or weights missing, should return empty safe defaults
    detector = ObjectDetector(enabled=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.process(frame, face_bbox=None, wrist_coords=None)
    assert res["is_phone_alert"] is False
    assert res["is_smoke_alert"] is False
    assert res["is_drink_alert"] is False
    assert res["detections"] == []


def test_yolo_configs_and_scripts_exist():
    # 1. driver_distraction.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "yolo", "driver_distraction.yaml")
    assert os.path.exists(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "names" in data
    assert data["names"][0] == "phone"
    assert data["names"][1] == "smoke"
    assert data["names"][2] == "drink"

    # 2. train.py and export.py
    train_path = os.path.join(os.path.dirname(__file__), "..", "yolo", "train.py")
    export_path = os.path.join(os.path.dirname(__file__), "..", "yolo", "export.py")
    readme_path = os.path.join(os.path.dirname(__file__), "..", "yolo", "README.md")

    assert os.path.exists(train_path)
    assert os.path.exists(export_path)
    assert os.path.exists(readme_path)
