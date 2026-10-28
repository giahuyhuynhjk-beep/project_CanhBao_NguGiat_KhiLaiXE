"""
Unit tests for Phase 3: Session Management, Session Logging (sessions.csv),
Session-based EAR Calibration, and Realistic Data Augmentation.
"""

import sys
import os
import csv
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.utils.event_logger import EventLogger
from src.core.preprocessor import apply_realistic_augmentation
from src.core.face_analyzer import DrowsinessDetector


def test_session_logging_and_summary(tmp_path):
    """Verify that sessions are recorded to sessions.csv with duration and alert summary."""
    test_events_csv = tmp_path / "events.csv"
    test_sessions_csv = tmp_path / "sessions.csv"
    test_snapshots_dir = tmp_path / "snapshots"

    logger = EventLogger(
        events_path=test_events_csv,
        sessions_path=test_sessions_csv,
        snapshots_dir=test_snapshots_dir,
        session_id="SES_PHASE3_001"
    )

    # Log 2 events in this session
    logger.log_event(alert_type="Ngủ gật", level="VERY_HIGH", duration=1.8, risk_score=40)
    logger.log_event(alert_type="Quay đầu / lệch hướng", level="HIGH", duration=2.6, risk_score=30)

    time.sleep(0.5)
    logger.stop()  # Ends session and writes to sessions.csv

    # Verify sessions.csv
    assert os.path.exists(test_sessions_csv)
    with open(test_sessions_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2
        header = reader[0]
        row = reader[1]
        assert header[0] == "Session_ID"
        assert row[0] == "SES_PHASE3_001"
        assert int(row[4]) == 2  # Total alerts = 2
        assert "Ngủ gật" in row[5]
        assert int(row[6]) == 40  # Max risk score = 40

    print("test_session_logging_and_summary PASSED!")


def test_realistic_data_augmentation():
    """Verify realistic augmentation preserves image geometry and produces valid pixels."""
    cv2_rect = np.zeros((200, 200), dtype=np.uint8)
    cv2_rect[50:150, 50:150] = 200

    augmented = apply_realistic_augmentation(cv2_rect)

    assert augmented.shape == (200, 200)
    assert augmented.dtype == np.uint8
    assert augmented.min() >= 0
    assert augmented.max() <= 255
    print("test_realistic_data_augmentation PASSED!")


def test_session_calibration_and_driver_absent_reset():
    """Verify session-based calibration, manual recalibration, and face absence handling."""
    detector = DrowsinessDetector()

    # 1. Start calibration on session launch
    t0 = 1000.0
    detector.start_calibration(now=t0)
    assert detector.is_calibrating is True
    assert not detector.is_calibrated

    # Collect samples
    for i in range(30):
        detector.update(raw_ear=0.30, yaw=0.0, pitch=0.0, now=t0 + i * 0.15)
    res = detector.update(raw_ear=0.30, yaw=0.0, pitch=0.0, now=t0 + 4.5)
    assert res["is_calibrating"] is False
    assert detector.is_calibrated is True
    calibrated_thresh = detector.current_threshold
    assert 0.18 <= calibrated_thresh <= 0.26

    # 2. Driver absence / face lost resets temporary timers
    detector.eye_closed_start_time = 1010.0
    detector.handle_face_lost()
    assert detector.eye_closed_start_time is None

    # 3. Manual recalibration (key 'C') resets and starts fresh
    detector.reset_calibration()
    assert detector.is_calibrated is False
    assert detector.current_threshold == config.EAR_THRESHOLD_DEFAULT
    detector.start_calibration(now=2000.0)
    assert detector.is_calibrating is True
    print("test_session_calibration_and_driver_absent_reset PASSED!")


if __name__ == "__main__":
    import pathlib
    test_session_logging_and_summary(pathlib.Path("scratch/test_phase3"))
    test_realistic_data_augmentation()
    test_session_calibration_and_driver_absent_reset()
    print("\nALL PHASE 3 TESTS PASSED!")
