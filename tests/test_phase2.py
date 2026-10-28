"""
Unit tests for Phase 2: State Machine, Alert System, and Event Logging.
"""

import sys
import os
import time
import csv
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.core.state_machine import DriverStateMachine, AlertLevel, BehaviorTracker
from src.utils.alert_system import AlertSystem, AudioSynthesizer
from src.utils.event_logger import EventLogger


def test_behavior_tracker_and_cooldown():
    """Verify behavior timer, activation, and cooldown throttling."""
    tracker = BehaviorTracker(key="TEST", time_limit=1.5, risk_weight=30, cooldown=3.0, severity="HIGH")

    t0 = 1000.0
    # Condition true for 1.0s (less than limit 1.5s)
    assert tracker.update(True, now=t0) is False
    assert tracker.update(True, now=t0 + 1.0) is False
    assert tracker.is_active is False

    # Condition true for 1.6s -> becomes active
    assert tracker.update(True, now=t0 + 1.6) is True
    assert tracker.is_active is True

    # First alert trigger passes
    assert tracker.can_trigger_alert(now=t0 + 1.6) is True

    # Immediate second check should be throttled by 3.0s cooldown
    assert tracker.can_trigger_alert(now=t0 + 2.0) is False
    assert tracker.can_trigger_alert(now=t0 + 4.5) is False

    # After cooldown (3.0s after 1.6s is 4.6s), can trigger again
    assert tracker.can_trigger_alert(now=t0 + 4.7) is True

    # When condition becomes False, resets
    tracker.update(False, now=t0 + 5.0)
    assert tracker.is_active is False
    assert tracker.duration == 0.0
    print("test_behavior_tracker_and_cooldown PASSED!")


def test_state_machine_transitions_and_risk_score():
    """Verify state transitions (NORMAL -> WARNING -> ALARM) and risk score summation."""
    sm = DriverStateMachine()
    t0 = 100.0

    # 1. Normal state
    conditions_normal = {k: False for k in config.ALERT_MATRIX.keys()}
    level, score, active, triggered = sm.process_all_conditions(conditions_normal, now=t0)
    assert level == AlertLevel.NORMAL
    assert score == 0
    assert len(active) == 0

    # 2. Trigger a MEDIUM violation (YAWN, weight 15)
    # Simulate condition lasting beyond limit
    sm.process_all_conditions({"YAWN": True}, now=t0)
    level, score, active, triggered = sm.process_all_conditions({"YAWN": True}, now=t0 + config.MAR_MIN_DURATION + 0.1)
    assert "YAWN" in active
    assert score == 15
    assert level == AlertLevel.WARNING  # Warning because of MEDIUM violation

    # 3. Add a HIGH violation (DISTRACTED, weight 30) -> Score = 15 + 30 = 45
    sm.process_all_conditions({"YAWN": True, "DISTRACTED": True}, now=t0)
    level, score, active, triggered = sm.process_all_conditions({"YAWN": True, "DISTRACTED": True}, now=t0 + config.DISTRACTED_TIME_LIMIT + 0.1)
    assert "DISTRACTED" in active
    assert score == 45
    assert level == AlertLevel.WARNING

    # 4. Trigger a VERY_HIGH violation (DROWSY, weight 40) -> ALARM state
    sm.process_all_conditions({"YAWN": True, "DISTRACTED": True, "DROWSY": True}, now=t0)
    level, score, active, triggered = sm.process_all_conditions({"YAWN": True, "DISTRACTED": True, "DROWSY": True}, now=t0 + config.DROWSY_TIME_LIMIT + 0.1)
    assert "DROWSY" in active
    assert level == AlertLevel.ALARM
    assert score >= config.RISK_THRESHOLD_ALARM
    print(f"State Machine transition PASSED! Final Score: {score}, Level: {level.name}")


def test_audio_synthesizer_and_alert_system():
    """Verify AudioSynthesizer initializes waveforms and AlertSystem handles queue."""
    synth = AudioSynthesizer()
    assert synth.is_ready is True
    assert "SIREN_LOUD" in synth.sounds
    assert "SIREN_RAPID" in synth.sounds
    assert "CHIME" in synth.sounds

    # AlertSystem queueing
    system = AlertSystem(enable_audio=True)
    system.trigger("CHIME", "Thử nghiệm âm thanh")
    system.trigger("SIREN_LOUD", None)
    time.sleep(0.3)
    system.stop()
    print("test_audio_synthesizer_and_alert_system PASSED!")


def test_event_logger_and_snapshot(tmp_path):
    """Verify that violation events, sessions, and snapshots are written to disk."""
    test_events_csv = tmp_path / "test_events.csv"
    test_sessions_csv = tmp_path / "test_sessions.csv"
    test_snapshots_dir = tmp_path / "snapshots"

    logger = EventLogger(
        events_path=test_events_csv,
        sessions_path=test_sessions_csv,
        snapshots_dir=test_snapshots_dir,
        session_id="SES_TEST_001"
    )

    # 1. Test Violation Event with mock image
    dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    logger.log_event(
        session_id="SES_TEST_001",
        alert_type="Ngủ gật",
        level="VERY_HIGH",
        duration=1.75,
        frame=dummy_frame,
        risk_score=40
    )

    # Allow worker thread to flush and end session
    time.sleep(0.6)
    logger.stop()

    # Verify events.csv
    assert os.path.exists(test_events_csv)
    with open(test_events_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2  # Header + 1 record
        header = reader[0]
        record = reader[1]
        assert "Timestamp" in header
        assert header[1] == "Session_ID"
        assert record[1] == "SES_TEST_001"
        assert record[2] == "Ngủ gật"
        assert record[3] == "VERY_HIGH"
        assert float(record[4]) == 1.75
        assert len(record[5]) > 0  # snapshot path

    # Verify sessions.csv
    assert os.path.exists(test_sessions_csv)
    with open(test_sessions_csv, "r", encoding="utf-8") as f:
        s_reader = list(csv.reader(f))
        assert len(s_reader) >= 2
        s_record = s_reader[1]
        assert s_record[0] == "SES_TEST_001"
        assert int(s_record[4]) >= 1  # Total alerts >= 1

    # Verify snapshot image exists
    snapshots = list(test_snapshots_dir.glob("*.jpg"))
    assert len(snapshots) >= 1
    print(f"test_event_logger_and_snapshot PASSED! Snapshot created: {snapshots[0].name}")


if __name__ == "__main__":
    test_behavior_tracker_and_cooldown()
    test_state_machine_transitions_and_risk_score()
    test_audio_synthesizer_and_alert_system()
    import pathlib
    test_event_logger_and_snapshot(pathlib.Path("scratch/test_phase2"))
    print("\nALL PHASE 2 TESTS PASSED!")
