"""
State Machine module for Driver Monitoring System.
Tracks alert states (NORMAL, WARNING, ALARM), behavior timers, cooldowns, and risk scores.
Implements ma trận cảnh báo theo quy chuẩn với bộ đệm chống báo lặp.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import src.config as config


class AlertLevel(Enum):
    NORMAL = 0
    WARNING = 1
    ALARM = 2


class BehaviorTracker:
    def __init__(self, key: str, time_limit: float, risk_weight: int, cooldown: float = config.COOLDOWN_SECONDS, severity: str = "MEDIUM"):
        self.key = key
        self.time_limit = time_limit
        self.risk_weight = risk_weight
        self.cooldown = cooldown
        self.severity = severity  # "VERY_HIGH", "HIGH", "MEDIUM"

        self.start_time: Optional[float] = None
        self.last_alert_time: float = 0.0
        self.is_active: bool = False
        self.duration: float = 0.0

    def update(self, condition: bool, now: Optional[float] = None) -> bool:
        """
        Update condition state.
        Returns True if condition has lasted >= time_limit and tracker is active.
        """
        if now is None:
            now = time.time()

        if condition:
            if self.start_time is None:
                self.start_time = now
            self.duration = now - self.start_time
            if self.duration >= self.time_limit:
                self.is_active = True
                return True
        else:
            self.start_time = None
            self.duration = 0.0
            self.is_active = False

        return False

    def can_trigger_alert(self, now: Optional[float] = None) -> bool:
        """Check if active violation can trigger alarm/audio based on cooldown."""
        if now is None:
            now = time.time()
        if self.is_active and (now - self.last_alert_time >= self.cooldown):
            self.last_alert_time = now
            return True
        return False

    def reset(self):
        self.start_time = None
        self.duration = 0.0
        self.is_active = False


class DriverStateMachine:
    def __init__(self):
        self.trackers: Dict[str, BehaviorTracker] = {}
        self._init_trackers()

        self.current_level = AlertLevel.NORMAL
        self.total_risk_score = 0

    def _init_trackers(self):
        """Initialize trackers from ALERT_MATRIX configuration."""
        for key, info in config.ALERT_MATRIX.items():
            # Derive time limits from config
            if key == "DROWSY":
                tl = config.DROWSY_TIME_LIMIT
            elif key == "YAWN":
                tl = config.MAR_MIN_DURATION
            elif key == "DISTRACTED":
                tl = config.DISTRACTED_TIME_LIMIT
            elif key == "PHONE":
                tl = config.DISTRACTION_PHONE_LIMIT
            elif key == "SMOKE":
                tl = config.DISTRACTION_SMOKE_LIMIT
            elif key == "DRINK":
                tl = config.DISTRACTION_DRINK_LIMIT
            elif key == "SLUMP":
                tl = config.SHOULDER_TILT_TIME_LIMIT
            elif key == "HANDS_OFF":
                tl = config.HANDS_OUT_OF_ROI_LIMIT
            elif key == "HANDS_NONE":
                tl = config.HANDS_OFF_TIME_LIMIT
            else:
                tl = 2.0

            self.trackers[key] = BehaviorTracker(
                key=key,
                time_limit=tl,
                risk_weight=info["risk_weight"],
                cooldown=info["cooldown"],
                severity=info["level"]
            )

    def update_condition(self, behavior_key: str, condition: bool, now: Optional[float] = None) -> bool:
        """Update a specific behavior condition."""
        if behavior_key in self.trackers:
            return self.trackers[behavior_key].update(condition, now)
        return False

    def process_all_conditions(self, conditions: Dict[str, bool], now: Optional[float] = None) -> Tuple[AlertLevel, int, List[str], List[Dict[str, Any]]]:
        """
        Process multiple behavior conditions in one call.
        Returns:
        - current_level: AlertLevel (NORMAL, WARNING, ALARM)
        - risk_score: int
        - active_violations: List[str]
        - newly_triggered_alerts: List[Dict] with info for audio/logging
        """
        if now is None:
            now = time.time()

        for key, active in conditions.items():
            if key in self.trackers:
                self.trackers[key].update(active, now)

        # Check for trackers that become active and pass cooldown
        triggered_alerts = []
        for key, tracker in self.trackers.items():
            if tracker.is_active and tracker.can_trigger_alert(now):
                meta = config.ALERT_MATRIX.get(key, {})
                triggered_alerts.append({
                    "key": key,
                    "name": meta.get("name", key),
                    "level": meta.get("level", "MEDIUM"),
                    "action": meta.get("action", "VOICE"),
                    "voice_msg": meta.get("voice_msg", ""),
                    "duration": tracker.duration
                })

        # Calculate combined risk score
        score = 0
        has_alarm = False
        has_warning = False

        for tracker in self.trackers.values():
            if tracker.is_active:
                score += tracker.risk_weight
                if tracker.severity == "VERY_HIGH":
                    has_alarm = True
                elif tracker.severity in ("HIGH", "MEDIUM"):
                    has_warning = True

        self.total_risk_score = score

        # State transition: ALARM if any VERY_HIGH or score >= ALARM threshold
        if has_alarm or score >= config.RISK_THRESHOLD_ALARM:
            self.current_level = AlertLevel.ALARM
        elif has_warning or score >= config.RISK_THRESHOLD_WARNING:
            self.current_level = AlertLevel.WARNING
        else:
            self.current_level = AlertLevel.NORMAL

        active_violations = [k for k, t in self.trackers.items() if t.is_active]
        return self.current_level, self.total_risk_score, active_violations, triggered_alerts

    def reset_all(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.current_level = AlertLevel.NORMAL
        self.total_risk_score = 0
