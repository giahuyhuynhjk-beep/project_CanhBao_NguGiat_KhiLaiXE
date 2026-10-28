"""
Event logger module for Driver Monitoring System.
Logs violation events (with snapshots) to events.csv and session summaries to sessions.csv asynchronously.
Anonymous, session-based logging without personal driver identification.
"""

import os
import csv
import time
import json
import queue
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import cv2
import numpy as np
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("EventLogger")


class EventLogger:
    def __init__(
        self,
        events_path=config.EVENTS_PATH,
        sessions_path=config.SESSIONS_PATH,
        snapshots_dir=config.SNAPSHOTS_DIR,
        session_id: Optional[str] = None
    ):
        self.events_path = str(events_path)
        self.sessions_path = str(sessions_path)
        self.snapshots_dir = str(snapshots_dir)

        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.sessions_path), exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self._init_events_csv()
        self._init_sessions_csv()

        # Session tracking state
        self.session_id = session_id or datetime.now().strftime("SES_%Y%m%d_%H%M%S")
        if getattr(config, "DRIVER_LABEL", ""):
            self.session_id = f"{config.DRIVER_LABEL}_{self.session_id}"

        self.session_start_time = datetime.now()
        self.session_alerts: Dict[str, int] = {}
        self.session_max_risk: int = 0
        self.session_ended = False

        self.log_queue = queue.Queue(maxsize=50)
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.worker_thread.start()

    def _init_events_csv(self):
        """Ensure events.csv exists with the updated session-based header."""
        if not os.path.exists(self.events_path) or os.path.getsize(self.events_path) == 0:
            try:
                with open(self.events_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Session_ID", "Alert_Type", "Level", "Duration_Sec", "Snapshot_Path"])
            except Exception as e:
                logger.warning(f"Không khởi tạo được events.csv: {e}")

    def _init_sessions_csv(self):
        """Ensure sessions.csv exists with correct summary header."""
        if not os.path.exists(self.sessions_path) or os.path.getsize(self.sessions_path) == 0:
            try:
                with open(self.sessions_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Session_ID", "Start_Time", "End_Time", "Duration_Sec", "Total_Alerts", "Alerts_Breakdown", "Max_Risk_Score"])
            except Exception as e:
                logger.warning(f"Không khởi tạo được sessions.csv: {e}")

    def update_risk_score(self, score: int):
        """Update maximum risk score observed in this session."""
        if score > self.session_max_risk:
            self.session_max_risk = score

    def log_event(
        self,
        session_id: Optional[str] = None,
        alert_type: str = "Violation",
        level: str = "MEDIUM",
        duration: float = 0.0,
        frame: Optional[np.ndarray] = None,
        risk_score: int = 0,
        driver_name: Optional[str] = None  # Backward-compatible parameter
    ):
        """Queue a violation event to be saved asynchronously."""
        sid = session_id or driver_name or self.session_id
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        snapshot_rel_path = ""

        # Update session metrics
        self.session_alerts[alert_type] = self.session_alerts.get(alert_type, 0) + 1
        if risk_score > self.session_max_risk:
            self.session_max_risk = risk_score

        if frame is not None:
            clean_sid = sid.replace(" ", "_")
            clean_alert = alert_type.replace(" ", "_").replace("/", "_")
            timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"{clean_sid}_{clean_alert}_{timestamp_str}.jpg"
            full_path = os.path.join(self.snapshots_dir, filename)
            snapshot_rel_path = os.path.relpath(full_path, config.PROJECT_ROOT)
            frame_copy = frame.copy()
        else:
            frame_copy = None
            full_path = ""

        try:
            self.log_queue.put_nowait({
                "timestamp": dt_string,
                "session_id": sid,
                "alert_type": alert_type,
                "level": level,
                "duration": round(duration, 2),
                "snapshot_path": snapshot_rel_path,
                "frame_to_save": frame_copy,
                "save_path": full_path
            })
        except queue.Full:
            pass

    def end_session(self, now: Optional[datetime] = None) -> bool:
        """Record session summary row into sessions.csv."""
        if self.session_ended:
            return False

        if now is None:
            now = datetime.now()

        duration_sec = round((now - self.session_start_time).total_seconds(), 1)
        total_alerts = sum(self.session_alerts.values())
        breakdown_str = json.dumps(self.session_alerts, ensure_ascii=False)

        try:
            with open(self.sessions_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.session_id,
                    self.session_start_time.strftime("%d/%m/%Y %H:%M:%S"),
                    now.strftime("%d/%m/%Y %H:%M:%S"),
                    duration_sec,
                    total_alerts,
                    breakdown_str,
                    self.session_max_risk
                ])
            self.session_ended = True
            logger.info(f"Đã lưu tổng kết phiên {self.session_id} vào sessions.csv (Thời lượng={duration_sec}s, Cảnh báo={total_alerts}, MaxRisk={self.session_max_risk})")
            return True
        except Exception as e:
            logger.warning(f"Không thể ghi sessions.csv: {e}")
            return False

    def _log_worker(self):
        """Worker thread to write events and save snapshot images."""
        while self.is_running:
            try:
                event = self.log_queue.get(timeout=0.3)
                # Save snapshot if frame is present
                if event["frame_to_save"] is not None and event["save_path"]:
                    try:
                        cv2.imwrite(event["save_path"], event["frame_to_save"])
                    except Exception:
                        pass

                # Write to CSV
                try:
                    with open(self.events_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            event["timestamp"],
                            event["session_id"],
                            event["alert_type"],
                            event["level"],
                            event["duration"],
                            event["snapshot_path"]
                        ])
                except Exception:
                    pass

                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.05)

    def stop(self):
        """Safely terminate worker thread and end session."""
        self.end_session()
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.8)
