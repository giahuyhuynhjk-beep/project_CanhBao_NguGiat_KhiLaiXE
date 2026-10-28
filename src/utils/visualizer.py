"""
Visualizer module for Driver Monitoring System (DOAN_CNPM_NHANDIENTAIXE).
Provides high-performance UI overlay, face bounding boxes, alerts, telemetry HUD,
and Steering Wheel ROI with full accented Vietnamese (Tiếng Việt có dấu) via Pillow.
Includes bitmap caching to maintain high FPS and supports headless mode / toggle.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import src.config as config
from src.utils.logger import get_logger

logger = get_logger("Visualizer")


class Visualizer:
    """
    Renders DMS visual interfaces onto OpenCV video frames.
    Supports accented Vietnamese via Pillow with an LRU-like text bitmap cache
    to avoid expensive full-frame PIL conversions on every frame.
    """

    def __init__(self, font_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.font_path = self._resolve_font_path(font_path)
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._text_cache: Dict[Tuple[str, int, Tuple[int, int, int], Optional[Tuple[int, int, int]], bool], Tuple[np.ndarray, np.ndarray, int, int]] = {}
        self._max_cache_size = 500

        # State caching for alert banners
        self._last_alert_keys: List[str] = []
        self._cached_alert_overlay: Optional[np.ndarray] = None
        self._cached_alert_pos: Tuple[int, int, int, int] = (0, 0, 0, 0)

        logger.info(f"Visualizer khởi tạo: Font='{self.font_path}', Bật={self.enabled}")

    def _resolve_font_path(self, font_path: Optional[str] = None) -> str:
        """Resolve a usable TrueType font path that supports Vietnamese accents."""
        candidates = [
            font_path,
            str(config.ASSETS_DIR / "fonts" / "arial.ttf"),
            str(config.ASSETS_DIR / "fonts" / "segoeui.ttf"),
            "assets/fonts/arial.ttf",
            "assets/fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return "arial.ttf"

    def _get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """Get or create cached PIL ImageFont instance for specified size."""
        if font_size not in self._font_cache:
            try:
                self._font_cache[font_size] = ImageFont.truetype(self.font_path, font_size)
            except Exception:
                try:
                    self._font_cache[font_size] = ImageFont.load_default()
                except Exception:
                    pass
        return self._font_cache.get(font_size)

    def draw_vietnamese_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        font_size: int = 16,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[Tuple[int, int, int]] = None,
        with_shadow: bool = False
    ) -> Tuple[int, int]:
        """
        Draw accented Vietnamese text onto frame at (x, y) using Pillow text cache.
        Returns (text_width, text_height).
        """
        if not self.enabled or frame is None or not text:
            return 0, 0

        cache_key = (text, font_size, color, bg_color, with_shadow)
        if cache_key not in self._text_cache:
            # Enforce max cache size
            if len(self._text_cache) > self._max_cache_size:
                self._text_cache.clear()

            font = self._get_font(font_size)
            # Create scratch draw to measure text size
            dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = max(1, bbox[2] - bbox[0] + 6)
            text_h = max(1, bbox[3] - bbox[1] + 6)

            # Render text onto transparent RGBA patch
            patch_img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
            p_draw = ImageDraw.Draw(patch_img)

            # Optional background box
            if bg_color is not None:
                p_draw.rectangle([0, 0, text_w, text_h], fill=(bg_color[2], bg_color[1], bg_color[0], 210))

            # Optional drop shadow for readability
            if with_shadow:
                p_draw.text((3, 3), text, font=font, fill=(0, 0, 0, 240))

            # Main text (PIL uses RGB, OpenCV uses BGR)
            rgb_color = (color[2], color[1], color[0], 255)
            p_draw.text((2, 2), text, font=font, fill=rgb_color)

            patch_np = np.array(patch_img)
            bgr_patch = patch_np[:, :, :3][:, :, ::-1]  # RGB to BGR
            alpha = (patch_np[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

            self._text_cache[cache_key] = (bgr_patch, alpha, text_w, text_h)

        bgr_patch, alpha, text_w, text_h = self._text_cache[cache_key]

        # Fast alpha blending onto target frame ROI
        x, y = pos
        fh, fw, _ = frame.shape
        if x >= fw or y >= fh:
            return text_w, text_h

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + text_w)
        y2 = min(fh, y + text_h)

        if x2 <= x1 or y2 <= y1:
            return text_w, text_h

        px1 = x1 - x
        py1 = y1 - y
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)

        sub_patch = bgr_patch[py1:py2, px1:px2]
        sub_alpha = alpha[py1:py2, px1:px2]
        roi = frame[y1:y2, x1:x2].astype(np.float32)

        # Vectorized alpha composite
        blended = (roi * (1.0 - sub_alpha) + sub_patch.astype(np.float32) * sub_alpha).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

        return text_w, text_h

    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        name: Optional[str] = None,
        is_calibrating: bool = False,
        calibration_progress: float = 0.0
    ):
        """Draw face rectangle, driver status badge, and calibration progress."""
        if not self.enabled or frame is None:
            return

        l, t, r, b = bbox
        color = (0, 255, 0)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        # Anonymous driver status label in Vietnamese
        label_text = f"Tài xế: {name}" if name else "Tài xế: Đang giám sát"
        self.draw_vietnamese_text(frame, label_text, (l + 2, max(5, t - 26)),
                                 font_size=15, color=color, bg_color=(20, 20, 20), with_shadow=True)

        # Calibration badge
        if is_calibrating:
            calib_text = f"Đang hiệu chuẩn: {int(calibration_progress * 100)}%"
            self.draw_vietnamese_text(frame, calib_text, (l + 2, b + 6),
                                     font_size=13, color=(0, 255, 255), bg_color=(30, 30, 30))
            # Mini progress bar below face
            bar_w = r - l
            prog_w = int(bar_w * calibration_progress)
            cv2.rectangle(frame, (l, b + 26), (r, b + 32), (50, 50, 50), -1)
            cv2.rectangle(frame, (l, b + 26), (l + prog_w, b + 32), (0, 255, 255), -1)

    def draw_roi(self, frame: np.ndarray, roi: Dict[str, float], steering_status: str = "OK"):
        """
        Draw Steering Wheel ROI in accented Vietnamese:
        - Green if status == 'OK'
        - Yellow if status == 'NO_HANDS'
        - Red if status == 'OUT_OF_ROI'
        """
        if not self.enabled or frame is None:
            return

        h, w, _ = frame.shape
        x1 = int(roi["x_min"] * w)
        y1 = int(roi["y_min"] * h)
        x2 = int(roi["x_max"] * w)
        y2 = int(roi["y_max"] * h)

        if steering_status == "OUT_OF_ROI":
            color = (0, 0, 255)  # Red
            status_text = "ROI: NGOÀI VÙNG TAY LÁI!"
        elif steering_status == "NO_HANDS":
            color = (0, 200, 255)  # Yellow
            status_text = "ROI: KHÔNG THẤY TAY"
        else:
            color = (0, 255, 0)  # Green
            status_text = "ROI: Tay trên vô-lăng"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.draw_vietnamese_text(frame, status_text, (x1 + 6, max(10, y1 - 22)),
                                 font_size=14, color=color, bg_color=(20, 20, 20), with_shadow=True)

    def draw_wrists(self, frame: np.ndarray, wrists: List[Tuple[int, int]]):
        """Draw indicators at detected wrist positions."""
        if not self.enabled or frame is None:
            return
        for (wx, wy) in wrists:
            cv2.circle(frame, (wx, wy), 8, (255, 0, 255), -1)
            cv2.circle(frame, (wx, wy), 10, (255, 255, 255), 1)

    def draw_alerts(self, frame: np.ndarray, alerts: List[str]):
        """
        Render active alert banners vertically on the top-left corner with caching.
        Only recalculates when the alert set changes to keep FPS high.
        """
        if not self.enabled or frame is None or not alerts:
            return

        y_pos = 25
        for alert_msg in alerts:
            # Map legacy ascii to accented Vietnamese if needed
            clean_msg = alert_msg
            if "CANH BAO" in clean_msg:
                clean_msg = clean_msg.replace("CANH BAO", "CẢNH BÁO")
            if "NGU GAT" in clean_msg:
                clean_msg = clean_msg.replace("NGU GAT", "NGỦ GẬT")
            if "NGAP LIEN TUC" in clean_msg:
                clean_msg = clean_msg.replace("NGAP LIEN TUC", "NGÁP LIÊN TỤC")
            if "ROI TAY" in clean_msg:
                clean_msg = clean_msg.replace("ROI TAY", "RỜI TAY KHỎI VÔ-LĂNG")

            self.draw_vietnamese_text(
                frame,
                clean_msg,
                (15, y_pos),
                font_size=18,
                color=(0, 0, 255),
                bg_color=(20, 20, 20),
                with_shadow=True
            )
            y_pos += 34

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        ear: float,
        ear_thresh: float,
        mar: float,
        yaw: float,
        pitch: float,
        perclos: float,
        yawns_count: int,
        state_level: str = "NORMAL",
        risk_score: int = 0,
        shoulder_tilt: float = 0.0,
        is_sunglasses: bool = False,
        is_low_light: bool = False
    ):
        """Draw telemetry metrics dashboard on top-right of the frame with Vietnamese labels."""
        if not self.enabled or frame is None:
            return

        h, w, _ = frame.shape
        panel_w = 270
        panel_h = 255

        # Semi-transparent dark background for HUD
        x_start = w - panel_w - 10
        y_start = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (w - 10, y_start + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        px = x_start + 10
        py = y_start + 8

        def put_hud_line(label: str, value: str, alert_cond: bool = False, custom_color: Optional[Tuple[int, int, int]] = None):
            nonlocal py
            color = custom_color if custom_color else ((0, 0, 255) if alert_cond else (0, 255, 200))
            text = f"{label}: {value}"
            self.draw_vietnamese_text(frame, text, (px, py), font_size=13, color=color)
            py += 21

        # Status level translation
        level_map = {
            "NORMAL": ("BÌNH THƯỜNG", (0, 255, 0)),
            "WARNING": ("CẢNH BÁO", (0, 180, 255)),
            "ALARM": ("BÁO ĐỘNG", (0, 0, 255))
        }
        status_str, status_color = level_map.get(state_level, (state_level, (0, 255, 0)))

        put_hud_line("TRẠNG THÁI", status_str, custom_color=status_color)
        put_hud_line("Điểm rủi ro", f"{risk_score}/100", alert_cond=risk_score >= config.RISK_THRESHOLD_WARNING)
        put_hud_line("Tốc độ xử lý", f"{fps:.1f} FPS")

        # Lighting & Eye mode badges
        if is_low_light:
            put_hud_line("Ánh sáng", "Thiếu sáng (CLAHE+Gamma)", custom_color=(0, 255, 255))
        if is_sunglasses:
            put_hud_line("Chế độ mắt", "Kính râm / Bị che", custom_color=(255, 180, 0))
        else:
            put_hud_line("EAR", f"{ear:.3f} (ngưỡng: {ear_thresh:.2f})", alert_cond=ear < ear_thresh)
            put_hud_line("PERCLOS", f"{perclos * 100:.1f}%", alert_cond=perclos >= config.PERCLOS_THRESHOLD)

        put_hud_line("MAR", f"{mar:.2f}", alert_cond=mar >= config.MAR_THRESHOLD)
        put_hud_line("Ngáp gần đây", f"{yawns_count}/phút", alert_cond=yawns_count >= config.YAWN_PER_MINUTE_LIMIT)
        is_distracted = (abs(yaw) > config.YAW_THRESHOLD_DEGREE or
                         pitch > config.PITCH_UP_THRESHOLD_DEGREE or
                         pitch < -config.PITCH_DOWN_THRESHOLD_DEGREE)
        put_hud_line("Góc đầu", f"Y:{yaw:+.0f}° P:{pitch:+.0f}°", alert_cond=is_distracted)
        put_hud_line("Góc vai", f"{shoulder_tilt:.1f}°", alert_cond=shoulder_tilt >= config.SHOULDER_TILT_THRESHOLD_DEGREE)

    def draw_objects(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Draw bounding boxes for detected objects (phone, smoke, drink) with Vietnamese tags."""
        if not self.enabled or frame is None:
            return

        label_map = {
            "phone": "ĐIỆN THOẠI",
            "cell": "ĐIỆN THOẠI",
            "mobile": "ĐIỆN THOẠI",
            "smoke": "HÚT THUỐC",
            "cigarette": "HÚT THUỐC",
            "drink": "UỐNG NƯỚC",
            "bottle": "UỐNG NƯỚC",
            "cup": "UỐNG NƯỚC"
        }

        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            raw_label = str(det.get("label", "Object")).lower()
            conf = det.get("conf", 0.0)
            is_near = det.get("is_near_driver", False)

            vn_label = label_map.get(raw_label, raw_label.upper())

            if is_near:
                box_color = (0, 165, 255)  # Orange for objects near driver
                thickness = 2
                tag = f"{vn_label} {conf:.2f} [GẦN]"
            else:
                box_color = (120, 120, 120)  # Gray for objects far from driver
                thickness = 1
                tag = f"{vn_label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            self.draw_vietnamese_text(frame, tag, (x1, max(5, y1 - 20)),
                                     font_size=13, color=box_color, bg_color=(20, 20, 20))
