"""
Image preprocessor module for Driver Monitoring System.
Handles adaptive low-light enhancement (CLAHE + Gamma Correction) and affine tilt normalization.
"""

import random
from typing import Tuple, Optional
import cv2
import numpy as np
import src.config as config


def apply_realistic_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Áp dụng các phép tăng cường dữ liệu thực tế (Realistic Data Augmentation):
    1. Đổi độ tương phản ngẫu nhiên (alpha trong [0.85, 1.25])
    2. Đổi độ sáng ngẫu nhiên (beta trong [-20, 20])
    3. Làm mờ nhẹ ngẫu nhiên (Gaussian blur 3x3) với xác suất 35%
    4. Lật ngang ngẫu nhiên (Horizontal flip) với xác suất 50%
    """
    augmented = image.copy()

    # 1. Độ sáng & Độ tương phản
    alpha = random.uniform(0.85, 1.25)
    beta = random.uniform(-20, 20)
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    # 2. Làm mờ nhẹ
    if random.random() < 0.35:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    # 3. Lật ngang
    if random.random() < 0.50:
        augmented = cv2.flip(augmented, 1)

    return augmented


class ImagePreprocessor:
    def __init__(self, low_light_thresh: float = config.LOW_LIGHT_THRESHOLD):
        self.low_light_thresh = low_light_thresh
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE
        )
        # Precompute gamma lookup table
        self.gamma_lut = self._build_gamma_lut(config.GAMMA_CORRECTION_VALUE)

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / max(0.1, gamma)
        return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def compute_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of a BGR frame."""
        if frame is None or frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply precomputed gamma correction."""
        return cv2.LUT(frame, self.gamma_lut)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization in YCrCb color space."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_clahe = self.clahe.apply(y)
        merged = cv2.merge((y_clahe, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Conditionally enhance frame if brightness is below low-light threshold.
        Returns: (output_frame, is_low_light_active)
        """
        brightness = self.compute_brightness(frame)
        if brightness < self.low_light_thresh:
            enhanced = self.apply_clahe(frame)
            gamma_enhanced = self.apply_gamma(enhanced)
            return gamma_enhanced, True
        return frame, False

    def align_face_affine(self, frame: np.ndarray, angle_deg: float = 0.0, center: Optional[Tuple[int, int]] = None, force: bool = False, **kwargs) -> np.ndarray:
        """
        Rotate frame by -angle_deg to horizontally level tilted shoulders/head.
        Applied when shoulder tilt exceeds threshold and affine normalization is enabled.
        """
        deg = kwargs.get("angle", angle_deg)
        if not force and (not config.ENABLE_AFFINE_NORMALIZATION or abs(deg) < 15.0):
            return frame

        h, w = frame.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        rot_mat = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        return cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# Backward-compatible alias
Preprocessor = ImagePreprocessor

