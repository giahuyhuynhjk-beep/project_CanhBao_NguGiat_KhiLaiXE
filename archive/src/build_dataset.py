import cv2
import os
import sys
import numpy as np
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Đảm bảo đường dẫn luôn đúng dù chạy từ đâu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Import config for path resolution and UTF-8 console output
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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

class DatasetCollector:
    def __init__(self, name, user_id):
        self.name = name
        self.user_id = user_id
        self.folder_path = os.path.join("dataset", f"{name}_{user_id}")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"[INFO] Đã tạo thư mục: {self.folder_path}")

        # Khởi tạo MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Định nghĩa các bước thu thập
        self.steps = [
            {"id": "STRAIGHT", "prompt": "Nhin THANG vao camera", "target_count": 20},
            {"id": "LEFT", "prompt": "Quay mat sang TRAI", "target_count": 15},
            {"id": "RIGHT", "prompt": "Quay mat sang PHAI", "target_count": 15},
            {"id": "UP", "prompt": "Nguoc mat len TREN", "target_count": 15},
            {"id": "DOWN", "prompt": "Cui mat xuong DUOI", "target_count": 15},
        ]
        self.current_step_idx = 0
        self.captured_count = 0
        self.total_captured = 0
        self.last_capture_time = 0

    def get_pose(self, landmarks):
        """Ước lượng tư thế đầu dựa trên các điểm mốc (landmarks)."""
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        top_head = landmarks[10]
        chin = landmarks[152]

        eye_dist = right_eye.x - left_eye.x
        if eye_dist == 0: return "UNKNOWN"
        yaw_ratio = (nose.x - left_eye.x) / eye_dist

        face_height = chin.y - top_head.y
        if face_height == 0: return "UNKNOWN"
        pitch_ratio = (nose.y - top_head.y) / face_height

        if yaw_ratio < 0.35: return "RIGHT"
        if yaw_ratio > 0.65: return "LEFT"
        if pitch_ratio < 0.45: return "UP"
        if pitch_ratio > 0.65: return "DOWN"
        
        if 0.4 <= yaw_ratio <= 0.6 and 0.5 <= pitch_ratio <= 0.6:
            return "STRAIGHT"
        
        return "CENTERING"

    def draw_guide(self, frame, pose_detected, is_correct_pose):
        if self.current_step_idx >= len(self.steps):
            return

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        rx, ry = 210, 270 

        color = (0, 255, 0) if is_correct_pose else (0, 165, 255)
        cv2.ellipse(frame, (center_x, center_y), (rx, ry), 0, 0, 360, color, 2)

        step = self.steps[self.current_step_idx]
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, step["prompt"], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        progress_text = f"Tien do: {self.captured_count}/{step['target_count']}"
        cv2.putText(frame, progress_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        status_color = (0, 255, 0) if is_correct_pose else (0, 0, 255)
        cv2.putText(frame, f"Pose: {pose_detected}", (w - 180, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if not is_correct_pose and pose_detected != "UNKNOWN":
            msg = "Hay giu dung tu the!"
            cv2.putText(frame, msg, (center_x - 100, center_y + ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Không thể mở webcam.")
            return

        while self.current_step_idx < len(self.steps):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            pose_detected = "NONE"
            is_correct_pose = False
            current_step = self.steps[self.current_step_idx]

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                pose_detected = self.get_pose(landmarks)
                
                if pose_detected == current_step["id"]:
                    is_correct_pose = True
                    if time.time() - self.last_capture_time > CAPTURE_DELAY:
                        self.save_image(frame, landmarks)
                        self.captured_count += 1
                        self.total_captured += 1
                        self.last_capture_time = time.time()

                        if self.captured_count >= current_step["target_count"]:
                            print(f"[SUCCESS] Hoàn thành bước: {current_step['id']}")
                            self.current_step_idx += 1
                            self.captured_count = 0
                            time.sleep(1) 

            self.draw_guide(frame, pose_detected, is_correct_pose)
            cv2.imshow("Thu thap du lieu tu dong - Nhan 'Q' de thoat", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_image(self, frame, landmarks):
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x1, y1 = int(min(x_coords) * w), int(min(y_coords) * h)
        x2, y2 = int(max(x_coords) * w), int(max(y_coords) * h)
        padding = 30
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        final_image = apply_realistic_augmentation(gray_face)
        file_name = f"{self.name}_{self.user_id}_{self.total_captured}.jpg"
        file_path = os.path.join(self.folder_path, file_name)
        cv2.imwrite(file_path, final_image)

def main():
    name = input("Nhap ten nhan vien: ").strip()
    user_id = input("Nhap ID nhan vien: ").strip()
    if not name or not user_id: return
    collector = DatasetCollector(name, user_id)
    collector.run()

if __name__ == "__main__":
    main()
