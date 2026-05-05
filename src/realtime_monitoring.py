import cv2
import face_recognition
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import argparse
import threading
import csv
import os
import sys
from datetime import datetime
from scipy.spatial import distance as dist

# Đảm bảo đường dẫn luôn đúng dù chạy từ đâu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

class RealtimeMonitor:
    def __init__(self, encodings_path="model/encodings.pickle", detection_method="hog"):
        # Fallback for encodings path
        if not os.path.exists(encodings_path) and os.path.exists("encodings.pickle"):
             encodings_path = "encodings.pickle"
             
        print("[INFO] Đang tải dữ liệu mã hóa khuôn mặt...")
        with open(encodings_path, "rb") as f:
            self.known_data = pickle.loads(f.read())
        
        self.detection_method = detection_method
        
        # Paths for MediaPipe models
        face_model = 'model/face_landmarker.task' if os.path.exists('model/face_landmarker.task') else 'face_landmarker.task'
        hand_model = 'model/hand_landmarker.task' if os.path.exists('model/hand_landmarker.task') else 'hand_landmarker.task'

        if not os.path.exists(face_model):
            print(f"[ERROR] Không tìm thấy file model: {face_model}")
        
        base_options_face = python.BaseOptions(model_asset_path=face_model)
        options_face = vision.FaceLandmarkerOptions(base_options=base_options_face, num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)
        
        base_options_hand = python.BaseOptions(model_asset_path=hand_model)
        options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        # Thresholds and Limits
        self.EAR_THRESHOLD = 0.30
        self.DROWSY_TIME_LIMIT = 1.0  # Cảnh báo sau 2 giây nhắm mắt
        self.DISTRACTED_TIME_LIMIT = 2.0
        self.HANDS_OFF_TIME_LIMIT = 2.0
        
        # States
        self.eye_closed_start_time = None
        self.distracted_start_time = None
        self.hands_off_start_time = None
        self.last_results = ([], [])
        self.attendance_list = set()
        
        # Threading for Face Recognition
        self.is_running = True
        self.frame_to_process = None
        self.recognition_thread = threading.Thread(target=self._face_recognition_worker, daemon=True)
        self.recognition_thread.start()

        # Eye landmarks indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _face_recognition_worker(self):
        while self.is_running:
            if self.frame_to_process is not None:
                rgb_small = self.frame_to_process
                self.frame_to_process = None # Clear it
                
                TOLERANCE = 0.45
                boxes = face_recognition.face_locations(rgb_small, model=self.detection_method)
                names = []
                
                if len(boxes) > 0:
                    encodings = face_recognition.face_encodings(rgb_small, boxes)
                    for encoding in encodings:
                        face_distances = face_recognition.face_distance(self.known_data["encodings"], encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] <= TOLERANCE:
                            name = self.known_data["names"][best_match_index]
                        else:
                            name = "Unknown"
                        names.append(name)
                
                self.last_results = (boxes, names)
            else:
                time.sleep(0.01)

    def calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append((lm.x * img_w, lm.y * img_h))
        pts = np.array(pts)
        v1 = dist.euclidean(pts[1], pts[5])
        v2 = dist.euclidean(pts[2], pts[4])
        h1 = dist.euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h1)

    def check_head_pose(self, landmarks):
        nose = landmarks[1]
        return not (nose.y > 0.7 or nose.x < 0.35 or nose.x > 0.65)

    def log_attendance(self, name):
        if name != "Unknown" and name != "Scanning..." and name not in self.attendance_list:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            try:
                # Tạo thư mục report nếu chưa có
                if not os.path.exists("report"):
                    os.makedirs("report")
                
                with open("report/attendance.csv", "a", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, dt_string])
                self.attendance_list.add(name)
                print(f"[INFO] Đã điểm danh: {name} lúc {dt_string}")
            except Exception as e:
                print(f"[ERROR] Không thể ghi file điểm danh: {e}")

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Send frame to recognition thread if it's ready
        if self.frame_to_process is None:
            self.frame_to_process = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        # Resize for MediaPipe once
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
        
        results_face = self.face_landmarker.detect(mp_image)
        results_hand = self.hand_landmarker.detect(mp_image)
        
        drowsy_alert = distract_alert = hands_off_alert = False
        ear = 0.0
        face_rect = None 

        if results_face.face_landmarks:
            landmarks = results_face.face_landmarks[0]
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            face_rect = (int(min(x_coords)*w), int(min(y_coords)*h), int(max(x_coords)*w), int(max(y_coords)*h))
            
            # EAR for drowsiness
            ear = (self.calculate_ear(landmarks, self.LEFT_EYE, w, h) + self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)) / 2.0
            
            if ear < self.EAR_THRESHOLD:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = time.time()
            else:
                self.eye_closed_start_time = None
            
            # Head pose for distraction
            if not self.check_head_pose(landmarks):
                if self.distracted_start_time is None:
                    self.distracted_start_time = time.time()
            else:
                self.distracted_start_time = None
        
        # Kiểm tra cảnh báo dựa trên bộ đếm thời gian
        if self.eye_closed_start_time and (time.time() - self.eye_closed_start_time > self.DROWSY_TIME_LIMIT):
            drowsy_alert = True
            
        if self.distracted_start_time and (time.time() - self.distracted_start_time > self.DISTRACTED_TIME_LIMIT):
            distract_alert = True

        # Kiểm tra tay lái (Yêu cầu cả 2 tay)
        if not results_hand.hand_landmarks or len(results_hand.hand_landmarks) < 2:
            if self.hands_off_start_time is None:
                self.hands_off_start_time = time.time()
        else:
            self.hands_off_start_time = None

        if self.hands_off_start_time and (time.time() - self.hands_off_start_time > self.HANDS_OFF_TIME_LIMIT):
            hands_off_alert = True

        # Display results
        boxes, names = self.last_results
        
        name = "Scanning..."
        if len(names) > 0:
            name = names[0]
            
        if face_rect:
            (l, t, r, b) = face_rect
            padding = 20
            l, t, r, b = max(0, l-padding), max(0, t-padding), min(w, r+padding), min(h, b+padding)
            color = (0, 255, 0) if name != "Unknown" and name != "Scanning..." else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, name, (l + 5, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.log_attendance(name)

        # Draw Alerts
        if drowsy_alert:
            cv2.putText(frame, "CANH BAO: BUON NGU!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if distract_alert:
            cv2.putText(frame, "CANH BAO: MAT TAP TRUNG!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if hands_off_alert:
            cv2.putText(frame, "CANH BAO: BUONG TAY!", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        return frame

    def stop(self):
        self.is_running = False
        if self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", default="model/encodings.pickle")
    args = vars(ap.parse_args())
    
    monitor = RealtimeMonitor(encodings_path=args["encodings"])
    vs = cv2.VideoCapture(0)
    
    if not vs.isOpened():
        print("[ERROR] Không thể mở camera.")
        return

    print("[INFO] Bắt đầu giám sát. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = vs.read()
        if not ret:
            break
            
        processed_frame = monitor.process_frame(frame)
        cv2.imshow("Driver Monitoring System", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    print("[INFO] Đang dừng hệ thống...")
    monitor.stop()
    vs.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát thành công.")

if __name__ == "__main__":
    main()
