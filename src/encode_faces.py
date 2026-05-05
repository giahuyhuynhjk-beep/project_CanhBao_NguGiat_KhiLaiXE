import os
import sys
import pickle
import cv2
import face_recognition
import argparse
from pathlib import Path

# Đảm bảo đường dẫn luôn đúng dù chạy từ đâu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def encode_faces(dataset_path="dataset", encodings_path="model/encodings.pickle", detection_method="hog"):
    if not os.path.exists(os.path.dirname(encodings_path)):
        os.makedirs(os.path.dirname(encodings_path))
        
    if os.path.exists(encodings_path):
        print(f"[WARNING] File '{encodings_path}' da ton tai.")
        ans = input("Ban co muon tiep tuc train lai de cap nhat du lieu moi khong? (y/n): ").strip().lower()
        if ans != 'y': return

    print("[INFO] Đang bắt đầu quá trình trích xuất đặc trưng khuôn mặt...")
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(list(Path(dataset_path).rglob(ext)))
    
    if len(image_paths) == 0:
        print("[ERROR] Không tìm thấy ảnh nào trong thư mục dataset!")
        return

    known_encodings = []
    known_names = []

    for (i, image_path) in enumerate(image_paths):
        image_path_str = str(image_path)
        name = image_path.parent.name
        image = cv2.imread(image_path_str)
        if image is None: continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    print(f"[INFO] Đang lưu {len(known_encodings)} mã hóa vào ổ đĩa...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        f.write(pickle.dumps(data))
    print(f"[HOÀN THÀNH] Đã trích xuất và lưu vào {encodings_path} thành công!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", default="dataset")
    ap.add_argument("-e", "--encodings", default="model/encodings.pickle")
    ap.add_argument("-d", "--detection-method", type=str, default="hog")
    args = vars(ap.parse_args())
    encode_faces(dataset_path=args["dataset"], encodings_path=args["encodings"], detection_method=args["detection_method"])
