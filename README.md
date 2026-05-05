# Hệ thống nhận diện khuôn mặt tích hợp điểm danh tự động, cảnh báo buồn ngủ, mất tập trung.

## 1. Cách dùng
*   **B1**: Cài đặt các thư viện cần thiết từ file requirements.txt
    ```bash
    pip install -r requirements.txt
    ```
*   **B2**: Chạy file `main.py` để bắt đầu sử dụng hệ thống
    ```bash
    python main.py
    ```

## 2. Cấu trúc source code
*   **main.py**: Trung tâm điều khiển, quản lý menu chính của ứng dụng.
*   **build_dataset.py**: Thu thập ảnh khuôn mặt tự động.
*   **encode_faces.py**: Train và lưu vào file encodings.pickle.
*   **realtime_monitoring.py**: Nhận diện khuôn mặt, cảnh báo buồn ngủ và mất tập trung.
*   **attendance.csv**: File lưu lịch sử điểm danh nhân viên.
*   **encodings.pickle**: File chứa dữ liệu khuôn mặt đã được huấn luyện.
*   **face_landmarker.task**: Model dùng để theo dõi các điểm mốc trên mặt.
*   **hand_landmarker.task**: Model dùng để theo dõi bàn tay.
