# HỆ THỐNG GIÁM SÁT TRẠNG THÁI TÀI XẾ (DRIVER MONITORING SYSTEM - DMS)

Hệ thống thị giác máy tính tích hợp trí tuệ nhân tạo (AI/Computer Vision) phục vụ giám sát an toàn người lái xe theo thời gian thực: phát hiện buồn ngủ (EAR & PERCLOS), ngáp liên tục (MAR), quay đầu lệch hướng (Head Pose 3D), buông tay khỏi vô-lăng (Steering Wheel ROI), gục người/lệch vai (Pose Slump), và các hành vi gây xao nhãng bằng YOLO (dùng điện thoại, hút thuốc, uống nước). Hệ thống tập trung hoàn toàn vào giám sát hành vi theo phiên làm việc ẩn danh (session-based), tối ưu hóa tải CPU và loại bỏ hoàn toàn các phụ thuộc nhận diện danh tính nặng nề (`dlib`/`face_recognition`).

---

## 1. Yêu cầu Hệ thống & Cài đặt trên Windows

### 1.1. Phiên bản Python khuyến nghị
*   **Python 3.10.x trở lên (64-bit)** (ví dụ: Python 3.10.11 hoặc 3.11).

### 1.2. Cài đặt các thành phần cốt lõi

1.  **Cài đặt PyTorch & Ultralytics (YOLO):**
    *   Dành cho máy tính dùng CPU:
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install ultralytics==8.4.183
        ```
    *   Dành cho máy tính có GPU NVIDIA (CUDA):
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        pip install ultralytics==8.4.183
        ```

2.  **Cài đặt toàn bộ phụ thuộc từ file đã ghim phiên bản:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Cách Chạy Các Chế Độ của Hệ Thống

### 2.1. Menu Điều khiển Trung tâm
Chạy menu chính quản lý toàn bộ tính năng:
```bash
python src/main.py
```
Menu cung cấp 4 chức năng tinh gọn:
1.  **Chạy giám sát thời gian thực (GUI):** Hiển thị màn hình theo dõi đầy đủ với HUD, thông số vi phạm và cảnh báo bằng giọng nói/âm thanh.
2.  **Chạy giám sát chế độ ẩn (Headless):** Chạy nền không mở cửa sổ đồ họa, tối ưu CPU tối đa, ghi log sự kiện đầy đủ.
3.  **Hiệu chuẩn vùng vô-lăng (ROI):** Dùng chuột kéo thả định vị vùng vô-lăng trên khung hình camera.
4.  **Thoát.**

### 2.2. Chạy Trực tiếp Module Giám sát Thời gian thực
*   **Chạy chế độ hiển thị thông thường:**
    ```bash
    python src/realtime_monitoring.py
    ```
*   **Chạy chế độ Headless (Dành cho thiết bị nhúng / server / chạy nền):**
    ```bash
    python src/realtime_monitoring.py --headless
    ```
*   **Tắt hiệu ứng vẽ UI để tối đa hóa FPS:**
    ```bash
    python src/realtime_monitoring.py --no-visualizer
    ```
*   **Chỉ định camera hoặc file video:**
    ```bash
    python src/realtime_monitoring.py -c 0
    python src/realtime_monitoring.py -c "path/to/video.mp4"
    ```

### 2.3. Chạy Kiểm thử Tự động (pytest)
Toàn bộ 35 unit test tự động (bao gồm kiểm thử dữ liệu giả lập cho EAR, MAR, Pose, ROI, YOLO, State Machine) có thể được thực thi bằng lệnh:
```bash
pytest -v
```

---

## 3. Cấu hình & Chỉnh sửa Ngưỡng (Thresholds Tuning)

Tất cả các ngưỡng số học và thời gian được cấu hình tập trung trong file [`src/config.py`](file:///c:/project11/src/config.py):

| Thông số trong `src/config.py` | Giá trị mặc định | Diễn giải & Hướng dẫn tinh chỉnh |
|---|---|---|
| `EAR_THRESHOLD_DEFAULT` | `0.22` | Ngưỡng tỷ lệ mắt mở (EAR). Nếu mắt mở bé/nheo, giảm xuống `0.20`. Nếu mắt to tròn, tăng lên `0.25`. |
| `DROWSY_TIME_LIMIT` | `1.5s` | Số giây nhắm mắt liên tục trước khi kích hoạt cảnh báo **🔴 Rất cao (BÁO ĐỘNG)**. |
| `PERCLOS_THRESHOLD` | `0.35` (35%) | Tỷ lệ thời gian mắt nhắm trong cửa sổ 60 giây. Đánh giá buồn ngủ tích lũy. |
| `MAR_THRESHOLD` | `0.60` | Ngưỡng tỷ lệ độ mở miệng (MAR). Lọc bỏ nói chuyện thông thường (MAR ~ 0.35). |
| `YAWN_PER_MINUTE_LIMIT` | `3` lần/phút | Ngưỡng số lần ngáp trong 60 giây để kích hoạt cảnh báo mệt mỏi. |
| `YAW_THRESHOLD_DEGREE` | `30.0°` | Góc quay đầu sang trái hoặc phải vượt quá 30 độ xem như quay đi chỗ khác. |
| `PITCH_UP_THRESHOLD_DEGREE` | `25.0°` | Ngưỡng ngửa đầu nhìn trần xe. |
| `PITCH_DOWN_THRESHOLD_DEGREE` | `20.0°` | Ngưỡng cúi gập đầu / gục mặt. |
| `DISTRACTED_TIME_LIMIT` | `2.5s` | Số giây quay đầu lệch hướng liên tục trước khi cảnh báo. |
| `HANDS_OUT_OF_ROI_LIMIT` | `3.0s` | Số giây cả hai cổ tay nằm ngoài vùng vô-lăng trước khi báo động rời tay. |
| `HANDS_OFF_TIME_LIMIT` | `5.0s` | Số giây không thấy bàn tay (camera bị che hoặc góc khuất) trước khi cảnh báo. |
| `SHOULDER_TILT_THRESHOLD_DEGREE` | `15.0°` | Góc nghiêng đường nối 2 vai so với phương ngang. |
| `SHOULDER_TILT_TIME_LIMIT` | `3.0s` | Thời gian duy trì tư thế ngồi vẹo / gục vai trước khi nhắc nhở. |
| `ENABLE_OBJECT_DETECTION` | `False/True` | Bật/tắt mô hình YOLO phát hiện đồ vật gây xao nhãng. |

---

## 4. Quy trình Hiệu chuẩn (Calibration)

### 4.1. Hiệu chuẩn Mắt Tự động theo Phiên (Session-based Calibration)
*   **Tự động kích hoạt:** Khi hệ thống khởi động và phát hiện khuôn mặt ổn định, chế độ hiệu chuẩn sẽ tự động chạy trong **4.0 giây đầu tiên**.
*   **Thanh tiến trình:** Màn hình hiển thị thanh tiến trình trực quan: `Đang hiệu chuẩn: xx%`. Tài xế chỉ cần nhìn thẳng về phía trước trong điều kiện mở mắt tự nhiên.
*   **Cơ chế ngưỡng thích ứng:** Ngưỡng EAR cá nhân hóa được tính bằng $72\%$ mức mắt mở cơ sở của tài xế (được giới hạn an toàn trong khoảng `[0.18, 0.26]`), loại bỏ hoàn toàn hiện tượng báo động giả khi nheo mắt nhẹ hoặc mắt một mí.
*   **Phím tắt hiệu chuẩn lại (`c` / `C`):** Người dùng có thể nhấn phím **`c`** bất kỳ lúc nào trên cửa sổ giám sát OpenCV để kích hoạt hiệu chuẩn lại ngay lập tức.
*   **Tự động reset khi vắng mặt:** Nếu tài xế rời khỏi ghế lái (mất khuôn mặt) quá `DRIVER_ABSENT_RESET_SEC = 10.0s`, hệ thống sẽ tự động reset state machine và tự động hiệu chuẩn lại ngay khi có tài xế quay lại trước camera.

### 4.2. Hiệu chuẩn Vùng Vô-lăng (Steering Wheel ROI)
1.  Chạy công cụ hiệu chuẩn:
    ```bash
    python src/core/hand_analyzer.py
    ```
2.  Cửa sổ webcam sẽ hiện ra. Sử dụng chuột nhấn giữ và kéo khung hình chữ nhật bao quanh vị trí vô-lăng trên xe.
3.  Nhấn phím **`s`** để lưu tọa độ vào file [`model/steering_roi.json`](file:///c:/project11/model/steering_roi.json).
4.  Nhấn phím **`q`** để thoát.

---

## 5. Huấn luyện & Export Mô hình YOLO Phát hiện Xao nhãng

Hệ thống hỗ trợ phát hiện 3 đối tượng gây xao nhãng: `phone` (điện thoại), `smoke` (hút thuốc), `drink` (uống nước).

### 5.1. Chuẩn bị Dữ liệu
Tạo cấu trúc thư mục theo chuẩn Ultralytics:
```
dataset_distraction/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
Nội dung file `data.yaml`:
```yaml
path: ../dataset_distraction
train: images/train
val: images/val
names:
  0: phone
  1: smoke
  2: drink
```

### 5.2. Huấn luyện với Ultralytics YOLO
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=80 imgsz=640 batch=16 device=0
```

### 5.3. Export & Triển khai vào Hệ thống
1.  Sau khi huấn luyện hoàn tất, copy file trọng số tốt nhất `best.pt` vào thư mục dự án:
    ```bash
    copy runs\detect\train\weights\best.pt model\yolo26n.pt
    ```
2.  Bật cờ nhận diện trong `src/config.py`:
    ```python
    ENABLE_OBJECT_DETECTION = True
    YOLO_MODEL_PATH = MODEL_DIR / "yolo26n.pt"
    ```
3.  *(Tùy chọn)* Export sang định dạng ONNX để tăng tốc:
    ```bash
    yolo export model=model/yolo26n.pt format=onnx
    ```

---

## 6. Báo cáo Hiệu năng & Tốc độ Khung hình (FPS Report)

Nhờ loại bỏ hoàn toàn tính toán nhận diện danh tính (`dlib`/`face_recognition`) nặng nề và áp dụng cơ chế **Bitmap Caching (Pillow LRU Cache)** cho văn bản tiếng Việt có dấu, hệ thống đạt tốc độ khung hình rất cao trên phần cứng CPU thông thường (Intel Core i5 thế hệ 10 trở lên):

| Kịch bản Thực thi | FPS trung bình | Thời gian xử lý mỗi frame | Nhận xét |
|---|---|---|---|
| **Vẽ UI tiếng Việt bằng Pillow (Đã Cache)** | **479.9 FPS** | **~2.08 ms** | Nhanh gấp ~35 lần so với vẽ PIL trực tiếp mỗi frame |
| **Pipeline Toàn diện (Chế độ Headless)** | **62.5 FPS** | **~16.0 ms** | Tối ưu tuyệt đối cho server, thiết bị nhúng và chạy nền |
| **Pipeline Toàn diện (Giao diện đồ họa GUI + HUD)** | **49.8 FPS** | **~20.1 ms** | Mượt mà, hiển thị đầy đủ thông số thời gian thực |
| **Pipeline khi bật thêm YOLO phân tích đồ vật** | **42.0 - 46.5 FPS** | **~22.5 ms** | YOLO chạy trong worker thread riêng mỗi 5 frame |

### Đặc tính Chống Báo Động Giả & Khả năng Chịu lỗi
*   **Không báo ngủ gật khi nheo mắt nhẹ:** Ngưỡng EAR cá nhân hóa được giới hạn tối đa `0.26`, đảm bảo khi tài xế nheo mắt hoặc nhìn tập trung (EAR ~ 0.24-0.27) hệ thống không bị kích hoạt nhầm.
*   **Không báo buông tay liên tục khi tay bị che một phần:** Chỉ cần ít nhất 1 cổ tay nằm trong vùng vô-lăng ROI, hệ thống đánh giá trạng thái an toàn (`status = "OK"`). Khi bàn tay bị khuất ngón, hệ thống tự động lấy tọa độ cổ tay dự phòng từ module Pose để bù trừ.
*   **Vận hành bền bỉ khi thiếu mô hình:** Nếu thiếu file model YOLO (`yolo26n.pt`) hoặc model Pose (`pose_landmarker.task`), hệ thống tự động tắt module tương ứng và duy trì các tính năng phân tích khuôn mặt, buồn ngủ, mất tập trung bình thường mà không gây lỗi ứng dụng.
*   **Ghi log an toàn theo phiên:** Mọi vi phạm được lưu trữ chuẩn xác vào [`report/events.csv`](file:///c:/project11/report/events.csv) (theo `Session_ID`) kèm ảnh chụp bằng chứng trong thư mục [`report/snapshots/`](file:///c:/project11/report/snapshots/), cùng nhật ký tổng kết phiên tại [`report/sessions.csv`](file:///c:/project11/report/sessions.csv).

---

## 7. Cấu trúc Thư mục Dự án

```
c:\project11\
├── archive/                 # Dữ liệu và mã nguồn lưu trữ (dataset cũ, dlib encodings, attendance)
├── assets/
│   └── fonts/               # Font chữ Arial / Segoe UI hỗ trợ tiếng Việt có dấu
├── docs/
│   ├── architecture.md      # Sơ đồ kiến trúc Mermaid & luồng xử lý
│   └── test_cases.md        # Bảng đặc tả 30+ test case chi tiết
├── model/
│   ├── face_landmarker.task # Mô hình MediaPipe Face (468 landmarks)
│   ├── hand_landmarker.task # Mô hình MediaPipe Hand (21 landmarks)
│   ├── pose_landmarker.task # Mô hình MediaPipe Pose (33 landmarks)
│   ├── steering_roi.json    # Tọa độ vùng vô-lăng đã hiệu chuẩn
│   └── yolo26n.pt           # Trọng số YOLO phát hiện vật thể gây xao nhãng
├── report/
│   ├── events.csv           # Nhật ký vi phạm an toàn lái xe theo Session_ID
│   ├── sessions.csv         # Tổng kết các phiên giám sát (bắt đầu, kết thúc, số vi phạm)
│   └── snapshots/           # Ảnh chụp bằng chứng lúc xảy ra cảnh báo
├── src/
│   ├── config.py            # Quản lý cấu hình, ngưỡng số học và đường dẫn
│   ├── main.py              # Menu điều khiển 4 mục chính của chương trình
│   ├── realtime_monitoring.py# Bộ giám sát thời gian thực đa luồng theo phiên
│   ├── core/
│   │   ├── face_analyzer.py # Phân tích EAR, MAR, Head Pose, PERCLOS & tự hiệu chuẩn
│   │   ├── hand_analyzer.py # Giám sát tay lái & ROI vô-lăng
│   │   ├── pose_analyzer.py # Phân tích góc vai & tư thế ngồi
│   │   ├── object_detector.py # Nhận diện điện thoại, thuốc lá, đồ uống
│   │   ├── preprocessor.py  # Tiền xử lý thiếu sáng CLAHE + Gamma
│   │   └── state_machine.py # Máy trạng thái 3 mức & tính điểm rủi ro
│   └── utils/
│       ├── alert_system.py  # Còi báo động đa âm & giọng nói tiếng Việt
│       ├── event_logger.py  # Ghi nhật ký CSV và lưu ảnh bất đồng bộ
│       ├── logger.py        # Logging chuẩn hóa
│       └── visualizer.py    # Giao diện tiếng Việt có dấu (Pillow Cache)
├── tests/
│   ├── test_phase1.py       # Test độ chính xác EAR, MAR, Pose, ROI
│   ├── test_phase2.py       # Test ma trận cảnh báo, âm thanh & events schema
│   ├── test_phase3.py       # Test tiền xử lý ánh sáng & session logging
│   ├── test_phase4.py       # Test YOLO & phân tích tư thế vai
│   ├── test_phase5.py       # Test tích hợp luồng xử lý E2E
│   └── test_phase6.py       # Test toàn diện chất lượng, góc, timer & headless
└── requirements.txt         # Danh mục thư viện ghim cố định phiên bản
```
