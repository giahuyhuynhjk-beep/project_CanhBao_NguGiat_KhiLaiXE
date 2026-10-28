# Sơ đồ Kiến trúc Hệ thống DMS (Driver Monitoring System)

Tài liệu mô tả kiến trúc tổng thể, luồng xử lý đa luồng (multi-threading), máy trạng thái (state machine), và các ràng buộc không gian trong dự án **Hệ thống giám sát trạng thái tài xế (DOAN_CNPM_NHANDIENTAIXE)**.

---

## 1. Kiến trúc Tổng thể (System Pipeline)

Hệ thống được thiết kế theo mô hình xử lý đường ống bất đồng bộ (Asynchronous Multimodal Pipeline), đảm bảo tốc độ khung hình (FPS) ổn định từ 45 - 60 FPS trên máy tính tiêu chuẩn (CPU).

```mermaid
flowchart TD
    A[Camera Input / Video Stream] --> B[ImagePreprocessor\nCLAHE + Gamma Night Mode]
    
    subgraph VisionInference [Xử lý thị giác & Phân tích hành vi]
        B --> C[MediaPipe FaceLandmarker\n468 Facial Landmarks]
        B --> D[MediaPipe HandLandmarker\n21 Hand Landmarks]
        B --> E[MediaPipe PoseLandmarker\n33 Body Landmarks]
        B -->|Mỗi 5 frames| G[YOLO Distraction Thread\nPhone / Smoke / Drink]
    end

    C --> H[FaceAnalyzer\n- EAR & PERCLOS\n- MAR & Tần suất ngáp\n- Head Pose: Yaw / Pitch\n- Tự hiệu chuẩn theo phiên]
    D --> I[HandAnalyzer\n- Cổ tay trong ROI vô-lăng\n- Fallback Pose Wrists khi che tay]
    E --> J[PoseAnalyzer\n- Góc nghiêng vai / Slump\n- Fallback Wrists]
    G --> L[ObjectDetector\nSpatial Proximity & Persistence Tracker]

    H --> M[DriverStateMachine & Risk Scoring]
    I --> M
    J --> M
    L --> M

    subgraph OutputDispatch [Phản hồi & Cảnh báo]
        M --> N{Cấp độ Cảnh báo}
        N -->|BÌNH THƯỜNG / CẢNH BÁO / BÁO ĐỘNG| O[Visualizer\nTiếng Việt có dấu Pillow + LRU Cache\nChế độ GUI / Headless]
        N -->|Có vi phạm mới & Hết Cooldown| P[AlertSystem\nAudioSynthesizer & pyttsx3 Voice]
        N -->|Ghi nhận vi phạm| Q[EventLogger\nevents.csv & sessions.csv + Ảnh Snapshot]
    end
```

---

## 2. Máy Trạng thái & Tính Điểm Rủi ro (State Machine Lifecycle)

Hệ thống sử dụng bộ máy trạng thái 3 mức với ngưỡng thời gian (persistence timers), bộ đệm chống báo lặp (cooldowns) và điểm rủi ro tổng hợp (Composite Risk Score: 0 - 100):

```mermaid
stateDiagram-v2
    [*] --> NORMAL: Khởi động hệ thống (Điểm rủi ro = 0)

    NORMAL --> WARNING: Nguy cơ Trung bình (Score >= 30 hoặc vi phạm Medium/High)
    NORMAL --> ALARM: Nguy cơ Nghiêm trọng (Score >= 60 hoặc vi phạm Very High)

    WARNING --> ALARM: Vi phạm kéo dài hoặc xuất hiện hành vi Rất cao (DROWSY, HANDS_OFF)
    WARNING --> NORMAL: Tài xế khắc phục hành vi & Điểm rủi ro < 30

    ALARM --> WARNING: Giảm dần cấp độ khi hành vi nguy hiểm chấm dứt
    ALARM --> NORMAL: Tài xế tập trung lái xe an toàn trở lại

    state WARNING {
        [*] --> YAWN_ALERT: MAR >= 0.60 & > 3 lần/phút
        [*] --> DISTRACTED_ALERT: |Yaw| > 30° hoặc Pitch lệch > 2.5s
        [*] --> SLUMP_ALERT: Góc nghiêng vai > 15° trong > 3.0s
        [*] --> HANDS_NONE_ALERT: Không phát hiện tay lái > 5.0s
        [*] --> DRINK_ALERT: Uống nước > 3.0s
    }

    state ALARM {
        [*] --> DROWSY_ALARM: EAR < ngưỡng > 1.5s hoặc PERCLOS >= 35%
        [*] --> HANDS_OFF_ALARM: Cả hai tay rời vô-lăng ngoài ROI > 3.0s
        [*] --> PHONE_ALARM: Dùng điện thoại gần mặt/tay > 1.5s
        [*] --> SMOKE_ALARM: Hút thuốc gần mặt > 1.5s
    }
```

---

## 3. Ràng buộc Không gian YOLO (Spatial Proximity Constraints)

Để triệt tiêu cảnh báo giả khi phát hiện đồ vật trong xe (ví dụ điện thoại để trên giá đỡ hay chai nước ở hộc để đồ):

```mermaid
flowchart LR
    YOLO[YOLO Bounding Box\nĐiện thoại / Thuốc lá / Cốc nước] --> COND{Kiểm tra Không gian}
    
    FACE[Vùng Mặt mở rộng\nEar / Cheek / Mouth Region] --> COND
    HANDS[Tọa độ Cổ tay\nWrists Coordinates < 120px] --> COND

    COND -->|Gần mặt hoặc Gần tay| VALID[is_near_driver = True\nKích hoạt đếm thời gian vi phạm]
    COND -->|Ở xa tài xế| IGNORE[is_near_driver = False\nBỏ qua, không cảnh báo]
```

---

## 4. Cơ chế Hiển thị Tiếng Việt Có Dấu Tối ưu (Pillow Cache)

```mermaid
sequenceDiagram
    participant Main as RealtimeMonitor
    participant Vis as Visualizer
    participant Cache as LRU Text Bitmap Cache
    participant PIL as Pillow Engine

    Main->>Vis: draw_vietnamese_text(frame, text, pos, font_size, color)
    Vis->>Cache: Kiểm tra key (text, font_size, color)
    alt Đã có trong Cache (Cache Hit)
        Cache-->>Vis: Trả về BGR patch & Alpha mask
    else Chưa có trong Cache (Cache Miss)
        Vis->>PIL: Render text với TrueType Font (Arial/Segoe UI)
        PIL-->>Vis: RGBA bitmap
        Vis->>Cache: Lưu (BGR patch, Alpha mask) vào Cache
    end
    Vis->>Main: Vectorized Alpha Blending trực tiếp lên ROI của frame (< 0.05ms)
```
