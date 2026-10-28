# Bảng Đặc tả Kịch bản Kiểm thử (Test Cases Specification)

Tài liệu chi tiết danh sách các Test Case kiểm thử tự động (`pytest`) và kiểm thử chức năng cho Hệ thống giám sát tài xế (DMS).

---

## 1. Bảng Tổng hợp Test Case Tự động (`tests/`)

| Test ID | Module | Kịch bản / Dữ liệu giả định | Kết quả mong đợi | Trạng thái | Ghi chú & Khả năng chịu lỗi |
|---|---|---|---|---|---|
| **TC-01** | `face_analyzer` | Mắt mở bình thường (EAR ~ 0.35) | EAR > 0.30, không báo buồn ngủ | **PASS** | Mốc điểm mắt 6 điểm chuẩn MediaPipe |
| **TC-02** | `face_analyzer` | Nheo mắt nhẹ / nhìn bình thường (EAR ~ 0.25) | EAR >= 0.22, `eye_closed=False`, KHÔNG cảnh báo giả | **PASS** | Giới hạn ngưỡng hiệu chuẩn an toàn `<= 0.26` |
| **TC-03** | `face_analyzer` | Mắt nhắm hoàn toàn (EAR ~ 0.05) | EAR < 0.10, `eye_closed=True`, báo ngủ khi > 1.5s | **PASS** | Cảnh báo mức `VERY_HIGH` |
| **TC-04** | `face_analyzer` | Miệng nói chuyện thông thường (MAR ~ 0.35) | MAR < 0.60, không tính là ngáp | **PASS** | Không nhầm lẫn giữa nói chuyện và ngáp |
| **TC-05** | `face_analyzer` | Miệng mở to ngáp (MAR ~ 0.75 >= 0.60) | Ghi nhận 1 lần ngáp khi kéo dài >= 1.0s | **PASS** | Lọc bỏ hành vi mở miệng chớp nhoáng |
| **TC-06** | `face_analyzer` | Đầu nhìn thẳng (Yaw 0°, Pitch 0°) | `is_distracted = False` | **PASS** | Không báo lệch hướng |
| **TC-07** | `face_analyzer` | Đầu quay sang trái/phải (\|Yaw\| = 35° > 30°) | `is_distracted = True` | **PASS** | Cảnh báo chú ý quan sát đường |
| **TC-08** | `face_analyzer` | Đầu gục xuống (Pitch = -25° < -20°) hoặc ngửa (Pitch = 28° > 25°) | `is_distracted = True` | **PASS** | Cảnh báo gục đầu / mất tập trung |
| **TC-09** | `pose_analyzer` | Hai vai ngang cân bằng (Góc nghiêng ~ 0°) | Tilt < 1°, không cảnh báo lệch vai | **PASS** | Mốc vai trái (11) và vai phải (12) |
| **TC-10** | `pose_analyzer` | Vai lệch nghiêng > 15° kéo dài > 3.0s | `is_slump_alert = True` | **PASS** | Cảnh báo chỉnh lại tư thế ngồi |
| **TC-11** | `state_machine` | Không có vi phạm | Trạng thái `NORMAL`, Điểm rủi ro = 0 | **PASS** | Hệ thống ở trạng thái giám sát nền |
| **TC-12** | `state_machine` | Xuất hiện vi phạm ngáp / uống nước | Chuyển sang `WARNING` | **PASS** | Điểm rủi ro tăng theo trọng số vi phạm |
| **TC-13** | `state_machine` | Xuất hiện vi phạm ngủ gật hoặc buông tay | Chuyển ngay sang `ALARM` | **PASS** | Ưu tiên hành vi nguy hiểm mức Rất cao |
| **TC-14** | `state_machine` | Vi phạm lặp lại liên tục trong thời gian ngắn | Bộ đệm Cooldown chặn cảnh báo liên tục | **PASS** | Tránh spam âm thanh và thông báo |
| **TC-15** | `face_analyzer` | Mắt nhắm >= 35% trong cửa sổ trượt 60 giây | `drowsy_perclos_alert = True` | **PASS** | Đánh giá buồn ngủ tích lũy (PERCLOS) |
| **TC-16** | `hand_analyzer` | Cả hai cổ tay nằm trong vùng vô-lăng ROI | `status = "OK"`, không cảnh báo | **PASS** | Trạng thái lái xe an toàn |
| **TC-17** | `hand_analyzer` | Một tay trên vô-lăng, một tay bị che khuất | `status = "OK"`, KHÔNG báo buông tay liên tục | **PASS** | Chỉ cần 1 tay trong ROI là an toàn |
| **TC-18** | `hand_analyzer` | Cả hai cổ tay nằm ngoài vùng vô-lăng ROI | `status = "OUT_OF_ROI"`, báo động sau 3.0s | **PASS** | Cảnh báo mức `VERY_HIGH` |
| **TC-19** | `hand_analyzer` | Bàn tay bị che hoàn toàn nhưng Pose thấy cổ tay | Nhận cổ tay từ Pose fallback -> `status = "OK"` | **PASS** | Chống cảnh báo giả khi tay che một phần |
| **TC-20** | `face_analyzer` | Ngáp liên tục 3 lần trong vòng 60 giây | `is_yawn_alert = True` | **PASS** | Cửa sổ trượt tự hủy các lần ngáp quá 60s |
| **TC-21** | `object_detector` | Điện thoại ở gần tai / mặt tài xế | `is_near_driver = True`, báo động sau 1.5s | **PASS** | Ràng buộc không gian vùng mặt mở rộng |
| **TC-22** | `object_detector` | Điện thoại ở vị trí xa (ghế phụ / hộc đồ) | `is_near_driver = False`, không cảnh báo | **PASS** | Triệt tiêu cảnh báo giả đồ vật tĩnh |
| **TC-23** | `object_detector` | Không tìm thấy file model YOLO | `is_ready = False`, trả về rỗng, không crash | **PASS** | Tự động hạ cấp an toàn khi thiếu model |
| **TC-24** | `pose_analyzer` | Không tìm thấy file model Pose | `is_available = False`, trả về mặc định, không crash | **PASS** | Giám sát vai tạm tắt, hệ thống vẫn chạy |
| **TC-25** | `face_analyzer` | Mất khuôn mặt khỏi khung hình | Reset bộ đếm nhắm mắt ngay lập tức | **PASS** | Không báo ngủ gật khi tài xế rời ghế |
| **TC-26** | `visualizer` | Vẽ chữ tiếng Việt có dấu ("CẢNH BÁO", "Tài xế") | Hiển thị sắc nét bằng Pillow | **PASS** | Tải từ `assets/fonts/` hoặc font hệ thống |
| **TC-27** | `visualizer` | Vẽ text lặp lại liên tục nhiều frame | Tận dụng Bitmap Cache, không render lại | **PASS** | Đạt tốc độ > 450 FPS cho riêng phần vẽ |
| **TC-28** | `visualizer` | Chạy chế độ Headless / tắt visualizer | Bỏ qua toàn bộ hàm vẽ, frame nguyên bản | **PASS** | Tối ưu tài nguyên CPU khi chạy nền |
| **TC-29** | `event_logger` | Ghi nhận phiên giám sát (`end_session`) | Ghi đầy đủ thống kê phiên vào `sessions.csv` | **PASS** | Theo dõi an toàn theo phiên ẩn danh |
| **TC-30** | `event_logger` | Phát sinh vi phạm kèm ảnh chụp khung hình | Ghi log vào `events.csv` với `Session_ID` và lưu ảnh vào `snapshots/` | **PASS** | Ghi log bất đồng bộ bằng worker thread |

---

## 2. Hướng dẫn Chạy Kiểm thử Toàn bộ

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Chạy toàn bộ 35 unit test
pytest -v

# Chỉ chạy bộ test chất lượng GĐ6
pytest tests/test_phase6.py -v
```
