# Hướng dẫn Huấn luyện & Triển khai Mô hình YOLO Nhận diện Hành vi Xao nhãng

Thư mục này chứa cấu hình và các kịch bản phụ trợ phục vụ việc huấn luyện mô hình YOLO phát hiện 3 hành vi xao nhãng của tài xế:
1. **phone** (Sử dụng điện thoại di động)
2. **smoke** (Hút thuốc lá / thuốc lá điện tử)
3. **drink** (Uống nước / cầm chai nước)

---

## 1. Cấu trúc thư mục dữ liệu đề xuất

Tổ chức tập dữ liệu theo định dạng YOLO chuẩn bên trong thư mục `dataset/distraction/`:

```
dataset/distraction/
├── images/
│   ├── train/        # Ảnh huấn luyện (.jpg, .png)
│   ├── val/          # Ảnh kiểm định (.jpg, .png)
│   └── test/         # Ảnh kiểm thử (tùy chọn)
└── labels/
    ├── train/        # Nhãn YOLO (.txt tương ứng từng ảnh)
    ├── val/          # Nhãn YOLO (.txt)
    └── test/         # Nhãn YOLO (.txt)
```

Mỗi dòng trong file nhãn `.txt` có định dạng:
```
<class_id> <x_center> <y_center> <width> <height>
```
Trong đó:
- `0`: phone
- `1`: smoke
- `2`: drink
- Các giá trị tọa độ được chuẩn hóa trong khoảng `[0.0, 1.0]`.

---

## 2. Huấn luyện mô hình (Training)

Chạy lệnh sau để huấn luyện mô hình (mặc định sử dụng pre-trained weights `yolo11n.pt` hoặc `yolov8n.pt`):

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Bắt đầu huấn luyện
python yolo/train.py --model yolo11n.pt --epochs 50 --batch 16 --imgsz 640
```

Sau khi huấn luyện hoàn tất, trọng số tối ưu nhất sẽ được tự động sao chép vào `model/distraction.pt`.

---

## 3. Xuất mô hình tối ưu hóa (Exporting)

Để đạt FPS cao nhất trên thiết bị nhúng hoặc CPU thông thường, xuất mô hình sang ONNX hoặc OpenVINO:

```bash
# Xuất sang định dạng ONNX
python yolo/export.py --weights model/distraction.pt --format onnx

# Hoặc OpenVINO (tối ưu hóa đặc biệt cho CPU Intel)
python yolo/export.py --weights model/distraction.pt --format openvino
```

---

## 4. Tích hợp với Hệ thống Real-time

Trong file `src/config.py`:
- Đặt `ENABLE_OBJECT_DETECTION = True` khi đã có file `model/distraction.pt` (hoặc `model/yolo11n.pt` / `model/yolov8n.pt`).
- Hệ thống sẽ tự động tải mô hình và kích hoạt các kênh giám sát `PHONE`, `SMOKE`, `DRINK` với cơ chế phân tích không gian (spatial overlap) quanh vùng mặt và bàn tay/vô lăng.
