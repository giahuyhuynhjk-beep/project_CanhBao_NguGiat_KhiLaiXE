"""
Export script for Driver Distraction YOLO Model.
Exports trained PyTorch weights (.pt) to high-performance inference formats: ONNX, OpenVINO, TorchScript.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics chưa được cài đặt. Chạy: pip install ultralytics")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO Distraction Model")
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "model" / "distraction.pt"),
                        help="Đường dẫn file .pt cần export")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "openvino", "torchscript", "tflite"],
                        help="Định dạng xuất (onnx, openvino, torchscript, tflite)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("      XUẤT MÔ HÌNH NHẬN DIỆN VẬT THỂ XAO NHÃNG (YOLO)")
    print("=" * 60)
    print(f"[INFO] Weights input: {args.weights}")
    print(f"[INFO] Định dạng đích: {args.format}")

    if not os.path.exists(args.weights):
        print(f"[ERROR] Không tìm thấy file model: {args.weights}")
        return

    model = YOLO(args.weights)
    export_path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"[SUCCESS] Đã xuất mô hình thành công tại: {export_path}")


if __name__ == "__main__":
    main()
