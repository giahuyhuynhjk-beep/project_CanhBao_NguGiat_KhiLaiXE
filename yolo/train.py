"""
Training script for Driver Distraction Detection using YOLO (Ultralytics).
Detects: phone (0), smoke (1), drink (2).
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
    parser = argparse.ArgumentParser(description="Train YOLO Driver Distraction Model")
    parser.add_argument("--data", type=str, default=str(Path(__file__).parent / "driver_distraction.yaml"),
                        help="Đường dẫn file cấu hình YAML của dataset")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Mô hình base (yolo11n.pt, yolov8n.pt, v.v.)")
    parser.add_argument("--epochs", type=int, default=50, help="Số lượng epoch")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "model"),
                        help="Thư mục lưu weights sau khi train")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("      HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN VẬT THỂ XAO NHÃNG")
    print("=" * 60)
    print(f"[INFO] Base Model: {args.model}")
    print(f"[INFO] Dataset Config: {args.data}")
    print(f"[INFO] Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")

    if not os.path.exists(args.data):
        print(f"[ERROR] Không tìm thấy file cấu hình dữ liệu: {args.data}")
        return

    # Load base model
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project="runs/distraction",
        name="train_exp",
        exist_ok=True
    )

    # Copy best weights to model/
    best_weights = Path("runs/distraction/train_exp/weights/best.pt")
    target_weights = Path(args.output_dir) / "distraction.pt"

    if best_weights.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"[SUCCESS] Đã lưu model tốt nhất vào: {target_weights}")
    else:
        print("[WARNING] Không tìm thấy file best.pt sau quá trình huấn luyện.")


if __name__ == "__main__":
    main()
