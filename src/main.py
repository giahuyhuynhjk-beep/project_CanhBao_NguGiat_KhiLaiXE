import os
import sys
import subprocess

# Đảm bảo đường dẫn luôn đúng dù chạy từ đâu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.config as config


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_menu():
    print("====================================================")
    print("   HỆ THỐNG GIÁM SÁT TÀI XẾ THỜI GIAN THỰC (DMS)     ")
    print("====================================================")
    print(" 1. Chạy hệ thống giám sát thời gian thực (GUI)     ")
    print(" 2. Chạy hệ thống giám sát chế độ ẩn (Headless)     ")
    print(" 3. Hiệu chuẩn vùng vô-lăng (Steering Wheel ROI)    ")
    print(" 4. Thoát                                           ")
    print("====================================================")


def main():
    while True:
        clear_screen()
        print_menu()
        choice = input("Chọn chức năng (1-4): ").strip()

        if choice == '1':
            print("\n[INFO] Đang mở module giám sát (Giao diện hiển thị GUI)...")
            subprocess.run([sys.executable, "src/realtime_monitoring.py"])
            input("\nNhấn Enter để quay lại menu...")

        elif choice == '2':
            print("\n[INFO] Đang mở module giám sát (Chế độ Headless - chạy nền)...")
            subprocess.run([sys.executable, "src/realtime_monitoring.py", "--headless"])
            input("\nNhấn Enter để quay lại menu...")

        elif choice == '3':
            print("\n[INFO] Đang mở module hiệu chuẩn vô-lăng (ROI)...")
            subprocess.run([sys.executable, "src/core/hand_analyzer.py"])
            input("\nNhấn Enter để quay lại menu...")

        elif choice in ('4', 'q', 'exit'):
            print("\nTạm biệt!")
            break
        else:
            print("\nLựa chọn không hợp lệ!")
            input("\nNhấn Enter để thử lại...")


if __name__ == "__main__":
    main()
