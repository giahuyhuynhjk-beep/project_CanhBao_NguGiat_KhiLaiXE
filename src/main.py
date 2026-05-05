import os
import sys
import subprocess

# Đảm bảo đường dẫn luôn đúng dù chạy từ đâu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    print("====================================================")
    print("   HE THONG GIAM SAT VA DIEM DANH KHUON MAT         ")
    print("====================================================")
    print(" 1. Them nhan vien moi (Thu thap du lieu)           ")
    print(" 2. Train du lieu (Cap nhat ma hoa khuon mat)       ")
    print(" 3. Chay he thong giam sat thoi gian thuc           ")
    print(" 4. Thoat                                           ")
    print("====================================================")

def main():
    while True:
        clear_screen()
        print_menu()
        choice = input("Chon chuc nang (1-4): ")

        if choice == '1':
            print("\n[INFO] Dang mo module thu thap du lieu...")
            subprocess.run([sys.executable, "src/build_dataset.py"])
            input("\nNhan Enter de quay lai menu...")

        elif choice == '2':
            print("\n[INFO] Dang mo module train du lieu...")
            subprocess.run([sys.executable, "src/encode_faces.py"])
            input("\nNhan Enter de quay lai menu...")

        elif choice == '3':
            print("\n[INFO] Dang mo module giam sat...")
            if not os.path.exists("model/encodings.pickle") and not os.path.exists("encodings.pickle"):
                print("\n[ERROR] Chua co du lieu ma hoa (encodings.pickle). Vui long train du lieu truoc.")
                input("\nNhan Enter de quay lai menu...")
                continue
            subprocess.run([sys.executable, "src/realtime_monitoring.py"])
            input("\nNhan Enter de quay lai menu...")

        elif choice == '4':
            print("\nTam biet!")
            break
        else:
            print("\nLua chon khong hop le!")
            input("\nNhan Enter de thu lai...")

if __name__ == "__main__":
    main()
