import subprocess
import sys
import os
import time

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f"🌙 BẬT CHẾ ĐỘ CÚ ĐÊM: {description}")
    print(f"{'='*50}")
    
    try:
        # Chạy script bằng môi trường python hiện tại
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ Hoàn thành: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy {script_name}: {e}")
    except KeyboardInterrupt:
        print(f"🛑 Đã dừng thủ công: {script_name}")
        sys.exit(0)

def night_shift():
    print(f"🌠 BẮT ĐẦU CA TRỰC ĐÊM TỰ ĐỘNG - CHÚC SẾP NGỦ NGON!")
    print(f"Thời gian bắt đầu: {time.ctime()}")
    
    # Bước 1: Quét sạch sành sanh khảo sát của bạn bè (Đã cài thuật toán bỏ ESTP list=RD)
    run_script("crawl/survey_mbti_processor.py", "Quay lại cày Khảo Sát Google Form")
    
    # Bước 2: Cút mồi, thu lưới các Playlist MBTI mới nhất năm 2025
    run_script("crawl/farm_modern_playlists.py", "Săn rải thảm Playlist MBTI Gen-Z 2025")
    
    # Bước 3: Cày Kaggle và các File Playlist vừa tải về (Chạy liên tục xuyên đêm)
    # Vì file này mình đã setup bắt lỗi quá tải và tiếp tục (retry), nên nó cứ tà tà chạy thôi.
    run_script("crawl/kaggle_mbti_reprocessor.py", "Cày chìm Master Training Data (Đến khi xong hoặc bị ngắt)")
    
    print(f"\n{'='*50}")
    print(f"🌅 BÌNH MINH LÊN RỒI! TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT.")
    print(f"Thời gian kết thúc: {time.ctime()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    night_shift()
