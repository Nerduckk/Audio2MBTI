import pandas as pd
import os
import sys
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

def process_google_form_survey(csv_path):
    print("==================================================")
    print(" TOOL TRÍCH XUẤT DATA TỪ KHẢO SÁT GOOGLE FORM")
    print("==================================================")
    
    if not os.path.exists(csv_path):
        print(f" Không tìm thấy file gốc: {csv_path}")
        print(" Vui lòng tải file Excel (.csv) từ Google Form và đổi tên thành 'survey_results.csv' rồi thả vào thư mục 'data/' của dự án.")
        return

    try:
        # Load CSV and skip header noise if necessary
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f" Không thể đọc file CSV: {e}")
        return

    # Tên cột chính xác từ file d:\\project\\data\\survey_results.csv của bạn
    # Pandas thỉnh thoảng sẽ gộp dòng xuống dòng thành \n, nên ta nhận diện qua chuỗi con
    mbti_col = next((c for c in df.columns if "nhóm tính cách mbti" in c.lower()), None)
    link_col = next((c for c in df.columns if "hãy dán link playlist" in c.lower()), None)
    conf_col = next((c for c in df.columns if "chắc chắn bao nhiêu phần trăm" in c.lower()), None)
    email_col = next((c for c in df.columns if "muốn nhận thông báo" in c.lower()), None)
            
    if not link_col or not mbti_col or not conf_col:
        print(" Script không thể mapping được các cột dữ liệu Form. Có đổi form không đó?")
        return

    output_db = r"data\mbti_database_survey.csv"
    
    # Lọc các cột có giá trị Tối Thiểu (Ko điền Link nhạc thì vứt)
    survey_data = df.dropna(subset=[link_col, mbti_col, conf_col])
    
    for index, row in survey_data.iterrows():
        url = str(row[link_col]).strip()
        mbti_label = str(row[mbti_col]).strip().upper()
        
        # Bỏ qua dòng rác (Vd: Form điền giỡn "Ko", "haha")
        if len(url) < 10 or ("spotify.com" not in url and "youtube.com" not in url and "youtu.be" not in url and "apple.com" not in url):
            print(f" KHÁCH HÀNG #{index+1}: Bỏ qua vì dán nhăng cuội (Không phải Link nhạc: {url})")
            continue
            
        # Chặn các Link Youtube Mix Tự Động Vô Hạn (list=RD...)
        if "list=RD" in url:
            print(f" KHÁCH HÀNG #{index+1}: Bỏ qua vì Link YouTube Mix Tự Động ({url}) là vô hạn, gây kẹt máy AI.")
            continue
            
        # Lọc độ tự tin (Confidence < 2 thì vứt vì Tạp Nham Data)
        try:
            confidence = int(row[conf_col])
            if confidence < 2:
                print(f" KHÁCH HÀNG #{index+1}: Bỏ qua vì chẩn đoán MBTI không chắc chắn ({confidence}/5 Điểm)")
                continue
        except (ValueError, KeyError) as e:
            print(f" Warning: Could not parse confidence value - {e}")
            confidence = 3  # Default to middle confidence
        
        email = str(row[email_col]).strip() if pd.notnull(row[email_col]) else "Không để lại"
        
        print(f"\n==================================================")
        print(f" KHÁCH HÀNG #{index+1} | MBTI: {mbti_label} (Tự tin {confidence}/5) | Email: {email}")
        print(f" Link: {url}")
        
        # Dùng một file tàng hình để hứng dữ liệu tạm thời
        raw_csv = r"data\.temp_scraper_raw.csv"
        
        # Gọi tool cào tương ứng
        cmd_args = None
        if "spotify.com" in url:
            try:
                pl_id = url.split("playlist/")[1].split("?")[0]
                cmd_args = [sys.executable, 'spotify_process.py', pl_id, raw_csv]
            except:
                print(" Link Spotify không hợp lệ.")
                continue
        elif "apple.com" in url:
            cmd_args = [sys.executable, 'apple_music_process.py', url, raw_csv]
        elif "youtube.com" in url or "youtu.be" in url:
            cmd_args = [sys.executable, 'youtube_process.py', url, raw_csv]
        else:
            print(f" Nền tảng nhạc lạ chưa hỗ trợ: {url}")
            continue
            
        # Xóa file raw cũ nếu còn tồn đọng
        if os.path.exists(raw_csv):
            os.remove(raw_csv)
            
        print(f" Bắt đầu cào dữ liệu từ đường link...")
        subprocess.run(cmd_args)
        
        if os.path.exists(raw_csv):
            # Nhồi nhãn MBTI vào cái đống nhạc vừa cào của User
            df_raw = pd.read_csv(raw_csv)
            df_raw['mbti_label'] = mbti_label
            
            # Gộp vào Tổng Cục Dự Trữ Khảo Sát
            header = not os.path.exists(output_db)
            df_raw.to_csv(output_db, mode='a', header=header, index=False, encoding='utf-8-sig')
            
            print(f" Khách hàng #{index+1} xong! Đã lưu {len(df_raw)} bài hát (mác {mbti_label}) vào Database!")
            # Dọn dẹp
            os.remove(raw_csv)
        else:
            print(f" Cào thất bại, khách hàng #{index+1} không có bài hát nào tải được.")
            
    print("\n==================================================")
    print(f" ĐÃ CHIẾT XUẤT XONG {len(survey_data)} KHÁCH HÀNG TỪ GOOGLE FORMS!")
    print(f"   => Dữ liệu đã được lưu thành phẩm tại: {output_db}")
    print("==================================================")

if __name__ == "__main__":
    process_google_form_survey(r"data\survey_results.csv")
