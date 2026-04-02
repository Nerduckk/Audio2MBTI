import pandas as pd
import os

def main():
    print("🔄 Đang tạo ánh xạ Track -> Playlist (Fixing mapping)...")
    data_dir = "2_process"
    input_path = os.path.join(data_dir, "mbti_cnn_metadata.csv")
    output_path = os.path.join(data_dir, "sample_to_playlist.csv")
    
    # Read without header to avoid index shift issues
    df = pd.read_csv(input_path, header=None, skiprows=1, on_bad_lines='skip')
    
    # Based on binary inspection:
    # 0: Title, 1: Artist, 2: Label, 9: Playlist ID
    print(f"   📊 Đã đọc {len(df)} dòng dữ liệu thô.")
    
    mapping = df[[0, 1, 2, 9]].dropna()
    mapping.columns = ['title', 'artists', 'label', 'playlist']
    
    # Clean data
    mapping['title'] = mapping['title'].str.strip()
    mapping['artists'] = mapping['artists'].str.strip()
    mapping['playlist'] = mapping['playlist'].str.strip()
    
    print(f"   ✅ Khớp thành công {len(mapping)} bài hát vào {mapping['playlist'].nunique()} playlists.")
    
    mapping.to_csv(output_path, index=False)
    print(f"   💾 Đã lưu ánh xạ tại: {output_path}")

if __name__ == "__main__":
    main()
