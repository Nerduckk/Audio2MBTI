import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config_loader import ConfigLoader, get_logger
from infrastructure.monitoring import PerformanceMonitor
from file_paths import get_data_dir, get_master_csv_path, get_merge_sources

config = ConfigLoader.load()
logger = get_logger(__name__)

def aggregate_all_data():
    logger.info("="*50)
    logger.info("Starting data aggregation - combining all sources")
    logger.info("="*50)
    
    output_master = get_master_csv_path()
    files_to_merge = get_merge_sources()
    
    all_dataframes = []
    total_rows = 0
    
    for file_path in files_to_merge:
        # Extract just the filename for logging
        filename = os.path.basename(file_path)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Bỏ qua các file trống rỗng (chỉ có header)
                if len(df) > 0:
                    logger.info(f"✓ Loaded: {filename} ({len(df)} rows)")
                    all_dataframes.append(df)
                    total_rows += len(df)
                else:
                    logger.warning(f"⊘ Empty file: {filename}")
            except Exception as e:
                logger.error(f"✗ Error: {e}")
        else:
            logger.warning(f"⊘ Not found: {filename}")

    if not all_dataframes:
        logger.error("No data to aggregate!")
        return

    logger.info(f"Merging {len(all_dataframes)} sources ({total_rows} total rows)...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # DỌN DẸP SƠ CẤP DATA MASTER
    # 1. Quét Trùng Lặp bằng (Tên Bài + Ca Sĩ)
    print(" Đang chà rửa, khử mụn trùng lặp...")
    master_df['title_clean'] = master_df['title'].astype(str).str.strip().str.lower()
    master_df['artist_clean'] = master_df['artists'].astype(str).str.strip().str.lower()
    
    initial_len = len(master_df)
    master_df = master_df.drop_duplicates(subset=['title_clean', 'artist_clean'], keep='first')
    master_df = master_df.drop(columns=['title_clean', 'artist_clean'])
    
    duplicates_removed = initial_len - len(master_df)
    if duplicates_removed > 0:
        logger.info(f"✓ Removed {duplicates_removed} duplicates")

    # Save to master CSV
    master_df.to_csv(output_master, index=False, encoding='utf-8-sig')
    
    logger.info("="*50)
    logger.info(f"✓ Aggregation complete!")
    logger.info(f"Output: {output_master}")
    logger.info(f"Total: {len(master_df)} songs (100% clean)")
    logger.info("="*50)

if __name__ == "__main__":
    aggregate_all_data()
