"""
File paths utility for Audio2MBTI crawlers
Centralizes CSV file path configuration from config.yaml
"""

import os
from pathlib import Path
import sys

# Add parent directory to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from infrastructure.config_loader import ConfigLoader
    config = ConfigLoader.load()
except Exception:
    config = {}


def get_data_dir():
    """Get the data directory path"""
    data_dir = config.get('paths', {}).get('data_dir', './data')
    return os.path.abspath(data_dir)


def get_csv_path(csv_key):
    """
    Get full path to a CSV file from config
    
    Args:
        csv_key: Key in config.crawlers (e.g. 'kaggle_csv', 'spotify_csv')
        
    Returns:
        Absolute path to CSV file
    """
    data_dir = get_data_dir()
    
    crawler_config = config.get('crawlers', {})
    csv_filename = crawler_config.get(csv_key, '')
    
    if not csv_filename:
        raise ValueError(f"CSV filename not found in config for key: {csv_key}")
    
    return os.path.join(data_dir, csv_filename)


def get_master_csv_path():
    """Get path to master training CSV"""
    data_dir = get_data_dir()
    master_csv = config.get('data', {}).get('master_csv', 'mbti_master_training_data.csv')
    return os.path.join(data_dir, master_csv)


def get_audio_dir():
    """Get path to raw audio files directory."""
    data_dir = get_data_dir()
    cnn_audio_dir = config.get('cnn', {}).get('paths', {}).get('audio_dir')
    if cnn_audio_dir:
        return os.path.abspath(cnn_audio_dir)
    return os.path.join(data_dir, 'audio_files')


def get_spectrograms_dir():
    """Get path to cached spectrogram directory."""
    data_dir = get_data_dir()
    cnn_cache_dir = config.get('cnn', {}).get('paths', {}).get('cache_dir')
    if cnn_cache_dir:
        return os.path.abspath(cnn_cache_dir)
    return os.path.join(data_dir, 'spectrograms')


def get_kaggle_csv():
    """Get path for Kaggle crawler output CSV"""
    return get_csv_path('kaggle_csv')


def get_spotify_csv():
    """Get path for Spotify crawler output CSV"""
    raise ValueError("Spotify CSV path is no longer used in the Kaggle-only pipeline")


def get_youtube_csv():
    """Get path for YouTube crawler output CSV"""
    raise ValueError("YouTube CSV path is no longer used in the Kaggle-only pipeline")


def get_applemusic_csv():
    """Get path for Apple Music crawler output CSV"""
    raise ValueError("Apple Music CSV path is no longer used in the Kaggle-only pipeline")


def get_survey_csv():
    """Get path for survey data CSV"""
    raise ValueError("Survey CSV path is no longer used in the Kaggle-only pipeline")


def get_merge_sources():
    """Get list of CSV files to merge for master database"""
    return [get_master_csv_path()]


def ensure_data_dir_exists():
    """Create data directory if it doesn't exist"""
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
