"""File path helpers for the active real-song pipeline."""

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
    """Get full path to a configured crawler CSV output."""
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


def get_kaggle_csv():
    """Get path for Kaggle crawler output CSV."""
    return get_csv_path('kaggle_csv')

def ensure_data_dir_exists():
    """Create data directory if it doesn't exist."""
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
