import sys
from pathlib import Path
import numpy as np
import torch
import json
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Add cnn folder to sys.path to resolve relative imports in trainer.py
CNN_DIR = PROJECT_ROOT / "3_train"
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from cnn.trainer import ModelTrainer

def load_config():
    config_path = PROJECT_ROOT / "3_train" / "cnn" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("cnn", config)

def main():
    print("🧪 Starting SANITY CHECK (10 samples)...")
    config = load_config()
    
    # Force minimal settings
    config['training']['batch_size'] = 2
    config['training']['epochs'] = 1
    config['training']['test_size'] = 0.5
    config['training']['val_size'] = 0.1
    
    x_path = str(PROJECT_ROOT / "2_process/cnn_embeddings/X_train.npy")
    y_path = str(PROJECT_ROOT / "2_process/cnn_embeddings/y_train.npy")
    output_dir = str(PROJECT_ROOT / "3_train/models/sanity_check")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = ModelTrainer(config)
    
    # Manual data loading for just 10 samples
    print("Loading tiny subset...")
    X_full = np.load(x_path, mmap_mode='r')
    y_full = np.load(y_path).astype(np.float32)
    
    X_tiny = X_full[:10]
    y_tiny = y_full[:10]
    
    # Save tiny versions to new test directory
    tiny_x_path = str(PROJECT_ROOT / "2_process/tests/X_tiny.npy")
    tiny_y_path = str(PROJECT_ROOT / "2_process/tests/y_tiny.npy")
    np.save(tiny_x_path, X_tiny)
    np.save(tiny_y_path, y_tiny)
    
    print("Running trainer...")
    try:
        result = trainer.train(tiny_x_path, tiny_y_path, output_dir)
        print("\n✅ SANITY CHECK PASSED!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n❌ SANITY CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
