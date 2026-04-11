"""CLI for training the CNN model."""

import argparse
import json
import sys
from pathlib import Path
import yaml

# Add current directory so we can import from cnn folder
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from cnn.trainer import ModelTrainer

def load_config():
    with open(current_dir / "cnn" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("cnn", config)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AudioCNN model.")
    parser.add_argument("--X-path", default=str(current_dir.parent / "2_process" / "cnn_embeddings" / "X_train.npy"))
    parser.add_argument("--y-path", default=str(current_dir.parent / "2_process" / "cnn_embeddings" / "y_train.npy"))
    parser.add_argument("--output-dir", default=str(current_dir / "models"))
    return parser

def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Starting Audio CNN training...")
    print(f"X_path: {args.X_path}")
    print(f"y_path: {args.y_path}")
    print(f"Output: {output_dir}")
    
    trainer = ModelTrainer(config)
    result = trainer.train(args.X_path, args.y_path, output_dir)
    print("Completed!")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
