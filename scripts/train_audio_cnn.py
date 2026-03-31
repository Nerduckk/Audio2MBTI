"""CLI for training the CNN model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.config_loader import load_cnn_config
from pipelines.cnn_pipeline import CNNPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AudioCNN model.")
    parser.add_argument("--X-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cnn_config()
    default_output_dir = config.get("paths", {}).get("models_dir", "./models/cnn")
    output_dir = args.output_dir or default_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = CNNPipeline(config)
    result = pipeline.train(args.X_path, args.y_path, output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
