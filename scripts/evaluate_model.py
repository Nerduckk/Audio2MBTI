"""CLI for evaluating a trained AudioCNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.config_loader import load_cnn_config
from pipelines.cnn_pipeline import CNNPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the AudioCNN model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--X-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--output-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cnn_config()
    pipeline = CNNPipeline(config)
    X = np.load(args.X_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    metrics = pipeline.evaluate(args.model_path, X, y)

    output_path = args.output_path or str(
        Path(config.get("paths", {}).get("results_dir", "./outputs/cnn")) / "metrics.json"
    )
    saved_path = pipeline.evaluator.save_metrics(metrics, output_path)
    print(json.dumps({"metrics_path": saved_path, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
