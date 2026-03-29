"""CLI for predicting MBTI from a single audio file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.CNN.feature_extraction import FeatureExtractor, vector_to_mbti
from infrastructure.config_loader import load_cnn_config
from infrastructure.cnn_pipeline import CNNPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict MBTI from one audio file.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cnn_config()
    pipeline = CNNPipeline(config)
    extractor = FeatureExtractor.from_config(config)
    spectrogram = extractor.extract(args.audio)[np.newaxis, ..., np.newaxis].astype(np.float32)
    model = pipeline.evaluator.load_model(args.model_path)
    probs = pipeline.evaluator.predict_logits(model, spectrogram)[0]
    payload = {
        "audio_path": str(Path(args.audio).resolve()),
        "mbti": vector_to_mbti(probs),
        "probabilities": {
            "E": float(probs[0]),
            "S": float(probs[1]),
            "T": float(probs[2]),
            "J": float(probs[3]),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
