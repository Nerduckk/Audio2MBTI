"""CLI for extracting Mel spectrogram datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawl.file_paths import get_audio_dir, get_master_csv_path, get_spectrograms_dir
from infrastructure.cnn_pipeline import CNNPipeline
from infrastructure.config_loader import load_cnn_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Mel spectrograms for CNN training.")
    parser.add_argument("--metadata-csv", default=get_master_csv_path())
    parser.add_argument("--audio-dir", default=get_audio_dir())
    parser.add_argument("--output-dir", default=get_spectrograms_dir())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cnn_config()
    pipeline = CNNPipeline(config)
    result = pipeline.extract_features(
        metadata_csv=args.metadata_csv,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
