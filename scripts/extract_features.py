"""CLI for extracting Mel spectrogram datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.CNN.feature_extraction import FeatureExtractor
from crawl.file_paths import get_audio_dir, get_master_csv_path, get_spectrograms_dir
from infrastructure.config_loader import load_cnn_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Mel spectrograms for CNN training.")
    parser.add_argument("--metadata-csv", default=get_master_csv_path())
    parser.add_argument("--audio-dir", default=get_audio_dir())
    parser.add_argument("--output-dir", default=get_spectrograms_dir())
    parser.add_argument("--output-prefix", default="cnn_dataset")
    parser.add_argument("--x-filename", default=None)
    parser.add_argument("--y-filename", default=None)
    parser.add_argument("--manifest-filename", default=None)
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Ignore CSV metadata and build the dataset purely from audio_dir/<MBTI>/*.ext.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cnn_config()
    extractor = FeatureExtractor.from_config(config)
    result = extractor.batch_extract(
        metadata_csv=None if args.ignore_metadata else args.metadata_csv,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        x_filename=args.x_filename,
        y_filename=args.y_filename,
        manifest_filename=args.manifest_filename,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
