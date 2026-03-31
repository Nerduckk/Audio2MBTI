"""Unified workflow entrypoint for crawl, feature build, and model training tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def command_crawl_metadata(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON_EXE,
        "scripts/run_data_pipeline.py",
        "--crawl-mode",
        "metadata",
        "--crawl-batch-size",
        str(args.crawl_batch_size),
        "--requests-per-second",
        str(args.requests_per_second),
        "--playlist-delay-min",
        str(args.playlist_delay_min),
        "--playlist-delay-max",
        str(args.playlist_delay_max),
    ]
    run_step("crawl-metadata", cmd)


def command_crawl_audio(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON_EXE,
        "scripts/build_audio_dataset.py",
        "--metadata-csv",
        args.metadata_csv,
        "--duration",
        str(args.duration),
        "--download-workers",
        str(args.download_workers),
        "--ffmpeg-workers",
        str(args.ffmpeg_workers),
        "--progress-every",
        str(args.progress_every),
        "--min-size-bytes",
        str(args.min_size_bytes),
    ]
    run_step("crawl-audio", cmd)


def command_extract_cnn(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON_EXE,
        "scripts/extract_features.py",
        "--ignore-metadata",
        "--audio-dir",
        args.audio_dir,
        "--output-dir",
        args.output_dir,
        "--x-filename",
        args.x_filename,
        "--y-filename",
        args.y_filename,
        "--manifest-filename",
        args.manifest_filename,
    ]
    run_step("extract-cnn", cmd)


def command_train_hybrid(args: argparse.Namespace) -> None:
    cmd = [
        PYTHON_EXE,
        "scripts/train_hybrid_tabular.py",
        "--old-csv",
        args.old_csv,
        "--manifest-path",
        args.manifest_path,
        "--feature-cache",
        args.feature_cache,
        "--metrics-path",
        args.metrics_path,
        "--duration",
        str(args.duration),
        "--workers",
        str(args.workers),
    ]
    run_step("train-hybrid", cmd)


def command_refresh_hybrid(args: argparse.Namespace) -> None:
    command_extract_cnn(args)
    command_train_hybrid(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project workflow wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_metadata = subparsers.add_parser("crawl-metadata", help="Refresh metadata CSV from the metadata crawler.")
    crawl_metadata.add_argument("--crawl-batch-size", type=int, default=20)
    crawl_metadata.add_argument("--requests-per-second", type=float, default=1.0)
    crawl_metadata.add_argument("--playlist-delay-min", type=float, default=0.5)
    crawl_metadata.add_argument("--playlist-delay-max", type=float, default=1.5)
    crawl_metadata.set_defaults(func=command_crawl_metadata)

    crawl_audio = subparsers.add_parser("crawl-audio", help="Download/normalize audio into data/audio_files.")
    crawl_audio.add_argument("--metadata-csv", default="data/mbti_cnn_metadata.csv")
    crawl_audio.add_argument("--duration", type=int, default=20)
    crawl_audio.add_argument("--download-workers", type=int, default=64)
    crawl_audio.add_argument("--ffmpeg-workers", type=int, default=12)
    crawl_audio.add_argument("--progress-every", type=int, default=25)
    crawl_audio.add_argument("--min-size-bytes", type=int, default=180000)
    crawl_audio.set_defaults(func=command_crawl_audio)

    extract_cnn = subparsers.add_parser("extract-cnn", help="Build X_train/y_train directly from audio_files.")
    extract_cnn.add_argument("--audio-dir", default="data/audio_files")
    extract_cnn.add_argument("--output-dir", default="data")
    extract_cnn.add_argument("--x-filename", default="X_train.npy")
    extract_cnn.add_argument("--y-filename", default="y_train.npy")
    extract_cnn.add_argument("--manifest-filename", default="train_manifest.json")
    extract_cnn.set_defaults(func=command_extract_cnn)

    train_hybrid = subparsers.add_parser("train-hybrid", help="Train the hybrid tree-based baseline.")
    train_hybrid.add_argument("--old-csv", default=r"data\(OLD) mbti_master_training_data.csv")
    train_hybrid.add_argument("--manifest-path", default="data/train_manifest.json")
    train_hybrid.add_argument("--feature-cache", default="data/audio_tabular_features.csv")
    train_hybrid.add_argument("--metrics-path", default="outputs/hybrid_tree_metrics.json")
    train_hybrid.add_argument("--duration", type=int, default=35)
    train_hybrid.add_argument("--workers", type=int, default=6)
    train_hybrid.set_defaults(func=command_train_hybrid)

    refresh_hybrid = subparsers.add_parser(
        "refresh-hybrid",
        help="Rebuild X_train/y_train from audio_files and retrain the hybrid tree baseline.",
    )
    refresh_hybrid.add_argument("--audio-dir", default="data/audio_files")
    refresh_hybrid.add_argument("--output-dir", default="data")
    refresh_hybrid.add_argument("--x-filename", default="X_train.npy")
    refresh_hybrid.add_argument("--y-filename", default="y_train.npy")
    refresh_hybrid.add_argument("--manifest-filename", default="train_manifest.json")
    refresh_hybrid.add_argument("--old-csv", default=r"data\(OLD) mbti_master_training_data.csv")
    refresh_hybrid.add_argument("--manifest-path", default="data/train_manifest.json")
    refresh_hybrid.add_argument("--feature-cache", default="data/audio_tabular_features.csv")
    refresh_hybrid.add_argument("--metrics-path", default="outputs/hybrid_tree_metrics.json")
    refresh_hybrid.add_argument("--duration", type=int, default=35)
    refresh_hybrid.add_argument("--workers", type=int, default=6)
    refresh_hybrid.set_defaults(func=command_refresh_hybrid)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
