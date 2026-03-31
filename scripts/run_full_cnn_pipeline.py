"""Run the full automated pipeline: crawl -> quality -> audio dataset -> extract -> train."""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawl.file_paths import get_data_dir, get_master_csv_path
from infrastructure.config_loader import load_cnn_config
from infrastructure.pipeline_runner import PipelineRunner


def save_test_arrays(test_split_path: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = np.load(test_split_path)
    X_path = output_dir / "X_test.npy"
    y_path = output_dir / "y_test.npy"
    np.save(X_path, payload["X_test"])
    np.save(y_path, payload["y_test"])
    return X_path, y_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end CNN data pipeline.")
    parser.add_argument("--per-label-limit", type=int, default=25)
    parser.add_argument("--total-limit", type=int, default=400)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--ffmpeg-workers", type=int, default=8)
    parser.add_argument("--crawl-batch-size", type=int, default=20)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--playlist-delay-min", type=float, default=0.5)
    parser.add_argument("--playlist-delay-max", type=float, default=1.5)
    parser.add_argument("--min-audio-duration", type=float, default=None)
    parser.add_argument("--min-audio-size-bytes", type=int, default=180000)
    parser.add_argument("--min-train-samples", type=int, default=64)
    parser.add_argument("--min-label-coverage", type=int, default=8)
    parser.add_argument("--cache-name", default="cnn_cache_auto")
    parser.add_argument("--model-name", default="cnn_auto")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--loop-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Stop after the current run if this file exists. Defaults to ./outputs/stop_full_cnn_pipeline.txt in continuous mode.",
    )
    return parser


def run_pipeline(args: argparse.Namespace, run_index: int | None = None) -> Path:
    python_exe = sys.executable
    data_dir = Path(get_data_dir())
    cnn_config = load_cnn_config()
    metadata_csv = Path(cnn_config.get("paths", {}).get("metadata_csv", str(Path(get_data_dir()) / "mbti_cnn_metadata.csv")))
    if not metadata_csv.is_absolute():
        metadata_csv = (PROJECT_ROOT / metadata_csv).resolve()
    audio_manifest = data_dir / "audio_manifest.csv"
    cache_dir = data_dir / args.cache_name
    model_dir = PROJECT_ROOT / "models" / args.model_name

    runner = PipelineRunner(
        project_root=PROJECT_ROOT,
        outputs_dir=PROJECT_ROOT / "outputs",
        summary_prefix="pipeline_run",
    )
    if run_index is not None:
        runner.attach("run_index", run_index)
    runner.attach("metadata_csv", str(metadata_csv.resolve()))

    if not args.skip_crawl:
        runner.run_step(
            "crawl",
            [
                python_exe,
                "crawl/kaggle_metadata_reprocessor.py",
                "--batch-size",
                str(args.crawl_batch_size),
                "--requests-per-second",
                str(args.requests_per_second),
                "--playlist-delay-min",
                str(args.playlist_delay_min),
                "--playlist-delay-max",
                str(args.playlist_delay_max),
            ],
            stream_output=True,
        )
        runner.attach("crawl", "completed")
        runner.attach("crawl_mode", "metadata")
    else:
        runner.attach("crawl", "skipped")

    if not args.skip_quality:
        runner.run_step(
            "quality_check",
            [python_exe, "crawl/check_data_quality.py", str(metadata_csv)],
            stream_output=True,
        )
        runner.attach("quality_check", "completed")
    else:
        runner.attach("quality_check", "skipped")

    build_cmd = [
        python_exe,
        "scripts/build_audio_dataset.py",
        "--metadata-csv",
        str(metadata_csv),
        "--manifest-path",
        str(audio_manifest),
        "--per-label-limit",
        str(args.per_label_limit),
        "--total-limit",
        str(args.total_limit),
        "--duration",
        str(args.duration),
        "--download-workers",
        str(args.download_workers),
        "--ffmpeg-workers",
        str(args.ffmpeg_workers),
        "--min-size-bytes",
        str(args.min_audio_size_bytes),
        "--cleanup-first",
        "--delete-invalid",
    ]
    if args.min_audio_duration is not None:
        build_cmd.extend(["--min-duration", str(args.min_audio_duration)])

    build_payload = runner.run_step("build_audio_dataset", build_cmd, parse_json=True)
    runner.attach("audio_dataset", build_payload)

    manifest_df = pd.read_csv(audio_manifest)
    valid_statuses = {"existing_valid", "downloaded_valid"}
    valid_manifest = manifest_df[manifest_df["status"].isin(valid_statuses)].copy()
    label_coverage = int(valid_manifest["mbti_label"].nunique()) if not valid_manifest.empty else 0
    runner.attach("audio_quality_gate", {
        "valid_audio_rows": int(len(valid_manifest)),
        "label_coverage": label_coverage,
        "rejected_rows": int((~manifest_df["status"].isin(valid_statuses)).sum()),
        "accepted_statuses": sorted(valid_statuses),
    })

    extract_payload = runner.run_step(
        "extract_features",
        [
            python_exe,
            "scripts/extract_features.py",
            "--metadata-csv",
            str(audio_manifest),
            "--output-dir",
            str(cache_dir),
        ],
        parse_json=True,
    )
    runner.attach("extract_features", extract_payload)

    X_path = Path(extract_payload["X_path"])
    y_path = Path(extract_payload["y_path"])
    X = np.load(X_path)
    y = np.load(y_path)
    sample_count = int(X.shape[0])
    runner.attach("sample_count", sample_count)
    extract_label_coverage = int(len({tuple(row.tolist()) for row in y}))
    runner.attach("extract_quality_gate", {
        "sample_count": sample_count,
        "label_coverage": extract_label_coverage,
    })

    if sample_count < args.min_train_samples or extract_label_coverage < args.min_label_coverage or args.skip_train:
        runner.attach("train", {
            "status": "skipped",
            "reason": (
                "sample_count_below_threshold"
                if sample_count < args.min_train_samples
                else "label_coverage_below_threshold"
                if extract_label_coverage < args.min_label_coverage
                else "skip_train_flag"
            ),
            "min_train_samples": args.min_train_samples,
            "min_label_coverage": args.min_label_coverage,
        })
    else:
        train_payload = runner.run_step(
            "train",
            [
                python_exe,
                "scripts/train_audio_cnn.py",
                "--X-path",
                str(X_path),
                "--y-path",
                str(y_path),
                "--output-dir",
                str(model_dir),
            ],
            parse_json=True,
        )
        runner.attach("train", train_payload)

        if not args.skip_eval:
            test_split_path = Path(train_payload["test_split_path"])
            X_test_path, y_test_path = save_test_arrays(test_split_path, model_dir)
            config = load_cnn_config()
            metrics_path = Path(config.get("paths", {}).get("results_dir", "./outputs/cnn")) / f"{args.model_name}_metrics.json"
            eval_payload = runner.run_step(
                "evaluate",
                [
                    python_exe,
                    "scripts/evaluate_model.py",
                    "--model-path",
                    str(train_payload["model_path"]),
                    "--X-path",
                    str(X_test_path),
                    "--y-path",
                    str(y_test_path),
                    "--output-path",
                    str(metrics_path),
                ],
                parse_json=True,
            )
            runner.attach("evaluate", eval_payload)
        else:
            runner.attach("evaluate", {"status": "skipped", "reason": "skip_eval_flag"})

    summary_path = runner.finalize()
    print("\n=== summary ===")
    print(json.dumps({**runner.summary, "summary_path": str(summary_path.resolve())}, indent=2))
    return summary_path


def resolve_stop_file(stop_file: str | None) -> Path:
    if stop_file:
        candidate = Path(stop_file)
    else:
        candidate = PROJECT_ROOT / "outputs" / "stop_full_cnn_pipeline.txt"
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    if args.per_label_limit is not None and args.per_label_limit <= 0:
        args.per_label_limit = None
    if args.total_limit is not None and args.total_limit <= 0:
        args.total_limit = None

    if not args.continuous:
        run_pipeline(args)
        return

    stop_file = resolve_stop_file(args.stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[continuous] enabled; create {stop_file} to stop after the current run")

    run_count = 0
    while True:
        run_count += 1
        print(f"\n[continuous] starting run {run_count}")
        try:
            run_pipeline(args, run_index=run_count)
        except Exception as exc:
            print(f"\n[continuous] run {run_count} failed: {exc}", file=sys.stderr)

        if stop_file.exists():
            print(f"\n[continuous] stop file detected: {stop_file}")
            break
        if args.max_runs is not None and run_count >= args.max_runs:
            print(f"\n[continuous] reached max runs: {args.max_runs}")
            break
        if args.loop_sleep_seconds > 0:
            print(f"[continuous] sleeping {args.loop_sleep_seconds:.1f}s before next run")
            time.sleep(args.loop_sleep_seconds)


if __name__ == "__main__":
    main()
