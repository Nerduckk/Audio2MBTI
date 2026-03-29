"""Run the full automated pipeline: crawl -> quality -> audio dataset -> extract -> train."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawl.file_paths import get_data_dir, get_master_csv_path
from infrastructure.config_loader import load_cnn_config


def run_step(name: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {name} ===")
    print(" ".join(args))
    result = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")
    return result


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python_exe = sys.executable
    data_dir = Path(get_data_dir())
    metadata_csv = Path(get_master_csv_path())
    audio_manifest = data_dir / "audio_manifest.csv"
    cache_dir = data_dir / args.cache_name
    model_dir = PROJECT_ROOT / "models" / args.model_name
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "metadata_csv": str(metadata_csv.resolve()),
    }

    if not args.skip_crawl:
        run_step("crawl", [python_exe, "crawl/kaggle_mbti_reprocessor.py"])
        summary["crawl"] = "completed"
    else:
        summary["crawl"] = "skipped"

    if not args.skip_quality:
        run_step("quality_check", [python_exe, "crawl/check_data_quality.py"])
        summary["quality_check"] = "completed"
    else:
        summary["quality_check"] = "skipped"

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
        "--min-size-bytes",
        str(args.min_audio_size_bytes),
        "--cleanup-first",
        "--delete-invalid",
    ]
    if args.min_audio_duration is not None:
        build_cmd.extend(["--min-duration", str(args.min_audio_duration)])

    build_result = run_step("build_audio_dataset", build_cmd)
    summary["audio_dataset"] = json.loads(build_result.stdout)

    manifest_df = pd.read_csv(audio_manifest)
    valid_statuses = {"existing_valid", "downloaded_valid"}
    valid_manifest = manifest_df[manifest_df["status"].isin(valid_statuses)].copy()
    label_coverage = int(valid_manifest["mbti_label"].nunique()) if not valid_manifest.empty else 0
    summary["audio_quality_gate"] = {
        "valid_audio_rows": int(len(valid_manifest)),
        "label_coverage": label_coverage,
        "rejected_rows": int((~manifest_df["status"].isin(valid_statuses)).sum()),
        "accepted_statuses": sorted(valid_statuses),
    }

    extract_result = run_step(
        "extract_features",
        [
            python_exe,
            "scripts/extract_features.py",
            "--metadata-csv",
            str(audio_manifest),
            "--output-dir",
            str(cache_dir),
        ],
    )
    extract_payload = json.loads(extract_result.stdout)
    summary["extract_features"] = extract_payload

    X_path = Path(extract_payload["X_path"])
    y_path = Path(extract_payload["y_path"])
    X = np.load(X_path)
    y = np.load(y_path)
    sample_count = int(X.shape[0])
    summary["sample_count"] = sample_count
    extract_label_coverage = int(len({tuple(row.tolist()) for row in y}))
    summary["extract_quality_gate"] = {
        "sample_count": sample_count,
        "label_coverage": extract_label_coverage,
    }

    if sample_count < args.min_train_samples or extract_label_coverage < args.min_label_coverage or args.skip_train:
        summary["train"] = {
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
        }
    else:
        train_result = run_step(
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
        )
        train_payload = json.loads(train_result.stdout)
        summary["train"] = train_payload

        if not args.skip_eval:
            test_split_path = Path(train_payload["test_split_path"])
            X_test_path, y_test_path = save_test_arrays(test_split_path, model_dir)
            config = load_cnn_config()
            metrics_path = Path(config.get("paths", {}).get("results_dir", "./outputs/cnn")) / f"{args.model_name}_metrics.json"
            eval_result = run_step(
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
            )
            summary["evaluate"] = json.loads(eval_result.stdout)
        else:
            summary["evaluate"] = {"status": "skipped", "reason": "skip_eval_flag"}

    summary["finished_at"] = datetime.now().isoformat()
    summary_path = outputs_dir / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== summary ===")
    print(json.dumps({**summary, "summary_path": str(summary_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
