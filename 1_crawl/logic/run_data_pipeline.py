"""Run the data-only pipeline: crawl -> quality check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add local logic folder to path
sys.path.insert(0, str(Path(__file__).parent))

from file_paths import get_master_csv_path
from infrastructure.config_loader import ConfigLoader
from infrastructure.pipeline_runner import PipelineRunner

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crawl and data quality stages.")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--crawl-batch-size", type=int, default=20)
    parser.add_argument("--crawl-mode", choices=["legacy", "metadata"], default="metadata")
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--playlist-delay-min", type=float, default=0.5)
    parser.add_argument("--playlist-delay-max", type=float, default=1.5)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    config = ConfigLoader.load()
    
    # Resolve roots
    logic_dir = Path(__file__).parent.resolve()
    project_root = logic_dir.parent.parent.resolve()
    
    metadata_csv = Path(get_master_csv_path())
    if args.crawl_mode == "metadata":
        # Default fallback
        metadata_csv = Path("./data/mbti_cnn_metadata.csv")
        if not metadata_csv.is_absolute():
            metadata_csv = (project_root / metadata_csv).resolve()

    runner = PipelineRunner(
        project_root=project_root,
        outputs_dir=project_root / "outputs",
        summary_prefix="data_pipeline_run",
    )
    runner.attach("metadata_csv", str(metadata_csv.resolve()))

    python_exe = sys.executable
    if not args.skip_crawl:
        crawl_script = (
            "1_crawl/logic/kaggle_metadata_reprocessor.py"
            if args.crawl_mode == "metadata"
            else "1_crawl/logic/kaggle_mbti_reprocessor.py"
        )
        runner.run_step(
            "crawl",
            [
                python_exe,
                str(project_root / crawl_script),
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
        runner.attach("crawl_mode", args.crawl_mode)
    else:
        runner.attach("crawl", "skipped")

    if not args.skip_quality:
        runner.run_step("quality_check", [python_exe, str(logic_dir / "check_data_quality.py"), str(metadata_csv)], stream_output=True)
        runner.attach("quality_check", "completed")
    else:
        runner.attach("quality_check", "skipped")

    summary_path = runner.finalize()
    print("\n=== summary ===")
    print({"summary_path": str(summary_path.resolve())})

if __name__ == "__main__":
    main()
