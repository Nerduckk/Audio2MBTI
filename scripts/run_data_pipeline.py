"""Run the data-only pipeline: crawl -> quality check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawl.file_paths import get_master_csv_path
from infrastructure.pipeline_runner import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crawl and data quality stages.")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = PipelineRunner(
        project_root=PROJECT_ROOT,
        outputs_dir=PROJECT_ROOT / "outputs",
        summary_prefix="data_pipeline_run",
    )
    runner.attach("metadata_csv", str(Path(get_master_csv_path()).resolve()))

    python_exe = sys.executable
    if not args.skip_crawl:
        runner.run_step("crawl", [python_exe, "crawl/kaggle_mbti_reprocessor.py"])
        runner.attach("crawl", "completed")
    else:
        runner.attach("crawl", "skipped")

    if not args.skip_quality:
        runner.run_step("quality_check", [python_exe, "crawl/check_data_quality.py"])
        runner.attach("quality_check", "completed")
    else:
        runner.attach("quality_check", "skipped")

    summary_path = runner.finalize()
    print("\n=== summary ===")
    print({"summary_path": str(summary_path.resolve())})


if __name__ == "__main__":
    main()
