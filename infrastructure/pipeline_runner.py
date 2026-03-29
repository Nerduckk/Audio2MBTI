"""Shared orchestration helpers for data and CNN pipeline scripts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from .monitoring import MetricsCollector


class PipelineRunner:
    """Run pipeline steps consistently and persist execution summaries."""

    def __init__(self, project_root: Path, outputs_dir: Path, summary_prefix: str):
        self.project_root = Path(project_root)
        self.outputs_dir = Path(outputs_dir)
        self.summary_prefix = summary_prefix
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsCollector(metrics_file=str(self.outputs_dir / f"{summary_prefix}_metrics.json"))
        self.summary: Dict[str, Any] = {"started_at": datetime.now().isoformat()}

    def run_step(
        self,
        name: str,
        args: Iterable[str],
        parse_json: bool = False,
        allow_empty_stdout: bool = True,
    ) -> Any:
        args = [str(arg) for arg in args]
        print(f"\n=== {name} ===")
        print(" ".join(args))

        timer_id = self.metrics.start_timer(name)
        result = subprocess.run(args, cwd=self.project_root, text=True, capture_output=True)
        duration = self.metrics.end_timer(timer_id)

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            self.metrics.increment_counter("steps_failed")
            raise RuntimeError(f"Step failed: {name}")

        self.metrics.increment_counter("steps_completed")
        self.summary.setdefault("steps", {})[name] = {"duration_seconds": duration}

        if not parse_json:
            return result

        stdout = result.stdout.strip()
        if not stdout and allow_empty_stdout:
            return {}
        return self._parse_json_payload(stdout)

    def attach(self, key: str, value: Any) -> None:
        self.summary[key] = value

    def finalize(self) -> Path:
        self.summary["finished_at"] = datetime.now().isoformat()
        self.metrics.save_metrics()
        summary_path = self.outputs_dir / f"{self.summary_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, indent=2)
        return summary_path

    @staticmethod
    def _parse_json_payload(stdout: str) -> Any:
        """Parse the final JSON object/array from noisy stdout."""
        stdout = stdout.strip()
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

        matches = re.findall(r"(\{[\s\S]*\}|\[[\s\S]*\])", stdout)
        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise json.JSONDecodeError("Could not parse JSON payload from stdout", stdout, 0)
