"""Download and normalize labeled audio files into `data/audio_files/<MBTI>/...`."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import pandas as pd
import yt_dlp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.CNN.feature_extraction import MBTI_LABELS
from crawl.file_paths import get_data_dir, get_master_csv_path


SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def sanitize_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^\w\s.-]", "", str(value), flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("._ ")
    value = value[:max_len].strip("._ ")
    return value or "track"


def build_track_slug(title: str, artists: str, max_len: int = 100) -> str:
    """Create a stable per-track filename suffix independent of run order."""
    return sanitize_filename(f"{title}_{artists}", max_len=max_len)


def normalize_rows(df: pd.DataFrame) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()

    for _, row in df.iterrows():
        label = str(row.get("mbti_label", "")).upper().strip()
        title = str(row.get("title", "")).strip()
        artists = str(row.get("artists", "")).strip()
        if label not in MBTI_LABELS:
            continue
        if not title or not artists:
            continue

        dedupe_key = (label, title.lower(), artists.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        query = f"{title} {artists} audio"
        rows.append(
            {
                "mbti_label": label,
                "title": title,
                "artists": artists,
                "query": query,
            }
        )

    return rows


def limit_rows(rows: Iterable[Dict[str, str]], per_label_limit: int | None, total_limit: int | None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mbti_label"]].append(row)

    selected = []
    label_counts = defaultdict(int)
    active = True
    while active:
        active = False
        for label in MBTI_LABELS:
            candidates = grouped.get(label, [])
            if not candidates:
                continue
            if per_label_limit is not None and label_counts[label] >= per_label_limit:
                continue
            selected.append(candidates.pop(0))
            label_counts[label] += 1
            active = True
            if total_limit is not None and len(selected) >= total_limit:
                return selected
    return selected


def find_existing_audio(target_dir: Path, stable_slug: str, stem_prefix: str | None = None) -> Path | None:
    patterns = [f"*_{stable_slug}*", f"{stable_slug}*"]
    if stem_prefix:
        patterns.insert(0, f"{stem_prefix}*")

    seen = set()
    for pattern in patterns:
        for path in target_dir.glob(pattern):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                return path
    return None


def remove_duplicate_audio_files(audio_dir: Path) -> int:
    """Remove duplicate files that map to the same stable track slug, keeping the largest one."""
    grouped: Dict[tuple[str, str], List[Path]] = defaultdict(list)
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        label = path.parent.name.upper()
        slug = re.sub(r"^\d+_", "", path.stem)
        grouped[(label, slug)].append(path)

    removed = 0
    for _key, paths in grouped.items():
        if len(paths) < 2:
            continue
        paths = sorted(paths, key=lambda item: (item.stat().st_size, item.name), reverse=True)
        for duplicate in paths[1:]:
            duplicate.unlink(missing_ok=True)
            removed += 1
    return removed


def remove_invalid_audio_files(audio_dir: Path, min_duration: float, min_size_bytes: int) -> int:
    """Remove files that do not satisfy the quality gate."""
    removed = 0
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        quality = inspect_audio_quality(path, min_duration=min_duration, min_size_bytes=min_size_bytes)
        if quality["is_valid"]:
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def dedupe_manifest_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Keep one manifest row per `(label, title, artists)` preferring valid rows."""
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("mbti_label", "")),
            str(row.get("title", "")).strip().lower(),
            str(row.get("artists", "")).strip().lower(),
        )
        grouped[key].append(row)

    priority = {
        "downloaded_valid": 4,
        "existing_valid": 3,
        "rejected_existing": 2,
        "rejected_downloaded": 2,
        "failed": 1,
    }

    deduped = []
    for rows_for_key in grouped.values():
        best = sorted(
            rows_for_key,
            key=lambda row: (
                priority.get(str(row.get("status", "")), 0),
                float(row.get("duration_seconds") or 0),
                int(row.get("file_size_bytes") or 0),
            ),
            reverse=True,
        )[0]
        deduped.append(best)
    return deduped


def cleanup_audio_directory(audio_dir: Path, min_duration: float, min_size_bytes: int) -> Dict[str, int]:
    """Remove obvious bad and duplicate audio artifacts before a new run."""
    invalid_removed = remove_invalid_audio_files(
        audio_dir=audio_dir,
        min_duration=min_duration,
        min_size_bytes=min_size_bytes,
    )
    duplicates_removed = remove_duplicate_audio_files(audio_dir=audio_dir)
    return {
        "invalid_removed": invalid_removed,
        "duplicates_removed": duplicates_removed,
    }


def download_audio(query: str, output_stem: Path, duration: int, ffmpeg_dir: Path) -> Path | None:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_stem),
        "quiet": True,
        "no_warnings": True,
        "default_search": "ytsearch",
        "download_ranges": lambda _, __: [{"start_time": 0, "end_time": duration}],
        "ffmpeg_location": str(ffmpeg_dir),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
    except Exception:
        return None

    for suffix in [".mp3", ".m4a", ".webm", ".wav", ".ogg", ".flac"]:
        candidate = output_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def inspect_audio_quality(audio_path: Path, min_duration: float, min_size_bytes: int) -> Dict[str, object]:
    """Return quality metadata for one audio file."""
    info: Dict[str, object] = {
        "is_valid": False,
        "reason": "",
        "duration_seconds": None,
        "file_size_bytes": 0,
    }

    try:
        file_size = int(audio_path.stat().st_size)
    except FileNotFoundError:
        info["reason"] = "missing_file"
        return info

    info["file_size_bytes"] = file_size
    if file_size < min_size_bytes:
        info["reason"] = "file_too_small"
        return info

    try:
        duration_seconds = float(librosa.get_duration(path=str(audio_path)))
    except Exception:
        info["reason"] = "duration_probe_failed"
        return info

    info["duration_seconds"] = duration_seconds
    if duration_seconds < min_duration:
        info["reason"] = "duration_too_short"
        return info

    info["is_valid"] = True
    info["reason"] = "ok"
    return info


def write_manifest(rows: List[Dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mbti_label",
                "title",
                "artists",
                "query",
                "audio_path",
                "status",
                "duration_seconds",
                "file_size_bytes",
                "quality_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized audio dataset folders.")
    parser.add_argument("--metadata-csv", default=get_master_csv_path())
    parser.add_argument("--output-dir", default=str(Path(get_data_dir()) / "audio_files"))
    parser.add_argument("--manifest-path", default=str(Path(get_data_dir()) / "audio_manifest.csv"))
    parser.add_argument("--duration", type=int, default=35)
    parser.add_argument("--per-label-limit", type=int, default=100)
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--min-size-bytes", type=int, default=180000)
    parser.add_argument("--ffmpeg-dir", default=str(PROJECT_ROOT / "ffmpeg-master-latest-win64-gpl" / "bin"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--cleanup-first", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.metadata_csv)
    rows = normalize_rows(df)
    rows = limit_rows(rows, args.per_label_limit, args.total_limit)

    output_dir = Path(args.output_dir)
    ffmpeg_dir = Path(args.ffmpeg_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_duration = float(args.min_duration if args.min_duration is not None else max(8.0, args.duration * 0.6))
    cleanup_summary = {"invalid_removed": 0, "duplicates_removed": 0}
    if args.cleanup_first:
        cleanup_summary = cleanup_audio_directory(
            audio_dir=output_dir,
            min_duration=min_duration,
            min_size_bytes=args.min_size_bytes,
        )

    manifest_rows: List[Dict[str, str]] = []
    downloaded = 0
    skipped = 0
    failed = 0
    rejected = 0

    for idx, row in enumerate(rows, start=1):
        label = row["mbti_label"]
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        stable_slug = build_track_slug(row["title"], row["artists"])
        stem_prefix = sanitize_filename(f"{idx:05d}_{stable_slug}")
        existing_path = find_existing_audio(target_dir, stable_slug, stem_prefix) if args.skip_existing else None

        manifest_entry = dict(row)
        if existing_path is not None:
            quality = inspect_audio_quality(existing_path, min_duration=min_duration, min_size_bytes=args.min_size_bytes)
            manifest_entry["audio_path"] = str(existing_path.resolve())
            manifest_entry["duration_seconds"] = quality["duration_seconds"]
            manifest_entry["file_size_bytes"] = quality["file_size_bytes"]
            manifest_entry["quality_reason"] = quality["reason"]
            manifest_entry["status"] = "existing_valid" if quality["is_valid"] else "rejected_existing"
            if not quality["is_valid"] and args.delete_invalid and existing_path.exists():
                os.remove(existing_path)
            manifest_rows.append(manifest_entry)
            if quality["is_valid"]:
                skipped += 1
            else:
                rejected += 1
            continue

        audio_path = download_audio(
            query=row["query"],
            output_stem=target_dir / stem_prefix,
            duration=args.duration,
            ffmpeg_dir=ffmpeg_dir,
        )
        if audio_path is None:
            manifest_entry["audio_path"] = ""
            manifest_entry["status"] = "failed"
            manifest_entry["duration_seconds"] = None
            manifest_entry["file_size_bytes"] = 0
            manifest_entry["quality_reason"] = "download_failed"
            manifest_rows.append(manifest_entry)
            failed += 1
            continue

        quality = inspect_audio_quality(audio_path, min_duration=min_duration, min_size_bytes=args.min_size_bytes)
        manifest_entry["audio_path"] = str(audio_path.resolve())
        manifest_entry["duration_seconds"] = quality["duration_seconds"]
        manifest_entry["file_size_bytes"] = quality["file_size_bytes"]
        manifest_entry["quality_reason"] = quality["reason"]
        manifest_entry["status"] = "downloaded_valid" if quality["is_valid"] else "rejected_downloaded"
        if not quality["is_valid"] and args.delete_invalid and audio_path.exists():
            os.remove(audio_path)
        manifest_rows.append(manifest_entry)
        if quality["is_valid"]:
            downloaded += 1
        else:
            rejected += 1

    manifest_rows = dedupe_manifest_rows(manifest_rows)
    write_manifest(manifest_rows, Path(args.manifest_path))
    print(
        json.dumps(
            {
                "selected_rows": len(rows),
                "downloaded": downloaded,
                "existing": skipped,
                "failed": failed,
                "rejected": rejected,
                "min_duration": min_duration,
                "min_size_bytes": args.min_size_bytes,
                "cleanup": cleanup_summary,
                "output_dir": str(output_dir.resolve()),
                "manifest_path": str(Path(args.manifest_path).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
