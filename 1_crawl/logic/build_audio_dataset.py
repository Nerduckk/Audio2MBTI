"""Download and normalize labeled audio files into `data/audio_files/<MBTI>/...`."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import pandas as pd
import yt_dlp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MBTI_LABELS = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
from file_paths import get_audio_dir, get_data_dir


SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MANIFEST_FIELDS = [
    "mbti_label",
    "title",
    "artists",
    "query",
    "audio_path",
    "status",
    "duration_seconds",
    "file_size_bytes",
    "quality_reason",
]

YOUTUBE_BOT_PATTERNS = (
    "sign in to confirm you're not a bot",
    "use --cookies-from-browser or --cookies",
)
FORMAT_UNAVAILABLE_PATTERNS = (
    "requested format is not available",
    "no video formats found",
)
VIDEO_UNAVAILABLE_PATTERNS = (
    "video unavailable",
    "private video",
    "this video is unavailable",
    "members-only content",
    "sign in if you've been granted access",
)


def sanitize_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^\w\s.-]", "", str(value), flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("._ ")
    value = value[:max_len].strip("._ ")
    return value or "track"


def build_track_slug(title: str, artists: str, max_len: int = 100) -> str:
    """Create a stable per-track filename suffix independent of run order."""
    return sanitize_filename(f"{title}_{artists}", max_len=max_len)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def build_unique_key(title: str, artists: str) -> str:
    return f"{normalize_text(title)}|{normalize_text(artists)}"


def build_query_variants(title: str, artists: str) -> List[str]:
    base = f"{title} {artists}".strip()
    return [
        f"{base} audio",
        f"{base} official audio",
        base,
        f"{base} lyrics",
    ]


def build_canonical_path(cache_dir: Path, unique_key: str, fmt: str) -> Path:
    digest = hashlib.sha1(unique_key.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.{fmt}"


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

        dedupe_key = (label, normalize_text(title), normalize_text(artists))
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


def filter_out_materialized_rows(rows: List[Dict[str, str]], output_dir: Path) -> List[Dict[str, str]]:
    """Skip rows whose label folder already contains the song."""
    filtered: List[Dict[str, str]] = []
    for row in rows:
        label_dir = output_dir / row["mbti_label"]
        stable_slug = build_track_slug(row["title"], row["artists"])
        existing_path = find_existing_audio(label_dir, stable_slug) if label_dir.exists() else None
        if existing_path is not None:
            continue
        filtered.append(row)
    return filtered


def group_rows_by_song(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for idx, row in enumerate(rows, start=1):
        unique_key = build_unique_key(row["title"], row["artists"])
        payload = grouped.setdefault(
            unique_key,
            {
                "unique_key": unique_key,
                "title": row["title"],
                "artists": row["artists"],
                "query_variants": build_query_variants(row["title"], row["artists"]),
                "entries": [],
            },
        )
        payload["entries"].append(
            {
                "index": idx,
                "mbti_label": row["mbti_label"],
                "title": row["title"],
                "artists": row["artists"],
                "query": row["query"],
            }
        )
    return list(grouped.values())


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
            normalize_text(str(row.get("title", ""))),
            normalize_text(str(row.get("artists", ""))),
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


def cleanup_stem_outputs(output_stem: Path) -> None:
    for suffix in [".mp3", ".m4a", ".webm", ".wav", ".ogg", ".flac", ".part", ".ytdl"]:
        candidate = output_stem.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink(missing_ok=True)


def find_downloaded_audio(output_stem: Path) -> Path | None:
    for suffix in [".mp3", ".m4a", ".webm", ".wav", ".ogg", ".flac"]:
        candidate = output_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def trim_audio_with_ffmpeg(audio_path: Path, duration: int, ffmpeg_dir: Path) -> bool:
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        return False
    temp_output = audio_path.with_name(f"{audio_path.stem}.trimmed{audio_path.suffix}")
    command = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(audio_path),
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "copy",
        str(temp_output),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_output.replace(audio_path)
        return True
    except Exception:
        temp_output.unlink(missing_ok=True)
        return False


def classify_yt_dlp_error(exc: BaseException) -> str:
    message = " ".join(str(part).strip() for part in exc.args if str(part).strip()) or str(exc)
    normalized = normalize_text(message)
    if any(pattern in normalized for pattern in YOUTUBE_BOT_PATTERNS):
        return "youtube_bot_check"
    if any(pattern in normalized for pattern in FORMAT_UNAVAILABLE_PATTERNS):
        return "format_unavailable"
    if any(pattern in normalized for pattern in VIDEO_UNAVAILABLE_PATTERNS):
        return "video_unavailable"
    return f"yt_dlp_error:{message[:160]}" if message else "yt_dlp_error"


def _bool_to_netscape(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def maybe_convert_cookie_json(cookie_path: Path) -> Path:
    try:
        raw_text = cookie_path.read_text(encoding="utf-8").strip()
    except Exception:
        return cookie_path
    if not raw_text.startswith("{") and not raw_text.startswith("["):
        return cookie_path

    try:
        payload = json.loads(raw_text)
    except Exception:
        return cookie_path

    cookies = payload.get("cookies") if isinstance(payload, dict) else payload
    if not isinstance(cookies, list):
        return cookie_path

    lines = ["# Netscape HTTP Cookie File"]
    for cookie in cookies:
        if not isinstance(cookie, dict):
            continue
        domain = str(cookie.get("domain") or "").strip()
        name = str(cookie.get("name") or "").strip()
        value = str(cookie.get("value") or "")
        path = str(cookie.get("path") or "/")
        if not domain or not name:
            continue
        include_subdomains = not bool(cookie.get("hostOnly", False))
        secure = bool(cookie.get("secure", False))
        expires = cookie.get("expirationDate")
        expires_int = int(float(expires)) if expires not in (None, "") else 0
        netscape_domain = f"#HttpOnly_{domain}" if bool(cookie.get("httpOnly", False)) else domain
        lines.append(
            "\t".join(
                [
                    netscape_domain,
                    _bool_to_netscape(include_subdomains),
                    path,
                    _bool_to_netscape(secure),
                    str(expires_int),
                    name,
                    value,
                ]
            )
        )

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="youtube_cookies_",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )
    temp_file.write("\n".join(lines) + "\n")
    temp_file.close()
    return Path(temp_file.name)


def resolve_query(
    query_variants: List[str],
    query_cache: Dict[str, Dict[str, str]],
    cache_lock: threading.Lock,
    ytdlp_semaphore: threading.BoundedSemaphore,
    args: argparse.Namespace,
) -> Dict[str, str] | None:
    for query in query_variants:
        cached = query_cache.get(query)
        if cached and cached.get("url"):
            return dict(cached)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "default_search": "ytsearch",
        "skip_download": True,
    }
    apply_yt_dlp_session_options(opts, args)
    for query in query_variants:
        try:
            with ytdlp_semaphore:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(f"ytsearch5:{query}", download=False)
        except Exception as exc:
            with cache_lock:
                query_cache.setdefault(
                    query,
                    {
                        "query": query,
                        "url": "",
                        "id": "",
                        "title": "",
                        "error": classify_yt_dlp_error(exc),
                    },
                )
            continue
        entries = info.get("entries") if isinstance(info, dict) else None
        if not entries:
            continue
        candidates = []
        for candidate in entries:
            if not isinstance(candidate, dict):
                continue
            url = str(candidate.get("webpage_url") or candidate.get("url") or "").strip()
            if not url:
                continue
            candidates.append(
                {
                    "url": url,
                    "id": str(candidate.get("id") or ""),
                    "title": str(candidate.get("title") or ""),
                }
            )
        if not candidates:
            continue
        payload = {
            "query": query,
            "url": candidates[0]["url"],
            "id": candidates[0]["id"],
            "title": candidates[0]["title"],
            "candidates": candidates,
        }
        with cache_lock:
            query_cache[query] = payload
        return payload
    return None


def apply_yt_dlp_session_options(ydl_opts: Dict[str, object], args: argparse.Namespace) -> None:
    if args.cookies_file:
        cookie_path = maybe_convert_cookie_json(Path(args.cookies_file))
        ydl_opts["cookiefile"] = str(cookie_path)
    if args.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = parse_browser_spec(str(args.cookies_from_browser))
    if args.sleep_interval_requests is not None:
        ydl_opts["sleep_interval_requests"] = float(args.sleep_interval_requests)
    if args.sleep_interval is not None:
        ydl_opts["sleep_interval"] = float(args.sleep_interval)
    if args.max_sleep_interval is not None:
        ydl_opts["max_sleep_interval"] = float(args.max_sleep_interval)


def parse_browser_spec(value: str) -> Tuple[str, str | None, str | None, str | None]:
    """Parse yt-dlp browser cookie syntax: BROWSER[+KEYRING][:PROFILE][::CONTAINER]."""
    match = re.fullmatch(
        r"""(?x)
        (?P<name>[^+:]+)
        (?:\s*\+\s*(?P<keyring>[^:]+))?
        (?:\s*:\s*(?!:)(?P<profile>.+?))?
        (?:\s*::\s*(?P<container>.+))?
        """,
        value.strip(),
    )
    if match is None:
        raise ValueError(f"invalid cookies-from-browser value: {value!r}")
    browser_name, keyring, profile, container = match.group("name", "keyring", "profile", "container")
    return (browser_name.lower(), profile, keyring.upper() if keyring else None, container)


def download_audio(
    url: str,
    output_stem: Path,
    duration: int,
    ffmpeg_dir: Path,
    args: argparse.Namespace,
    ytdlp_semaphore: threading.BoundedSemaphore,
) -> tuple[Path | None, str | None]:
    cleanup_stem_outputs(output_stem)
    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_stem),
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(ffmpeg_dir),
        "force_overwrites": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    attempts = []

    primary_opts = dict(base_opts)
    attempts.append(primary_opts)

    fallback_opts = dict(base_opts)
    fallback_opts["format"] = "bestaudio*/best"
    attempts.append(fallback_opts)

    last_error = None
    for ydl_opts in attempts:
        apply_yt_dlp_session_options(ydl_opts, args)
        try:
            with ytdlp_semaphore:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            audio_path = find_downloaded_audio(output_stem)
            if audio_path is None:
                cleanup_stem_outputs(output_stem)
                continue
            trim_audio_with_ffmpeg(audio_path, duration, ffmpeg_dir)
            return find_downloaded_audio(output_stem), None
        except Exception as exc:
            last_error = classify_yt_dlp_error(exc)
            cleanup_stem_outputs(output_stem)
            if last_error != "format_unavailable":
                return None, last_error

    return None, last_error or "download_failed"


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


def ensure_link_or_copy(source: Path, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def load_query_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_query_cache(cache: Dict[str, Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def format_rate_per_hour(done: int, elapsed_seconds: float) -> float:
    if done <= 0 or elapsed_seconds <= 0:
        return 0.0
    return done * 3600.0 / elapsed_seconds


def maybe_log_progress(
    *,
    processed: int,
    total: int,
    downloaded: int,
    existing: int,
    failed: int,
    rejected: int,
    started_at: float,
    report_every: int,
) -> None:
    if processed <= 0:
        return
    if processed != total and processed % report_every != 0:
        return
    elapsed = max(0.001, time.perf_counter() - started_at)
    rate = format_rate_per_hour(processed, elapsed)
    remaining = max(0, total - processed)
    eta_seconds = (remaining / processed * elapsed) if processed else 0.0
    print(
        "[progress] "
        f"{processed}/{total} "
        f"downloaded={downloaded} existing={existing} failed={failed} rejected={rejected} "
        f"rate={rate:.1f}_songs_per_hour "
        f"elapsed={elapsed/60:.1f}m "
        f"eta={eta_seconds/60:.1f}m"
    )


def process_song(
    song: Dict[str, object],
    *,
    cache_dir: Path,
    ffmpeg_dir: Path,
    duration: int,
    min_duration: float,
    min_size_bytes: int,
    fmt: str,
    args: argparse.Namespace,
    query_cache: Dict[str, Dict[str, str]],
    query_cache_lock: threading.Lock,
    ytdlp_semaphore: threading.BoundedSemaphore,
    ffmpeg_semaphore: threading.BoundedSemaphore,
) -> Dict[str, object]:
    canonical_path = build_canonical_path(cache_dir, str(song["unique_key"]), fmt)
    try:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)

        quality = inspect_audio_quality(canonical_path, min_duration=min_duration, min_size_bytes=min_size_bytes)
        if quality["is_valid"]:
            return {"song": song, "canonical_path": canonical_path, "quality": quality, "status": "existing_valid", "query": str(song["query_variants"][0])}

        resolution = resolve_query(
            list(song["query_variants"]),
            query_cache,
            query_cache_lock,
            ytdlp_semaphore,
            args,
        )
        if resolution is None:
            return {
                "song": song,
                "canonical_path": canonical_path,
                "quality": {
                    "is_valid": False,
                    "reason": "resolve_failed",
                    "duration_seconds": None,
                    "file_size_bytes": 0,
                },
                "status": "failed",
                "query": str(song["query_variants"][0]),
            }

        audio_path = None
        error_reason = None
        candidates = resolution.get("candidates") or [{"url": resolution["url"]}]
        for candidate in candidates:
            with ffmpeg_semaphore:
                audio_path, error_reason = download_audio(
                    url=str(candidate["url"]),
                    output_stem=canonical_path.with_suffix(""),
                    duration=duration,
                    ffmpeg_dir=ffmpeg_dir,
                    args=args,
                    ytdlp_semaphore=ytdlp_semaphore,
                )
            if audio_path is not None:
                break
            if error_reason not in {"format_unavailable", "video_unavailable"}:
                break

        if audio_path is None:
            return {
                "song": song,
                "canonical_path": canonical_path,
                "quality": {
                    "is_valid": False,
                    "reason": error_reason or "download_failed",
                    "duration_seconds": None,
                    "file_size_bytes": 0,
                },
                "status": "failed",
                "query": resolution["query"],
            }

        if audio_path.resolve() != canonical_path.resolve():
            audio_path.replace(canonical_path)
        quality = inspect_audio_quality(canonical_path, min_duration=min_duration, min_size_bytes=min_size_bytes)
        return {
            "song": song,
            "canonical_path": canonical_path,
            "quality": quality,
            "status": "downloaded_valid" if quality["is_valid"] else "rejected_downloaded",
            "query": resolution["query"],
        }
    except BaseException as exc:
        return {
            "song": song,
            "canonical_path": canonical_path,
            "quality": {
                "is_valid": False,
                "reason": f"unexpected_error:{str(exc)[:160]}",
                "duration_seconds": None,
                "file_size_bytes": 0,
            },
            "status": "failed",
            "query": str(song["query_variants"][0]),
        }


def write_manifest(rows: List[Dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized audio dataset folders.")
    parser.add_argument("--metadata-csv", default=str(Path(get_data_dir()) / "mbti_cnn_metadata.csv"))
    parser.add_argument("--output-dir", default=str(Path(get_data_dir()) / "audio_files"))
    parser.add_argument("--cache-dir", default=str(Path(get_data_dir()) / "audio_cache"))
    parser.add_argument("--manifest-path", default=str(Path(get_data_dir()) / "audio_manifest.csv"))
    parser.add_argument("--query-cache-path", default=str(Path(get_data_dir()) / "youtube_query_cache.json"))
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--per-label-limit", type=int, default=100)
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--min-size-bytes", type=int, default=180000)
    parser.add_argument("--format", default="mp3", choices=["mp3"])
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--ffmpeg-workers", type=int, default=8)
    parser.add_argument("--yt-dlp-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--cookies-file", default=None)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--sleep-interval-requests", type=float, default=0.75)
    parser.add_argument("--sleep-interval", type=float, default=2.0)
    parser.add_argument("--max-sleep-interval", type=float, default=5.0)
    parser.add_argument("--ffmpeg-dir", default=str(PROJECT_ROOT / "ffmpeg-master-latest-win64-gpl" / "bin"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--cleanup-first", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.metadata_csv)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    ffmpeg_dir = Path(args.ffmpeg_dir)
    manifest_path = Path(args.manifest_path)
    query_cache_path = Path(args.query_cache_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = normalize_rows(df)
    if args.skip_existing:
        candidate_rows = filter_out_materialized_rows(candidate_rows, output_dir)
    rows = limit_rows(candidate_rows, args.per_label_limit, args.total_limit)
    songs = group_rows_by_song(rows)
    min_duration = float(args.min_duration if args.min_duration is not None else max(8.0, args.duration * 0.6))
    cleanup_summary = {"invalid_removed": 0, "duplicates_removed": 0}
    if args.cleanup_first:
        cleanup_summary = cleanup_audio_directory(
            audio_dir=output_dir,
            min_duration=min_duration,
            min_size_bytes=args.min_size_bytes,
        )

    query_cache = load_query_cache(query_cache_path)
    query_cache_lock = threading.Lock()
    ytdlp_semaphore = threading.BoundedSemaphore(max(1, args.yt_dlp_workers))
    ffmpeg_semaphore = threading.BoundedSemaphore(max(1, args.ffmpeg_workers))

    song_results: List[Dict[str, object]] = []
    song_downloaded = 0
    song_existing = 0
    song_failed = 0
    song_rejected = 0
    progress_started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as executor:
        futures = [
            executor.submit(
                process_song,
                song,
                cache_dir=cache_dir,
                ffmpeg_dir=ffmpeg_dir,
                duration=args.duration,
                min_duration=min_duration,
                min_size_bytes=args.min_size_bytes,
                fmt=args.format,
                args=args,
                query_cache=query_cache,
                query_cache_lock=query_cache_lock,
                ytdlp_semaphore=ytdlp_semaphore,
                ffmpeg_semaphore=ffmpeg_semaphore,
            )
            for song in songs
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "song": {"entries": [], "query_variants": [""]},
                    "canonical_path": cache_dir / "unknown.mp3",
                    "quality": {
                        "is_valid": False,
                        "reason": f"future_error:{str(exc)[:160]}",
                        "duration_seconds": None,
                        "file_size_bytes": 0,
                    },
                    "status": "failed",
                    "query": "",
                }
            song_results.append(result)
            song_quality = dict(result["quality"])
            song_status = str(result["status"])
            if song_status == "downloaded_valid":
                song_downloaded += 1
            elif song_quality.get("is_valid"):
                song_existing += 1
            elif song_status == "failed":
                song_failed += 1
            else:
                song_rejected += 1
            maybe_log_progress(
                processed=len(song_results),
                total=len(songs),
                downloaded=song_downloaded,
                existing=song_existing,
                failed=song_failed,
                rejected=song_rejected,
                started_at=progress_started_at,
                report_every=max(1, args.progress_every),
            )

    save_query_cache(query_cache, query_cache_path)

    manifest_rows: List[Dict[str, object]] = []
    downloaded = 0
    skipped = 0
    failed = 0
    rejected = 0
    materialized = 0

    for result in song_results:
        quality = dict(result["quality"])
        song_status = str(result["status"])
        canonical_path = Path(result["canonical_path"])
        if song_status == "downloaded_valid":
            downloaded += 1
        elif quality.get("is_valid"):
            skipped += 1
        elif song_status == "failed":
            failed += 1
        else:
            rejected += 1

        for entry in result["song"]["entries"]:
            label = entry["mbti_label"]
            target_dir = output_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            stable_slug = build_track_slug(entry["title"], entry["artists"])
            stem_prefix = sanitize_filename(f"{int(entry['index']):05d}_{stable_slug}")
            target_path = target_dir / f"{stem_prefix}.{args.format}"
            existing_path = find_existing_audio(target_dir, stable_slug, stem_prefix) if args.skip_existing else None

            manifest_entry = {
                "mbti_label": entry["mbti_label"],
                "title": entry["title"],
                "artists": entry["artists"],
            }
            if existing_path is not None:
                label_quality = inspect_audio_quality(existing_path, min_duration=min_duration, min_size_bytes=args.min_size_bytes)
                manifest_entry["audio_path"] = str(existing_path.resolve())
                manifest_entry["duration_seconds"] = label_quality["duration_seconds"]
                manifest_entry["file_size_bytes"] = label_quality["file_size_bytes"]
                manifest_entry["quality_reason"] = label_quality["reason"]
                manifest_entry["status"] = "existing_valid" if label_quality["is_valid"] else "rejected_existing"
                if not label_quality["is_valid"] and args.delete_invalid and existing_path.exists():
                    os.remove(existing_path)
                manifest_rows.append(manifest_entry)
                continue

            if quality.get("is_valid"):
                linked_path = ensure_link_or_copy(canonical_path, target_path)
                materialized += 1
                label_quality = inspect_audio_quality(linked_path, min_duration=min_duration, min_size_bytes=args.min_size_bytes)
                manifest_entry["audio_path"] = str(linked_path.resolve())
                manifest_entry["duration_seconds"] = label_quality["duration_seconds"]
                manifest_entry["file_size_bytes"] = label_quality["file_size_bytes"]
                manifest_entry["quality_reason"] = label_quality["reason"]
                manifest_entry["status"] = "downloaded_valid" if song_status == "downloaded_valid" else "existing_valid"
            else:
                manifest_entry["audio_path"] = ""
                manifest_entry["duration_seconds"] = None
                manifest_entry["file_size_bytes"] = 0
                manifest_entry["quality_reason"] = quality.get("reason") or "download_failed"
                manifest_entry["status"] = "failed" if song_status == "failed" else "rejected_downloaded"
            manifest_entry["query"] = result["query"]
            manifest_rows.append(manifest_entry)

    manifest_rows = dedupe_manifest_rows(manifest_rows)
    write_manifest(manifest_rows, manifest_path)
    summary = {
        "selected_rows": len(rows),
        "unique_songs": len(songs),
        "downloaded": downloaded,
        "existing": skipped,
        "failed": failed,
        "rejected": rejected,
        "materialized": materialized,
        "min_duration": min_duration,
        "min_size_bytes": args.min_size_bytes,
        "download_workers": args.download_workers,
        "ffmpeg_workers": args.ffmpeg_workers,
        "yt_dlp_workers": args.yt_dlp_workers,
        "cleanup": cleanup_summary,
        "output_dir": str(output_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "query_cache_path": str(query_cache_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    print("__AUDIO_DATASET_SUMMARY__=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
