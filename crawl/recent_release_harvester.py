"""Harvest recent albums/playlists from seed URLs and append real-song rows to master."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .file_paths import get_master_csv_path
    from .playlist_feature_pipeline import process_track_to_feature_row
    from .spotify_process import fetch_spotify_album, fetch_spotify_playlist
    from .youtube_process import fetch_youtube_playlist
    from .apple_music_process import fetch_apple_music_album, fetch_apple_music_playlist
except ImportError:
    from file_paths import get_master_csv_path
    from playlist_feature_pipeline import process_track_to_feature_row
    from spotify_process import fetch_spotify_album, fetch_spotify_playlist
    from youtube_process import fetch_youtube_playlist
    from apple_music_process import fetch_apple_music_album, fetch_apple_music_playlist


FETCHERS = {
    ("spotify", "album"): fetch_spotify_album,
    ("spotify", "playlist"): fetch_spotify_playlist,
    ("youtube", "album"): fetch_youtube_playlist,
    ("youtube", "playlist"): fetch_youtube_playlist,
    ("apple_music", "album"): fetch_apple_music_album,
    ("apple_music", "playlist"): fetch_apple_music_playlist,
}


def load_seed_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def harvest_recent_sources(manifest_path: str, output_csv: str, min_year: int, max_year: int, limit_per_source: int | None = None):
    seeds = load_seed_manifest(manifest_path)
    output_path = Path(output_csv)
    if output_path.exists() and output_path.stat().st_size > 20:
        master_df = pd.read_csv(output_path)
    else:
        master_df = pd.DataFrame()

    seen_keys = set()
    if not master_df.empty and {"title", "artists"} <= set(master_df.columns):
        for _, row in master_df.dropna(subset=["title", "artists"]).iterrows():
            seen_keys.add(f"{str(row['title']).strip().lower()} - {str(row['artists']).strip().lower()}")

    rows = []
    for seed in seeds:
        if not seed.get("enabled", True):
            continue
        platform = seed["platform"]
        kind = seed.get("kind", "playlist")
        mbti_label = seed["mbti_label"]
        url = seed["url"]
        fetcher = FETCHERS[(platform, kind)]
        collection = fetcher(url)
        tracks = list(collection.get("tracks", []))
        if limit_per_source is not None:
            tracks = tracks[:limit_per_source]

        for track in tracks:
            row = process_track_to_feature_row(track)
            if not row:
                continue
            release_year = int(row.get("release_year") or 0)
            if release_year < min_year or release_year > max_year:
                continue
            key = f"{row['title'].strip().lower()} - {row['artists'].strip().lower()}"
            if key in seen_keys:
                continue
            row["mbti_label"] = mbti_label
            row["source_seed_url"] = url
            row["source_seed_kind"] = kind
            rows.append(row)
            seen_keys.add(key)

    if not rows:
        print("No new rows harvested.")
        return 0

    new_df = pd.DataFrame(rows)
    combined = pd.concat([master_df, new_df], ignore_index=True) if not master_df.empty else new_df
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Added {len(new_df)} rows to {output_path}")
    return len(new_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"D:\project\config\recent_source_seeds.json")
    parser.add_argument("--output", default=get_master_csv_path())
    parser.add_argument("--min-year", type=int, default=max(2020, datetime.now().year - 2))
    parser.add_argument("--max-year", type=int, default=datetime.now().year)
    parser.add_argument("--limit-per-source", type=int, default=None)
    args = parser.parse_args()

    harvest_recent_sources(
        manifest_path=args.manifest,
        output_csv=args.output,
        min_year=args.min_year,
        max_year=args.max_year,
        limit_per_source=args.limit_per_source,
    )
