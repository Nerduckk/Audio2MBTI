"""Fast Kaggle metadata-only crawler for the CNN pipeline."""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from spotify_scraper import SpotifyClient

sys.path.insert(0, str(Path(__file__).parent))
from file_paths import ensure_data_dir_exists

sys.path.insert(0, str(Path(__file__).parent.parent))
from infrastructure.batch_processor import BatchProcessor
from infrastructure.retry_logic import RateLimiter

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

spotify_rate_limiter = RateLimiter(requests_per_second=1.0)


def get_cnn_metadata_csv() -> str:
    """Dedicated metadata CSV for the CNN pipeline."""
    ensure_data_dir_exists()
    return str(Path("data") / "mbti_cnn_metadata.csv")


def load_playlist_ids(kaggle_dir: str) -> list[tuple[str, str]]:
    playlist_ids: list[tuple[str, str]] = []
    for file in os.listdir(kaggle_dir):
        if not file.endswith("_df.csv") or file.startswith("combined"):
            continue
        temp_df = pd.read_csv(os.path.join(kaggle_dir, file))
        if "playlist_id" not in temp_df.columns or "mbti" not in temp_df.columns:
            continue
        label_values = temp_df["mbti"].dropna().astype(str).str.upper().unique().tolist()
        if not label_values:
            continue
        label = label_values[0]
        for pid in temp_df["playlist_id"].dropna().astype(str).unique().tolist():
            playlist_ids.append((pid, label))
    return playlist_ids


def normalize_track(track: dict, mbti_label: str, playlist_id: str, playlist_url: str) -> dict | None:
    title = str(track.get("name") or track.get("title") or "").strip()
    artists_data = track.get("artists", [])
    if isinstance(artists_data, list):
        artists = ", ".join(
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in artists_data
            if a
        ).strip()
    else:
        artists = str(artists_data).strip()

    if not title or not artists:
        return None

    external_url = str(track.get("url") or track.get("external_url") or "").strip()
    track_id = str(track.get("id") or track.get("track_id") or "").strip()

    return {
        "title": title,
        "artists": artists,
        "mbti_label": mbti_label,
        "source_platform": "spotify",
        "external_url": external_url,
        "source_track_id": track_id,
        "source_track_uri": f"spotify:track:{track_id}" if track_id else "",
        "source_seed_url": playlist_url,
        "source_seed_kind": "playlist",
        "playlist_id": playlist_id,
    }


def mass_reprocess_kaggle_metadata(max_playlists=None, max_tracks_per_playlist=None, batch_size=50):
    print("==================================================")
    print(" KAGGLE METADATA-ONLY REPROCESSOR FOR CNN ")
    print("==================================================")

    kaggle_dir = r"data\kaggle data set"
    output_csv = get_cnn_metadata_csv()
    batch_processor = BatchProcessor(batch_size=batch_size, output_file=output_csv)
    random.seed(42)

    if not os.path.exists(kaggle_dir):
        print(f"Không tìm thấy thư mục {kaggle_dir}")
        return

    playlist_ids = load_playlist_ids(kaggle_dir)
    if not playlist_ids:
        print("Không tìm thấy playlist_id trong dữ liệu Kaggle.")
        return

    random.shuffle(playlist_ids)
    if max_playlists is not None:
        playlist_ids = playlist_ids[:max_playlists]

    processed_songs = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 20:
        try:
            df_out = pd.read_csv(output_csv)
            if "title" in df_out.columns and "artists" in df_out.columns:
                df_out = df_out.dropna(subset=["title", "artists", "mbti_label"])
                for _, row in df_out.iterrows():
                    song_key = (
                        f"{str(row['mbti_label']).strip().upper()}|"
                        f"{str(row['title']).strip().lower()}|"
                        f"{str(row['artists']).strip().lower()}"
                    )
                    processed_songs.add(song_key)
                print(f"Tìm thấy {output_csv}, đã có {len(processed_songs)} bài metadata.")
        except pd.errors.EmptyDataError:
            pass
    else:
        df_empty = pd.DataFrame(
            columns=[
                "title",
                "artists",
                "mbti_label",
                "source_platform",
                "external_url",
                "source_track_id",
                "source_track_uri",
                "source_seed_url",
                "source_seed_kind",
                "playlist_id",
            ]
        )
        df_empty.to_csv(output_csv, index=False, encoding="utf-8-sig")

    spotify_scraper = SpotifyClient()
    total_playlists = len(playlist_ids)

    for idx, (pid, mbti_label) in enumerate(playlist_ids, 1):
        try:
            percentage = (idx / total_playlists) * 100
            print(f"\n==================================================")
            print(f"TIẾN ĐỘ: {percentage:.2f}% ({idx}/{total_playlists})")
            print(f"Playlist ID: {pid} | MBTI: {mbti_label}")
            print("==================================================")

            playlist_url = f"https://open.spotify.com/playlist/{pid}"
            tracks_data = []
            attempts = 0
            while attempts < 3:
                try:
                    spotify_rate_limiter.wait()
                    playlist_info = spotify_scraper.get_playlist_info(playlist_url)
                    tracks_data = playlist_info.get("tracks", []) or []
                    if tracks_data:
                        break
                except Exception as exc:
                    print(f"  [!] Spotify scraper error: {str(exc)[:120]}")
                attempts += 1
                time.sleep(2 ** attempts)

            if not tracks_data:
                print("  [!] SKIPPED: không lấy được track list.")
                continue

            if max_tracks_per_playlist is not None:
                tracks_data = tracks_data[:max_tracks_per_playlist]

            new_rows = 0
            duplicates = 0
            for track in tracks_data:
                if not isinstance(track, dict):
                    continue
                row = normalize_track(track, mbti_label, pid, playlist_url)
                if row is None:
                    continue

                song_key = f"{row['mbti_label']}|{row['title'].lower()}|{row['artists'].lower()}"
                if song_key in processed_songs:
                    duplicates += 1
                    continue

                processed_songs.add(song_key)
                batch_processor.add(row)
                new_rows += 1

            print(f"  ✓ new_rows={new_rows} | duplicates={duplicates} | total_saved={batch_processor.total_saved}")
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as exc:
            print(f"  [!] Playlist error {pid}: {exc}")
            time.sleep(3)

    batch_processor.flush()
    spotify_scraper.close()
    print(f"\n✓ Hoàn thành! Metadata lưu tại {output_csv} | total_saved={batch_processor.total_saved}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-playlists", type=int, default=None)
    parser.add_argument("--max-tracks-per-playlist", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    mass_reprocess_kaggle_metadata(
        max_playlists=args.max_playlists,
        max_tracks_per_playlist=args.max_tracks_per_playlist,
        batch_size=args.batch_size,
    )
