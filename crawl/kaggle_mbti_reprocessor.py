# -*- coding: utf-8 -*-
"""Reprocess Kaggle MBTI playlists into real-song training rows without Spotify API."""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from spotify_scraper import SpotifyClient

try:
    from .file_paths import ensure_data_dir_exists, get_master_csv_path
    from .mbti_genre_processor import calculate_genre_mbti_scores
    from .processing_utils import (
        analyze_audio_features,
        analyze_lyrics_sentiment,
        download_audio_segment,
        fetch_track_metadata,
    )
except ImportError:
    from file_paths import ensure_data_dir_exists, get_master_csv_path
    from mbti_genre_processor import calculate_genre_mbti_scores
    from processing_utils import (
        analyze_audio_features,
        analyze_lyrics_sentiment,
        download_audio_segment,
        fetch_track_metadata,
    )

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from infrastructure.batch_processor import BatchProcessor
from infrastructure.parallel_processor import ParallelProcessor
from infrastructure.retry_logic import RateLimiter

load_dotenv()

spotify_rate_limiter = RateLimiter(requests_per_second=1.0)


def _normalize_track(track: dict, mbti_label: str, processed_songs: set[str]) -> dict | None:
    title = str(track.get("name") or track.get("title") or "").strip()
    artists_data = track.get("artists", [])
    if isinstance(artists_data, list):
        artists = ", ".join(
            a.get("name", "").strip() if isinstance(a, dict) else str(a).strip()
            for a in artists_data
            if a
        )
    else:
        artists = str(artists_data or "").strip()

    if not title or not artists:
        return None

    return {
        "name": title,
        "artists": artists,
        "mbti_label": mbti_label,
        "processed_songs": processed_songs,
    }


def process_track_with_all_features(track_info: dict, batch_processor: BatchProcessor, processed_songs_lock: threading.Lock):
    name = track_info.get("name", "").strip()
    artists = track_info.get("artists", "").replace("\xa0", " ").strip()
    mbti_label = track_info.get("mbti_label", "")
    processed_songs = track_info.get("processed_songs", set())

    if not name or not artists:
        return None

    song_key = f"{name.lower()} - {artists.lower()}"
    with processed_songs_lock:
        if song_key in processed_songs:
            return None
        processed_songs.add(song_key)

    try:
        print(f"   Xu ly: {name} - {artists} (MBTI: {mbti_label})")

        metadata = fetch_track_metadata(name, artists)
        genres_list = metadata["genres_list"]
        genre_scores = calculate_genre_mbti_scores(genres_list)

        audio_path = download_audio_segment(f"{name} {artists} audio")
        if not audio_path:
            with processed_songs_lock:
                processed_songs.discard(song_key)
            return None

        try:
            features = analyze_audio_features(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        if not features:
            with processed_songs_lock:
                processed_songs.discard(song_key)
            return None

        lyrics = analyze_lyrics_sentiment(name, artists)

        row = {
            "title": name,
            "artists": artists,
            "popularity_proxy": metadata["popularity_proxy"],
            "release_year": metadata["release_year"],
            "artist_genres": metadata["artist_genres"],
            "genre_ei_score": genre_scores["genre_ei"],
            "genre_sn_score": genre_scores["genre_sn"],
            "genre_tf_score": genre_scores["genre_tf"],
            "tempo_bpm": features["tempo_bpm"],
            "energy": features["energy"],
            "danceability": features["danceability"],
            "spectral_centroid": features["spectral_centroid"],
            "spectral_flatness": features["spectral_flatness"],
            "zero_crossing_rate": features["zero_crossing_rate"],
            "spectral_bandwidth": features["spectral_bandwidth"],
            "spectral_rolloff": features["spectral_rolloff"],
            "mfcc_mean": features["mfcc_mean"],
            "chroma_mean": features["chroma_mean"],
            "tempo_strength": features["tempo_strength"],
            "lyrics_polarity": lyrics["lyrics_polarity"],
            "lyrics_joy": lyrics["lyrics_joy"],
            "lyrics_sadness": lyrics["lyrics_sadness"],
            "lyrics_anger": lyrics["lyrics_anger"],
            "lyrics_love": lyrics["lyrics_love"],
            "lyrics_fear": lyrics["lyrics_fear"],
            "mbti_label": mbti_label,
        }

        batch_processor.add(row)
        return row
    except Exception as exc:
        print(f"     [!] Loi xu ly track: {exc}")
        with processed_songs_lock:
            processed_songs.discard(song_key)
        return None


def _load_playlist_ids(kaggle_dir: str) -> list[tuple[str, str]]:
    playlist_ids: list[tuple[str, str]] = []
    for file in os.listdir(kaggle_dir):
        if file.endswith("_df.csv") and not file.startswith("combined"):
            temp_df = pd.read_csv(os.path.join(kaggle_dir, file))
            if "playlist_id" in temp_df.columns and "mbti" in temp_df.columns:
                label = file.split("_")[0]
                for pid in temp_df["playlist_id"].dropna().unique().tolist():
                    playlist_ids.append((pid, label))
    return playlist_ids


def mass_reprocess_kaggle(max_playlists: int | None = None, max_tracks_per_playlist: int | None = None):
    print("==================================================")
    print(" TOOL REPROCESS DU LIEU KAGGLE (NO SPOTIFY API) ")
    print("==================================================")

    kaggle_dir = r"data\kaggle data set"
    ensure_data_dir_exists()
    output_csv = get_master_csv_path()

    if not os.path.exists(kaggle_dir):
        print(f" Khong tim thay thu muc {kaggle_dir}")
        return

    playlist_ids = _load_playlist_ids(kaggle_dir)
    if not playlist_ids:
        print(" Khong tim thay du lieu playlist_id trong cac file CSV.")
        return

    random.seed(42)
    random.shuffle(playlist_ids)
    if max_playlists is not None:
        playlist_ids = playlist_ids[:max_playlists]

    processed_songs: set[str] = set()
    expected_columns = [
        "title",
        "artists",
        "popularity_proxy",
        "release_year",
        "artist_genres",
        "genre_ei_score",
        "genre_sn_score",
        "genre_tf_score",
        "tempo_bpm",
        "energy",
        "danceability",
        "spectral_centroid",
        "spectral_flatness",
        "zero_crossing_rate",
        "spectral_bandwidth",
        "spectral_rolloff",
        "mfcc_mean",
        "chroma_mean",
        "tempo_strength",
        "lyrics_polarity",
        "lyrics_joy",
        "lyrics_sadness",
        "lyrics_anger",
        "lyrics_love",
        "lyrics_fear",
        "mbti_label",
    ]

    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 20:
        try:
            df_out = pd.read_csv(output_csv)
            if "title" in df_out.columns and "artists" in df_out.columns:
                df_out = df_out.dropna(subset=["title", "artists"])
                for _, row in df_out.iterrows():
                    processed_songs.add(
                        f"{str(row['title']).strip().lower()} - {str(row['artists']).strip().lower()}"
                    )
                print(f" Tim thay {output_csv}, da tai {len(processed_songs)} bai hat da xu ly.")
        except pd.errors.EmptyDataError:
            pass
    else:
        pd.DataFrame(columns=expected_columns).to_csv(output_csv, index=False, encoding="utf-8-sig")

    batch_processor = BatchProcessor(batch_size=50, output_file=output_csv)
    parallel_processor = ParallelProcessor(num_workers=4, use_multiprocessing=False, chunk_size=2)
    processed_songs_lock = threading.Lock()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    spotify_scraper = SpotifyClient()

    total_playlists = len(playlist_ids)
    for idx, (pid, mbti_label) in enumerate(playlist_ids, 1):
        try:
            percentage = (idx / total_playlists) * 100
            print("\n========================================================================")
            print(f" TIEN DO TONG THE: {percentage:.2f}% (Dang xu ly Playlist thu {idx} / {total_playlists})")
            print("========================================================================")
            print(f" Dang cao Track List cho Playlist ID: {pid} (MBTI: {mbti_label})")

            playlist_url = f"https://open.spotify.com/playlist/{pid}"
            tracks_data = []
            for attempt in range(3):
                try:
                    spotify_rate_limiter.wait()
                    playlist_info = spotify_scraper.get_playlist_info(playlist_url)
                    tracks_data = playlist_info.get("tracks", []) or []
                    if tracks_data:
                        break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        print(f"   [!] Playlist retry {attempt + 1}: {str(exc)[:80]}")

            if not tracks_data:
                print("   [!] SKIPPED: Khong lay duoc playlist.")
                continue

            tracks = []
            for track in tracks_data:
                if not isinstance(track, dict):
                    continue
                normalized = _normalize_track(track, mbti_label, processed_songs)
                if normalized:
                    tracks.append(normalized)

            if max_tracks_per_playlist is not None:
                tracks = tracks[:max_tracks_per_playlist]

            if tracks:
                results = parallel_processor.map(
                    lambda track_info: process_track_with_all_features(
                        track_info, batch_processor, processed_songs_lock
                    ),
                    tracks,
                )
                valid_results = [item for item in results if item is not None]
                print(f"     OK Xu ly xong {len(valid_results)}/{len(tracks)} tracks tu playlist.")

            time.sleep(random.uniform(2, 4))
        except Exception as exc:
            print(f" Loi xu ly playlist {pid}: {exc}")
            time.sleep(5)
            continue

    batch_processor.flush()
    spotify_scraper.close()
    print(f"\nOK Hoan thanh! Tong cong {batch_processor.total_saved} tracks duoc luu vao {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-playlists", type=int, default=None)
    parser.add_argument("--max-tracks-per-playlist", type=int, default=None)
    args = parser.parse_args()
    mass_reprocess_kaggle(
        max_playlists=args.max_playlists,
        max_tracks_per_playlist=args.max_tracks_per_playlist,
    )
