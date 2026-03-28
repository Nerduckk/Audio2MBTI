"""Unified playlist-to-feature pipeline for Spotify, YouTube, and Apple Music."""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import pandas as pd

try:
    from .apple_music_process import fetch_apple_music_album
    from .apple_music_process import fetch_apple_music_playlist
    from .processing_utils import (
        analyze_audio_features,
        analyze_lyrics_sentiment,
        download_audio_segment,
        fetch_track_metadata,
    )
    from .spotify_process import fetch_spotify_album
    from .spotify_process import fetch_spotify_playlist
    from .youtube_process import fetch_youtube_playlist
except ImportError:
    from apple_music_process import fetch_apple_music_album
    from apple_music_process import fetch_apple_music_playlist
    from processing_utils import (
        analyze_audio_features,
        analyze_lyrics_sentiment,
        download_audio_segment,
        fetch_track_metadata,
    )
    from spotify_process import fetch_spotify_album
    from spotify_process import fetch_spotify_playlist
    from youtube_process import fetch_youtube_playlist

PLAYLIST_FETCHERS: Dict[str, Callable[[str], Dict[str, object]]] = {
    "spotify": fetch_spotify_playlist,
    "youtube": fetch_youtube_playlist,
    "apple_music": fetch_apple_music_playlist,
}

COLLECTION_FETCHERS: Dict[tuple[str, str], Callable[[str], Dict[str, object]]] = {
    ("spotify", "playlist"): fetch_spotify_playlist,
    ("spotify", "album"): fetch_spotify_album,
    ("youtube", "playlist"): fetch_youtube_playlist,
    ("youtube", "album"): fetch_youtube_playlist,
    ("apple_music", "playlist"): fetch_apple_music_playlist,
    ("apple_music", "album"): fetch_apple_music_album,
}


def detect_platform(playlist_url: str) -> str:
    lowered = playlist_url.lower()
    if "open.spotify.com/playlist/" in lowered:
        return "spotify"
    if "youtube.com/playlist" in lowered or "youtu.be/" in lowered and "list=" in lowered:
        return "youtube"
    if "music.apple.com/" in lowered and "/playlist/" in lowered:
        return "apple_music"
    raise ValueError("Unsupported playlist URL.")


def detect_kind(collection_url: str) -> str:
    lowered = collection_url.lower()
    if "/album/" in lowered:
        return "album"
    return "playlist"


def process_track_to_feature_row(track: Dict[str, object]) -> Dict[str, object] | None:
    title = str(track.get("title") or "").strip()
    artists = list(track.get("artists") or [])
    artists_text = str(track.get("artists_text") or ", ".join(artists)).strip()
    if not title:
        return None

    metadata = fetch_track_metadata(title, artists_text)
    audio_path = download_audio_segment(f"{title} {artists_text} audio")
    if not audio_path:
        return None

    try:
        audio_features = analyze_audio_features(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if not audio_features:
        return None

    lyrics_features = analyze_lyrics_sentiment(title, artists_text)
    return {
        "title": title,
        "artists": artists_text,
        "popularity_proxy": metadata["popularity_proxy"],
        "release_year": metadata["release_year"],
        "artist_genres": metadata["artist_genres"],
        "genre_ei_score": metadata["genre_ei_score"],
        "genre_sn_score": metadata["genre_sn_score"],
        "genre_tf_score": metadata["genre_tf_score"],
        "tempo_bpm": audio_features["tempo_bpm"],
        "energy": audio_features["energy"],
        "danceability": audio_features["danceability"],
        "spectral_centroid": audio_features["spectral_centroid"],
        "spectral_flatness": audio_features["spectral_flatness"],
        "zero_crossing_rate": audio_features["zero_crossing_rate"],
        "spectral_bandwidth": audio_features["spectral_bandwidth"],
        "spectral_rolloff": audio_features["spectral_rolloff"],
        "mfcc_mean": audio_features["mfcc_mean"],
        "chroma_mean": audio_features["chroma_mean"],
        "tempo_strength": audio_features["tempo_strength"],
        "lyrics_polarity": lyrics_features["lyrics_polarity"],
        "lyrics_joy": lyrics_features["lyrics_joy"],
        "lyrics_sadness": lyrics_features["lyrics_sadness"],
        "lyrics_anger": lyrics_features["lyrics_anger"],
        "lyrics_love": lyrics_features["lyrics_love"],
        "lyrics_fear": lyrics_features["lyrics_fear"],
        "source_platform": track.get("source_platform"),
        "external_url": track.get("external_url"),
        "source_track_id": track.get("track_id") or track.get("video_id"),
        "source_track_uri": track.get("track_uri"),
    }


def process_playlist(playlist_url: str, platform: str | None = None, limit: int | None = None) -> Dict[str, object]:
    resolved_platform = platform or detect_platform(playlist_url)
    kind = detect_kind(playlist_url)
    fetcher = COLLECTION_FETCHERS[(resolved_platform, kind)]
    playlist = fetcher(playlist_url)
    tracks = list(playlist.get("tracks", []))
    if limit is not None:
        tracks = tracks[:limit]

    rows: List[Dict[str, object]] = []
    for track in tracks:
        row = process_track_to_feature_row(track)
        if row:
            rows.append(row)

    return {
        "platform": resolved_platform,
        "kind": kind,
        "playlist_id": playlist.get("playlist_id"),
        "playlist_title": playlist.get("title"),
        "playlist_url": playlist_url,
        "processed_count": len(rows),
        "requested_count": len(tracks),
        "rows": rows,
    }


def playlist_to_dataframe(playlist_url: str, platform: str | None = None, limit: int | None = None) -> pd.DataFrame:
    result = process_playlist(playlist_url, platform=platform, limit=limit)
    return pd.DataFrame(result["rows"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("playlist_url")
    parser.add_argument("--platform", choices=sorted(PLAYLIST_FETCHERS), default=None)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    df = playlist_to_dataframe(args.playlist_url, platform=args.platform, limit=args.limit)
    print(df.to_json(orient="records", force_ascii=False, indent=2))
