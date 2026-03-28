"""YouTube playlist crawler using yt-dlp metadata extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import yt_dlp


def extract_youtube_playlist_id(playlist_url: str) -> str | None:
    match = re.search(r"[?&]list=([A-Za-z0-9_-]+)", playlist_url)
    return match.group(1) if match else None


def _split_title_and_artist(video_title: str, uploader: str) -> tuple[str, List[str]]:
    clean_title = str(video_title or "").strip()
    if " - " in clean_title:
        artist_part, song_part = clean_title.split(" - ", 1)
        artists = [artist_part.strip()] if artist_part.strip() else []
        title = song_part.strip() or clean_title
        return title, artists
    artists = [uploader.strip()] if uploader and uploader.strip() else []
    return clean_title, artists


def _normalize_entry(entry: Dict[str, Any], position: int) -> Dict[str, Any] | None:
    video_title = str(entry.get("title") or "").strip()
    if not video_title:
        return None

    uploader = str(entry.get("uploader") or entry.get("channel") or "").strip()
    title, artists = _split_title_and_artist(video_title, uploader)

    return {
        "position": position,
        "title": title,
        "artists": artists,
        "artists_text": ", ".join(artists),
        "album": "",
        "duration_ms": int(entry.get("duration", 0) * 1000) if entry.get("duration") else None,
        "external_url": str(entry.get("webpage_url") or entry.get("url") or "").strip(),
        "video_id": str(entry.get("id") or "").strip(),
        "source_platform": "youtube",
    }


def fetch_youtube_playlist(playlist_url: str) -> Dict[str, Any]:
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": None,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    entries = info.get("entries", []) or []
    tracks = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_entry(entry, index)
        if normalized:
            tracks.append(normalized)

    return {
        "platform": "youtube",
        "playlist_id": extract_youtube_playlist_id(playlist_url),
        "playlist_url": playlist_url,
        "title": str(info.get("title") or "").strip(),
        "description": str(info.get("description") or "").strip(),
        "owner": str(info.get("uploader") or info.get("channel") or "").strip(),
        "track_count": len(tracks),
        "tracks": tracks,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("playlist_url")
    args = parser.parse_args()
    print(json.dumps(fetch_youtube_playlist(args.playlist_url), ensure_ascii=False, indent=2))
