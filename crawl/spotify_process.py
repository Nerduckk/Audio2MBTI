"""Spotify playlist crawler using no-auth web scraping."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from spotify_scraper import SpotifyClient


def extract_spotify_playlist_id(playlist_url: str) -> str | None:
    match = re.search(r"playlist/([A-Za-z0-9]+)", playlist_url)
    return match.group(1) if match else None


def _normalize_artists(artists_data: Any) -> List[str]:
    if isinstance(artists_data, list):
        names = []
        for artist in artists_data:
            if isinstance(artist, dict):
                name = str(artist.get("name", "")).strip()
            else:
                name = str(artist).strip()
            if name:
                names.append(name)
        return names

    if isinstance(artists_data, str):
        return [part.strip() for part in artists_data.split(",") if part.strip()]

    return []


def _normalize_track(track: Dict[str, Any], position: int) -> Dict[str, Any] | None:
    title = str(track.get("name") or track.get("title") or "").strip()
    artists = _normalize_artists(track.get("artists", []))
    if not title:
        return None

    return {
        "position": position,
        "title": title,
        "artists": artists,
        "artists_text": ", ".join(artists),
        "album": str(track.get("album_name") or track.get("album") or "").strip(),
        "duration_ms": track.get("duration_ms"),
        "external_url": str(track.get("url") or track.get("external_url") or "").strip(),
        "source_platform": "spotify",
    }


def fetch_spotify_playlist(playlist_url: str, client: SpotifyClient | None = None) -> Dict[str, Any]:
    owns_client = client is None
    client = client or SpotifyClient()
    try:
        playlist = client.get_playlist_info(playlist_url)
        raw_tracks = playlist.get("tracks", []) or []
        tracks = []
        for index, track in enumerate(raw_tracks, start=1):
            if not isinstance(track, dict):
                continue
            normalized = _normalize_track(track, index)
            if normalized:
                tracks.append(normalized)

        return {
            "platform": "spotify",
            "playlist_id": extract_spotify_playlist_id(playlist_url),
            "playlist_url": playlist_url,
            "title": str(playlist.get("name") or "").strip(),
            "description": str(playlist.get("description") or "").strip(),
            "owner": str(playlist.get("owner_name") or playlist.get("owner") or "").strip(),
            "track_count": len(tracks),
            "tracks": tracks,
        }
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("playlist_url")
    args = parser.parse_args()
    print(json.dumps(fetch_spotify_playlist(args.playlist_url), ensure_ascii=False, indent=2))
