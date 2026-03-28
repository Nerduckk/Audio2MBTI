"""Spotify playlist and album crawler using no-auth web scraping."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from spotify_scraper import SpotifyClient


def extract_spotify_playlist_id(playlist_url: str) -> str | None:
    match = re.search(r"playlist/([A-Za-z0-9]+)", playlist_url)
    return match.group(1) if match else None


def extract_spotify_album_id(album_url: str) -> str | None:
    match = re.search(r"album/([A-Za-z0-9]+)", album_url)
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
        "track_id": str(track.get("id") or "").strip(),
        "track_uri": str(track.get("uri") or "").strip(),
        "title": title,
        "artists": artists,
        "artists_text": ", ".join(artists),
        "album": str(track.get("album_name") or track.get("album") or "").strip(),
        "duration_ms": track.get("duration_ms"),
        "external_url": str(track.get("url") or track.get("external_url") or "").strip(),
        "source_platform": "spotify",
    }


def _normalize_collection_tracks(raw_tracks: list[dict]) -> List[Dict[str, Any]]:
    tracks = []
    for index, track in enumerate(raw_tracks, start=1):
        if not isinstance(track, dict):
            continue
        normalized = _normalize_track(track, index)
        if normalized:
            tracks.append(normalized)
    return tracks


def fetch_spotify_playlist(playlist_url: str, client: SpotifyClient | None = None) -> Dict[str, Any]:
    owns_client = client is None
    client = client or SpotifyClient()
    try:
        playlist = client.get_playlist_info(playlist_url)
        tracks = _normalize_collection_tracks(playlist.get("tracks", []) or [])

        return {
            "platform": "spotify",
            "kind": "playlist",
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


def fetch_spotify_album(album_url: str, client: SpotifyClient | None = None) -> Dict[str, Any]:
    owns_client = client is None
    client = client or SpotifyClient()
    try:
        album = client.get_album_info(album_url)
        tracks = _normalize_collection_tracks(album.get("tracks", []) or [])
        artists = album.get("artists", []) or []
        owner = ""
        if artists:
            first_artist = artists[0]
            owner = str(first_artist.get("name") or "").strip() if isinstance(first_artist, dict) else str(first_artist).strip()

        return {
            "platform": "spotify",
            "kind": "album",
            "playlist_id": extract_spotify_album_id(album_url),
            "playlist_url": album_url,
            "title": str(album.get("name") or "").strip(),
            "description": "",
            "owner": owner,
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
    parser.add_argument("url")
    parser.add_argument("--kind", choices=["playlist", "album"], default="playlist")
    args = parser.parse_args()
    fetcher = fetch_spotify_playlist if args.kind == "playlist" else fetch_spotify_album
    print(json.dumps(fetcher(args.url), ensure_ascii=False, indent=2))
