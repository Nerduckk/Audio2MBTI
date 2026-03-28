"""Apple Music playlist and album crawler using public web page parsing."""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

APPLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def extract_apple_playlist_id(playlist_url: str) -> str | None:
    match = re.search(r"/playlist/[^/]+/(pl\.[A-Za-z0-9]+)", playlist_url)
    return match.group(1) if match else None


def extract_apple_album_id(album_url: str) -> str | None:
    match = re.search(r"/album/[^/]+/(\d+)", album_url)
    return match.group(1) if match else None


def _load_html(playlist_url: str) -> str:
    response = requests.get(playlist_url, timeout=20, headers=APPLE_HEADERS)
    response.raise_for_status()
    return response.text


def _parse_collection_schema(html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("@type") in {"MusicPlaylist", "MusicAlbum"}:
            return payload
    raise ValueError("Could not locate Apple Music collection schema.")


def _parse_artist_names(html_text: str) -> List[str]:
    matches = re.findall(r'"artistName":"(.*?)"', html_text)
    artists = []
    for value in matches:
        clean_value = html.unescape(value).strip()
        if clean_value:
            artists.append(clean_value)
    return artists


def _normalize_tracks(raw_tracks: list[dict], artist_names: list[str]) -> list[dict]:
    tracks = []
    for index, raw_track in enumerate(raw_tracks, start=1):
        if not isinstance(raw_track, dict):
            continue
        title = str(raw_track.get("name") or "").strip()
        if not title:
            continue

        artist = artist_names[index - 1] if index - 1 < len(artist_names) else ""
        external_url = str(raw_track.get("url") or "").strip()
        tracks.append(
            {
                "position": index,
                "title": title,
                "artists": [artist] if artist else [],
                "artists_text": artist,
                "album": "",
                "duration_ms": None,
                "external_url": external_url,
                "source_platform": "apple_music",
            }
        )
    return tracks


def fetch_apple_music_playlist(playlist_url: str) -> Dict[str, Any]:
    html_text = _load_html(playlist_url)
    playlist_schema = _parse_collection_schema(html_text)
    artist_names = _parse_artist_names(html_text)
    raw_tracks = playlist_schema.get("track", []) or playlist_schema.get("tracks", []) or []
    tracks = _normalize_tracks(raw_tracks, artist_names)

    author = playlist_schema.get("author", {})
    owner = ""
    if isinstance(author, dict):
        owner = str(author.get("name") or "").strip()

    return {
        "platform": "apple_music",
        "kind": "playlist",
        "playlist_id": extract_apple_playlist_id(playlist_url),
        "playlist_url": playlist_url,
        "title": str(playlist_schema.get("name") or "").strip(),
        "description": str(playlist_schema.get("description") or "").strip(),
        "owner": owner,
        "track_count": len(tracks),
        "tracks": tracks,
    }


def fetch_apple_music_album(album_url: str) -> Dict[str, Any]:
    html_text = _load_html(album_url)
    album_schema = _parse_collection_schema(html_text)
    artist_names = _parse_artist_names(html_text)
    raw_tracks = album_schema.get("track", []) or album_schema.get("tracks", []) or []
    tracks = _normalize_tracks(raw_tracks, artist_names)

    by_artist = album_schema.get("byArtist", []) or []
    owner = ""
    if by_artist:
        first_artist = by_artist[0]
        if isinstance(first_artist, dict):
            owner = str(first_artist.get("name") or "").strip()

    return {
        "platform": "apple_music",
        "kind": "album",
        "playlist_id": extract_apple_album_id(album_url),
        "playlist_url": album_url,
        "title": str(album_schema.get("name") or "").strip(),
        "description": str(album_schema.get("description") or "").strip(),
        "owner": owner,
        "track_count": len(tracks),
        "tracks": tracks,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--kind", choices=["playlist", "album"], default="playlist")
    args = parser.parse_args()
    fetcher = fetch_apple_music_playlist if args.kind == "playlist" else fetch_apple_music_album
    print(json.dumps(fetcher(args.url), ensure_ascii=False, indent=2))
