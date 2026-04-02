"""Apple Music playlist crawler using public web page parsing."""

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


def _load_html(playlist_url: str) -> str:
    response = requests.get(playlist_url, timeout=20, headers=APPLE_HEADERS)
    response.raise_for_status()
    return response.text


def _parse_playlist_schema(html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("@type") == "MusicPlaylist":
            return payload
    raise ValueError("Could not locate Apple Music playlist schema.")


def _parse_artist_names(html_text: str) -> List[str]:
    matches = re.findall(r'"artistName":"(.*?)"', html_text)
    artists = []
    for value in matches:
        clean_value = html.unescape(value).strip()
        if clean_value:
            artists.append(clean_value)
    return artists


def fetch_apple_music_playlist(playlist_url: str) -> Dict[str, Any]:
    html_text = _load_html(playlist_url)
    playlist_schema = _parse_playlist_schema(html_text)
    artist_names = _parse_artist_names(html_text)

    raw_tracks = playlist_schema.get("track", []) or []
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

    author = playlist_schema.get("author", {})
    owner = ""
    if isinstance(author, dict):
        owner = str(author.get("name") or "").strip()

    return {
        "platform": "apple_music",
        "playlist_id": extract_apple_playlist_id(playlist_url),
        "playlist_url": playlist_url,
        "title": str(playlist_schema.get("name") or "").strip(),
        "description": str(playlist_schema.get("description") or "").strip(),
        "owner": owner,
        "track_count": len(tracks),
        "tracks": tracks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("playlist_url")
    args = parser.parse_args()
    print(json.dumps(fetch_apple_music_playlist(args.playlist_url), ensure_ascii=False, indent=2))
