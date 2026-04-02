"""Fast Kaggle metadata-only crawler for the CNN pipeline."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent))
from infrastructure.batch_processor import BatchProcessor
from infrastructure.retry_logic import RateLimiter

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

def get_cnn_metadata_csv() -> str:
    """Dedicated metadata CSV for the CNN pipeline."""
    ensure_data_dir_exists()
    return str(Path("data") / "mbti_cnn_metadata.csv")


def get_cnn_metadata_state_path() -> Path:
    """State file used to resume playlist crawling."""
    ensure_data_dir_exists()
    return Path("data") / "mbti_cnn_metadata_state.json"


def load_playlist_ids(kaggle_dir: str) -> list[tuple[str, str]]:
    playlist_ids: list[tuple[str, str]] = []
    for file in sorted(os.listdir(kaggle_dir)):
        if not file.endswith("_df.csv") or file.startswith("combined"):
            continue
        temp_df = pd.read_csv(os.path.join(kaggle_dir, file))
        if "playlist_id" not in temp_df.columns or "mbti" not in temp_df.columns:
            continue
        label_values = temp_df["mbti"].dropna().astype(str).str.upper().unique().tolist()
        if not label_values:
            continue
        label = label_values[0]
        for pid in sorted(temp_df["playlist_id"].dropna().astype(str).unique().tolist()):
            playlist_ids.append((pid, label))
    return playlist_ids


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"completed_playlists": [], "failed_playlists": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed_playlists": [], "failed_playlists": {}}

    if not isinstance(payload, dict):
        return {"completed_playlists": [], "failed_playlists": {}}

    payload.setdefault("completed_playlists", [])
    payload.setdefault("failed_playlists", {})
    return payload


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_state_payload(completed_playlists: set[str], failed_playlists: dict[str, dict]) -> dict:
    return {
        "completed_playlists": sorted(str(pid) for pid in completed_playlists),
        "failed_playlists": {
            str(pid): details for pid, details in sorted(failed_playlists.items(), key=lambda item: item[0])
        },
    }


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


def mass_reprocess_kaggle_metadata(
    max_playlists=None,
    max_tracks_per_playlist=None,
    batch_size=50,
    requests_per_second=1.0,
    playlist_delay_min=0.5,
    playlist_delay_max=1.5,
    resume=True,
):
    kaggle_dir = r"data\kaggle data set"
    output_csv = get_cnn_metadata_csv()
    state_path = get_cnn_metadata_state_path()
    batch_processor = BatchProcessor(batch_size=batch_size, output_file=output_csv)
    spotify_rate_limiter = RateLimiter(requests_per_second=requests_per_second)
    random.seed(42)

    if not os.path.exists(kaggle_dir):
        print(f"Không tìm thấy thư mục {kaggle_dir}")
        return

    playlist_ids = load_playlist_ids(kaggle_dir)
    if not playlist_ids:
        print("Không tìm thấy playlist_id trong dữ liệu Kaggle.")
        return

    state = load_state(state_path)
    completed_playlists = set(str(pid) for pid in state.get("completed_playlists", []))
    failed_playlists = {
        str(pid): details
        for pid, details in state.get("failed_playlists", {}).items()
        if isinstance(details, dict)
    }
    skipped_playlists = set(completed_playlists) | set(failed_playlists)

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
                if "playlist_id" in df_out.columns and resume:
                    completed_playlists.update(
                        df_out["playlist_id"].dropna().astype(str).unique().tolist()
                    )
                    skipped_playlists = set(completed_playlists) | set(failed_playlists)
                print(f"existing_metadata={len(processed_songs)}")
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

    if resume:
        playlist_ids = [(pid, label) for pid, label in playlist_ids if str(pid) not in skipped_playlists]

    spotify_scraper = SpotifyClient()
    total_playlists = len(playlist_ids)
    print(
        f"playlists={total_playlists} output={output_csv} resume={'on' if resume else 'off'} "
        f"completed={len(completed_playlists)} failed={len(failed_playlists)}"
    )

    for idx, (pid, mbti_label) in enumerate(playlist_ids, 1):
        try:
            percentage = (idx / total_playlists) * 100
            print(f"[{idx}/{total_playlists}] {percentage:.1f}% playlist={pid} mbti={mbti_label}")

            playlist_url = f"https://open.spotify.com/playlist/{pid}"
            tracks_data = []
            attempts = 0
            last_error = ""
            while attempts < 3:
                try:
                    spotify_rate_limiter.wait()
                    playlist_info = spotify_scraper.get_playlist_info(playlist_url)
                    tracks_data = playlist_info.get("tracks", []) or []
                    if tracks_data:
                        break
                except Exception as exc:
                    last_error = str(exc)[:200]
                    print(f"  scraper_error={last_error}")
                attempts += 1
                time.sleep(2 ** attempts)

            if not tracks_data:
                failure_reason = "no_tracks"
                if last_error:
                    failure_reason = f"scraper_error:{last_error}"
                failed_playlists[str(pid)] = {
                    "mbti_label": mbti_label,
                    "playlist_url": playlist_url,
                    "reason": failure_reason,
                    "attempts": attempts,
                }
                save_state(state_path, build_state_payload(completed_playlists, failed_playlists))
                print(f"  skipped={failure_reason}")
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

            print(f"  new_rows={new_rows} duplicates={duplicates} saved={batch_processor.total_saved}")
            completed_playlists.add(str(pid))
            failed_playlists.pop(str(pid), None)
            save_state(state_path, build_state_payload(completed_playlists, failed_playlists))
            time.sleep(random.uniform(playlist_delay_min, playlist_delay_max))
        except Exception as exc:
            failed_playlists[str(pid)] = {
                "mbti_label": mbti_label,
                "playlist_url": f"https://open.spotify.com/playlist/{pid}",
                "reason": f"playlist_error:{str(exc)[:200]}",
            }
            save_state(state_path, build_state_payload(completed_playlists, failed_playlists))
            print(f"  playlist_error={pid} {exc}")
            time.sleep(3)

    batch_processor.flush()
    spotify_scraper.close()
    save_state(state_path, build_state_payload(completed_playlists, failed_playlists))
    print(f"done saved={batch_processor.total_saved} metadata={output_csv} state={state_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-playlists", type=int, default=None)
    parser.add_argument("--max-tracks-per-playlist", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--playlist-delay-min", type=float, default=0.5)
    parser.add_argument("--playlist-delay-max", type=float, default=1.5)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    mass_reprocess_kaggle_metadata(
        max_playlists=args.max_playlists,
        max_tracks_per_playlist=args.max_tracks_per_playlist,
        batch_size=args.batch_size,
        requests_per_second=args.requests_per_second,
        playlist_delay_min=args.playlist_delay_min,
        playlist_delay_max=args.playlist_delay_max,
        resume=args.resume,
    )
