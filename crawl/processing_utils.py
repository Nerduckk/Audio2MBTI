"""Shared audio, lyrics, and metadata processing helpers."""

from __future__ import annotations

import os
import re
import math
import urllib.parse
import uuid
from typing import Dict, List

import librosa
import numpy as np
import requests
import syncedlyrics
import yt_dlp
from deep_translator import GoogleTranslator
from transformers import pipeline

try:
    from .mbti_genre_processor import calculate_genre_mbti_scores, match_genre_to_mbti
except ImportError:
    from mbti_genre_processor import calculate_genre_mbti_scores, match_genre_to_mbti

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_emotion_pipeline = None

EMOTION_WEIGHTS = {
    "excitement": 1.0,
    "joy": 0.9,
    "amusement": 0.8,
    "optimism": 0.7,
    "pride": 0.6,
    "admiration": 0.5,
    "gratitude": 0.4,
    "relief": 0.3,
    "love": 0.3,
    "caring": 0.2,
    "approval": 0.2,
    "desire": 0.2,
    "realization": 0.1,
    "surprise": 0.1,
    "curiosity": 0.1,
    "neutral": 0.0,
    "confusion": -0.1,
    "embarrassment": -0.2,
    "nervousness": -0.3,
    "annoyance": -0.4,
    "disapproval": -0.4,
    "disgust": -0.5,
    "anger": -0.6,
    "fear": -0.7,
    "disappointment": -0.8,
    "remorse": -0.8,
    "sadness": -0.9,
    "grief": -1.0,
}

EMOTION_GROUPS = {
    "lyrics_joy": ["excitement", "joy", "amusement", "optimism", "pride"],
    "lyrics_sadness": ["sadness", "grief", "disappointment", "remorse"],
    "lyrics_anger": ["anger", "annoyance", "disgust", "disapproval"],
    "lyrics_love": ["love", "caring", "admiration", "desire", "gratitude"],
    "lyrics_fear": ["fear", "nervousness", "confusion", "embarrassment"],
}


def get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
        )
    return _emotion_pipeline


def fetch_track_metadata(title: str, artists_text: str) -> Dict[str, object]:
    found_genres: List[str] = []
    release_year = 2020
    popularity = 50

    try:
        search_query = urllib.parse.quote(f"{title} {artists_text}")
        url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        if data.get("resultCount", 0) > 0:
            result = data["results"][0]
            release_date = str(result.get("releaseDate") or "")
            if len(release_date) >= 4 and release_date[:4].isdigit():
                release_year = int(release_date[:4])

            apple_genres = result.get("genres", []) or []
            primary = result.get("primaryGenreName")
            if primary and primary not in apple_genres:
                apple_genres.append(primary)

            for genre in apple_genres:
                genre_name = ""
                if isinstance(genre, dict):
                    genre_name = str(genre.get("name") or "").lower()
                elif isinstance(genre, str):
                    genre_name = genre.lower()
                if not genre_name or genre_name == "music":
                    continue
                matched = match_genre_to_mbti(genre_name)
                if matched and matched not in found_genres:
                    found_genres.append(matched)
    except Exception:
        pass

    if not found_genres:
        found_genres = ["pop"]

    popularity = estimate_youtube_popularity(title, artists_text)

    scores = calculate_genre_mbti_scores(found_genres)
    return {
        "artist_genres": ", ".join(found_genres).upper(),
        "genre_ei_score": scores["genre_ei"],
        "genre_sn_score": scores["genre_sn"],
        "genre_tf_score": scores["genre_tf"],
        "release_year": release_year,
        "popularity_proxy": popularity,
        "genres_list": found_genres,
    }


def estimate_youtube_popularity(title: str, artists_text: str) -> int:
    query = f"{title} {artists_text} audio"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "default_search": "ytsearch",
        "extract_flat": False,
        "noplaylist": True,
        "socket_timeout": 10,
        "js_runtimes": {"node": {}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        entries = info.get("entries", []) or []
        if not entries:
            return 50
        top_entry = entries[0] or {}
        views = int(top_entry.get("view_count") or 0)
        if views <= 0:
            return 50
        scaled = int(round(min(100, max(1, (math.log10(views + 1) - 3) * 15))))
        return scaled
    except Exception:
        return 50


def download_audio_segment(query: str, duration: int = 35, output_basename: str | None = None) -> str | None:
    output_basename = output_basename or f"temp_audio_{uuid.uuid4().hex}"
    audio_path = f"{output_basename}.mp3"
    for candidate in [output_basename, audio_path, f"{output_basename}.webm", f"{output_basename}.m4a"]:
        if os.path.exists(candidate):
            try:
                os.remove(candidate)
            except OSError:
                pass

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_basename,
        "quiet": True,
        "no_warnings": True,
        "extract_audio": True,
        "audio_format": "mp3",
        "ffmpeg_location": r"d:\project\ffmpeg-master-latest-win64-gpl\bin",
        "default_search": "ytsearch",
        "download_ranges": lambda _, __: [{"start_time": 0, "end_time": duration}],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
        return audio_path if os.path.exists(audio_path) else None
    except Exception:
        return None


def analyze_audio_features(audio_path: str, duration: int = 35) -> Dict[str, float] | None:
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        rms = librosa.feature.rms(y=y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        return {
            "tempo_bpm": round(tempo, 2),
            "energy": round(min(float(np.mean(rms)) / 0.3, 1.0), 4),
            "danceability": round(min(float(np.var(onset_env)) / 10.0, 1.0), 4),
            "spectral_centroid": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 1),
            "spectral_flatness": round(float(np.mean(librosa.feature.spectral_flatness(y=y))), 4),
            "zero_crossing_rate": round(float(np.mean(librosa.feature.zero_crossing_rate(y=y))), 4),
            "spectral_bandwidth": round(
                min(1.0, float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) / 4000.0), 4
            ),
            "spectral_rolloff": round(
                min(1.0, float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) / 8000.0), 4
            ),
            "mfcc_mean": round(float(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))), 4),
            "chroma_mean": round(float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))), 4),
            "tempo_strength": round(min(1.0, (float(np.max(onset_env)) if len(onset_env) else 0.0) / 5.0), 4),
        }
    except Exception:
        return None


def analyze_lyrics_sentiment(track_name: str, artist_name: str) -> Dict[str, float]:
    result = {
        "lyrics_polarity": 0.0,
        "lyrics_joy": 0.0,
        "lyrics_sadness": 0.0,
        "lyrics_anger": 0.0,
        "lyrics_love": 0.0,
        "lyrics_fear": 0.0,
    }

    try:
        raw_lyrics = syncedlyrics.search(
            f"{track_name} {artist_name}",
            providers=["Lrclib", "NetEase", "MegLyrics"],
        )
        if not raw_lyrics:
            return result

        clean_lyrics = re.sub(r"\[\d{2}:\d{2}\.\d{2}\]", "", raw_lyrics).strip()[:2000]
        translated = GoogleTranslator(source="auto", target="en").translate(clean_lyrics[:1500])
        raw_results = get_emotion_pipeline()(translated[:1500], top_k=10)
        ai_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results

        polarity_score = 0.0
        for item in ai_results:
            if not isinstance(item, dict) or "label" not in item:
                continue
            label = item["label"].lower()
            score = float(item["score"])
            polarity_score += EMOTION_WEIGHTS.get(label, 0.0) * score
            for group_name, labels in EMOTION_GROUPS.items():
                if label in labels:
                    result[group_name] += score

        result["lyrics_polarity"] = round(max(-1.0, min(1.0, polarity_score)), 4)
        for key in list(EMOTION_GROUPS):
            result[key] = round(result[key], 4)
        return result
    except Exception:
        return result
