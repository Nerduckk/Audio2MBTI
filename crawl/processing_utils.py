"""Shared audio and NLP processing helpers for crawlers."""

from __future__ import annotations

import os
import re
from typing import Dict

import librosa
import numpy as np
import syncedlyrics
from deep_translator import GoogleTranslator
from transformers import pipeline

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
        print("=> Loading HuggingFace emotion model...")
        _emotion_pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
        )
    return _emotion_pipeline


def analyze_audio_features(audio_path: str, duration: int = 35) -> Dict[str, float] | None:
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)

        rms = librosa.feature.rms(y=y)
        energy = min(float(np.mean(rms)) / 0.3, 1.0)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        danceability = min(float(np.var(onset_env)) / 10.0, 1.0)

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        spectral_bandwidth = min(
            1.0, float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) / 4000.0
        )
        spectral_rolloff = min(
            1.0, float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) / 8000.0
        )
        mfcc_mean = float(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)))
        chroma_mean = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        tempo_strength = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
        tempo_strength = min(1.0, tempo_strength / 5.0)

        return {
            "tempo_bpm": round(tempo, 2),
            "energy": round(energy, 4),
            "danceability": round(danceability, 4),
            "spectral_centroid": round(spectral_centroid, 1),
            "spectral_flatness": round(spectral_flatness, 4),
            "zero_crossing_rate": round(zero_crossing_rate, 4),
            "spectral_bandwidth": round(spectral_bandwidth, 4),
            "spectral_rolloff": round(spectral_rolloff, 4),
            "mfcc_mean": round(mfcc_mean, 4),
            "chroma_mean": round(chroma_mean, 4),
            "tempo_strength": round(tempo_strength, 4),
        }
    except Exception as exc:
        print(f"     Audio analysis failed: {exc}")
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
        for res in ai_results:
            if not isinstance(res, dict) or "label" not in res:
                continue
            label = res["label"].lower()
            score = float(res["score"])
            polarity_score += EMOTION_WEIGHTS.get(label, 0.0) * score
            for group_name, labels in EMOTION_GROUPS.items():
                if label in labels:
                    result[group_name] += score

        result["lyrics_polarity"] = round(max(-1.0, min(1.0, polarity_score)), 4)
        for key in list(EMOTION_GROUPS):
            result[key] = round(result[key], 4)
        return result
    except Exception as exc:
        print(f"     Lyrics analysis failed: {exc}")
        return result
