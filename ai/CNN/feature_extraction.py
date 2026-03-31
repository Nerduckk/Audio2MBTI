"""Audio feature extraction for CNN training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd


MBTI_LABELS = tuple(
    sorted(
        [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP",
        ]
    )
)


def mbti_to_vector(label: str) -> np.ndarray:
    """Convert MBTI label to four binary dimensions."""
    label = str(label).upper().strip()
    if len(label) != 4:
        raise ValueError(f"Invalid MBTI label: {label}")

    return np.asarray(
        [
            1.0 if label[0] == "E" else 0.0,
            1.0 if label[1] == "S" else 0.0,
            1.0 if label[2] == "T" else 0.0,
            1.0 if label[3] == "J" else 0.0,
        ],
        dtype=np.float32,
    )


def vector_to_mbti(vector: np.ndarray) -> str:
    """Convert four probabilities or binary values to MBTI text."""
    values = np.asarray(vector).reshape(-1)
    if values.shape[0] != 4:
        raise ValueError("MBTI vector must have four values")

    return "".join(
        [
            "E" if values[0] >= 0.5 else "I",
            "S" if values[1] >= 0.5 else "N",
            "T" if values[2] >= 0.5 else "F",
            "J" if values[3] >= 0.5 else "P",
        ]
    )


@dataclass
class DatasetItem:
    audio_path: str
    label: str
    sample_id: str


class FeatureExtractor:
    """Extract fixed-size Mel spectrograms from audio files."""

    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: int = 30,
        target_shape: Tuple[int, int] = (128, 1290),
    ) -> None:
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.duration = int(duration)
        self.target_shape = tuple(target_shape)

    @classmethod
    def from_config(cls, config: Dict) -> "FeatureExtractor":
        feature_cfg = config.get("feature_extraction", {})
        return cls(
            sr=feature_cfg.get("sr", 22050),
            n_mels=feature_cfg.get("n_mels", 128),
            n_fft=feature_cfg.get("n_fft", 2048),
            hop_length=feature_cfg.get("hop_length", 512),
            duration=feature_cfg.get("duration", 30),
            target_shape=tuple(feature_cfg.get("target_shape", [128, 1290])),
        )

    def extract(self, audio_path: str) -> np.ndarray:
        """Load one audio file and return a standardized Mel spectrogram."""
        signal, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
        spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        mel_bins, time_bins = self.target_shape
        if spectrogram_db.shape[0] != mel_bins:
            raise ValueError(
                f"Unexpected mel bin count {spectrogram_db.shape[0]} for {audio_path}; expected {mel_bins}"
            )

        if spectrogram_db.shape[1] < time_bins:
            pad_width = time_bins - spectrogram_db.shape[1]
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode="constant")
        else:
            spectrogram_db = spectrogram_db[:, :time_bins]

        return spectrogram_db.astype(np.float32)

    def extract_to_cache(self, audio_path: str, output_path: str) -> np.ndarray:
        """Extract one sample and persist the `.npy` cache file."""
        spectrogram = self.extract(audio_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, spectrogram)
        return spectrogram

    def build_dataset_index(
        self,
        metadata_csv: Optional[str] = None,
        audio_dir: Optional[str] = None,
    ) -> List[DatasetItem]:
        """
        Build a training manifest.

        Supported inputs:
        - CSV with one of: `audio_path`, `audio_file`, `file_path` and `mbti_label`
        - Directory structure: `audio_dir/<MBTI>/*.mp3`
        """
        items: List[DatasetItem] = []

        if metadata_csv:
            df = pd.read_csv(metadata_csv)
            audio_col = next(
                (col for col in ("audio_path", "audio_file", "file_path") if col in df.columns),
                None,
            )
            if audio_col and "mbti_label" in df.columns:
                base_dir = Path(audio_dir).resolve() if audio_dir else Path(metadata_csv).resolve().parent
                for idx, row in df.iterrows():
                    raw_value = row.get(audio_col)
                    label = str(row.get("mbti_label", "")).upper().strip()
                    if pd.isna(raw_value) or pd.isna(label):
                        continue
                    if label not in MBTI_LABELS:
                        continue

                    path_obj = Path(str(raw_value))
                    if not path_obj.is_absolute():
                        path_obj = base_dir / path_obj

                    if not path_obj.exists():
                        continue

                    items.append(
                        DatasetItem(
                            audio_path=str(path_obj.resolve()),
                            label=label,
                            sample_id=str(row.get("source_track_id") or row.get("title") or idx),
                        )
                    )

        if items:
            return items

        if not audio_dir:
            raise ValueError(
                "Could not build dataset index. Provide a CSV with an audio path column or "
                "an audio directory structured as audio_dir/<MBTI>/*.mp3."
            )

        base_dir = Path(audio_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        supported_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        for label_dir in sorted(base_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name.upper()
            if label not in MBTI_LABELS:
                continue

            for audio_file in sorted(label_dir.rglob("*")):
                if audio_file.suffix.lower() not in supported_exts:
                    continue
                items.append(
                    DatasetItem(
                        audio_path=str(audio_file.resolve()),
                        label=label,
                        sample_id=audio_file.stem,
                    )
                )

        if not items:
            raise ValueError(
                "No labeled audio files found. Expected CSV path columns or folders like "
                "data/audio_files/INTJ/*.mp3."
            )

        return items

    def batch_extract(
        self,
        metadata_csv: Optional[str],
        audio_dir: Optional[str],
        output_dir: str,
        output_prefix: str = "cnn_dataset",
        x_filename: Optional[str] = None,
        y_filename: Optional[str] = None,
        manifest_filename: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build cached dataset arrays and a manifest file."""
        items = self.build_dataset_index(metadata_csv=metadata_csv, audio_dir=audio_dir)
        output_root = Path(output_dir)
        cache_dir = output_root / "spectrograms"
        cache_dir.mkdir(parents=True, exist_ok=True)

        X_rows: List[np.ndarray] = []
        y_rows: List[np.ndarray] = []
        manifest: List[Dict[str, str]] = []

        for item in items:
            cache_path = cache_dir / f"{item.sample_id}.npy"
            try:
                spectrogram = self.extract_to_cache(item.audio_path, str(cache_path))
            except Exception:
                continue

            X_rows.append(spectrogram[..., np.newaxis])
            y_rows.append(mbti_to_vector(item.label))
            manifest.append(
                {
                    "sample_id": item.sample_id,
                    "label": item.label,
                    "audio_path": item.audio_path,
                    "spectrogram_path": str(cache_path.resolve()),
                }
            )

        if not X_rows:
            raise ValueError("No spectrograms were extracted successfully.")

        X = np.stack(X_rows).astype(np.float32)
        y = np.stack(y_rows).astype(np.float32)

        X_path = output_root / (x_filename or f"{output_prefix}_X.npy")
        y_path = output_root / (y_filename or f"{output_prefix}_y.npy")
        manifest_path = output_root / (manifest_filename or f"{output_prefix}_manifest.json")

        np.save(X_path, X)
        np.save(y_path, y)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        return {
            "X_path": str(X_path.resolve()),
            "y_path": str(y_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "count": str(len(manifest)),
        }
