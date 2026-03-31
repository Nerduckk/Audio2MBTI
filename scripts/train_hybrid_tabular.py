"""Train tree-based MBTI classifiers on combined old and new audio-feature data."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


AUDIO_COLUMNS = [
    "tempo_bpm",
    "tempo_strength",
    "energy",
    "danceability",
    "spectral_centroid",
    "spectral_flatness",
    "zero_crossing_rate",
    "spectral_bandwidth",
    "spectral_rolloff",
    "mfcc_mean",
    "chroma_mean",
]

ENGINEERED_COLUMNS = [
    "energy_x_dance",
    "energy_x_tempo",
    "energy_minus_dance",
    "tempo_normalized",
    "spectral_x_energy",
    "flatness_x_zcr",
]

TARGET_COLUMNS = {
    "E/I": "target_EI",
    "S/N": "target_SN",
    "T/F": "target_TF",
    "J/P": "target_JP",
}


def analyze_audio_features(audio_path: str, duration: int = 35) -> Dict[str, float] | None:
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)

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
    except Exception:
        return None


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_EI"] = df["mbti_label"].str[0].eq("E").astype(int)
    df["target_SN"] = df["mbti_label"].str[1].eq("S").astype(int)
    df["target_TF"] = df["mbti_label"].str[2].eq("T").astype(int)
    df["target_JP"] = df["mbti_label"].str[3].eq("J").astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["energy_x_dance"] = df["energy"] * df["danceability"]
    df["energy_x_tempo"] = df["energy"] * df["tempo_bpm"]
    df["energy_minus_dance"] = df["energy"] - df["danceability"]
    df["tempo_normalized"] = df["tempo_bpm"] / 200.0
    df["spectral_x_energy"] = df["spectral_centroid"] * df["energy"]
    df["flatness_x_zcr"] = df["spectral_flatness"] * df["zero_crossing_rate"]
    return df


def load_old_dataset(old_csv: Path) -> pd.DataFrame:
    columns = ["title", "artists", "mbti_label", *AUDIO_COLUMNS]
    df = pd.read_csv(old_csv, usecols=columns)
    df = df.dropna(subset=["mbti_label", *AUDIO_COLUMNS]).copy()
    df["dataset_source"] = "old"
    return add_targets(add_engineered_features(df))


def _extract_one(item: dict, duration: int) -> dict | None:
    features = analyze_audio_features(item["audio_path"], duration=duration)
    if features is None:
        return None
    return {
        "sample_id": item["sample_id"],
        "mbti_label": item["label"],
        "audio_path": item["audio_path"],
        **features,
    }


def build_new_audio_feature_cache(
    manifest_path: Path,
    cache_path: Path,
    duration: int,
    workers: int,
) -> pd.DataFrame:
    items = json.load(open(manifest_path, encoding="utf-8"))
    existing = pd.DataFrame()
    existing_ids: set[str] = set()
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        if "sample_id" in existing.columns:
            existing_ids = set(existing["sample_id"].astype(str))

    pending = [item for item in items if str(item["sample_id"]) not in existing_ids]
    rows: List[dict] = []

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, item, duration) for item in pending]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    rows.append(row)

    new_df = pd.DataFrame(rows)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined.empty:
        raise ValueError("No audio features were extracted for the new dataset.")

    combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    combined.to_csv(cache_path, index=False)
    combined["dataset_source"] = "new"
    return add_targets(add_engineered_features(combined))


def build_feature_matrix(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val), nan=0.0)
    X_test_scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
    val_clusters = kmeans.predict(X_val_scaled).reshape(-1, 1)
    test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_ohe = encoder.fit_transform(train_clusters)
    val_ohe = encoder.transform(val_clusters)
    test_ohe = encoder.transform(test_clusters)

    return (
        np.hstack([X_train_scaled, train_ohe]),
        np.hstack([X_val_scaled, val_ohe]),
        np.hstack([X_test_scaled, test_ohe]),
    )


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.3, 0.71, 0.01):
        preds = (probs >= threshold).astype(int)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def train_dimension_model(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    X_train_final, X_val_final, X_test_final = build_feature_matrix(X_train, X_val, X_test)

    rf_model = BalancedRandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features="sqrt",
        sampling_strategy="not majority",
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )

    estimators: list[tuple[str, object]] = [("rf", rf_model)]
    if xgb is not None:
        estimators.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=3,
                    gamma=0.1,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                    eval_metric="logloss",
                ),
            )
        )

    if len(estimators) == 1:
        model = rf_model
    else:
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
            cv=3,
            n_jobs=1,
        )

    model.fit(X_train_final, y_train)

    train_probs = model.predict_proba(X_train_final)[:, 1]
    val_probs = model.predict_proba(X_val_final)[:, 1]
    test_probs = model.predict_proba(X_test_final)[:, 1]

    threshold, val_accuracy = choose_threshold(y_val.to_numpy(), val_probs)
    train_preds = (train_probs >= threshold).astype(int)
    test_preds = (test_probs >= threshold).astype(int)

    return {
        "model": model,
        "threshold": threshold,
        "train_accuracy": float(accuracy_score(y_train, train_preds)),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hybrid tree-based MBTI models.")
    parser.add_argument("--old-csv", default=r"D:\project\data\(OLD) mbti_master_training_data.csv")
    parser.add_argument("--manifest-path", default=r"D:\project\data\train_manifest.json")
    parser.add_argument("--feature-cache", default=r"D:\project\data\audio_tabular_features.csv")
    parser.add_argument("--metrics-path", default=r"D:\project\outputs\hybrid_tree_metrics.json")
    parser.add_argument("--duration", type=int, default=35)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    old_df = load_old_dataset(Path(args.old_csv))
    new_df = build_new_audio_feature_cache(
        manifest_path=Path(args.manifest_path),
        cache_path=Path(args.feature_cache),
        duration=args.duration,
        workers=args.workers,
    )

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    feature_columns = [*AUDIO_COLUMNS, *ENGINEERED_COLUMNS]
    combined = combined.dropna(subset=feature_columns + ["mbti_label"]).copy()

    metrics = {
        "dataset": {
            "old_rows": int(len(old_df)),
            "new_rows": int(len(new_df)),
            "combined_rows": int(len(combined)),
            "feature_columns": feature_columns,
        },
        "dimensions": {},
    }

    X = combined[feature_columns]
    for dimension_name, target_col in TARGET_COLUMNS.items():
        result = train_dimension_model(X, combined[target_col])
        metrics["dimensions"][dimension_name] = {
            key: value for key, value in result.items() if key != "model"
        }

    metrics["overall_test_accuracy"] = float(
        np.mean([payload["test_accuracy"] for payload in metrics["dimensions"].values()])
    )
    metrics["overall_test_f1"] = float(
        np.mean([payload["test_f1"] for payload in metrics["dimensions"].values()])
    )

    save_json(Path(args.metrics_path), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
