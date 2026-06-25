# -*- coding: utf-8 -*-
"""
Train Audio2MBTI report-era XGBoost playlist classifiers.

The submitted report was produced from the playlist feature matrix:
`2_process/playlist_hybrid_features.csv`.

This script keeps that training path reproducible with normal XGBoost training
and writes models to `3_train/models`, matching the submitted report structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "2_process" / "playlist_hybrid_features.csv"
MODEL_DIR = PROJECT_ROOT / "3_train" / "models"
TARGET_LABELS = ["E_I", "S_N", "T_F", "J_P"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing report-era dataset: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in TARGET_LABELS + ["playlist"]]
    x_data = df[feature_cols].fillna(df[feature_cols].median())

    split_seed = 5
    meta = {
        "target_labels": TARGET_LABELS,
        "features_used": {},
        "accuracy": {},
        "f1_macro": {},
        "dataset": str(DATA_PATH),
        "split_seed": split_seed,
        "model_random_state": 42,
        "note": (
            "Internal validation models using normal XGBoost training and calculated metrics."
        ),
    }

    print("=" * 72)
    print("Training Audio2MBTI internal-validation models")
    print("=" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Output: {MODEL_DIR}\n")

    for dim in TARGET_LABELS:
        y_data = df[dim].astype(int).values
        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.20,
            random_state=split_seed,
            stratify=y_data,
        )

        model = xgb.XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=25,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        preds = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, preds))
        f1_macro = float(f1_score(y_test, preds, average="macro"))

        model.save_model(str(MODEL_DIR / f"hybrid_playlist_{dim}.json"))
        meta["features_used"][dim] = feature_cols
        meta["accuracy"][dim] = accuracy
        meta["f1_macro"][dim] = f1_macro

        print(f"{dim}: accuracy={accuracy:.4f} ({accuracy:.2%}), f1={f1_macro:.4f}")

    meta["avg_accuracy"] = float(np.mean(list(meta["accuracy"].values())))
    meta["avg_f1_macro"] = float(np.mean(list(meta["f1_macro"].values())))
    (MODEL_DIR / "hybrid_playlist_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "-" * 72)
    print(f"Average accuracy: {meta['avg_accuracy']:.4f} ({meta['avg_accuracy']:.2%})")
    print(f"Average F1-macro: {meta['avg_f1_macro']:.4f}")
    print(f"Saved meta: {MODEL_DIR / 'hybrid_playlist_meta.json'}")


if __name__ == "__main__":
    main()
