# -*- coding: utf-8 -*-
"""
Audio2MBTI - clean internal validation report.

This script reproduces the report-era internal validation using the trained
XGBoost artifacts in `3_train/models`.
It uses normal estimator behavior and calculates predictions directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "3_train" / "models"
DATA_PATH = PROJECT_ROOT / "2_process" / "playlist_hybrid_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_train" / "outputs"

DIM_LABELS = {
    "E_I": ("Extraversion", "Introversion"),
    "S_N": ("Sensing", "Intuition"),
    "T_F": ("Thinking", "Feeling"),
    "J_P": ("Judging", "Perceiving"),
}


def main() -> None:
    meta_path = MODEL_DIR / "hybrid_playlist_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing trained model metadata: {meta_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing report-era dataset: {DATA_PATH}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    target_labels = meta["target_labels"]
    feature_names = meta["features_used"]["E_I"]
    split_seed = int(meta.get("split_seed", 5))

    df = pd.read_csv(DATA_PATH)
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols[:10]}")

    x_data = df[feature_names].fillna(df[feature_names].median())
    dim_results = {}

    print("=" * 72)
    print("Audio2MBTI - INTERNAL VALIDATION REPORT")
    print("=" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {len(df):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Split seed: {split_seed}")
    print("Note: report-era internal validation.\n")

    for dim in target_labels:
        y_data = df[dim].astype(int).values
        _, x_test, _, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.20,
            random_state=split_seed,
            stratify=y_data,
        )

        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_DIR / f"hybrid_playlist_{dim}.json"))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        dim_results[dim] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        left, right = DIM_LABELS[dim]
        print(f"[{dim}] {left} vs {right}")
        print(f"  Accuracy    : {acc:.4f} ({acc:.2%})")
        print(f"  F1-Macro    : {f1_macro:.4f}")
        print(f"  F1-Weighted : {f1_weighted:.4f}")
        print(f"  ROC-AUC     : {auc:.4f}")
        print(f"  Confusion   : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")

    avg_accuracy = float(np.mean([dim_results[d]["accuracy"] for d in target_labels]))
    avg_f1 = float(np.mean([dim_results[d]["f1_macro"] for d in target_labels]))
    avg_auc = float(np.mean([dim_results[d]["roc_auc"] for d in target_labels]))

    print("-" * 72)
    print(f"Average Accuracy : {avg_accuracy:.4f} ({avg_accuracy:.2%})")
    print(f"Average F1-Macro : {avg_f1:.4f}")
    print(f"Average ROC-AUC  : {avg_auc:.4f}")
    print("-" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test_accuracy_report.json"
    payload = {
        "summary": {
            "dataset": str(DATA_PATH),
            "model_dir": str(MODEL_DIR),
            "split_seed": split_seed,
            "avg_accuracy": avg_accuracy,
            "avg_f1_macro": avg_f1,
            "avg_roc_auc": avg_auc,
            "note": "Report-era internal validation.",
        },
        "dimensions": dim_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
