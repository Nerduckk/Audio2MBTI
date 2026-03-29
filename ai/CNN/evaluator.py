"""Evaluation helpers for CNN predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .feature_extraction import vector_to_mbti
from .model import AudioCNN


class ModelEvaluator:
    """Evaluate checkpoints on held-out arrays or single samples."""

    dimension_names = ("E/I", "S/N", "T/F", "J/P")

    def __init__(self, config: dict):
        self.config = config
        training_cfg = config.get("training", {})
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_cfg.get("device", "auto") != "cpu"
            else "cpu"
        )

    def load_model(self, model_path: str) -> AudioCNN:
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", self.config)
        model = AudioCNN.from_config(config).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def predict_logits(self, model: AudioCNN, X: np.ndarray) -> np.ndarray:
        features = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def evaluate(self, model_path: str, X: np.ndarray, y: np.ndarray) -> Dict:
        model = self.load_model(model_path)
        probs = self.predict_logits(model, X)
        preds = (probs >= 0.5).astype(np.float32)

        dimension_metrics = {}
        accuracies = []
        f1_scores = []

        for idx, name in enumerate(self.dimension_names):
            true_col = y[:, idx]
            pred_col = preds[:, idx]
            prob_col = probs[:, idx]
            accuracy = accuracy_score(true_col, pred_col)
            f1 = f1_score(true_col, pred_col, zero_division=0)
            try:
                roc_auc = roc_auc_score(true_col, prob_col)
            except ValueError:
                roc_auc = None

            accuracies.append(accuracy)
            f1_scores.append(f1)
            dimension_metrics[name] = {
                "accuracy": float(accuracy),
                "f1": float(f1),
                "roc_auc": None if roc_auc is None else float(roc_auc),
            }

        true_types = [vector_to_mbti(row) for row in y]
        pred_types = [vector_to_mbti(row) for row in probs]
        type_accuracy = accuracy_score(true_types, pred_types)

        return {
            "overall_dimension_accuracy": float(np.mean(accuracies)),
            "overall_dimension_f1": float(np.mean(f1_scores)),
            "full_type_accuracy": float(type_accuracy),
            "dimensions": dimension_metrics,
            "sample_count": int(len(X)),
        }

    def save_metrics(self, metrics: Dict, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return str(Path(output_path).resolve())
