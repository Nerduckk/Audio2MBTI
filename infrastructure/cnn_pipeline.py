"""Orchestration helpers for the CNN audio pipeline."""

from __future__ import annotations

import numpy as np

from ai.CNN import FeatureExtractor, ModelEvaluator, ModelTrainer


class CNNPipeline:
    """Thin orchestration layer around extraction, training and evaluation."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_extractor = FeatureExtractor.from_config(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

    def extract_features(self, metadata_csv: str | None, audio_dir: str | None, output_dir: str):
        return self.feature_extractor.batch_extract(
            metadata_csv=metadata_csv,
            audio_dir=audio_dir,
            output_dir=output_dir,
        )

    def train(self, X_path: str, y_path: str, output_dir: str):
        return self.trainer.train(X_path=X_path, y_path=y_path, output_dir=output_dir)

    def evaluate(self, model_path: str, X: np.ndarray, y: np.ndarray):
        return self.evaluator.evaluate(model_path=model_path, X=X, y=y)
