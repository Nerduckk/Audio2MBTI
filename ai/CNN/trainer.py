"""Training utilities for the AudioCNN model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .augmentation import SpecAugment
from .model import AudioCNN


class SpectrogramDataset(Dataset):
    """Dataset wrapper around precomputed spectrogram arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: Optional[nn.Module] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        features = self.X[index].permute(2, 0, 1)
        if self.augment is not None:
            features = self.augment(features)
        return features, self.y[index]


class ModelTrainer:
    """Train and persist AudioCNN checkpoints."""

    def __init__(self, config: dict):
        self.config = config
        self.training_cfg = config.get("training", {})
        self.augmentation_cfg = config.get("augmentation", {})
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.training_cfg.get("device", "auto") != "cpu"
            else "cpu"
        )

    def load_arrays(self, X_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        if X.ndim != 4:
            raise ValueError(f"Expected X with 4 dims (N, H, W, C), got {X.shape}")
        if y.ndim != 2 or y.shape[1] != 4:
            raise ValueError(f"Expected y shape (N, 4), got {y.shape}")
        return X, y

    def prepare_data(self, X: np.ndarray, y: np.ndarray):
        test_size = float(self.training_cfg.get("test_size", 0.1))
        val_size = float(self.training_cfg.get("val_size", 0.1))
        random_state = int(self.training_cfg.get("random_state", 42))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        adjusted_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=adjusted_val_size,
            random_state=random_state,
            shuffle=True,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        batch_size = int(self.training_cfg.get("batch_size", 16))
        augment = None
        if self.augmentation_cfg.get("enabled", False):
            augment = SpecAugment(
                freq_mask_param=self.augmentation_cfg.get("freq_mask_param", 24),
                time_mask_param=self.augmentation_cfg.get("time_mask_param", 48),
                p=self.augmentation_cfg.get("probability", 0.5),
            )

        train_dataset = SpectrogramDataset(X_train, y_train, augment=augment)
        val_dataset = SpectrogramDataset(X_val, y_val)
        test_dataset = SpectrogramDataset(X_test, y_test)

        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )

    def train(self, X_path: str, y_path: str, output_dir: str) -> Dict[str, str]:
        X, y = self.load_arrays(X_path, y_path)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        train_loader, val_loader, test_loader = self._build_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        model = AudioCNN.from_config(self.config).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(self.training_cfg.get("learning_rate", 1e-3))
        )

        epochs = int(self.training_cfg.get("epochs", 20))
        patience = int(self.training_cfg.get("early_stopping_patience", 5))
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_root / "audio_cnn.pt"
        history_path = output_root / "training_history.json"
        split_path = output_root / "test_split.npz"

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": []}

        for _epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * features.size(0)

            train_loss /= max(len(train_loader.dataset), 1)
            val_loss = self._evaluate_loss(model, val_loader, criterion)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": self.config,
                        "input_shape": list(model.input_shape),
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        np.savez_compressed(split_path, X_test=X_test, y_test=y_test)
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        test_loss = self._evaluate_loss(model, test_loader, criterion)
        return {
            "model_path": str(checkpoint_path.resolve()),
            "history_path": str(history_path.resolve()),
            "test_split_path": str(split_path.resolve()),
            "test_loss": f"{test_loss:.6f}",
        }

    def _evaluate_loss(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                logits = model(features)
                loss = criterion(logits, targets)
                total_loss += loss.item() * features.size(0)
        return total_loss / max(len(loader.dataset), 1)
