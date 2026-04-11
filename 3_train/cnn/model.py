"""PyTorch CNN architecture for MBTI prediction from spectrograms."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolutional block with normalization and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNN(nn.Module):
    """Configurable CNN that outputs four MBTI dimension logits."""

    def __init__(
        self,
        input_shape: Iterable[int] = (1, 128, 1290),
        channels: Iterable[int] = (32, 64, 128, 256),
        conv_dropout: float = 0.25,
        dense_hidden: Iterable[int] = (512, 256),
        dense_dropout: Iterable[float] = (0.3, 0.2),
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)

        layers: List[nn.Module] = []
        in_channels = self.input_shape[0]
        for out_channels in channels:
            layers.append(ConvBlock(in_channels, out_channels, conv_dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        dense_layers: List[nn.Module] = []
        prev_dim = in_channels
        dropout_values = list(dense_dropout)
        hidden_values = list(dense_hidden)
        for index, hidden_dim in enumerate(hidden_values):
            dense_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_values[index] if index < len(dropout_values) else 0.2),
                ]
            )
            prev_dim = hidden_dim

        dense_layers.append(nn.Linear(prev_dim, output_dim))
        self.classifier = nn.Sequential(*dense_layers)

    @classmethod
    def from_config(cls, config: dict) -> "AudioCNN":
        feature_cfg = config.get("feature_extraction", {})
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        target_shape = feature_cfg.get("target_shape", [128, 1290])
        return cls(
            input_shape=(1, target_shape[0], target_shape[1]),
            channels=model_cfg.get("channels", [32, 64, 128, 256]),
            conv_dropout=model_cfg.get("conv_dropout", 0.25),
            dense_hidden=model_cfg.get("dense_hidden", [512, 256]),
            dense_dropout=model_cfg.get("dense_dropout", [0.3, 0.2]),
            output_dim=training_cfg.get("output_dim", 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
