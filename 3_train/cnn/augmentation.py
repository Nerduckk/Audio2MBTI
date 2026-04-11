"""Spectrogram augmentation utilities."""

from __future__ import annotations

import random

import torch
from torch import nn


class SpecAugment(nn.Module):
    """Apply simple time and frequency masking to spectrogram tensors."""

    def __init__(self, freq_mask_param: int = 24, time_mask_param: int = 48, p: float = 0.5):
        super().__init__()
        self.freq_mask_param = max(0, int(freq_mask_param))
        self.time_mask_param = max(0, int(time_mask_param))
        self.p = float(p)

    def _mask_axis(self, tensor: torch.Tensor, axis: int, max_width: int) -> torch.Tensor:
        if max_width <= 0 or random.random() > self.p:
            return tensor

        axis_size = tensor.shape[axis]
        if axis_size <= 1:
            return tensor

        width = random.randint(0, min(max_width, axis_size - 1))
        if width == 0:
            return tensor

        start = random.randint(0, axis_size - width)
        result = tensor.clone()
        slicer = [slice(None)] * result.ndim
        slicer[axis] = slice(start, start + width)
        result[tuple(slicer)] = 0.0
        return result

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor shape (C, H, W), got {tuple(tensor.shape)}")

        augmented = self._mask_axis(tensor, axis=1, max_width=self.freq_mask_param)
        augmented = self._mask_axis(augmented, axis=2, max_width=self.time_mask_param)
        return augmented
