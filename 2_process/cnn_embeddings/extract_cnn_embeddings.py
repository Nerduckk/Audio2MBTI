"""CLI for extracting dense embeddings from a trained AudioCNN model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.CNN.model import AudioCNN
from infrastructure.config_loader import load_cnn_config


def extract_embeddings(
    x_path: Path,
    model_path: Path,
    output_path: Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    """Load X.npy, run through CNN (up to classification head), save embeddings.npy."""
    print(f"Loading data from {x_path}...")
    X = np.load(x_path, mmap_mode='r')
    # X shape is usually (N, H, W, 1), model expects (N, 1, H, W)
    tensor_x = torch.from_numpy(X).permute(0, 3, 1, 2).float()
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading CNN model from {model_path}...")
    config = load_cnn_config()
    model = AudioCNN.from_config(config)
    
    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model = checkpoint # In case they saved the whole model object
    except Exception as e:
        print(f"Warning: Could not load state dict: {e}")
        # Final fallback in case it's a raw model object but torch.load failed earlier
        pass

    model.to(device)
    model.eval()

    embeddings = []
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            # Use the new extract_embedding method bypassing the MLP classifier
            if hasattr(model, "extract_embedding"):
                out = model.extract_embedding(inputs)
            else:
                out = model.features(inputs)
                out = model.pool(out)
                out = torch.flatten(out, 1)
            embeddings.append(out.cpu().numpy())

    embeddings_array = np.vstack(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_array)
    print(f"Successfully saved {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]} to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CNN embeddings.")
    parser.add_argument("--x-path", default="data/X_train.npy", help="Path to spectrograms numpy array.")
    parser.add_argument("--model-path", default="models/cnn_model_best.pth", help="Path to trained PyTorch model.")
    parser.add_argument("--output-path", default="data/cnn_embeddings.npy", help="Path to save embeddings.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_embeddings(
        x_path=Path(args.x_path),
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
