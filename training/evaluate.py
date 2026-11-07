"""Evaluación del Transformer Nano en conjunto de prueba (Nivel 2)."""

import math
import torch

from model_config import CONFIG
from model import build_model
from dataset import create_dataloaders
from checkpoint import load_safetensors


def evaluate(
    weights_path: str = "weights/model.safetensors",
    test_path: str = "data/synthetic_train.txt",
    batch_size: int = 8,
    device: str = "cpu",
):
    _, _, test_loader = create_dataloaders(
        train_path=test_path,
        batch_size=batch_size,
    )

    model = build_model(CONFIG).to(device)
    load_safetensors(model, weights_path)
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for input_ids, targets in test_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits, loss = model(input_ids, targets=targets)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    print("Evaluación en test:")
    print(f"  Batches: {total_batches}")
    print(f"  Loss promedio: {avg_loss:.4f}")
    print(f"  Perplejidad: {perplexity:.2f}")

    return avg_loss, perplexity


if __name__ == "__main__":
    evaluate()
