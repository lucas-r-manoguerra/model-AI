"""Entrenamiento causal del Transformer Nano (Nivel 2).

Bucle de entrenamiento con:
- Next-token prediction (causal language modeling)
- Pérdida cross-entropy
- Optimizador AdamW con weight decay
- Scheduler CosineAnnealingLR
- Gradient clipping
- Checkpoints periódicos y exportación SafeTensors del mejor modelo
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_config import CONFIG
from model import build_model
from dataset import create_dataloaders
from checkpoint import save_checkpoint, save_safetensors


def evaluate_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            _, loss = model(input_ids, targets=targets)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train(
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    save_every: int = 500,
    log_every: int = 50,
    device: str = "cpu",
    train_path: str = "data/synthetic_train.txt",
    checkpoint_dir: str = "checkpoints",
    weights_path: str = "weights/model.safetensors",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    train_loader, val_loader, _ = create_dataloaders(
        train_path=train_path,
        batch_size=batch_size,
    )

    model = build_model(CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    global_step = 0
    best_val_loss = float("inf")

    print(f"Iniciando entrenamiento: {epochs} épocas, batch_size={batch_size}, lr={learning_rate}")
    print(f"Parámetros entrenables: {model.count_parameters():,}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (input_ids, targets) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, targets=targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % log_every == 0:
                avg = running_loss / log_every
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss {avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0

            if global_step % save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    path=f"{checkpoint_dir}/checkpoint_step_{global_step}.pt",
                )

        # Validación al final de cada época
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch} completa | Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_safetensors(model, weights_path)
            print(f"  >>> Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")

    save_checkpoint(model, optimizer, epochs, global_step,
                    path=f"{checkpoint_dir}/checkpoint_final.pt")
    save_safetensors(model, weights_path)
    print("Entrenamiento completado.")


if __name__ == "__main__":
    train()
