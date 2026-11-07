"""Smoke test de entrenamiento end-to-end (Nivel 2).

Usa una configuración miniatura para verificar en segundos que el pipeline
completo funciona: dataset, entrenamiento, checkpoints, SafeTensors, evaluación y generación.
"""

import os
import sys
import torch

sys.path.insert(0, ".")

from model_config import ModelConfig
from model import build_model
from dataset import SyntheticTextDataset, create_dataloaders
from checkpoint import save_checkpoint, save_safetensors, load_safetensors
from train import evaluate_loss
from generate import generate


def main():
    # Configuración miniatura para prueba rápida
    cfg = ModelConfig(
        vocab_size=260,
        max_seq_len=32,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        hidden_size=64,
        intermediate_size=256,
    )

    print(f"Config mini: {cfg.n_layers} capas, hidden={cfg.hidden_size}, "
          f"seq_len={cfg.max_seq_len}, params={build_model(cfg).count_parameters():,}")

    # Usar el dataset pequeño existente
    data_path = "data/synthetic_small.txt"
    if not os.path.exists(data_path):
        print(f"Creando {data_path}...")
        from generate_synthetic import generate_examples
        exs = generate_examples(seed=42, n_examples=50)
        with open(data_path, "w", encoding="utf-8") as f:
            for line in exs:
                f.write(line + "\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_path,
        batch_size=4,
        seed=42,
        max_seq_len=cfg.max_seq_len,
    )

    device = "cpu"
    model = build_model(cfg).to(device)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.nn as nn

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=2 * len(train_loader))

    # Entrenamiento 2 épocas
    for epoch in range(1, 3):
        model.train()
        running = 0.0
        for batch_idx, (input_ids, targets) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets=targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()

        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch} | Train loss {running/len(train_loader):.4f} | Val loss {val_loss:.4f}")

    # Guardar pesos
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    save_checkpoint(model, optimizer, 2, 999, path="checkpoints/smoke.pt")
    save_safetensors(model, path="weights/smoke.safetensors")

    # Cargar desde SafeTensors y generar
    model2 = build_model(cfg).to(device)
    load_safetensors(model2, path="weights/smoke.safetensors")

    texto = generate("Hola", weights_path="weights/smoke.safetensors", max_new_tokens=10,
                     device=device, temperature=0.8, config=cfg)
    print(f"Generación: 'Hola' -> '{texto}'")

    print("\nSMOKE TEST OK: pipeline completo funcionando")


if __name__ == "__main__":
    main()
