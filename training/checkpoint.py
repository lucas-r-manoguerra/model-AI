"""Guardado y recuperación de checkpoints y pesos SafeTensors (Nivel 2)."""

import os
import torch
from safetensors.torch import save_file, load_file

from model_config import CONFIG


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    path: str = "checkpoints/checkpoint.pt",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": CONFIG.vocab_size,
            "max_seq_len": CONFIG.max_seq_len,
            "n_layers": CONFIG.n_layers,
            "n_heads": CONFIG.n_heads,
            "head_dim": CONFIG.head_dim,
            "hidden_size": CONFIG.hidden_size,
            "intermediate_size": CONFIG.intermediate_size,
        },
    }
    torch.save(state, path)
    print(f"Checkpoint guardado en {path} (epoch={epoch}, step={step})")


def load_checkpoint(
    model,
    optimizer=None,
    path: str = "checkpoints/checkpoint.pt",
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    step = state.get("step", 0)
    print(f"Checkpoint cargado desde {path} (epoch={epoch}, step={step})")
    return epoch, step


def save_safetensors(model, path: str = "weights/model.safetensors"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # SafeTensors no permite tensores con memoria compartida (embeddings atados).
    # Clonamos para romper el compartir y guardar copias independientes.
    tensors = {k: v.contiguous().cpu().clone() for k, v in model.state_dict().items()}
    save_file(tensors, path)
    print(f"Pesos exportados a SafeTensors: {path}")


def load_safetensors(model, path: str = "weights/model.safetensors"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontraron pesos en: {path}")
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
    print(f"Pesos cargados desde SafeTensors: {path}")
