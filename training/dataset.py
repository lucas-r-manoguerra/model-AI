"""Carga y preparación de datos para el Transformer Nano (Nivel 2).

Lee el corpus sintético, tokeniza cada línea y crea ejemplos de
entrenamiento causal: input = tokens[:-1], target = tokens[1:].
"""

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from model_config import CONFIG
from train_tokenizer import encode_text, PAD_ID


class SyntheticTextDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int = CONFIG.max_seq_len):
        self.max_seq_len = max_seq_len
        self.examples: List[List[int]] = []
        self._load(file_path)

    def _load(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = encode_text(line)
                # Truncar o padding a max_seq_len
                if len(ids) > self.max_seq_len:
                    ids = ids[: self.max_seq_len]
                else:
                    ids = ids + [PAD_ID] * (self.max_seq_len - len(ids))
                self.examples.append(ids)

        if len(self.examples) == 0:
            raise ValueError(f"El archivo {file_path} no produjo ejemplos válidos.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, targets


def create_dataloaders(
    train_path: str = "data/synthetic_train.txt",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 8,
    seed: int = 42,
    max_seq_len: int = CONFIG.max_seq_len,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    random.seed(seed)

    full_dataset = SyntheticTextDataset(train_path, max_seq_len=max_seq_len)
    n = len(full_dataset)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {n} ejemplos totales")
    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

    return train_loader, val_loader, test_loader
