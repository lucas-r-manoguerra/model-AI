"""Prueba de humo del Transformer Nano (Nivel 2).

Ejecuta un forward pass, verifica dimensiones, cuenta parámetros entrenables
y comprueba que no hay valores NaN.
"""

import sys
import torch

# Permitir importar model_config y model desde este directorio
sys.path.insert(0, ".")

from model_config import CONFIG
from model import build_model


def main() -> None:
    torch.manual_seed(42)

    model = build_model(CONFIG)
    model.eval()

    n_params = model.count_parameters()
    print(f"Parámetros entrenables: {n_params:,}")

    batch_size = 1
    seq_len = 24
    input_ids = torch.randint(0, CONFIG.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids)

    print(f"Forma de logits: {tuple(logits.shape)}")
    print(f"Dispositivo: {logits.device}")
    print(f"Tiene NaN: {bool(torch.isnan(logits).any())}")

    # Verificar shape esperada
    expected = (batch_size, seq_len, CONFIG.vocab_size)
    assert tuple(logits.shape) == expected, f"Shape inesperada: {tuple(logits.shape)} vs {expected}"
    assert not torch.isnan(logits).any(), "Hay NaN en los logits"

    # Con targets para calcular loss
    targets = torch.randint(0, CONFIG.vocab_size, (batch_size, seq_len))
    logits2, loss2 = model(input_ids, targets=targets)
    print(f"Loss con targets: {loss2.item():.4f}")
    assert loss2 is not None

    print("OK: forward pass")


if __name__ == "__main__":
    main()
