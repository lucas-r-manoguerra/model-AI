"""Generación de texto autoregresiva con el Transformer Nano (Nivel 2).

Soporta temperatura, top-k y top-p para muestreo.
"""

import os
import torch

from model_config import CONFIG
from model import build_model
from train_tokenizer import (
    encode_text,
    decode_ids,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB,
    ID_TO_TOKEN,
)


def _top_k_top_p_filter(logits, top_k: int, top_p: float):
    """Aplica filtro top-k y top-p sobre los logits del último token."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Tokens con probabilidad acumulada > top_p se eliminan
        sorted_indices_to_remove = cumulative_probs > top_p
        # Desplazar para mantener el primer token por encima del umbral
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def generate(
    prompt: str,
    weights_path: str = "weights/model.safetensors",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
    config: ModelConfig = CONFIG,
):
    model = build_model(config).to(device)

    if os.path.exists(weights_path):
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        print(f"Pesos cargados desde {weights_path}")
    else:
        print("Advertencia: no se encontraron pesos; se genera con pesos aleatorios.")

    model.eval()

    # Codificar prompt: quitar BOS y EOS, dejar solo los bytes del contenido
    full_ids = encode_text(prompt)
    content_ids = full_ids[1:-1] if len(full_ids) > 2 else []
    input_ids = [BOS_ID] + content_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncar si excede el contexto máximo
            if input_tensor.size(1) > CONFIG.max_seq_len:
                input_tensor = input_tensor[:, -CONFIG.max_seq_len:]

            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :] / temperature
            next_logits = _top_k_top_p_filter(next_logits.unsqueeze(0), top_k, top_p)[0]
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)],
                dim=1,
            )

            if next_token == EOS_ID:
                break

    return decode_ids(generated)


def generate_interactive(
    weights_path: str = "weights/model.safetensors",
    device: str = "cpu",
):
    print("Generación interactiva. Escribe 'salir' para terminar.\n")
    while True:
        try:
            prompt = input("Tú: ")
        except EOFError:
            break
        if prompt.strip().lower() in ("salir", "exit", "quit"):
            break
        if not prompt.strip():
            continue
        respuesta = generate(prompt, weights_path=weights_path, device=device)
        print(f"Modelo: {respuesta}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        print(generate(prompt))
    else:
        generate_interactive()
