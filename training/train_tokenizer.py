"""Tokenizer byte-level para el Nivel 2.

Vocabulario: 256 bytes + 4 especiales = 260 tokens.
  ID 0: <pad>
  ID 1: <bos>
  ID 2: <eos>
  ID 3:  (vacío)
  IDs 4..259: <byte_000> .. <byte_255>
"""

import json
import os

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", ""]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 260


def build_byte_level_vocab() -> dict:
    vocab = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i
    for b in range(256):
        vocab[f"<byte_{b:03d}>"] = len(SPECIAL_TOKENS) + b
    return vocab


VOCAB = build_byte_level_vocab()
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}


def save_tokenizer_json(path: str = "tokenizer/tokenizer.json"):
    id_to_token = {v: k for k, v in VOCAB.items()}
    tokenizer_dict = {
        "version": "1.0",
        "truncation": None,
        "padding": {"pad_token": "<pad>", "pad_id": PAD_ID},
        "added_tokens": [
            {
                "id": i,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
            for i, tok in enumerate(SPECIAL_TOKENS)
        ],
        "normalizer": {"type": "ByteLevel"},
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "WordLevel",
            "vocab": id_to_token,
            "unk_token": "",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)


def encode_text(text: str) -> list:
    """Codifica texto a lista de IDs: [BOS, bytes..., EOS]."""
    ids = [BOS_ID]
    for byte in text.encode("utf-8"):
        ids.append(VOCAB[f"<byte_{byte:03d}>"])
    ids.append(EOS_ID)
    return ids


def decode_ids(ids: list) -> str:
    """Decodifica lista de IDs a texto, ignorando tokens no byte."""
    bytes_list = []
    for tid in ids:
        token = ID_TO_TOKEN.get(tid, "")
        if token.startswith("<byte_"):
            try:
                b = int(token[6:9])
                bytes_list.append(b)
            except ValueError:
                continue
    return bytes(bytes_list).decode("utf-8", errors="replace")


def main():
    save_tokenizer_json()
    print(f"Tokenizer guardado. Vocabulario: {VOCAB_SIZE}")
    prueba = "Hola mundo"
    ids = encode_text(prueba)
    dec = decode_ids(ids)
    print(f"Prueba: '{prueba}' -> {ids} -> '{dec}'")


if __name__ == "__main__":
    main()
