"""Configuración centralizada del Transformer Nano (Nivel 2)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Arquitectura
    vocab_size: int = 260          # 256 bytes + 4 especiales
    max_seq_len: int = 512         # contexto máximo
    n_layers: int = 9              # bloques Transformer
    n_heads: int = 7               # cabezas de atención
    head_dim: int = 64             # dimensión por cabeza
    hidden_size: int = 448         # d_model
    intermediate_size: int = 1792  # SwiGLU intermedio (aprox 4x hidden)

    # Componentes
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    causal: bool = True

    # Embeddings
    tie_embeddings: bool = True    # compartir embedding con LM Head

    # Derivados
    @property
    def embedding_dim(self) -> int:
        return self.hidden_size

    @property
    def num_special_tokens(self) -> int:
        return 4  # <pad>, <bos>, <eos>, 

    @property
    def num_byte_tokens(self) -> int:
        return 256

    def total_parameters_estimate(self) -> int:
        """Aproximación de parámetros entrenables."""
        cfg = self
        # Embedding compartido
        emb = cfg.vocab_size * cfg.hidden_size
        # Por capa Transformer
        # Attention: Q, K, V, O
        attn = 4 * cfg.hidden_size * cfg.hidden_size
        # SwiGLU: W1, W2, W3 (gate, down, up)
        ffn = 3 * cfg.hidden_size * cfg.intermediate_size
        # RMSNorm x2 (attention + ffn)
        norms = 2 * cfg.hidden_size
        per_layer = attn + ffn + norms
        total = emb + cfg.n_layers * per_layer
        # LM Head no cuenta si está atado al embedding
        return total


CONFIG = ModelConfig()
