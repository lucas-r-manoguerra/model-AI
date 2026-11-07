# Nivel 2 — Motor Neuronal Propio

Este paquete contiene el punto de partida para construir un Transformer
Decoder-only desde cero, en PyTorch, con exportación posterior a Rust/Candle.

## Estructura

```
training/
├── model_config.py        # Configuración centralizada del modelo
├── generate_synthetic.py  # Generador de corpus sintético en español
├── train_tokenizer.py     # Creador del tokenizer byte-level (260 tokens)
├── model.py               # Transformer PyTorch (RMSNorm, RoPE, SwiGLU, atención causal)
└── smoke_test.py          # Prueba de forward pass y conteo de parámetros
```

## Arquitectura elegida (prototipo ~30M parámetros)

| Componente          | Valor                   |
| ------------------- | ----------------------- |
| Arquitectura        | Decoder-only            |
| Capas Transformer   | 9                       |
| Hidden size         | 448                     |
| Attention heads     | 7                       |
| Head dimension      | 64                      |
| Intermediate size   | 1.792                   |
| Vocabulario         | 260 (256 bytes + 4 esp.)|
| Contexto máximo     | 512                     |
| Posiciones          | RoPE                    |
| Normalización       | RMSNorm                 |
| Feed-forward        | SwiGLU                  |
| Atención            | Causal                  |
| Embeddings          | Compartidos con LM Head |

## Pasos para ejecutar

### 1. Entorno Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch tokenizers safetensors
```

### 2. Generar corpus sintético
```bash
python training/generate_synthetic.py --examples 50000
```

### 3. Crear tokenizer
```bash
python training/train_tokenizer.py
```

### 4. Prueba del modelo
```bash
python training/smoke_test.py
```

Salida esperada:

```
Parámetros entrenables: 29,026,368
Forma de logits: (1, 24, 260)
Dispositivo: cpu
Tiene NaN: False
Loss con targets: 5.57xx
OK: forward pass
```

## Siguientes hitos

1. Confirmar forward pass en PyTorch (este archivo).
2. Implementar entrenamiento causal (dataset, batches, next-token prediction).
3. Entrenar y exportar pesos a SafeTensors.
4. Reproducir el modelo en Rust/Candle y comparar logits contra PyTorch.
