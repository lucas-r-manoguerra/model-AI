# model-AI — Motor neuronal propio desde cero

Proyecto para construir un Transformer decoder-only entrenado desde cero, con
motor de inferencia en Rust (Candle) y entrenamiento en PyTorch.

## Estado actual

| Nivel | Estado | Descripción |
|-------|--------|-------------|
| Nivel 1 | ✅ Completado | Motor de inferencia en Rust + Candle para CPU. Carga SafeTensors, tokeniza y ejecuta forward pass. |
| Nivel 2 | 🔄 Entrenando | Transformer PyTorch (~29M parámetros, RoPE, RMSNorm, SwiGLU, atención causal). Corpus sintético de 50.000 ejemplos. Entrenamiento en curso. |

## Estructura

```
model-AI/
├── Cargo.toml
├── src/main.rs          # Motor de inferencia Rust (Nivel 1)
├── README.md
├── README_NIVEL2.md
├── training/
│   ├── model_config.py  # Configuración del modelo
│   ├── model.py         # Transformer PyTorch
│   ├── train.py         # Entrenamiento causal
│   ├── evaluate.py      # Evaluación en test
│   ├── generate.py      # Generación autoregresiva
│   ├── dataset.py       # Carga y tokenización de datos
│   ├── checkpoint.py    # Checkpoints y SafeTensors
│   ├── train_tokenizer.py
│   ├── generate_synthetic.py
│   ├── smoke_test.py    # Prueba de forward pass
│   └── train_smoke.py   # Prueba rápida end-to-end
├── data/
│   ├── synthetic_train.txt   # 50.000 ejemplos sintéticos
│   └── synthetic_small.txt   # 200 ejemplos (pruebas rápidas)
├── tokenizer/
│   └── tokenizer.json        # Tokenizer byte-level (260 tokens)
├── weights/                  # Pesos SafeTensors (se genera al entrenar)
├── checkpoints/              # Checkpoints de entrenamiento
└── .gitignore
```

## Ejecutar el entrenamiento (Nivel 2)

```bash
source .venv/bin/activate
python training/train.py --epochs 5 --batch_size 8
```

## Ejecutar el motor de inferencia (Nivel 1)

```bash
cargo run --release
```

Requiere `weights/model.safetensors` y `tokenizer/tokenizer.json`.

## Roadmap (10 fases)

0. Preparación del entorno y estructura ← aquí
1. Modelo base en Python ← entrenando
2. Tokenizer y corpus ← completado
3. Entrenamiento desde cero ← en curso
4. Generación de texto
5. Portabilidad a Rust (comparar logits PyTorch vs Candle)
6. Arquitectura cognitiva ICL
7. Razonamiento y refinamiento
8. Agentes, memoria y herramientas
9. RSI controlada (auto-mejora supervisada)
