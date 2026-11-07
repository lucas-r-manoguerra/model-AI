"""Generador de corpus sintético en español para el Nivel 2.

Produce ejemplos con operaciones aritméticas, comparaciones y preguntas
con respuestas verificables. Se guarda en data/synthetic_train.txt.
"""

import argparse
import os
import random


ESP_NUMBERS = {
    0: "cero", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro",
    5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
    10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce",
    15: "quince", 16: "dieciséis", 17: "diecisiete", 18: "dieciocho",
    19: "diecinueve", 20: "veinte",
}


def num_to_word(n: int) -> str:
    if n in ESP_NUMBERS:
        return ESP_NUMBERS[n]
    return str(n)


def generate_examples(seed: int = 42, n_examples: int = 50000) -> list[str]:
    random.seed(seed)
    examples: list[str] = []

    # 1. Sumas simples
    for _ in range(n_examples // 5):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        q = f"Pregunta: ¿Cuánto es {num_to_word(a)} más {num_to_word(b)}?"
        r = f"Respuesta: {a + b}."
        examples.append(f"{q} {r}")

    # 2. Restas simples (resultado >= 0)
    for _ in range(n_examples // 5):
        a = random.randint(0, 30)
        b = random.randint(0, a)
        q = f"Pregunta: ¿Cuánto es {a} menos {b}?"
        r = f"Respuesta: {a - b}."
        examples.append(f"{q} {r}")

    # 3. Comparaciones
    for _ in range(n_examples // 5):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        if a > b:
            r = f"Respuesta: {a} es mayor que {b}."
        elif a < b:
            r = f"Respuesta: {b} es mayor que {a}."
        else:
            r = f"Respuesta: {a} es igual a {b}."
        q = f"Pregunta: ¿Cuál es mayor, {a} o {b}?"
        examples.append(f"{q} {r}")

    # 4. Multiplicaciones pequeñas
    for _ in range(n_examples // 5):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        q = f"Pregunta: ¿Cuánto es {a} por {b}?"
        r = f"Respuesta: {a * b}."
        examples.append(f"{q} {r}")

    # 5. Secuencias y patrones
    for _ in range(n_examples // 5):
        start = random.randint(1, 5)
        step = random.randint(1, 3)
        seq = [start + i * step for i in range(5)]
        q = "Pregunta: Completa la secuencia: " + ", ".join(str(x) for x in seq[:4]) + "."
        r = f"Respuesta: {seq[4]}."
        examples.append(f"{q} {r}")

    random.shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar corpus sintético en español")
    parser.add_argument("--examples", type=int, default=50000, help="Número de ejemplos")
    parser.add_argument("--output", type=str, default="data/synthetic_train.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    examples = generate_examples(n_examples=args.examples)

    with open(args.output, "w", encoding="utf-8") as f:
        for line in examples:
            f.write(line + "\n")

    print(f"Generados {len(examples)} ejemplos en {args.output}")


if __name__ == "__main__":
    main()
