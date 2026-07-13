"""
Prueba completa del pipeline RAG con Salamandra.

Uso:
    cd C:\\Users\\Usuario\\TFM
    python scripts\\test_rag.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

from llm import RAGChain

# ─── Preguntas de prueba ─────────────────────────────────────────
PREGUNTAS_TEST = [
    "¿Cuál es la jornada máxima de trabajo semanal en España?",
    "¿Cuáles son las causas del despido disciplinario?",
    "¿Qué prestaciones cubre la incapacidad temporal?",
]

SEPARADOR = "═" * 65


def mostrar_resultado(resultado: dict, num: int, total: int) -> None:
    """Muestra una respuesta RAG formateada en consola."""
    print(f"\n{SEPARADOR}")
    print(f"  CONSULTA {num}/{total}")
    print(SEPARADOR)
    print(f"\n  Q: {resultado['pregunta']}\n")

    print("  RESPUESTA:")
    print(f"  {'─' * 60}")
    for linea in resultado["respuesta"].split("\n"):
        print(f"  {linea}")

    print(f"\n  FUENTES RECUPERADAS ({len(resultado['fuentes'])}):")
    for i, fuente in enumerate(resultado["fuentes"], 1):
        print(f"    {i}. {fuente.get('titulo_bloque', '')[:55]}")
        print(f"       {fuente.get('ley_titulo', '')[:60]}")
        print(f"       {fuente.get('url_boe', '')}")

    print(f"\n  MÉTRICAS:")
    print(f"    Modelo:    {resultado['modelo']}")
    print(f"    Tokens:    {resultado['tokens_entrada']} in "
          f"| {resultado['tokens_salida']} out")
    print(f"    Coste:     ${resultado['coste_usd']:.5f} USD")
    print(f"    Latencia:  {resultado['latencia_ms']} ms "
          f"({resultado['latencia_ms']//1000}s)")


if __name__ == "__main__":
    print(f"\n{SEPARADOR}")
    print("  TEST RAG — ASISTENTE JURÍDICO BOE (Salamandra)")
    print(SEPARADOR)

    # ── Advertencia modo CPU ──────────────────────────────────────
    import torch
    if not torch.cuda.is_available():
        print("\n  ⚠️  AVISO: No se detectó GPU.")
        print("  La inferencia en CPU puede tardar 2-5 min por respuesta.")
        print("  Ejecutando solo 1 pregunta de prueba para validar.\n")
        PREGUNTAS_TEST = PREGUNTAS_TEST[:1]

    # ── Inicializar pipeline ──────────────────────────────────────
    print("\n  Cargando pipeline RAG con Salamandra...")
    print("  (Esto puede tardar varios minutos la primera vez)\n")

    top_k = 3 if not torch.cuda.is_available() else 5
    rag   = RAGChain(backend="salamandra", top_k=top_k, verbose=True)
    print(f"\n  ✅ Pipeline listo\n")

    # ── Ejecutar consultas de prueba ──────────────────────────────
    for i, pregunta in enumerate(PREGUNTAS_TEST, start=1):
        resultado = rag.consultar(pregunta)
        mostrar_resultado(resultado, i, len(PREGUNTAS_TEST))

    # ── Modo interactivo ──────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("  MODO INTERACTIVO (Ctrl+C para salir)")
    if not torch.cuda.is_available():
        print("  ⚠️  CPU detectada: cada respuesta tardará 2-5 minutos")
    print(SEPARADOR)

    while True:
        try:
            pregunta = input("\n  Consulta: ").strip()
            if not pregunta:
                continue

            print("  ⏳ Procesando...")
            resultado = rag.consultar(pregunta)

            print(f"\n  RESPUESTA:")
            print(f"  {'─' * 60}")
            for linea in resultado["respuesta"].split("\n"):
                print(f"  {linea}")

            print(f"\n  Fuentes: ", end="")
            print(" | ".join(
                f.get("titulo_bloque", "")[:30]
                for f in resultado["fuentes"]
            ))
            print(
                f"  Latencia: {resultado['latencia_ms']}ms "
                f"({resultado['latencia_ms']//1000}s) "
                f"| Coste: $0.00 (local)"
            )

        except KeyboardInterrupt:
            print("\n\n  Saliendo del modo interactivo...")
            break