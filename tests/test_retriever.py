"""
Prueba interactiva del retriever híbrido.

Uso:
    cd C:\\Users\\Usuario\\TFM
    python scripts\\test_retriever.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,   # Solo warnings durante el test
    format="%(asctime)s [%(levelname)s] %(message)s"
)

from retriever import HybridRetriever

# ─── Preguntas de prueba ─────────────────────────────────────────
PREGUNTAS_TEST = [
    "¿Cuál es la jornada máxima de trabajo semanal?",
    "¿Cuánto dura el período de prueba en un contrato indefinido?",
    "¿Qué prestaciones cubre la incapacidad temporal?",
    "¿Cuáles son las causas de despido disciplinario?",
    "¿Qué es un ERTE y cuándo se puede aplicar?",
    "¿Cuánto cobra un trabajador en situación de desempleo?",
]

def mostrar_resultado(rank: int, chunk: dict) -> None:
    print(f"\n  {rank}. [{chunk['score_rrf']:.5f}] {chunk.get('titulo_bloque', 'Sin título')}")
    print(f"     Norma:  {chunk.get('norma_id', '')}")
    print(f"     Tipo:   {chunk.get('tipo_bloque', '')}")
    print(f"     Vector: {'✅' if chunk.get('en_vector') else '❌'} | "
          f"BM25: {'✅' if chunk.get('en_bm25') else '❌'}")
    texto_preview = chunk.get("texto", "")[:120].replace("\n", " ")
    print(f"     Texto:  {texto_preview}...")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST RETRIEVER HÍBRIDO — ASISTENTE JURÍDICO BOE")
    print("=" * 65)

    # Inicializar retriever
    print("\nCargando retriever...")
    retriever = HybridRetriever(
        top_k_per_sistema=20,
        top_k_final=5,
    )
    print("✅ Retriever listo\n")

    # Ejecutar preguntas de prueba
    for pregunta in PREGUNTAS_TEST:
        print("\n" + "─" * 65)
        print(f"  Q: {pregunta}")
        print("─" * 65)

        resultados = retriever.buscar(pregunta, detalle=True)

        if not resultados:
            print("  ⚠️  Sin resultados")
            continue

        for r in resultados:
            mostrar_resultado(r["rank_final"], r)

    # Modo interactivo
    print("\n" + "=" * 65)
    print("  MODO INTERACTIVO (Ctrl+C para salir)")
    print("=" * 65)

    while True:
        try:
            pregunta = input("\n  Consulta: ").strip()
            if not pregunta:
                continue

            resultados = retriever.buscar(pregunta, detalle=True)
            for r in resultados:
                mostrar_resultado(r["rank_final"], r)

        except KeyboardInterrupt:
            print("\n\n  Saliendo...")
            break