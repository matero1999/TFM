"""
Script de ejecución de la ingesta.

Uso:
    cd C:\\Users\\Usuario\\TFM
    python -m ingesta.run_ingesta
"""

import logging
import json
from pathlib import Path
from .corpus_builder import CorpusBuilder

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/ingesta.log", encoding="utf-8"),
    ],
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  INGESTA CORPUS — ASISTENTE JURÍDICO-LABORAL BOE")
    print("=" * 60)

    builder = CorpusBuilder(
        output_dir="./data/corpus",
        delay=1.5,
    )

    chunks = builder.build()

    # ─── Resumen final ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ INGESTA COMPLETADA")
    print(f"  📦 Chunks generados:  {len(chunks)}")
    print(f"  📁 Ruta del corpus:   ./data/corpus/")
    print("=" * 60)

    # Estadísticas rápidas
    tipos = {}
    for c in chunks:
        t = c.get("tipo_bloque", "desconocido")
        tipos[t] = tipos.get(t, 0) + 1

    print("\n  Distribución por tipo de bloque:")
    for tipo, count in sorted(tipos.items()):
        print(f"    {tipo:20} {count} chunks")

    # Longitud media de texto
    if chunks:
        media = sum(len(c["texto"]) for c in chunks) / len(chunks)
        print(f"\n  Longitud media de texto: {int(media)} caracteres")