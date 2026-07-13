"""
Genera embeddings del corpus y construye el índice FAISS.

Uso:
    cd C:\\Users\\Usuario\\TFM
    python -m embeddings.generate

Salida:
    data/embeddings/corpus_embeddings.npy   ← vectores numpy
    data/embeddings/corpus_chunks.json      ← chunks con metadata
    data/embeddings/faiss_index.bin         ← índice FAISS listo
"""

import json
import logging
import faiss
import numpy as np
from pathlib import Path
from .embedding_manager import MODEL_ID, EmbeddingManager

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/embeddings.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Rutas ───────────────────────────────────────────────────────
CORPUS_DIR       = Path("./data/corpus/processed")
OUTPUT_DIR       = Path("./data/embeddings")
EMBEDDINGS_PATH  = OUTPUT_DIR / "corpus_embeddings.npy"
CHUNKS_PATH      = OUTPUT_DIR / "corpus_chunks.json"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
LOCAL_MODEL_PATH = "./models/bge-m3-legal-spanish"


def cargar_chunks() -> list[dict]:
    """Carga todos los chunks del corpus desde /processed."""
    chunks = []
    ficheros = sorted(CORPUS_DIR.glob("*.json"))

    if not ficheros:
        raise FileNotFoundError(
            f"No se encontraron chunks en {CORPUS_DIR}. "
            "Ejecute primero: python -m ingesta.run_ingesta"
        )

    logger.info(f"Cargando {len(ficheros)} chunks desde {CORPUS_DIR}...")

    for fichero in ficheros:
        try:
            with open(fichero, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                chunks.append(chunk)
        except json.JSONDecodeError as e:
            logger.warning(f"Error leyendo {fichero.name}: {e}")
            continue

    logger.info(f"Chunks cargados correctamente: {len(chunks)}")
    return chunks


def construir_indice_faiss(
    embeddings: np.ndarray,
    dimension:  int,
) -> faiss.Index:
    """
    Construye índice FAISS de similitud coseno.

    Usamos IndexFlatIP (Inner Product) porque los embeddings
    ya están normalizados L2 → IP == cosine similarity.

    Para corpus pequeños (<100K) IndexFlatIP es óptimo:
    exacto, sin aproximación, latencia <10ms.
    """
    logger.info(f"Construyendo índice FAISS (IndexFlatIP, dim={dimension})...")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    logger.info(f"Índice construido. Vectores indexados: {index.ntotal}")
    return index


def mostrar_estadisticas(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Muestra estadísticas del corpus generado."""
    print("\n" + "=" * 60)
    print("  ESTADÍSTICAS DEL CORPUS EMBEBIDO")
    print("=" * 60)
    print(f"  Total chunks indexados:   {len(chunks)}")
    print(f"  Dimensión embeddings:     {embeddings.shape[1]}")
    print(f"  Memoria (float32):        "
          f"{embeddings.nbytes / 1e6:.1f} MB")

    # Desglose por norma
    print("\n  Desglose por norma:")
    normas = {}
    for c in chunks:
        norma = c.get("ley_rango", "") + " " + c.get("ley_titulo", "")[:40]
        norma = norma.strip() or c.get("norma_id", "desconocida")
        normas[c.get("norma_id", "?")] = normas.get(
            c.get("norma_id", "?"), {"nombre": norma[:50], "count": 0}
        )
        normas[c.get("norma_id", "?")]["count"] += 1

    for norma_id, info in normas.items():
        print(f"    [{norma_id}]")
        print(f"      {info['nombre']}")
        print(f"      {info['count']} chunks")

    # Longitudes
    longitudes = [len(c.get("texto", "")) for c in chunks]
    print(f"\n  Longitud texto (chars):")
    print(f"    Mínima:   {min(longitudes)}")
    print(f"    Máxima:   {max(longitudes)}")
    print(f"    Media:    {int(sum(longitudes)/len(longitudes))}")
    print(f"    Mediana:  {sorted(longitudes)[len(longitudes)//2]}")
    print("=" * 60)


def test_busqueda_rapida(
    manager:    EmbeddingManager,
    chunks:     list[dict],
    index:      faiss.Index,
) -> None:
    """
    Prueba rápida de búsqueda semántica con 3 preguntas reales.
    Verifica que el índice funciona antes de guardar.
    """
    preguntas_test = [
        "¿Cuál es la jornada máxima de trabajo semanal?",
        "¿Cuánto dura el período de prueba en un contrato?",
        "¿Qué prestaciones cubre la incapacidad temporal?",
    ]

    print("\n  TEST DE BÚSQUEDA SEMÁNTICA")
    print("  " + "─" * 56)

    for pregunta in preguntas_test:
        query_emb = manager.encode_query(pregunta)
        query_vec = np.array([query_emb], dtype=np.float32)

        scores, indices = index.search(query_vec, k=3)

        print(f"\n  Q: {pregunta}")
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                continue
            chunk = chunks[idx]
            titulo = chunk.get("titulo_bloque", "Sin título")[:55]
            norma  = chunk.get("norma_id", "")
            print(f"    {rank}. [{score:.4f}] {titulo}")
            print(f"          Norma: {norma}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERACIÓN DE EMBEDDINGS — CORPUS JURÍDICO BOE")
    print("=" * 60)

    # ── 1. Crear directorio de salida ────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Cargar chunks ─────────────────────────────────────────
    chunks = cargar_chunks()

    # ── 3. Inicializar modelo de embeddings ──────────────────────────
    local_path = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else None

    if local_path:
        logger.info(f"Modelo local encontrado: {local_path}")
    else:
        logger.info("Modelo local no encontrado. Se descargará de HuggingFace.")

    manager = EmbeddingManager(
        model_id=MODEL_ID,
        model_path=local_path,
    )

    # ── 4. Preparar textos (usar texto_embedding enriquecido) ────
    # texto_embedding incluye: rango + título ley + título bloque + texto
    # Esto mejora la recuperación al añadir contexto legal al vector
    textos = [
        c.get("texto_embedding") or c.get("texto", "")
        for c in chunks
    ]

    # ── 5. Generar embeddings ────────────────────────────────────
    if EMBEDDINGS_PATH.exists():
        logger.info("Embeddings ya existentes encontrados. Cargando desde disco...")
        embeddings = manager.load(str(EMBEDDINGS_PATH))
    else:
        embeddings = manager.encode_corpus(textos)

    # ── 6. Construir índice FAISS ────────────────────────────────
    index = construir_indice_faiss(embeddings, manager.dimension)

    # ── 7. Test de búsqueda antes de guardar ─────────────────────
    test_busqueda_rapida(manager, chunks, index)

    # ── 8. Guardar todo en disco ──────────────────────────────────
    logger.info("Guardando artefactos en disco...")

    # Embeddings numpy
    manager.save(embeddings, str(EMBEDDINGS_PATH))

    # Chunks con metadata (mismo orden que embeddings → índice = posición)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks guardados: {CHUNKS_PATH}")

    # Índice FAISS
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"Índice FAISS guardado: {FAISS_INDEX_PATH}")

    # ── 9. Estadísticas finales ───────────────────────────────────
    mostrar_estadisticas(chunks, embeddings)

    print("\n  ✅ EMBEDDINGS COMPLETADOS")
    print(f"  📁 Artefactos en: {OUTPUT_DIR}")
    print("=" * 60)