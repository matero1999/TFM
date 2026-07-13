"""
Búsqueda léxica BM25 sobre el corpus jurídico.

BM25 complementa la búsqueda vectorial para:
- Términos exactos: "artículo 34", "BOE-A-2015-11430"
- Siglas y números: "ERTE", "IT", "SMI"
- Palabras poco frecuentes pero jurídicamente clave
"""

import json
import logging
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = "./data/embeddings/corpus_chunks.json"


def tokenizar_juridico(texto: str) -> list[str]:
    """
    Tokenizador adaptado a texto jurídico español.

    - Convierte a minúsculas
    - Preserva números y siglas (importantes en textos legales)
    - Mantiene términos compuestos con guión (e.g. "socio-laboral")
    - Elimina stopwords jurídicas vacías
    - Preserva términos clave: artículo, disposición, real decreto...
    """
    STOPWORDS_JURIDICAS = {
        "de", "la", "el", "en", "y", "a", "los", "las", "se",
        "del", "al", "que", "por", "con", "para", "un", "una",
        "es", "su", "sus", "o", "no", "si", "lo", "le", "les",
        "como", "más", "pero", "este", "esta", "estos", "estas",
    }

    texto = texto.lower()

    # Preservar siglas y acrónimos (mayúsculas → proteger antes del lower)
    texto = re.sub(r"([A-ZÁÉÍÓÚÑ]{2,})", lambda m: m.group(0).lower(), texto)

    # Tokenizar: alfanumérico + guiones internos
    tokens = re.findall(r"[a-záéíóúñü0-9]+(?:-[a-záéíóúñü0-9]+)*", texto)

    # Filtrar stopwords y tokens muy cortos (< 2 chars)
    tokens = [
        t for t in tokens
        if t not in STOPWORDS_JURIDICAS and len(t) >= 2
    ]

    return tokens


class BM25Retriever:
    """
    Índice BM25 sobre el corpus jurídico-laboral.

    BM25Okapi es la variante estándar de BM25 con parámetros:
        k1 = 1.5  (saturación de frecuencia de términos)
        b  = 0.75 (normalización por longitud del documento)
    """

    def __init__(self, chunks_path: str = DEFAULT_CHUNKS_PATH):
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Fichero de chunks no encontrado: {chunks_path}"
            )

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(f"Construyendo índice BM25 sobre {len(self.chunks)} chunks...")

        # Tokenizar corpus completo
        # Usamos texto_embedding (enriquecido con título y rango)
        # para que BM25 también capture términos del encabezado legal
        corpus_tokenizado = [
            tokenizar_juridico(
                c.get("texto_embedding") or c.get("texto", "")
            )
            for c in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus_tokenizado)
        logger.info("Índice BM25 construido correctamente.")

    def buscar(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Búsqueda léxica BM25.

        Args:
            query: Consulta en lenguaje natural
            top_k: Número de resultados

        Returns:
            Lista de chunks con score BM25 y ranking
        """
        tokens_query = tokenizar_juridico(query)

        if not tokens_query:
            logger.warning(f"Query sin tokens tras tokenización: '{query}'")
            return []

        scores = self.bm25.get_scores(tokens_query)

        # Obtener top_k índices con mayor score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        resultados = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue  # Omitir chunks sin ninguna coincidencia

            chunk = self.chunks[idx].copy()
            chunk["score_bm25"]    = float(scores[idx])
            chunk["rank_bm25"]     = rank
            chunk["retrieval_idx"] = idx
            resultados.append(chunk)

        return resultados