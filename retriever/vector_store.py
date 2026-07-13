"""
Búsqueda vectorial semántica sobre el índice FAISS construido en Paso 3.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Rutas por defecto ───────────────────────────────────────────
DEFAULT_FAISS_PATH  = "./data/embeddings/faiss_index.bin"
DEFAULT_CHUNKS_PATH = "./data/embeddings/corpus_chunks.json"


class VectorStore:
    """
    Encapsula el índice FAISS y los metadatos del corpus.
    Proporciona búsqueda semántica por similitud coseno.
    """

    def __init__(
        self,
        faiss_path:  str = DEFAULT_FAISS_PATH,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
    ):
        # Cargar índice FAISS
        faiss_path = Path(faiss_path)
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"Índice FAISS no encontrado: {faiss_path}\n"
                "Ejecute primero: python -m embeddings.generate"
            )

        self.index = faiss.read_index(str(faiss_path))
        logger.info(f"Índice FAISS cargado: {self.index.ntotal} vectores")

        # Cargar chunks con metadatos
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Fichero de chunks no encontrado: {chunks_path}"
            )

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(f"Chunks cargados: {len(self.chunks)}")

        # Verificar consistencia índice ↔ chunks
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Desajuste: {self.index.ntotal} vectores en FAISS "
                f"pero {len(self.chunks)} chunks en JSON. "
                "Regenere los embeddings."
            )

    def buscar(
        self,
        query_embedding: np.ndarray,
        top_k:           int = 10,
    ) -> list[dict]:
        """
        Búsqueda semántica por similitud coseno.

        Args:
            query_embedding: Vector de consulta (1024,) normalizado L2
            top_k:           Número de resultados a devolver

        Returns:
            Lista de chunks ordenados por score descendente:
            [{chunk_data..., "score": float, "rank_vector": int}]
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_vec, k=top_k)

        resultados = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:  # FAISS devuelve -1 si no hay suficientes vectores
                continue

            chunk = self.chunks[idx].copy()
            chunk["score_vector"]  = float(score)
            chunk["rank_vector"]   = rank
            chunk["retrieval_idx"] = int(idx)
            resultados.append(chunk)

        return resultados