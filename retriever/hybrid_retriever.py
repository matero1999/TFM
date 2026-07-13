"""
Retriever híbrido: BM25 + FAISS con Reciprocal Rank Fusion (RRF).

RRF combina los rankings de ambos sistemas sin necesidad de
normalizar sus scores (que tienen escalas completamente distintas).

Fórmula: RRF_score(d) = Σ 1 / (k + rank_i(d))
donde k=60 es la constante estándar de RRF (Cormack et al., 2009).
"""

import logging
from embeddings.embedding_manager import EmbeddingManager
from .vector_store  import VectorStore
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

# Constante RRF estándar (Cormack et al., 2009)
# k=60 penaliza menos los documentos en posiciones bajas del ranking
RRF_K = 60


class HybridRetriever:
    """
    Retriever híbrido que combina búsqueda semántica y léxica.

    Pipeline de recuperación:
        1. BM25: captura términos exactos, siglas, referencias
        2. FAISS: captura semántica y paráfrasis
        3. RRF Fusion: combina rankings sin normalizar scores
        4. Output: top_k chunks más relevantes con trazabilidad completa
    """

    def __init__(
        self,
        faiss_path:        str = "./data/embeddings/faiss_index.bin",
        chunks_path:       str = "./data/embeddings/corpus_chunks.json",
        model_path:        str = "./models/bge-m3-legal-spanish",
        top_k_per_sistema: int = 20,   # Candidatos por sistema antes de fusión
        top_k_final:       int = 5,    # Resultados finales tras fusión
    ):
        self.top_k_per_sistema = top_k_per_sistema
        self.top_k_final       = top_k_final

        logger.info("Inicializando HybridRetriever...")

        # Inicializar componentes
        self.vector_store = VectorStore(faiss_path, chunks_path)
        self.bm25         = BM25Retriever(chunks_path)
        self.embedder     = EmbeddingManager(model_path=model_path)

        logger.info("HybridRetriever listo.")

    # ─── Búsqueda principal ──────────────────────────────────────
    def buscar(
        self,
        query:   str,
        top_k:   int = None,
        detalle: bool = False,
    ) -> list[dict]:
        """
        Búsqueda híbrida completa.

        Args:
            query:   Consulta del usuario en lenguaje natural
            top_k:   Nº de resultados finales (usa default si None)
            detalle: Si True, incluye scores intermedios en el resultado

        Returns:
            Lista de chunks ordenados por relevancia combinada:
            [
                {
                    ...datos del chunk...,
                    "score_rrf":    float,   # score combinado
                    "rank_final":   int,     # posición en ranking final
                    "score_vector": float,   # score FAISS (si detalle=True)
                    "score_bm25":   float,   # score BM25  (si detalle=True)
                    "en_vector":    bool,    # si apareció en FAISS
                    "en_bm25":      bool,    # si apareció en BM25
                }
            ]
        """
        k_final = top_k or self.top_k_final

        # ── 1. Búsqueda vectorial ────────────────────────────────
        query_embedding = self.embedder.encode_query(query)
        resultados_vector = self.vector_store.buscar(
            query_embedding,
            top_k=self.top_k_per_sistema,
        )

        # ── 2. Búsqueda BM25 ────────────────────────────────────
        resultados_bm25 = self.bm25.buscar(
            query,
            top_k=self.top_k_per_sistema,
        )

        # ── 3. RRF Fusion ────────────────────────────────────────
        fusionados = self._rrf_fusion(
            resultados_vector,
            resultados_bm25,
            k_final=k_final,
        )

        # ── 4. Limpiar campos internos si no se pide detalle ─────
        if not detalle:
            for r in fusionados:
                r.pop("rank_vector",   None)
                r.pop("rank_bm25",     None)
                r.pop("retrieval_idx", None)

        return fusionados

    # ─── RRF Fusion ──────────────────────────────────────────────
    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results:   list[dict],
        k_final:        int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        Para cada documento en cualquiera de los dos rankings:
            score_rrf += 1 / (RRF_K + rank)

        Un documento que aparece en ambos rankings suma
        contribuciones de ambos → sube en el ranking final.
        """
        rrf_scores:   dict[str, float] = {}
        chunk_by_id:  dict[str, dict]  = {}
        en_vector:    set[str]         = set()
        en_bm25:      set[str]         = set()

        # Contribución del ranking vectorial
        for resultado in vector_results:
            cid  = resultado["chunk_id"]
            rank = resultado.get("rank_vector", 999)
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_id[cid] = resultado
            en_vector.add(cid)

        # Contribución del ranking BM25
        for resultado in bm25_results:
            cid  = resultado["chunk_id"]
            rank = resultado.get("rank_bm25", 999)
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = resultado
            en_bm25.add(cid)

        # Ordenar por score RRF descendente
        ids_ordenados = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )[:k_final]

        # Construir resultados finales
        resultados_finales = []
        for rank_final, cid in enumerate(ids_ordenados, start=1):
            chunk = chunk_by_id[cid].copy()
            chunk["score_rrf"]  = round(rrf_scores[cid], 6)
            chunk["rank_final"] = rank_final
            chunk["en_vector"]  = cid in en_vector
            chunk["en_bm25"]    = cid in en_bm25
            resultados_finales.append(chunk)

        return resultados_finales