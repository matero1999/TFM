"""
Endpoints de la API del asistente jurídico-laboral.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio

from .schemas import (
    ConsultaRequest,
    ConsultaResponse,
    FuenteResponse,
    HealthResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Health check ────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del sistema",
    tags=["Sistema"],
)
async def health_check(request: Request):
    """
    Verifica que el sistema está operativo.
    Devuelve el estado del RAG pipeline y el corpus.
    """
    rag = request.app.state.rag

    # Contar chunks del corpus
    try:
        n_chunks = len(rag.retriever.bm25.chunks)
    except Exception:
        n_chunks = 0

    return HealthResponse(
        status="ok",
        version="1.0.0",
        backend_activo=rag.backend,
        corpus_chunks=n_chunks,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ─── Consulta principal ──────────────────────────────────────────
@router.post(
    "/consulta",
    response_model=ConsultaResponse,
    summary="Realizar una consulta jurídico-laboral",
    tags=["Asistente"],
)
async def realizar_consulta(
    body:    ConsultaRequest,
    request: Request,
):
    rag = request.app.state.rag

    logger.info(f"Consulta recibida: '{body.pregunta[:80]}'")

    try:
        # ── CORRECCIÓN: ejecutar en thread pool ──────────────────
        # asyncio.to_thread() libera el event loop durante la
        # inferencia síncrona del LLM, permitiendo que el servidor
        # siga respondiendo a health checks y otras peticiones.
        resultado = await asyncio.to_thread(
            rag.consultar,
            body.pregunta,
        )

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Error interno al procesar la consulta: {str(e)}",
        )

    fuentes = [
        FuenteResponse(
            norma_id=      f.get("norma_id",      ""),
            titulo_bloque= f.get("titulo_bloque", ""),
            ley_titulo=    f.get("ley_titulo",    ""),
            url_boe=       f.get("url_boe",       ""),
            score_rrf=     f.get("score_rrf",     0.0),
        )
        for f in resultado.get("fuentes", [])
    ]

    return ConsultaResponse(
        pregunta=       resultado["pregunta"],
        respuesta=      resultado["respuesta"],
        fuentes=        fuentes,
        tokens_entrada= resultado["tokens_entrada"],
        tokens_salida=  resultado["tokens_salida"],
        coste_usd=      resultado["coste_usd"],
        latencia_ms=    resultado["latencia_ms"],
        modelo=         resultado["modelo"],
        backend=        resultado["backend"],
        timestamp=      datetime.now(timezone.utc).isoformat(),
    )


# ─── Listado de fuentes disponibles ──────────────────────────────
@router.get(
    "/corpus/info",
    summary="Información sobre el corpus disponible",
    tags=["Corpus"],
)
async def corpus_info(request: Request):
    """
    Devuelve información sobre las normas indexadas en el corpus.
    """
    rag = request.app.state.rag

    try:
        chunks = rag.retriever.bm25.chunks

        # Agrupar por norma
        normas = {}
        for chunk in chunks:
            nid = chunk.get("norma_id", "desconocida")
            if nid not in normas:
                normas[nid] = {
                    "norma_id":   nid,
                    "ley_titulo": chunk.get("ley_titulo",  ""),
                    "ley_rango":  chunk.get("ley_rango",   ""),
                    "url_boe":    chunk.get("url_boe",     ""),
                    "n_chunks":   0,
                }
            normas[nid]["n_chunks"] += 1

        return {
            "total_chunks": len(chunks),
            "total_normas": len(normas),
            "normas":       list(normas.values()),
        }

    except Exception as e:
        logger.error(f"Error obteniendo info del corpus: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# ─── Búsqueda directa (sin LLM) ──────────────────────────────────
@router.get(
    "/buscar",
    summary="Búsqueda directa en el corpus sin LLM",
    tags=["Corpus"],
)
async def buscar_chunks(
    q:     str,
    top_k: int = 5,
    request: Request = None,
):
    """
    Búsqueda híbrida directa en el corpus (BM25 + FAISS).
    Útil para depuración y evaluación del retriever.
    """
    if not q or len(q.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="La consulta debe tener al menos 3 caracteres."
        )

    rag = request.app.state.rag

    try:
        chunks = rag.retriever.buscar(q, top_k=min(top_k, 10))
        return {
            "query":    q,
            "top_k":    top_k,
            "chunks": [
                {
                    "titulo_bloque": c.get("titulo_bloque", ""),
                    "norma_id":      c.get("norma_id",      ""),
                    "ley_titulo":    c.get("ley_titulo",     ""),
                    "score_rrf":     c.get("score_rrf",     0.0),
                    "en_vector":     c.get("en_vector",     False),
                    "en_bm25":       c.get("en_bm25",       False),
                    "texto_preview": c.get("texto", "")[:200],
                    "url_boe":       c.get("url_boe",       ""),
                }
                for c in chunks
            ],
        }
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        raise HTTPException(status_code=503, detail=str(e))