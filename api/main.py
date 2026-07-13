"""
Punto de entrada del backend FastAPI.

Uso:
    cd C:\\Users\\Usuario\\TFM
    uvicorn api.main:app --reload --port 8000

Documentación automática:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from api.routes import router
from llm.rag_chain import RAGChain

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Lifespan: carga y descarga del modelo ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida del servidor.
    Carga el RAG pipeline al arrancar y lo libera al parar.
    """
    logger.info("=" * 60)
    logger.info("  Iniciando Asistente Jurídico-Laboral BOE")
    logger.info("=" * 60)

    # Determinar backend desde .env (por defecto salamandra)
    import os
    backend = os.getenv("LLM_BACKEND", "salamandra")
    top_k   = int(os.getenv("TOP_K_RETRIEVAL", "3"))

    logger.info(f"Backend LLM: {backend}")
    logger.info(f"Top-K retrieval: {top_k}")

    try:
        app.state.rag = RAGChain(
            backend=backend,
            top_k=top_k,
            verbose=False,
        )
        logger.info("✅ RAG pipeline cargado correctamente")
    except Exception as e:
        logger.error(f"❌ Error cargando RAG pipeline: {e}")
        raise

    yield  # ← Servidor activo y atendiendo peticiones

    # Limpieza al apagar
    logger.info("Cerrando servidor...")
    if hasattr(app.state, "rag"):
        del app.state.rag
    logger.info("✅ Servidor cerrado limpiamente")


# ─── Aplicación FastAPI ──────────────────────────────────────────
app = FastAPI(
    title="Asistente Jurídico-Laboral BOE",
    description=(
        "API REST para consultas de Derecho Laboral y Seguridad Social "
        "española basadas en el BOE mediante RAG.\n\n"
        "**Corpus**: Estatuto de los Trabajadores, LGSS, RDL 32/2021\n\n"
        "**Modelos**: BGE-M3-Legal-Spanish (embeddings) + "
        "Salamandra-2B / GPT-4o-mini (generación)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Desarrollo local — todas las variantes posibles
        "http://localhost:5500",     # ← AÑADIDO (python http.server)
        "http://127.0.0.1:5500",     # ← ya existía
        "http://localhost:3000",     # React dev server
        "http://localhost:5173",     # Vite dev server
        "http://localhost:8080",     # Alternativo
        "http://127.0.0.1:8080",
        "null",                      # ← file:// origin (abrir HTML directo)
    ],
    allow_credentials=False,         # ← False es más permisivo para desarrollo
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,                    # Cache del preflight 1 hora
)

# ─── Rutas ───────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ─── Handler de errores globales ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no controlado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Error interno del servidor",
            "detalle": str(exc),
            "codigo":  500,
        },
    )


# ─── Ruta raíz ───────────────────────────────────────────────────
@app.get("/", tags=["Sistema"])
async def root():
    return {
        "nombre":     "Asistente Jurídico-Laboral BOE",
        "version":    "1.0.0",
        "docs":       "/docs",
        "health":     "/api/v1/health",
        "consulta":   "/api/v1/consulta",
        "corpus":     "/api/v1/corpus/info",
    }