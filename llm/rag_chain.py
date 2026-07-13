"""
Cadena RAG completa: query → retrieval → generación → respuesta.
Soporta Salamandra (local) y OpenAI (API) de forma intercambiable.
"""

import os
import logging
from dotenv import load_dotenv

from retriever.hybrid_retriever import HybridRetriever
from .salamandra_client import SalamandraClient

load_dotenv()
logger = logging.getLogger(__name__)


class RAGChain:
    """
    Pipeline RAG completo con soporte multi-modelo.

    Flujo por consulta:
        1. Recibe pregunta del usuario
        2. HybridRetriever recupera top-k chunks relevantes
        3. LLM genera respuesta citando las fuentes
        4. Devuelve respuesta + fuentes + métricas de trazabilidad

    Uso:
        # Con Salamandra (local, coste cero)
        rag = RAGChain(backend="salamandra")

        # Con OpenAI (requiere clave API)
        rag = RAGChain(backend="openai")

        resultado = rag.consultar("¿Cuál es la jornada máxima?")
    """

    BACKENDS_DISPONIBLES = ("salamandra", "openai")

    def __init__(
        self,
        backend:    str  = "salamandra",
        top_k:      int  = 5,
        verbose:    bool = False,
    ):
        if backend not in self.BACKENDS_DISPONIBLES:
            raise ValueError(
                f"Backend '{backend}' no soportado. "
                f"Opciones: {self.BACKENDS_DISPONIBLES}"
            )

        self.backend = backend
        self.top_k   = top_k
        self.verbose = verbose

        logger.info(f"Inicializando RAGChain (backend={backend})...")

        # ── Inicializar retriever ─────────────────────────────────
        self.retriever = HybridRetriever(
            top_k_per_sistema=top_k * 4,
            top_k_final=top_k,
        )

        # ── Inicializar cliente LLM ───────────────────────────────
        if backend == "salamandra":
            self.llm = SalamandraClient()

        elif backend == "openai":
            # Importación diferida para no requerir openai si no se usa
            try:
                from .openai_client import OpenAIClient
                self.llm = OpenAIClient()
            except ImportError:
                raise ImportError(
                    "Instale openai: pip install openai"
                )
            except ValueError as e:
                raise ValueError(
                    f"Error configurando OpenAI: {e}\n"
                    "Asegúrese de tener OPENAI_API_KEY en el fichero .env"
                )

        logger.info(f"✅ RAGChain lista. Backend: {backend}")

    def consultar(self, pregunta: str) -> dict:
        """
        Procesa una consulta de principio a fin.

        Args:
            pregunta: Texto de la consulta del usuario

        Returns:
            {
                pregunta:       str,
                respuesta:      str,   # Con citas del BOE
                fuentes:        list,  # Metadatos de chunks usados
                chunks_usados:  list,  # Para trazabilidad completa
                tokens_entrada: int,
                tokens_salida:  int,
                coste_usd:      float, # 0.0 si Salamandra
                latencia_ms:    int,
                modelo:         str,
                backend:        str,
            }
        """
        if self.verbose:
            print(f"\n[RAG] Consulta recibida: {pregunta}")
            print(f"[RAG] Recuperando chunks (top_k={self.top_k})...")

        # ── 1. Retrieval ──────────────────────────────────────────
        chunks = self.retriever.buscar(pregunta, top_k=self.top_k)

        if self.verbose:
            print(f"[RAG] {len(chunks)} chunks recuperados:")
            for c in chunks:
                print(
                    f"  [{c.get('score_rrf', 0):.5f}] "
                    f"{c.get('titulo_bloque', '')[:50]} "
                    f"| {c.get('norma_id', '')}"
                )
            print("[RAG] Generando respuesta con LLM...")

        # ── 2. Generación ─────────────────────────────────────────
        resultado_llm = self.llm.generar_respuesta(
            pregunta=pregunta,
            chunks=chunks,
        )

        # ── 3. Resultado completo con trazabilidad ────────────────
        return {
            "pregunta":       pregunta,
            "respuesta":      resultado_llm["respuesta"],
            "fuentes":        resultado_llm["fuentes"],
            "chunks_usados":  chunks,
            "tokens_entrada": resultado_llm["tokens_entrada"],
            "tokens_salida":  resultado_llm["tokens_salida"],
            "coste_usd":      resultado_llm["coste_usd"],
            "latencia_ms":    resultado_llm["latencia_ms"],
            "modelo":         resultado_llm["modelo"],
            "backend":        self.backend,
        }