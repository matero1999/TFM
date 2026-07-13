"""
Gestor de embeddings jurídicos con BGE-M3-Legal-Spanish.

Modelo seleccionado: wilfredomartel/BGE-M3-Legal-Spanish
Base:   BAAI/bge-m3 (XLMRoberta)
Dims:   1024 (o 768 con Matryoshka truncation)
Tokens: 8192 máx — permite artículos legales completos
NDCG:   0.9002 en eval jurídico español
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─── Constantes ─────────────────────────────────────────────────
MODEL_ID     = "wilfredomartel/BGE-M3-Legal-Spanish"
EMBED_DIM    = 1024          # Dimensión completa
EMBED_DIM_S  = 768           # Matryoshka reduced (si hay memory constraints)
MAX_TOKENS   = 8192          # Ventana máxima del modelo
BATCH_SIZE_CPU = 4
BATCH_SIZE_GPU = 16
QUERY_INSTRUCTION = (
    "Representa esta consulta jurídica para buscar "
    "artículos legales relevantes en español: "
)


class EmbeddingManager:
    """
    Gestor centralizado de embeddings para el corpus jurídico.

    Características clave del modelo seleccionado:
    - Fine-tuned en 600k pares Q-A jurídicos en español
    - Distingue entre encode_query() y encode_document()
      (asimétrico: óptimo para RAG)
    - Matryoshka Loss: permite truncar a 768 dims sin reentrenar
    - Compatible con pgvector (cosine similarity)
    """

    def __init__(
    self,
    model_id:   str = MODEL_ID,
    model_path: Optional[str] = None,
    device:     Optional[str] = None,
):
        self.truncate_dim = None  # Por ahora no aplicamos truncation, pero se puede activar si hay limitaciones de memoria
        # Determinar dispositivo
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Batch size según dispositivo
        self.batch_size = (
            BATCH_SIZE_GPU if self.device == "cuda" else BATCH_SIZE_CPU
        )

        # Si se proporciona ruta local y existe → cargar desde disco
        # Si no → descargar desde HuggingFace
        if model_path and Path(model_path).exists():
            source = model_path
            logger.info(f"Cargando modelo desde ruta local: {model_path}")
        else:
            source = model_id
            logger.info(f"Cargando modelo desde HuggingFace: {model_id}")

        logger.info(f"Dispositivo: {self.device} | Batch size: {self.batch_size}")

        self.model = SentenceTransformer(source, device=self.device)

        logger.info(
            f"Modelo cargado. "
            f"Dimensión: {self.model.get_sentence_embedding_dimension()}"
        )
    # ─── Encoding de corpus (documentos) ────────────────────────
    def encode_corpus(
        self,
        textos:        list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Codifica los textos del corpus para indexación.

        Args:
            textos:        Lista de texto_embedding de cada chunk
            show_progress: Mostrar barra de progreso

        Returns:
            Array numpy (N, 1024) normalizado L2
            (normalizado → similitud coseno = producto escalar)
        """
        logger.info(f"Generando embeddings para {len(textos)} chunks...")

        embeddings = self.model.encode(
            textos,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        logger.info(f"Embeddings generados: shape={embeddings.shape}")
        return embeddings

    # ─── Encoding de query (consulta de usuario) ─────────────────
    def encode_query(self, query: str) -> np.ndarray:
        """
        Codifica una consulta de usuario para búsqueda.

        Aplica instrucción de query BGE-M3 para mejorar
        la precisión en recuperación asimétrica doc↔query.

        Returns:
            Array numpy (1024,) normalizado L2
        """
        texto_query = f"{QUERY_INSTRUCTION}{query}"

        embedding = self.model.encode(
            [texto_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding[0]  # shape: (1024,)

    # ─── Persistencia ─────────────────────────────────────────────
    def save(self, embeddings: np.ndarray, path: str) -> None:
        """Guarda embeddings en disco en formato numpy."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        logger.info(f"Embeddings guardados: {embeddings.shape} → {path}")

    @staticmethod
    def load(path: str) -> np.ndarray:
        """Carga embeddings desde disco."""
        embeddings = np.load(path)
        logger.info(f"Embeddings cargados: shape={embeddings.shape}")
        return embeddings

    @property
    def dimension(self) -> int:
        """Dimensión efectiva del espacio de embeddings."""
        if self.truncate_dim:
            return self.truncate_dim
        return self.model.get_sentence_embedding_dimension()

    # def encode_query(self, query: str) -> np.ndarray:
    #     """
    #     Codifica una consulta de usuario.

    #     IMPORTANTE: BGE-M3 usa codificación ASIMÉTRICA.
    #     Las queries se procesan distinto a los documentos.
    #     Usar siempre encode_query() para preguntas, nunca encode().
    #     """
    #     embedding = self.model.encode_query(
    #         [query],
    #         show_progress_bar=False,
    #         convert_to_numpy=True,
    #     )
    #     return embedding[0]  # shape: (1024,)

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 0,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Codifica una lista de documentos (chunks del corpus).

        Usar para indexación del corpus, NO para queries de usuario.
        """
        embeddings = self.model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings  # shape: (N, 1024)

    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula similitud coseno entre query y documentos.
        El modelo aplica normalización L2 en output → cosine = dot product.
        """
        # Los embeddings ya están normalizados (normalize=True en el modelo)
        return np.dot(doc_embeddings, query_embedding)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: str = "./data/embeddings/corpus_embeddings.npy"
    ) -> None:
        """Persiste embeddings en disco para reutilización."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        logger.info(f"Embeddings guardados: {embeddings.shape} → {path}")

    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        """Carga embeddings previamente generados."""
        embeddings = np.load(path)
        logger.info(f"Embeddings cargados: {embeddings.shape}")
        return embeddings