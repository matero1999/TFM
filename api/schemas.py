"""
Modelos Pydantic para request/response de la API.
Define la estructura contractual entre backend y frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── Request ────────────────────────────────────────────────────
class ConsultaRequest(BaseModel):
    """Cuerpo de una consulta al asistente."""
    pregunta: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Consulta jurídica del usuario",
        examples=["¿Cuál es la jornada máxima de trabajo semanal?"]
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Número de artículos a recuperar"
    )
    backend: Optional[str] = Field(
        default="salamandra",
        description="Modelo LLM a usar: 'salamandra' u 'openai'"
    )


# ─── Response ────────────────────────────────────────────────────
class FuenteResponse(BaseModel):
    """Metadatos de una fuente citada en la respuesta."""
    norma_id:      str
    titulo_bloque: str
    ley_titulo:    str
    url_boe:       str
    score_rrf:     float


class ConsultaResponse(BaseModel):
    """Respuesta completa del asistente jurídico."""
    pregunta:       str
    respuesta:      str
    fuentes:        list[FuenteResponse]
    tokens_entrada: int
    tokens_salida:  int
    coste_usd:      float
    latencia_ms:    int
    modelo:         str
    backend:        str
    timestamp:      str


class HealthResponse(BaseModel):
    """Estado del sistema."""
    status:          str
    version:         str
    backend_activo:  str
    corpus_chunks:   int
    timestamp:       str


class ErrorResponse(BaseModel):
    """Respuesta de error estructurada."""
    error:   str
    detalle: str
    codigo:  int