"""
Cliente OpenAI para el asistente jurídico-laboral.
Modelo: gpt-4o-mini (referencia SOTA del TFM).
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from .prompt_templates import (
    SYSTEM_PROMPT,
    formatear_contexto,
    construir_mensaje_usuario,
)

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Cliente GPT-4o-mini para generación de respuestas RAG.
    Interfaz diseñada para ser intercambiable con SalamandraClient.
    """

    def __init__(
        self,
        modelo:      str   = None,
        temperatura: float = None,
        max_tokens:  int   = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-..."):
            raise ValueError(
                "OPENAI_API_KEY no configurada en .env\n"
                "Añada su clave: OPENAI_API_KEY=sk-..."
            )

        self.client      = OpenAI(api_key=api_key)
        self.modelo      = modelo      or os.getenv("LLM_MODELO",      "gpt-4o-mini")
        self.temperatura = temperatura or float(os.getenv("LLM_TEMPERATURA", "0.1"))
        self.max_tokens  = max_tokens  or int(os.getenv("LLM_MAX_TOKENS",    "600"))

        logger.info(f"OpenAIClient listo: modelo={self.modelo}")

    def generar_respuesta(
        self,
        pregunta:     str,
        chunks:       list[dict],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> dict:
        """
        Genera una respuesta RAG con citas obligatorias.

        Args:
            pregunta:      Consulta del usuario
            chunks:        Lista de chunks recuperados del BOE
            system_prompt: Instrucciones del sistema

        Returns:
            {
                respuesta:      str,   # Texto de la respuesta con citas
                fuentes:        list,  # Metadatos de chunks usados
                tokens_entrada: int,
                tokens_salida:  int,
                coste_usd:      float,
                latencia_ms:    int,
                modelo:         str,
            }
        """
        contexto        = formatear_contexto(chunks)
        mensaje_usuario = construir_mensaje_usuario(pregunta, contexto)

        t0 = time.time()

        response = self.client.chat.completions.create(
            model=self.modelo,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": mensaje_usuario},
            ],
            temperature=self.temperatura,
            max_tokens=self.max_tokens,
        )

        latencia_ms = int((time.time() - t0) * 1000)
        respuesta   = response.choices[0].message.content.strip()

        # Calcular coste estimado (precios gpt-4o-mini a abril 2026)
        tokens_in  = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        coste_usd  = (tokens_in * 0.00000015) + (tokens_out * 0.0000006)

        # Preparar metadatos de fuentes para trazabilidad
        fuentes = [
            {
                "norma_id":      c.get("norma_id",      ""),
                "titulo_bloque": c.get("titulo_bloque", ""),
                "ley_titulo":    c.get("ley_titulo",    ""),
                "url_boe":       c.get("url_boe",       ""),
                "score_rrf":     c.get("score_rrf",     0.0),
            }
            for c in chunks
        ]

        logger.info(
            f"Respuesta generada | tokens={tokens_in}+{tokens_out} "
            f"| coste=${coste_usd:.5f} | latencia={latencia_ms}ms"
        )

        return {
            "respuesta":      respuesta,
            "fuentes":        fuentes,
            "tokens_entrada": tokens_in,
            "tokens_salida":  tokens_out,
            "coste_usd":      coste_usd,
            "latencia_ms":    latencia_ms,
            "modelo":         self.modelo,
        }