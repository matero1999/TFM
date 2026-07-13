"""
Cliente Python para la API oficial del BOE (Legislación Consolidada).
Documentación oficial: https://www.boe.es/datosabiertos/api/api.php

Sin autenticación requerida. Rate limit cortés: 1.5s entre peticiones.
"""

import json
import time
import logging
from typing import Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

BASE_URL         = "https://boe.es/datosabiertos/api"
DEFAULT_HEADERS  = {"Accept": "application/json"}
RATE_LIMIT_DELAY = 1.5  # segundos entre peticiones


@dataclass
class BloqueTexto:
    """Unidad mínima del corpus: un artículo o sección legal."""
    norma_id:           str
    bloque_id:          str
    tipo:               str
    titulo:             str
    texto:              str
    fecha_publicacion:  str
    norma_modificadora: str
    url_boe:            str
    ley_titulo:         str
    ley_rango:          str


class BOEClient:
    """
    Cliente para la API de Legislación Consolidada del BOE.
    Implementa reintentos automáticos y control de errores.
    """

    def __init__(self, delay: float = RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay   = delay

    # ─── Petición base ──────────────────────────────────────────
    def _get(
        self,
        endpoint:   str,
        params:     dict = None,
        accept_xml: bool = False,
    ) -> dict | str | None:
        """GET con reintentos y backoff exponencial."""
        url     = f"{BASE_URL}{endpoint}"
        headers = {"Accept": "application/xml"} if accept_xml else {}

        for intento in range(3):
            try:
                resp = self.session.get(
                    url, params=params, headers=headers, timeout=30
                )
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp.text if accept_xml else resp.json()

            except requests.HTTPError as e:
                codigo = e.response.status_code
                if codigo == 404:
                    logger.warning(f"[404] No encontrado: {url}")
                    return None
                if codigo == 429:
                    espera = 60 * (intento + 1)
                    logger.warning(f"[429] Rate limit. Esperando {espera}s...")
                    time.sleep(espera)
                else:
                    logger.error(f"[{codigo}] Error HTTP en {url}")
                    raise

            except requests.RequestException as e:
                logger.error(f"Error de red (intento {intento + 1}/3): {e}")
                time.sleep(5 * (intento + 1))

        logger.error(f"Todos los reintentos fallaron para: {url}")
        return None

    # ─── Metadatos de una norma ──────────────────────────────────
    def obtener_metadatos(self, norma_id: str) -> Optional[dict]:
        """Metadatos completos de una norma por su identificador BOE."""
        data = self._get(f"/legislacion-consolidada/id/{norma_id}/metadatos")
        if not data:
            return None
        if data.get("status", {}).get("code") != "200":
            logger.warning(f"Respuesta no 200 para metadatos de {norma_id}")
            return None

        resultado = data.get("data")

        # ── CORRECCIÓN: La API puede devolver lista o dict ───────────
        if isinstance(resultado, list):
            return resultado[0] if resultado else None

        return resultado if isinstance(resultado, dict) else None

    # ─── Análisis (referencias cruzadas y materias) ───────────────
    def obtener_analisis(self, norma_id: str) -> Optional[dict]:
        """Referencias cruzadas, materias y notas de la norma."""
        data = self._get(f"/legislacion-consolidada/id/{norma_id}/analisis")
        if not data:
            return None
        if data.get("status", {}).get("code") != "200":
            return None

        resultado = data.get("data")

        # ── CORRECCIÓN: Misma defensa contra lista ───────────────────
        if isinstance(resultado, list):
            return resultado[0] if resultado else None

        return resultado if isinstance(resultado, dict) else None


    def obtener_indice_texto(self, norma_id: str) -> list[dict]:
        """Lista todos los bloques (artículos, secciones) de una norma."""
        data = self._get(
            f"/legislacion-consolidada/id/{norma_id}/texto/indice"
        )
        if not data:
            return []
        if data.get("status", {}).get("code") != "200":
            return []

        contenido = data.get("data", {})

        # ── CORRECCIÓN: data puede ser lista o dict ──────────────────
        if isinstance(contenido, list):
            contenido = contenido[0] if contenido else {}

        bloques = contenido.get("bloque", [])

        # La API devuelve dict si solo hay 1 bloque
        if isinstance(bloques, dict):
            bloques = [bloques]

        return bloques if isinstance(bloques, list) else []

    # ─── Texto de un bloque concreto ─────────────────────────────
    def obtener_bloque_xml(
        self, norma_id: str, bloque_id: str
    ) -> Optional[str]:
        """
        XML de un artículo/bloque concreto.
        Devuelve string XML raw para parsear con boe_parser.py
        """
        return self._get(
            f"/legislacion-consolidada/id/{norma_id}/texto/bloque/{bloque_id}",
            accept_xml=True,
        )