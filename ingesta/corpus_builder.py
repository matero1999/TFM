"""
Pipeline de construcción del corpus jurídico-laboral.
Orquesta: metadatos → índice → bloques → almacenamiento JSON.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from .boe_client import BOEClient
from .boe_parser import extraer_texto_bloque_xml

logger = logging.getLogger(__name__)


# ─── Seed list: normas laborales prioritarias ────────────────────
NORMAS_SEED = {
    "BOE-A-2015-11430": "Estatuto de los Trabajadores (ET)",
    "BOE-A-2015-11724": "Ley General de la Seguridad Social (LGSS)",
    "BOE-A-2021-21788": "RDL 32/2021 - Reforma Laboral",
    "BOE-A-1977-7840": "RDL 17/1977 - Derecho de Huelga y Conflictos Colectivos",   
}

# Tipos de bloque incluidos en el corpus RAG
TIPOS_INCLUIDOS = {"precepto", "parte_final", "preambulo"}

# Longitud mínima de texto para considerar un chunk válido
MIN_CHARS = 80


class CorpusBuilder:
    """
    Construye el corpus RAG desde la API del BOE.

    Estructura de salida:
        data/corpus/
            raw/        → JSON completo por norma
            processed/  → Un JSON por artículo (listo para embedding)
            index.json  → Índice global del corpus
    """

    def __init__(
        self,
        output_dir: str = "./data/corpus",
        delay:      float = 1.5,
    ):
        self.client     = BOEClient(delay=delay)
        self.output_dir = Path(output_dir)

        # Crear directorios si no existen
        (self.output_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(parents=True, exist_ok=True)

    # ─── Pipeline principal ──────────────────────────────────────
    def build(self, normas: dict = None) -> list[dict]:
        """
        Construye el corpus completo.

        Args:
            normas: dict {id_boe: descripcion}.
                    Si es None, usa NORMAS_SEED por defecto.

        Returns:
            Lista de todos los chunks generados.
        """
        normas = normas or NORMAS_SEED
        todos_chunks = []
        indice       = []

        for norma_id, descripcion in normas.items():
            logger.info(f"\n{'─'*60}")
            logger.info(f"Procesando: {descripcion}")
            logger.info(f"ID BOE:     {norma_id}")

            chunks = self._procesar_norma(norma_id, descripcion)

            if chunks:
                todos_chunks.extend(chunks)
                indice.append({
                    "id":          norma_id,
                    "descripcion": descripcion,
                    "chunks":      len(chunks),
                    "procesado":   datetime.utcnow().isoformat() + "Z",
                })
                logger.info(f"✅ {len(chunks)} chunks extraídos")
            else:
                logger.warning(f"⚠️  Sin chunks para {norma_id}")

        # Guardar índice global
        indice_path = self.output_dir / "index.json"
        with open(indice_path, "w", encoding="utf-8") as f:
            json.dump(indice, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Corpus completo: {len(todos_chunks)} chunks totales")
        logger.info(f"Índice guardado: {indice_path}")

        return todos_chunks

    # ─── Procesamiento de una norma ──────────────────────────────
    def _procesar_norma(
        self, norma_id: str, descripcion: str
    ) -> list[dict]:

        # 1. Metadatos
        meta = self.client.obtener_metadatos(norma_id)
        if not meta:
            logger.warning(f"Sin metadatos: {norma_id}")
            return []
        # ── CORRECCIÓN: Defensa extra por si acaso llega lista ───────
        if isinstance(meta, list):
            meta = meta[0] if meta else None
        if not isinstance(meta, dict):
            logger.error(f"Formato inesperado en metadatos de {norma_id}: {type(meta)}")
            return []

        # 2. Verificar que la norma está vigente
        if meta.get("vigencia_agotada") == "S":
            logger.info(f"Norma derogada, omitiendo: {norma_id}")
            return []

        # 3. Materias (enriquecimiento de metadatos)
        materias = self._extraer_materias(norma_id)

        # 4. Índice de bloques
        bloques_indice = self.client.obtener_indice_texto(norma_id)
        if not bloques_indice:
            logger.warning(f"Sin bloques en índice: {norma_id}")
            return []

        logger.info(f"Bloques encontrados: {len(bloques_indice)}")

        # 5. Extraer cada bloque
        chunks       = []
        raw_bloques  = []

        for i, bloque_info in enumerate(bloques_indice, start=1):
            bloque_id = (
                bloque_info.get("id")
                or bloque_info.get("bloque_id")
                or ""
            )
            if not bloque_id:
                continue

            logger.debug(f"  Bloque {i}/{len(bloques_indice)}: {bloque_id}")

            xml_raw = self.client.obtener_bloque_xml(norma_id, bloque_id)
            if not xml_raw:
                continue

            bloque_data = extraer_texto_bloque_xml(xml_raw)
            raw_bloques.append(bloque_data)

            # Filtrar por tipo y longitud mínima
            if bloque_data.get("tipo") not in TIPOS_INCLUIDOS:
                continue
            if len(bloque_data.get("texto", "")) < MIN_CHARS:
                continue

            chunk = self._construir_chunk(norma_id, meta, bloque_data, materias)
            chunks.append(chunk)

            # Guardar chunk individual en /processed
            chunk_path = (
                self.output_dir / "processed" / f"{chunk['chunk_id']}.json"
            )
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)

        # Guardar raw completo de la norma en /raw
        raw_path = self.output_dir / "raw" / f"{norma_id}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {"metadatos": meta, "bloques": raw_bloques},
                f, ensure_ascii=False, indent=2
            )

        return chunks

    # ─── Construcción del chunk final ────────────────────────────
    def _construir_chunk(
        self,
        norma_id:  str,
        meta:      dict,
        bloque:    dict,
        materias:  list[str],
    ) -> dict:
        """
        Construye el objeto chunk con todos los metadatos
        necesarios para RAG, citación y trazabilidad.
        """
        texto       = bloque.get("texto", "")
        titulo_blq  = bloque.get("titulo", "")
        bloque_id   = bloque.get("bloque_id", "")

        # ID reproducible del chunk
        chunk_id = (
            f"{norma_id}_{bloque_id}_"
            + hashlib.md5(f"{norma_id}{bloque_id}".encode()).hexdigest()[:8]
        )

        rango = ""
        if isinstance(meta.get("rango"), dict):
            rango = meta["rango"].get("texto", "")
        elif isinstance(meta.get("rango"), str):
            rango = meta["rango"]

        return {
            # ── Identificación ───────────────────────────────────
            "chunk_id":   chunk_id,
            "norma_id":   norma_id,
            "bloque_id":  bloque_id,

            # ── Contenido ────────────────────────────────────────
            "texto": texto,

            # Texto enriquecido para embedding
            # Incluye contexto legal → mejora calidad de recuperación
            "texto_embedding": (
                f"{rango} {meta.get('titulo', '')}. "
                f"{titulo_blq}. {texto}"
            ),

            # ── Metadatos legales ─────────────────────────────────
            "titulo_bloque":        titulo_blq,
            "tipo_bloque":          bloque.get("tipo", ""),
            "ley_titulo":           meta.get("titulo", ""),
            "ley_rango":            rango,
            "ley_numero_oficial":   meta.get("numero_oficial", ""),
            "departamento":         (
                meta.get("departamento", {}).get("texto", "")
                if isinstance(meta.get("departamento"), dict)
                else ""
            ),
            "fecha_disposicion":    meta.get("fecha_disposicion", ""),
            "fecha_publicacion_ley": meta.get("fecha_publicacion", ""),
            "fecha_vigencia":       meta.get("fecha_vigencia", ""),
            "vigencia_agotada":     meta.get("vigencia_agotada", "N"),
            "materias":             materias,
            "url_boe":              meta.get("url_html_consolidada", ""),
            "url_eli":              meta.get("url_eli", ""),

            # ── Versión del bloque ────────────────────────────────
            "norma_modificadora":   bloque.get("norma_modificadora", ""),
            "fecha_version":        bloque.get("fecha_publicacion", ""),

            # ── Trazabilidad ──────────────────────────────────────
            "fecha_cosecha": datetime.utcnow().isoformat() + "Z",
            "fuente":        "BOE_API_LEGISLACION_CONSOLIDADA",
        }

    # ─── Extracción de materias ──────────────────────────────────
    def _extraer_materias(self, norma_id: str) -> list[str]:
        """Extrae lista de materias temáticas de la norma."""
        analisis = self.client.obtener_analisis(norma_id)
        if not analisis:
            return []

        materias_raw = analisis.get("materias")
        if not materias_raw:
            return []

        if isinstance(materias_raw, dict):
            items = materias_raw.get("materia", [])
            if isinstance(items, dict):
                items = [items]
            return [m.get("texto", "") for m in items if m.get("texto")]

        return []