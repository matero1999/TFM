"""
Parser XML de bloques del BOE.
Extrae texto limpio con estructura jurídica preservada.
"""

import logging
import re
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


def limpiar_texto(texto: str) -> str:
    """
    Limpia espacios, saltos de línea múltiples y
    caracteres de control del texto extraído del XML.
    """
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.strip()
    return texto


def extraer_texto_bloque_xml(xml_raw: str) -> dict:
    """
    Parsea el XML de un bloque consolidado del BOE.

    Estructura XML que procesa:
        <bloque id="a34" tipo="precepto" titulo="Artículo 34">
            <version id_norma="..." fecha_publicacion="..." fecha_vigencia="...">
                <p class="articulo">Artículo 34. Jornada de trabajo.</p>
                <p class="parrafo">1. La duración máxima...</p>
                <blockquote>
                    <p class="nota_pie">Se modifica el apartado...</p>
                </blockquote>
            </version>
        </bloque>

    Devuelve:
        {
            bloque_id, tipo, titulo, texto,
            norma_modificadora, fecha_publicacion,
            fecha_vigencia, versiones, notas_pie
        }
    """
    resultado = {
        "bloque_id":          "",
        "tipo":               "",
        "titulo":             "",
        "texto":              "",
        "norma_modificadora": "",
        "fecha_publicacion":  "",
        "fecha_vigencia":     "",
        "versiones":          [],
        "notas_pie":          [],
    }

    try:
        root    = ET.fromstring(xml_raw)
        bloque  = root.find(".//bloque")
        if bloque is None:
            logger.warning("No se encontró elemento <bloque> en el XML")
            return resultado

        resultado["bloque_id"] = bloque.get("id", "")
        resultado["tipo"]      = bloque.get("tipo", "")
        resultado["titulo"]    = bloque.get("titulo", "")

        versiones = bloque.findall("version")
        if not versiones:
            return resultado

        # La versión más reciente es la primera en el XML
        version_vigente = versiones[0]
        resultado["norma_modificadora"] = version_vigente.get("id_norma", "")
        resultado["fecha_publicacion"]  = version_vigente.get("fecha_publicacion", "")
        resultado["fecha_vigencia"]     = version_vigente.get("fecha_vigencia", "")

        # Extraer párrafos de la versión vigente (excluir blockquotes = notas de pie)
        parrafos = []
        for elemento in version_vigente:
            if elemento.tag == "p":
                clase = elemento.get("class", "")
                # Excluir clases puramente estructurales
                if clase not in ("titulo",):
                    texto_p = limpiar_texto("".join(elemento.itertext()))
                    if texto_p:
                        parrafos.append(texto_p)

            elif elemento.tag == "blockquote":
                # Guardar notas de pie (modificaciones históricas)
                nota_texto = limpiar_texto("".join(elemento.itertext()))
                if nota_texto:
                    resultado["notas_pie"].append({
                        "norma":  version_vigente.get("id_norma", ""),
                        "nota":   nota_texto,
                    })

        resultado["texto"] = "\n".join(parrafos)

        # Historial de versiones para trazabilidad
        resultado["versiones"] = [
            {
                "id_norma":          v.get("id_norma", ""),
                "fecha_publicacion": v.get("fecha_publicacion", ""),
                "fecha_vigencia":    v.get("fecha_vigencia", ""),
            }
            for v in versiones
        ]

    except ET.ParseError as e:
        logger.error(f"Error parseando XML: {e}")

    return resultado