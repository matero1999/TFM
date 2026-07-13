"""
Plantillas de prompt para el asistente jurídico-laboral.

Principios de diseño:
- Obligar a citar fuente en cada afirmación
- Prohibir explícitamente inventar artículos o fechas
- Forzar declaración de incertidumbre cuando no hay fuente
- Limitar scope a España y al corpus disponible
"""

# ─── Prompt de sistema principal ────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente especializado en Derecho Laboral \
y Seguridad Social español. Tu función es responder consultas \
basándote EXCLUSIVAMENTE en los artículos del BOE que se te \
proporcionan como contexto.

NORMAS OBLIGATORIAS:
1. CITA SIEMPRE la fuente tras cada afirmación usando el formato:
   [Nombre de la norma, Artículo X]
2. Si la información NO está en el contexto proporcionado, responde
   exactamente: "No dispongo de información suficiente en las fuentes
   disponibles para responder esta consulta."
3. NUNCA inventes artículos, números de BOE, fechas ni referencias.
4. Si existe ambigüedad o la norma admite varias interpretaciones,
   indícalo con: "⚠️ Esta cuestión puede tener matices..."
5. No uses expresiones vagas: "generalmente", "probablemente",
   "suele ser". Cíñete al texto de los artículos.
6. Indica siempre al final: "Para aplicar esta norma a un caso
   concreto, consulte con un profesional del derecho."
7. El ámbito es exclusivamente España. No apliques normativa
   de otras jurisdicciones.

FORMATO DE RESPUESTA:
- Respuesta directa y estructurada.
- Citas inline tras cada afirmación: [Norma, Art. X].
- Nota de advertencia si procede.
- Aviso de consulta profesional al final."""


# ─── Template de consulta con contexto ──────────────────────────
def formatear_contexto(chunks: list[dict]) -> str:
    """
    Formatea los chunks recuperados como contexto para el LLM.
    Incluye: número de fuente, nombre de norma, artículo y texto.
    """
    if not chunks:
        return "No se han encontrado artículos relevantes en el corpus."

    lineas = []
    for i, chunk in enumerate(chunks, start=1):
        norma   = chunk.get("ley_titulo",    "Norma desconocida")
        titulo  = chunk.get("titulo_bloque", "Sin título")
        texto   = chunk.get("texto",         "").strip()
        url     = chunk.get("url_boe",       "")

        lineas.append(
            f"[FUENTE {i}]\n"
            f"Norma:    {norma}\n"
            f"Artículo: {titulo}\n"
            f"URL BOE:  {url}\n"
            f"Texto:\n{texto}\n"
        )

    return "\n" + ("─" * 60 + "\n").join(lineas)


def construir_mensaje_usuario(
    pregunta: str,
    contexto: str,
) -> str:
    """Construye el mensaje completo de usuario con contexto."""
    return (
        f"ARTÍCULOS DEL BOE DISPONIBLES COMO FUENTE:\n"
        f"{'═' * 60}\n"
        f"{contexto}\n"
        f"{'═' * 60}\n\n"
        f"CONSULTA: {pregunta}"
    )