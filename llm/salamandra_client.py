"""
Cliente para BSC-LT/salamandra-7b-instruct (modelo local).

Modelo: BSC-LT/salamandra-7b-instruct
Params: 7.7B parámetros
Legal:  Pre-entrenado con Legal-ES (BOE, BORME, Senado, Congreso)
Idioma: Español upsampled 2x
Plantilla: ChatML (<|im_start|> / <|im_end|>)

Modos de carga (automático según hardware):
    - GPU ≥16GB VRAM → bfloat16 completo
    - GPU  ~8GB VRAM → 4-bit NF4 con bitsandbytes
    - Solo CPU       → float32 sin quantización
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ─── Constantes ─────────────────────────────────────────────────
# Modelo y configuración de carga
# MODEL_ID         = "BSC-LT/salamandra-7b-instruct"
# LOCAL_MODEL_PATH = "./models/salamandra-7b-instruct"

# VRAM_THRESHOLD_FULL  = 15.0   # GB → carga completa bfloat16
# VRAM_THRESHOLD_4BIT  =  7.0   # GB → quantización 4-bit viable

MODEL_ID         = "BSC-LT/salamandra-2b-instruct"
LOCAL_MODEL_PATH = "./models/salamandra-2b-instruct"

VRAM_THRESHOLD_FULL = 6.0   # GPU ≥6GB  → bfloat16 completo
VRAM_THRESHOLD_4BIT = 3.0   # GPU ≥3GB  → quantización 4-bit

MAX_NEW_TOKENS = 384
TEMPERATURE    = 0.0
REPETITION_PEN = 1.1

# ─── Prompt jurídico-laboral ─────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente especializado en Derecho Laboral \
y Seguridad Social español.

REGLA ABSOLUTA NÚMERO 1:
Responde ÚNICAMENTE con información que aparezca literalmente en los
artículos del BOE proporcionados como FUENTE. Si la información no
está en las fuentes, responde EXACTAMENTE esto y nada más:
"No dispongo de información suficiente en las fuentes disponibles
para responder esta consulta. Esta materia puede no estar cubierta
en el corpus actual."

REGLA ABSOLUTA NÚMERO 2:
NUNCA uses conocimiento propio sobre leyes, artículos o normas.
Todo lo que afirmes debe estar en el texto de las fuentes.

REGLA ABSOLUTA NÚMERO 3:
Cita SIEMPRE la fuente tras cada afirmación con este formato exacto:
[Nombre de la norma, Artículo X]

OTRAS NORMAS:
- NUNCA inventes artículos, fechas ni referencias legales.
- Si hay ambigüedad indica: "⚠️ Esta cuestión puede tener matices..."
- No uses: "generalmente", "probablemente", "suele ser".
- Termina siempre con: "Para aplicar esta norma a un caso concreto,
  consulte con un profesional del derecho."
- Ámbito exclusivo: España."""


# ─── Detección de hardware ───────────────────────────────────────
def detectar_hardware() -> dict:
    """
    Detecta el hardware disponible y determina el modo de carga.

    Returns:
        {
            dispositivo:   "cuda" | "cpu"
            vram_gb:       float  (0.0 si no hay GPU)
            modo:          "bfloat16" | "4bit" | "cpu"
            usar_4bit:     bool
        }
    """
    if not torch.cuda.is_available():
        logger.info("GPU no disponible. Modo: CPU (inferencia lenta)")
        return {
            "dispositivo": "cpu",
            "vram_gb":     0.0,
            "modo":        "cpu",
            "usar_4bit":   False,
        }

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb    = vram_bytes / 1e9
    gpu_nombre = torch.cuda.get_device_name(0)

    logger.info(f"GPU detectada: {gpu_nombre} ({vram_gb:.1f} GB VRAM)")

    if vram_gb >= VRAM_THRESHOLD_FULL:
        modo     = "bfloat16"
        usar_4bit = False
    else:
        modo      = "4bit"
        usar_4bit  = True

    logger.info(f"Modo de carga seleccionado: {modo}")

    return {
        "dispositivo": "cuda",
        "vram_gb":     vram_gb,
        "modo":        modo,
        "usar_4bit":   usar_4bit,
    }


# ─── Cliente Salamandra ──────────────────────────────────────────
class SalamandraClient:
    """
    Cliente para inferencia local con Salamandra-7B-Instruct.

    Detecta automáticamente el hardware disponible y selecciona
    el modo de carga óptimo. Interfaz compatible con OpenAIClient
    para permitir comparativa transparente en el TFM.
    """

    def __init__(
        self,
        model_path:     str   = LOCAL_MODEL_PATH,
        model_id:       str   = MODEL_ID,
        max_new_tokens: int   = MAX_NEW_TOKENS,
        temperature:    float = TEMPERATURE,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        # Detectar hardware
        hw = detectar_hardware()
        self.dispositivo = hw["dispositivo"]
        self.modo        = hw["modo"]

        # Determinar fuente del modelo
        source = model_path if Path(model_path).exists() else model_id
        if Path(model_path).exists():
            logger.info(f"Cargando desde ruta local: {model_path}")
        else:
            logger.info(f"Ruta local no encontrada. "
                        f"Descargando desde HuggingFace: {model_id}")

        # Cargar tokenizador
        logger.info("Cargando tokenizador...")
        self.tokenizer = AutoTokenizer.from_pretrained(source)

        # Cargar modelo según hardware disponible
        self.model = self._cargar_modelo(source, hw)

        logger.info("✅ Salamandra-7B cargado y listo.")

    def _cargar_modelo(self, source: str, hw: dict):
        """Carga el modelo con la configuración óptima para el hardware."""

        # ── Modo CPU ─────────────────────────────────────────────────
        if hw["modo"] == "cpu":
            logger.warning(
                "Cargando en CPU. La inferencia será lenta "
                "(2-5 minutos por respuesta). "
                "Considere usar la API de OpenAI si necesita velocidad."
            )
            return AutoModelForCausalLM.from_pretrained(
                source,
                dtype=torch.float32,   # ✅ float32 correcto en CPU
                device_map="cpu",      # ✅ CPU correcto
            )

        # ── Modo GPU 4-bit ────────────────────────────────────────────
        if hw["modo"] == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                logger.info(
                    f"Cargando en 4-bit NF4 "
                    f"(VRAM disponible: {hw['vram_gb']:.1f} GB)"
                )

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                return AutoModelForCausalLM.from_pretrained(
                    source,
                    quantization_config=quant_config,  # ✅ se pasa la config
                    device_map="auto",                 # ✅ GPU automático
                    # No se especifica dtype: lo gestiona BitsAndBytesConfig
                )

            except ImportError:
                logger.warning(
                    "bitsandbytes no disponible. "
                    "Intentando carga en bfloat16..."
                )
                # Fallback a bfloat16 en GPU
                return AutoModelForCausalLM.from_pretrained(
                    source,
                    dtype=torch.bfloat16,   # ✅ bfloat16 en GPU
                    device_map="auto",      # ✅ GPU automático
                )

        # ── Modo GPU bfloat16 completo ────────────────────────────────
        logger.info(
            f"Cargando en bfloat16 completo "
            f"(VRAM disponible: {hw['vram_gb']:.1f} GB)"
        )
        return AutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.bfloat16,   # ✅ bfloat16 en GPU
            device_map="auto",      # ✅ GPU automático
        )

    def generar_respuesta(
        self,
        pregunta:      str,
        chunks:        list[dict],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> dict:
        """
        Genera una respuesta RAG con Salamandra.

        Args:
            pregunta:      Consulta del usuario
            chunks:        Lista de chunks recuperados del BOE
            system_prompt: Instrucciones del sistema

        Returns:
            {
                respuesta:      str,
                fuentes:        list[dict],
                tokens_entrada: int,
                tokens_salida:  int,
                latencia_ms:    int,
                modelo:         str,
                coste_usd:      float,  # Siempre 0.0 (modelo local)
            }
        """
        # Formatear contexto con los chunks recuperados (truncado a 400 chars)
        contexto = self._formatear_contexto(chunks)

        # Construir mensajes en formato ChatML
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"ARTÍCULOS DEL BOE DISPONIBLES COMO FUENTE:\n"
                    f"{'═' * 60}\n"
                    f"{contexto}\n"
                    f"{'═' * 60}\n\n"
                    f"CONSULTA: {pregunta}"
                ),
            },
        ]

        # Aplicar plantilla ChatML de Salamandra
        fecha  = datetime.today().strftime("%Y-%m-%d")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            date_string=fecha,
        )

        # ── CAMBIO 1: Tokenizar con attention_mask explícita ─────────
        # Corrige el warning: "attention mask not set"
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs         = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Mover al dispositivo correcto
        if self.dispositivo == "cuda":
            inputs         = inputs.to("cuda")
            attention_mask = attention_mask.to("cuda")

        n_tokens_entrada = inputs.shape[1]

        logger.info(
            f"Generando respuesta "
            f"({n_tokens_entrada} tokens entrada, "
            f"modo={self.modo})..."
        )

        t0 = time.time()

        # ── CAMBIO 2: Pasar attention_mask y corregir temperature ────
        # do_sample=False con temperature=0.0 → greedy decoding
        # (más rápido y más determinista en CPU)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                repetition_penalty=REPETITION_PEN,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        latencia_ms = int((time.time() - t0) * 1000)

        # Extraer solo los tokens generados (excluir prompt)
        generated_ids = outputs[0][n_tokens_entrada:]
        respuesta     = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        n_tokens_salida = len(generated_ids)

        logger.info(
            f"Respuesta generada | "
            f"tokens={n_tokens_entrada}+{n_tokens_salida} | "
            f"latencia={latencia_ms}ms"
        )

        # Preparar metadatos de fuentes
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

        return {
            "respuesta":      respuesta,
            "fuentes":        fuentes,
            "tokens_entrada": n_tokens_entrada,
            "tokens_salida":  n_tokens_salida,
            "latencia_ms":    latencia_ms,
            # ── CAMBIO 3: Nombre correcto del modelo ─────────────────
            "modelo":         f"salamandra-2b-instruct ({self.modo})",
            "coste_usd":      0.0,
        }

    @staticmethod
    def _formatear_contexto(
    chunks:         list[dict],
    max_chars_chunk: int = 400,    # ← Truncar cada chunk a 400 chars
    ) -> str:
        """
        Formatea los chunks como contexto estructurado para el LLM.
        Trunca cada chunk para mantener el contexto total manejable.

        Con 5 chunks × 400 chars ≈ 500 tokens de contexto
        vs los 6500 tokens anteriores → 13x más rápido en CPU.
        """
        if not chunks:
            return "No se han encontrado artículos relevantes en el corpus."

        lineas = []
        for i, chunk in enumerate(chunks, start=1):
            norma  = chunk.get("ley_titulo",    "Norma desconocida")
            titulo = chunk.get("titulo_bloque", "Sin título")
            url    = chunk.get("url_boe",       "")

            # Truncar texto del chunk para reducir tokens
            texto_completo = chunk.get("texto", "").strip()
            if len(texto_completo) > max_chars_chunk:
                texto = texto_completo[:max_chars_chunk] + "..."
            else:
                texto = texto_completo

            lineas.append(
                f"[FUENTE {i}]\n"
                f"Norma:    {norma[:80]}\n"
                f"Artículo: {titulo}\n"
                f"URL BOE:  {url}\n"
                f"Texto:    {texto}\n"
            )

        return ("\n" + "─" * 40 + "\n").join(lineas)