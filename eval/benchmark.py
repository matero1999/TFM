"""
eval/benchmark.py — Script de benchmarking comparativo entre los backends
OpenAI y Salamandra del sistema RAG jurídico-laboral (Asistente BOE),
evaluado mediante las métricas del framework RAGAS (Faithfulness,
Context Precision, Context Recall, Answer Relevancy) y métricas operativas
(coste, latencia y consumo de tokens).

Proyecto: TFM - Asistente Jurídico-Laboral BOE

Se apoya en la clase real `RAGChain` de `llm/rag_chain.py`:

    from llm.rag_chain import RAGChain
    rag = RAGChain(backend="openai", top_k=5)
    resultado = rag.consultar("¿Cuál es la jornada máxima?")
    # resultado: {pregunta, respuesta, fuentes, chunks_usados, tokens_entrada,
    #             tokens_salida, coste_usd, latencia_ms, modelo, backend}

Dependencias requeridas (añadir a requirements.txt si no están ya):
    ragas, datasets, langchain-openai, python-dotenv, pandas
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuración del PATH para importaciones modulares del proyecto
# -----------------------------------------------------------------------------
# Permite ejecutar el script tanto desde la raíz del proyecto (`python eval/benchmark.py`)
# como desde dentro de la carpeta `eval/` (`python benchmark.py`).
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
DIRECTORIO_RAIZ = DIRECTORIO_ACTUAL.parent
if str(DIRECTORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_RAIZ))

# Importación de métricas desde el módulo hermano `eval/metrics.py`
try:
    from eval.metrics import build_dataset, compute_metrics
except ImportError:
    # Fallback si se ejecuta directamente dentro del directorio eval/
    from metrics import build_dataset, compute_metrics

# Importación de la cadena RAG real del proyecto (llm/rag_chain.py)
from llm.rag_chain import RAGChain


# =============================================================================
# EXTRACCIÓN DE TEXTO DE LOS CHUNKS RECUPERADOS
# =============================================================================
# CONFIRMADO: cada chunk de `corpus_chunks.json` contiene el campo "texto"
# (texto original del chunk) y "texto_embedding" (variante enriquecida con
# título de bloque y rango normativo, usada por BM25Retriever para indexar).
# Para RAGAS (retrieved_contexts) se prioriza "texto" -el contenido real y
# limpio del chunk-, cayendo a "texto_embedding" solo si "texto" viniera
# vacío, y como último recurso a metadatos (titulo_bloque + norma_id) para
# no interrumpir la ejecución ante un chunk incompleto.
CLAVES_TEXTO_CHUNK = ("texto", "texto_embedding")


def extraer_texto_chunk(chunk: Dict[str, Any]) -> str:
    """Extrae el texto real del chunk (campo "texto", con fallback a "texto_embedding")."""
    for clave in CLAVES_TEXTO_CHUNK:
        valor = chunk.get(clave)
        if valor:
            return str(valor)
    # Fallback de emergencia (chunk sin texto ni texto_embedding): no debería
    # ocurrir en un corpus_chunks.json bien formado, pero evita que el
    # benchmark se detenga por un registro incompleto.
    titulo = chunk.get("titulo_bloque", "")
    norma = chunk.get("norma_id", "")
    return f"{titulo} ({norma})".strip()


# =============================================================================
# DATASET DE EVALUACIÓN (PREGUNTAS DE DOMINIO JURÍDICO-LABORAL BOE)
# =============================================================================
# EDITAR: sustituir o ampliar con el dataset definitivo de evaluación del TFM.
#
# NOTA SOBRE 'reference':
# Cada pregunta incluye ahora una respuesta canónica (ground truth) redactada
# a partir del texto literal del Estatuto de los Trabajadores / LGSS, con el
# fin de habilitar el cálculo de `Context Recall` y `Context Precision` en
# RAGAS (ambas métricas requieren "ground_truth" / "reference" no vacío).
# Estas referencias NO han sido generadas por los LLM evaluados (OpenAI /
# Salamandra), para evitar sesgo circular en la evaluación.
# =============================================================================
DATASET_EVALUACION: List[Dict[str, Optional[str]]] = [
    {
        "user_input": "¿Cuántas horas semanales puede trabajar un empleado a tiempo completo?",
        "reference": (
            "La duración máxima de la jornada ordinaria de trabajo será de cuarenta "
            "horas semanales de trabajo efectivo de promedio en cómputo anual, según "
            "el artículo 34.1 del Estatuto de los Trabajadores. La duración concreta "
            "puede fijarse en convenio colectivo o contrato de trabajo, y puede "
            "distribuirse de forma irregular a lo largo del año conforme al "
            "artículo 34.2."
        ),
    },
    {
        "user_input": "¿Cuánto dura el período de prueba en un contrato indefinido?",
        "reference": (
            "En defecto de pacto en convenio colectivo, la duración del período de "
            "prueba no podrá exceder de seis meses para los técnicos titulados, ni "
            "de dos meses para los demás trabajadores. En las empresas de menos de "
            "veinticinco trabajadores, el período de prueba no podrá exceder de tres "
            "meses para los trabajadores que no sean técnicos titulados "
            "(artículo 14.1 del Estatuto de los Trabajadores). Será nulo el pacto de "
            "período de prueba si el trabajador ya desempeñó las mismas funciones "
            "anteriormente en la empresa."
        ),
    },
    {
        "user_input": "¿Qué causas justifican un despido disciplinario?",
        "reference": (
            "El despido disciplinario debe basarse en un incumplimiento grave y "
            "culpable del trabajador (artículo 54.1 ET). Se consideran "
            "incumplimientos contractuales: las faltas repetidas e injustificadas de "
            "asistencia o puntualidad; la indisciplina o desobediencia; las ofensas "
            "verbales o físicas al empresario o a otras personas de la empresa o sus "
            "familiares convivientes; la transgresión de la buena fe contractual o "
            "abuso de confianza; la disminución continuada y voluntaria del "
            "rendimiento de trabajo; la embriaguez habitual o toxicomanía que "
            "repercutan negativamente en el trabajo; y el acoso por razón de origen "
            "racial o étnico, religión o convicciones, discapacidad, edad u "
            "orientación sexual, así como el acoso sexual o por razón de sexo "
            "(artículo 54.2 ET)."
        ),
    },
    {
        "user_input": "¿Cuándo tiene derecho un trabajador a la prestación por incapacidad temporal?",
        "reference": (
            "Cuando el trabajador se encuentre en situación de incapacidad temporal "
            "derivada de contingencias comunes y durante la misma se extinga su "
            "contrato, seguirá percibiendo la prestación por incapacidad temporal en "
            "cuantía igual a la prestación por desempleo hasta que se extinga dicha "
            "situación, pasando entonces a la situación legal de desempleo si "
            "procede. Si la incapacidad temporal deriva de contingencias "
            "profesionales, seguirá percibiendo la prestación en la cuantía que "
            "tuviera reconocida hasta su extinción (artículo 283 de la Ley General "
            "de la Seguridad Social)."
        ),
    },
    {
        "user_input": "¿Cuánto cobra un trabajador en situación de desempleo?",
        "reference": (
            "La cuantía de la prestación por desempleo se determina aplicando a la "
            "base reguladora el 70 % durante los ciento ochenta primeros días y el "
            "50 % a partir del día ciento ochenta y uno. La cuantía máxima será del "
            "175 % del indicador público de rentas de efectos múltiples (IPREM), o "
            "del 200 % / 225 % si el trabajador tiene uno o más hijos a su cargo. La "
            "cuantía mínima será del 107 % o del 80 % del IPREM según tenga o no "
            "hijos a cargo (artículo 270 de la Ley General de la Seguridad Social)."
        ),
    },
    {
        "user_input": "¿Qué derechos tienen los trabajadores en caso de huelga?",
        "reference": (
            "Los trabajadores tienen reconocido como derecho básico el derecho a la "
            "huelga, con el contenido y alcance que establezca su normativa "
            "específica (artículo 4.1.e del Estatuto de los Trabajadores). Este "
            "derecho se enmarca junto con otros derechos básicos como la libre "
            "sindicación, la negociación colectiva y la adopción de medidas de "
            "conflicto colectivo, reconocidos en el mismo artículo 4.1 ET."
        ),
    },
]


# =============================================================================
# FUNCIONES DE EJECUCIÓN Y EVALUACIÓN
# =============================================================================

def ejecutar_backend(
    preguntas: List[Dict[str, Optional[str]]],
    backend: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ejecuta el pipeline RAG real (RAGChain) para un conjunto de preguntas bajo
    un backend específico ('openai' o 'salamandra'), midiendo métricas
    operativas y capturando errores por pregunta sin detener el benchmark.

    Args:
        preguntas: Lista de diccionarios con 'user_input' y opcionalmente 'reference'.
        backend: Backend LLM a utilizar ('openai' o 'salamandra').
        top_k: Número de chunks a recuperar por consulta (se pasa a RAGChain).

    Returns:
        Lista de diccionarios con la estructura requerida por
        `eval.metrics.build_dataset`, junto a las métricas operativas.
    """
    resultados: List[Dict[str, Any]] = []
    total_preguntas = len(preguntas)

    print(f"\n{'='*70}")
    print(f"Iniciando evaluación para backend: [{backend.upper()}] ({total_preguntas} consultas)")
    print(f"{'='*70}")

    # Instanciación única de RAGChain para todo el backend (evita recargar
    # el retriever y el cliente LLM en cada pregunta).
    rag = RAGChain(backend=backend, top_k=top_k)

    for idx, item in enumerate(preguntas, start=1):
        pregunta = item.get("user_input", "")
        referencia = item.get("reference", None)

        print(f"[{backend.upper()}] ({idx}/{total_preguntas}) Procesando: \"{pregunta[:65]}...\"")

        inicio_t = time.perf_counter()
        try:
            salida = rag.consultar(pregunta)
            fin_t = time.perf_counter()

            respuesta_texto = salida.get("respuesta", "")
            chunks_usados = salida.get("chunks_usados", []) or []
            contextos_recuperados = [extraer_texto_chunk(c) for c in chunks_usados]

            tokens_in = salida.get("tokens_entrada", 0) or 0
            tokens_out = salida.get("tokens_salida", 0) or 0
            coste = salida.get("coste_usd", 0.0) or 0.0
            latencia = salida.get("latencia_ms", (fin_t - inicio_t) * 1000.0)

            registro = {
                "user_input": pregunta,
                "response": respuesta_texto,
                "retrieved_contexts": contextos_recuperados,
                "reference": referencia,
                "modelo": salida.get("modelo", ""),
                "tokens_entrada": tokens_in,
                "tokens_salida": tokens_out,
                "tokens_totales": tokens_in + tokens_out,
                "coste_usd": coste,
                "latencia_ms": round(float(latencia), 2),
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": None,
                "ragas_context_precision": None,
                "ragas_context_recall": None,
                "error": None,
                "exito": True,
            }

        except Exception as exc:
            fin_t = time.perf_counter()
            error_msg = f"Error en la ejecución: {exc}"
            print(f"  --> [ERROR en pregunta {idx}]: {error_msg}")

            registro = {
                "user_input": pregunta,
                "response": "",
                "retrieved_contexts": [],
                "reference": referencia,
                "modelo": "",
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "tokens_totales": 0,
                "coste_usd": 0.0,
                "latencia_ms": round((fin_t - inicio_t) * 1000.0, 2),
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": None,
                "ragas_context_precision": None,
                "ragas_context_recall": None,
                "error": error_msg,
                "exito": False,
            }

        resultados.append(registro)

    return resultados


def ejecutar_benchmark_completo(
    backends: Optional[List[str]] = None,
    dataset: Optional[List[Dict[str, Optional[str]]]] = None,
    metricas_ragas: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Ejecuta el ciclo completo de benchmarking para todos los backends especificados:
    1. Inferencia y recuperación con RAGChain (pipeline real del proyecto).
    2. Construcción del dataset compatible con RAGAS.
    3. Cálculo de métricas de calidad RAGAS (Faithfulness, Context Precision, etc.).
    4. Agregación de métricas operativas (coste, latencia media, tokens).

    Args:
        backends: Backends a contrastar. Deben coincidir con
            `RAGChain.BACKENDS_DISPONIBLES` = ("salamandra", "openai").
        dataset: Lista de preguntas de prueba (por defecto `DATASET_EVALUACION`).
        metricas_ragas: Lista opcional de métricas a calcular con RAGAS.
        top_k: Chunks a recuperar por consulta.

    Returns:
        Diccionario estructurado con los resultados por backend.
    """
    if backends is None:
        backends = ["openai", "salamandra"]
    if dataset is None:
        dataset = DATASET_EVALUACION

    resultados_globales: Dict[str, Any] = {}

    for backend in backends:
        registros_backend = ejecutar_backend(preguntas=dataset, backend=backend, top_k=top_k)

        muestras_validas = [r for r in registros_backend if r["exito"] and r["response"]]

        ragas_scores: Dict[str, float] = {}
        if muestras_validas:
            print(f"\n[RAGAS] Evaluando métricas automáticas para backend '{backend}'...")
            try:
                ragas_dataset = build_dataset(muestras_validas)
                # Se fija explícitamente gpt-4o-mini como LLM juez de RAGAS porque los modelos
                # gpt-5.x/gpt-6 usados como backend generador (RAGChain) solo admiten temperature=1
                # y RAGAS necesita temperaturas bajas deterministas para el cálculo de varias métricas.
                # Al haber rellenado el campo "reference" en DATASET_EVALUACION, compute_metrics()
                # detectará "ground_truth" no vacío y calculará también context_precision y
                # context_recall (excluidas automáticamente cuando reference es None).
                ragas_scores = compute_metrics(
                    ragas_dataset,
                    metrics=metricas_ragas,
                    evaluator_llm_model="gpt-4o-mini",
                )

                # Incorporar scores individuales por pregunta al registro de detalle
                detail_df = ragas_scores.get("detail_df") if isinstance(ragas_scores, dict) else None
                if isinstance(detail_df, pd.DataFrame) and len(detail_df) == len(muestras_validas):
                    metricas_objetivo = (
                        metricas_ragas
                        if metricas_ragas is not None
                        else ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
                    )
                    cols_normalizadas = {
                        str(c).strip().lower().replace(" ", "_"): c for c in detail_df.columns
                    }
                    for m_name in metricas_objetivo:
                        clave_m = m_name.strip().lower().replace(" ", "_")
                        col_real = None
                        for c_norm, c_orig in cols_normalizadas.items():
                            if clave_m in c_norm:
                                col_real = c_orig
                                break
                        if col_real is not None:
                            for idx_m, reg in enumerate(muestras_validas):
                                val = detail_df[col_real].iloc[idx_m]
                                if pd.notna(val):
                                    try:
                                        reg[f"ragas_{clave_m}"] = round(float(val), 4)
                                    except (ValueError, TypeError):
                                        reg[f"ragas_{clave_m}"] = None
                                else:
                                    reg[f"ragas_{clave_m}"] = None
                elif detail_df is not None:
                    print(
                        f"[RAGAS] Aviso: No se pudo alinear detail_df ({len(detail_df)} filas) "
                        f"con muestras_validas ({len(muestras_validas)})."
                    )
            except Exception as e:
                print(f"[RAGAS] Advertencia: Falló el cálculo de métricas RAGAS para '{backend}': {e}")
                ragas_scores = {}
        else:
            print(f"[RAGAS] No hay muestras exitosas suficientes para evaluar '{backend}'.")

        total_consultas = len(registros_backend)
        exitosas = len(muestras_validas)
        latencias = [r["latencia_ms"] for r in registros_backend if r["exito"]]
        latencia_media = (sum(latencias) / len(latencias)) if latencias else 0.0
        coste_acumulado = sum(r["coste_usd"] for r in registros_backend)
        tokens_acumulados = sum(r["tokens_totales"] for r in registros_backend)

        metricas_operativas = {
            "total_consultas": total_consultas,
            "consultas_exitosas": exitosas,
            "tasa_exito_pct": round((exitosas / total_consultas) * 100.0, 2) if total_consultas > 0 else 0.0,
            "coste_total_usd": round(coste_acumulado, 5),
            "latencia_media_ms": round(latencia_media, 2),
            "tokens_totales": tokens_acumulados,
        }

        resultados_globales[backend] = {
            "detalle_preguntas": registros_backend,
            "metricas_ragas": ragas_scores,
            "metricas_operativas": metricas_operativas,
        }

    return resultados_globales


# =============================================================================
# GENERACIÓN DE TABLAS Y EXPORTACIÓN DE RESULTADOS
# =============================================================================

def generar_tabla_comparativa(resultados: Dict[str, Any]) -> pd.DataFrame:
    """
    Construye un DataFrame comparativo con las métricas clave de cada backend,
    a partir de los resultados REALES obtenidos en `ejecutar_benchmark_completo`.
    No contiene valores precalculados ni de ejemplo.
    """
    filas = []

    for backend, datos in resultados.items():
        ragas = datos.get("metricas_ragas", {}) or {}
        ops = datos.get("metricas_operativas", {}) or {}

        fila = {
            "Backend": backend.upper(),
            "Faithfulness": round(float(ragas["faithfulness"]), 4) if ragas.get("faithfulness") is not None else "N/A",
            "Context Precision": round(float(ragas["context_precision"]), 4) if ragas.get("context_precision") is not None else "N/A",
            "Context Recall": round(float(ragas["context_recall"]), 4) if ragas.get("context_recall") is not None else "N/A",
            "Answer Relevancy": round(float(ragas["answer_relevancy"]), 4) if ragas.get("answer_relevancy") is not None else "N/A",
            "Coste Total ($)": ops.get("coste_total_usd", 0.0),
            "Latencia Media (ms)": ops.get("latencia_media_ms", 0.0),
            "Tokens Totales": ops.get("tokens_totales", 0),
            "Tasa Éxito (%)": ops.get("tasa_exito_pct", 0.0),
        }
        filas.append(fila)

    return pd.DataFrame(filas)


def _json_default(obj: Any) -> Any:
    """
    Convierte de forma segura a tipos nativos de JSON cualquier objeto
    no serializable de forma nativa (DataFrame, Series, tipos numpy,
    Path, etc.), evitando que `json.dump` falle con TypeError.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    # Tipos numpy (float64, int64, bool_, ndarray...)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def exportar_resultados(
    resultados: Dict[str, Any],
    tabla_comparativa: pd.DataFrame,
    carpeta_salida: str = "eval/resultados",
) -> None:
    """
    Persiste los resultados REALES de la evaluación en CSV y JSON:
    - `comparativa_backends.csv` / `.json` (resumen agregado).
    - `detalle_{backend}.csv` / `.json` (pregunta por pregunta con contextos).
    """
    path_salida = Path(carpeta_salida)
    path_salida.mkdir(parents=True, exist_ok=True)

    ruta_csv_comp = path_salida / "comparativa_backends.csv"
    ruta_json_comp = path_salida / "comparativa_backends.json"

    tabla_comparativa.to_csv(ruta_csv_comp, index=False, encoding="utf-8")
    tabla_comparativa.to_json(ruta_json_comp, orient="records", indent=2, force_ascii=False)
    print("\n[EXPORTACIÓN] Resumen comparativo guardado en:")
    print(f"  • CSV:  {ruta_csv_comp}")
    print(f"  • JSON: {ruta_json_comp}")

    for backend, datos in resultados.items():
        detalle_preguntas = datos.get("detalle_preguntas", [])

        if isinstance(datos.get("metricas_ragas"), pd.DataFrame):
            datos["metricas_ragas"] = datos["metricas_ragas"].to_dict(orient="records")

        ruta_json_det = path_salida / f"detalle_{backend.lower()}.json"
        with open(ruta_json_det, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2, default=_json_default)

        ruta_csv_det = path_salida / f"detalle_{backend.lower()}.csv"
        df_detalle = pd.DataFrame(detalle_preguntas)

        if "retrieved_contexts" in df_detalle.columns:
            df_detalle["retrieved_contexts"] = df_detalle["retrieved_contexts"].apply(
                lambda ctxs: " |---CHUNK---| ".join(ctxs) if isinstance(ctxs, list) else str(ctxs)
            )

        df_detalle.to_csv(ruta_csv_det, index=False, encoding="utf-8")
        print(f"  • Detalle '{backend}': {ruta_csv_det} y {ruta_json_det}")


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    print("=" * 80)
    print("   BENCHMARK COMPARATIVO RAG: OPENAI vs SALAMANDRA (TFM ASISTENTE BOE)   ")
    print("=" * 80)
    print(f"Dataset de prueba cargado: {len(DATASET_EVALUACION)} consultas del ámbito jurídico-laboral.")
    print("Backends a evaluar: OpenAI, Salamandra")
    print("-" * 80)

    resultados_benchmark = ejecutar_benchmark_completo(
        backends=["openai", "salamandra"],
        dataset=DATASET_EVALUACION,
    )

    df_resumen = generar_tabla_comparativa(resultados_benchmark)

    exportar_resultados(
        resultados=resultados_benchmark,
        tabla_comparativa=df_resumen,
        carpeta_salida="eval/resultados",
    )

    print("\n" + "=" * 80)
    print("   TABLA COMPARATIVA FINAL DE RESULTADOS (RAGAS & RENDIMIENTO)   ")
    print("=" * 80)
    print(df_resumen.to_string(index=False))
    print("=" * 80)
    print("Benchmark finalizado satisfactoriamente. Resultados basados en ejecución real, sin datos de ejemplo.\n")