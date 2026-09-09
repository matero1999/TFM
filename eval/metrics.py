"""Módulo de evaluación RAGAS para el pipeline RAG Jurídico-Laboral (BOE).

Fijado a la API REAL de ragas==0.1.14 (confirmada por requirements.txt del
proyecto), SIN lógica de detección de versión "moderna/legada" — esa doble
vía fue la causa de errores encadenados difíciles de depurar (NameError
disfrazado, ImportError silencioso por incompatibilidad con langchain_community).

Dependencias exactas requeridas (requirements.txt del proyecto):
    ragas==0.1.14
    langchain-openai>=0.1.0
    langchain-community<0.3   # ← necesario: ragas 0.1.14 importa
                               #   incondicionalmente ChatVertexAI desde
                               #   langchain_community.chat_models.vertexai,
                               #   módulo retirado en langchain_community>=0.3
    datasets>=2.19.0
    python-dotenv>=1.0.0
    pandas==2.2.2

Métricas implementadas (API legada/funcional de ragas 0.1.x - objetos ya
instanciados en ragas.metrics, no clases a instanciar):
    1. faithfulness       - Fidelidad de la respuesta al contexto recuperado.
                             No requiere ground_truth. Escala [0, 1].
    2. answer_relevancy    - Pertinencia semántica de la respuesta a la pregunta.
                             No requiere ground_truth. Escala [0, 1].
    3. context_precision   - Precisión del contexto recuperado (posiciones
                             relevantes al principio). REQUIERE ground_truth
                             en esta versión de ragas. Escala [0, 1].
    4. context_recall      - Exhaustividad del contexto recuperado respecto a
                             la respuesta de referencia. REQUIERE ground_truth.
                             Escala [0, 1].

Referencia oficial: https://docs.ragas.io (branch/tag correspondiente a 0.1.x)
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Carga de variables de entorno desde .env (ubicado en la raíz del proyecto)
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Mapa nombre -> objeto métrica ya instanciado (API funcional de ragas 0.1.x)
_METRIC_REGISTRY: dict[str, Any] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

# Métricas que en ragas==0.1.14 requieren obligatoriamente 'ground_truth'
_REQUIRES_GROUND_TRUTH = {"context_precision", "context_recall"}


def _check_dependencies() -> None:
    """Verifica que las librerías necesarias estén disponibles.

    Con la versión fijada (ragas==0.1.14 + langchain-community<0.3), el
    propio bloque de imports de este módulo ya habría fallado de forma
    explícita en tiempo de carga si algo faltase. Esta función queda como
    punto único de comprobación adicional de la clave de API.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "No se encontró la variable de entorno 'OPENAI_API_KEY'. "
            "Defínela en tu archivo .env (debe ser una clave real de OpenAI, "
            "con prefijo 'sk-', NO una API key de AiKit con prefijo 'aik_')."
        )


def get_evaluator_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Instancia el LLM que actuará como 'juez evaluador' en RAGAS.

    En ragas==0.1.14 el objeto langchain (ChatOpenAI) se pasa DIRECTAMENTE a
    evaluate(llm=...) — esta versión envuelve internamente el modelo, por lo
    que NO debe usarse aquí LangchainLLMWrapper (esa clase pertenece a la API
    de ragas>=0.2, y no está expuesta de la misma forma en 0.1.14).

    Args:
        model_name: Identificador del modelo para el juez LLM.
        temperature: Temperatura de muestreo (0.0 para máxima reproducibilidad).

    Returns:
        Instancia de ChatOpenAI lista para ser consumida por evaluate().
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró la variable de entorno 'OPENAI_API_KEY'. "
            "Defínela en tu archivo .env para ejecutar la evaluación."
        )
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def get_evaluator_embeddings(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Instancia el modelo de embeddings usado por métricas semánticas (answer_relevancy).

    Igual que con el LLM, en ragas==0.1.14 se pasa el objeto langchain
    directamente a evaluate(embeddings=...), sin wrapper adicional.

    Args:
        model_name: Nombre del modelo de embeddings de OpenAI.

    Returns:
        Instancia de OpenAIEmbeddings.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró la variable de entorno 'OPENAI_API_KEY' para embeddings."
        )
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


def build_dataset(samples: list[dict[str, Any]]) -> Dataset:
    """Construye un datasets.Dataset (HuggingFace) compatible con ragas==0.1.14.

    Cada diccionario de entrada usa la nomenclatura moderna de campos
    (user_input/response/retrieved_contexts/reference) por consistencia con
    el resto del pipeline de evaluación del proyecto, pero aquí se traduce
    a las columnas EXACTAS que ragas==0.1.14 espera internamente:
        - 'question'      <- user_input
        - 'answer'        <- response
        - 'contexts'      <- retrieved_contexts (list[str])
        - 'ground_truth'  <- reference (opcional; requerido solo para
                             context_precision / context_recall)

    Args:
        samples: Lista de diccionarios con los resultados de las inferencias.
            Claves obligatorias: 'user_input', 'response', 'retrieved_contexts'.
            Clave opcional: 'reference'.

    Returns:
        datasets.Dataset con columnas question/answer/contexts/ground_truth.

    Raises:
        ValueError: Si la lista está vacía o falta alguna clave obligatoria.
        TypeError: Si 'retrieved_contexts' no es una lista.
    """
    if not samples:
        raise ValueError("La lista de 'samples' no puede estar vacía.")

    required_keys = {"user_input", "response", "retrieved_contexts"}
    for idx, s in enumerate(samples):
        missing = required_keys - set(s.keys())
        if missing:
            raise ValueError(
                f"La muestra en el índice {idx} carece de las claves obligatorias: {missing}"
            )
        if not isinstance(s["retrieved_contexts"], list):
            raise TypeError(
                f"En la muestra {idx}, 'retrieved_contexts' debe ser una lista de strings."
            )

    formatted_data = {
        "question": [str(s["user_input"]) for s in samples],
        "answer": [str(s["response"]) for s in samples],
        "contexts": [[str(c) for c in s["retrieved_contexts"]] for s in samples],
        "ground_truth": [str(s.get("reference") or "") for s in samples],
    }
    return Dataset.from_dict(formatted_data)


def compute_metrics(
    dataset: Dataset,
    metrics: list[str] | None = None,
    evaluator_llm_model: str = "gpt-4o-mini",
    evaluator_embed_model: str = "text-embedding-3-small",
    raise_exceptions: bool = False,
) -> dict[str, Any]:
    """Ejecuta el pipeline de evaluación RAGAS (API funcional de ragas==0.1.14).

    Si el dataset no contiene 'ground_truth' (columna vacía en todas las
    filas), las métricas 'context_precision' y 'context_recall' se excluyen
    automáticamente con un aviso, ya que en esta versión de ragas producen
    error si se les pide calcular sin referencia.

    Args:
        dataset: Instancia de datasets.Dataset generada por `build_dataset`.
        metrics: Lista opcional de nombres a calcular, de entre:
            'faithfulness', 'context_precision', 'context_recall',
            'answer_relevancy'. Si es None, se intentan todas las aplicables.
        evaluator_llm_model: Nombre del modelo LLM juez (OpenAI).
        evaluator_embed_model: Nombre del modelo de embeddings del juez (OpenAI).
        raise_exceptions: Si es True, propaga errores internos de ragas.evaluate().

    Returns:
        Diccionario con las claves de las métricas calculadas (float, media
        global) más la clave 'detail_df' (pandas.DataFrame con el detalle
        fila a fila).

    Raises:
        ValueError: Si no queda ninguna métrica activa tras los filtros.
        RuntimeError: Si ragas.evaluate() falla internamente.
    """
    _check_dependencies()

    evaluator_llm = get_evaluator_llm(evaluator_llm_model)
    evaluator_embeddings = get_evaluator_embeddings(evaluator_embed_model)

    has_reference = "ground_truth" in dataset.column_names and any(
        bool(gt and str(gt).strip()) for gt in dataset["ground_truth"]
    )

    all_available = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    selected_metrics = [m.lower().strip() for m in metrics] if metrics else list(all_available)

    if not has_reference:
        excluded = [m for m in selected_metrics if m in _REQUIRES_GROUND_TRUTH]
        if excluded:
            warnings.warn(
                f"No se detectó 'ground_truth' (reference) en el dataset. "
                f"Se omitirán las métricas dependientes de referencia: {excluded}.",
                UserWarning,
            )
            selected_metrics = [m for m in selected_metrics if m not in _REQUIRES_GROUND_TRUTH]

    active_metric_objects = [_METRIC_REGISTRY[m] for m in selected_metrics if m in _METRIC_REGISTRY]

    if not active_metric_objects:
        raise ValueError("No hay métricas activas seleccionadas para la evaluación.")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=active_metric_objects,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=raise_exceptions,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Error al ejecutar ragas.evaluate(). Verifique 'OPENAI_API_KEY' "
            f"(debe ser una clave real de OpenAI, prefijo 'sk-') y la conectividad "
            f"con la API. Detalle técnico: {exc}"
        ) from exc

    try:
        detail_df: pd.DataFrame = result.to_pandas()
    except Exception:
        detail_df = pd.DataFrame(result)

    summary_scores: dict[str, Any] = {}
    for metric_name in selected_metrics:
        matching_cols = [c for c in detail_df.columns if metric_name in c.lower().replace(" ", "_")]
        if matching_cols:
            summary_scores[metric_name] = float(detail_df[matching_cols[0]].dropna().mean())
        elif hasattr(result, "get") and result.get(metric_name) is not None:
            summary_scores[metric_name] = float(result[metric_name])

    summary_scores["detail_df"] = detail_df
    return summary_scores


# ---------------------------------------------------------------------------
# TEST Y EJEMPLO DE USO RÁPIDO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("EJECUCIÓN DE PRUEBA: Módulo de Métricas RAGAS (Asistente Jurídico BOE)")
    print("Fijado a ragas==0.1.14 (API funcional/legada)")
    print("=" * 80)

    dummy_samples = [
        {
            "user_input": "¿Cuántos días de preaviso son necesarios para una dimisión voluntaria?",
            "response": (
                "De acuerdo con el artículo 49 del Estatuto de los Trabajadores, el trabajador "
                "debe dar el preaviso que señalen los convenios colectivos o la costumbre del "
                "lugar, siendo habitualmente de 15 días si no se especifica otra cosa."
            ),
            "retrieved_contexts": [
                (
                    "Artículo 49. Extinción del contrato. 1. El contrato de trabajo se extinguirá: "
                    "d) Por dimisión del trabajador, debiendo mediar el preaviso que señalen los "
                    "convenios colectivos o la costumbre del lugar."
                ),
                (
                    "Jurisprudencia laboral consolidada: En defecto de pacto en convenio colectivo, "
                    "se considera prudencial y conforme a la costumbre un preaviso de 15 días naturales."
                ),
            ],
            "reference": (
                "El preaviso para la dimisión voluntaria será el establecido en el convenio "
                "colectivo aplicable o por la costumbre del lugar (habitualmente 15 días naturales), "
                "conforme al artículo 49.1.d del Estatuto de los Trabajadores."
            ),
        },
        {
            "user_input": "¿Cuál es la duración máxima de la jornada laboral ordinaria semanal?",
            "response": (
                "La duración máxima de la jornada ordinaria de trabajo es de 40 horas semanales "
                "de trabajo efectivo de promedio en cómputo anual, según el artículo 34 del Estatuto "
                "de los Trabajadores."
            ),
            "retrieved_contexts": [
                (
                    "Artículo 34. Jornada. 1. La duración de la jornada de trabajo será la pactada "
                    "en los convenios colectivos o contrato de trabajo. La duración máxima de la "
                    "jornada ordinaria de trabajo será de cuarenta horas semanales de trabajo efectivo "
                    "de promedio en cómputo anual."
                )
            ],
            "reference": (
                "La jornada máxima ordinaria de trabajo en España es de 40 horas semanales de trabajo "
                "efectivo de promedio en cómputo anual (art. 34.1 ET)."
            ),
        },
    ]

    print(f"\n1. Construyendo Dataset con {len(dummy_samples)} muestras...")
    try:
        eval_dataset = build_dataset(dummy_samples)
        print("   ✓ Dataset generado con éxito.")
    except Exception as e:
        print(f"   ✗ Error al construir dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n2. Verificando configuración de OpenAI API Key...")
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "   [AVISO] 'OPENAI_API_KEY' no está configurada en .env. "
            "Para ejecutar la llamada real al juez evaluador, añade tu clave en el archivo .env "
            "(debe ser una clave real de OpenAI, prefijo 'sk-').",
            file=sys.stderr,
        )
        print("   Fin de la comprobación sintáctica del módulo.")
        sys.exit(0)

    print("   ✓ API Key detectada. Ejecutando evaluación con RAGAS...")
    try:
        results = compute_metrics(
            dataset=eval_dataset,
            metrics=["faithfulness", "context_precision", "context_recall", "answer_relevancy"],
        )

        print("\n" + "=" * 80)
        print("RESULTADOS DE LA EVALUACIÓN (PROMEDIOS GLOBALES):")
        print("=" * 80)
        for metric_name, score in results.items():
            if metric_name != "detail_df":
                print(f"  • {metric_name.replace('_', ' ').title():<25}: {score:.4f}")

        print("\nDETALLE POR MUESTRA (DataFrame Pandas):")
        print(results["detail_df"][["question", "faithfulness", "answer_relevancy"]])
        print("=" * 80)

    except Exception as err:
        print(f"   ✗ Error durante la evaluación: {err}", file=sys.stderr)
        sys.exit(1)