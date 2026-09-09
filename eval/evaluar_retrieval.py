"""
eval/evaluar_retrieval.py — Script de evaluación intrínseca del componente de
recuperación híbrida (HybridRetriever con VectorStore/BGE-M3-Legal-Spanish + BM25Retriever
+ Reciprocal Rank Fusion RRF) mediante métricas de Information Retrieval clásicas
(Precision@1 y Recall@5).

Proyecto: TFM - Asistente Jurídico-Laboral BOE

Clases del proyecto utilizadas:
    - retriever.hybrid_retriever.HybridRetriever (atributos: .vector_store, .bm25, .embedder)
    - retriever.bm25_retriever.BM25Retriever (método: .buscar(query, top_k))

Objetivo:
    Aislar el rendimiento del retriever (Sección 3.3.3 y 3.3.5 de la memoria)
    frente a la evaluación end-to-end con LLMs (que realiza `eval/benchmark.py`
    vía RAGAS). Permite comparar empíricamente:
        1. Recuperación Densa pura (FAISS / VectorStore + BGE-M3-Legal-Spanish)
        2. Recuperación Léxica pura (BM25Retriever / BM25Okapi sobre texto_embedding)
        3. Recuperación Híbrida (Fusión RRF k=60, vía HybridRetriever.buscar)

Dependencias requeridas (mismo entorno virtual del proyecto):
    pandas, python-dotenv, rank_bm25, faiss-cpu (o faiss-gpu), sentence-transformers / torch

Configuración externa:
    - NO requiere API keys de OpenAI ni Salamandra: la evaluación es puramente
      local sobre el índice vectorial (FAISS) y el índice léxico (BM25).
    - Los índices FAISS y BM25 deben estar construidos previamente en el proyecto
      (por defecto en `./data/embeddings/faiss_index.bin` y `./data/embeddings/corpus_chunks.json`).
    - Ejecutable desde la raíz del proyecto:
          python -m eval.evaluar_retrieval
          # o bien:
          python eval/evaluar_retrieval.py
      o desde dentro del directorio eval/:
          python evaluar_retrieval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuración del PATH para importaciones modulares del proyecto
# -----------------------------------------------------------------------------
# Permite ejecutar el script tanto desde la raíz del proyecto (`python eval/evaluar_retrieval.py`)
# como desde dentro de la carpeta `eval/` (`python evaluar_retrieval.py`).
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
DIRECTORIO_RAIZ = DIRECTORIO_ACTUAL.parent

if str(DIRECTORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_RAIZ))

# Carga de variables de entorno si existiera un archivo .env en la raíz
load_dotenv(DIRECTORIO_RAIZ / ".env")

# =============================================================================
# IMPORTACIÓN DE LAS CLASES REALES DE RECUPERACIÓN (retriever/)
# =============================================================================
try:
    from retriever.hybrid_retriever import HybridRetriever
    from retriever.bm25_retriever import BM25Retriever
except ImportError as err:
    raise ImportError(
        f"No se pudieron importar las clases de recuperación del proyecto ({err}).\n"
        f"Verifica que la carpeta raíz del proyecto contiene el paquete `retriever/`\n"
        f"con `hybrid_retriever.py` y `bm25_retriever.py`, y que ejecutas el script\n"
        f"desde la raíz del repositorio (`python eval/evaluar_retrieval.py`)."
    ) from err


def instanciar_retriever_real() -> HybridRetriever:
    """
    Instancia y retorna la clase HybridRetriever del proyecto con sus rutas por defecto:
        - faiss_path: './data/embeddings/faiss_index.bin'
        - chunks_path: './data/embeddings/corpus_chunks.json'
        - model_path: './models/bge-m3-legal-spanish'
        - top_k_per_sistema: 20
        - top_k_final: 5
    """
    try:
        return HybridRetriever()
    except Exception as err:
        raise RuntimeError(
            f"Error al instanciar HybridRetriever() con rutas por defecto: {err}.\n"
            f"Comprueba que los ficheros `./data/embeddings/faiss_index.bin` y\n"
            f"`./data/embeddings/corpus_chunks.json` existen, o ejecuta el script\n"
            f"desde la raíz del proyecto. Si usas rutas personalizadas, edita los\n"
            f"argumentos de `HybridRetriever(...)` en `instanciar_retriever_real()`."
        ) from err


# =============================================================================
# EXTRACCIÓN Y NORMALIZACIÓN DE IDENTIFICADORES Y TEXTOS DE CHUNKS
# =============================================================================
# CONFIRMADO: cada chunk de `corpus_chunks.json` contiene los campos:
#   - "texto": contenido literal del artículo/bloque normativo.
#   - "texto_embedding": variante enriquecida con metadatos para indexación.
#   - "titulo_bloque": título de la sección o artículo (ej. "Artículo 34. Jornada").
#   - "norma_id": identificador de la norma (ej. "ET", "LGSS").
#   - "chunk_id" o "id": identificador único del chunk en el corpus indexado.

def extraer_id_chunk(chunk: Union[Dict[str, Any], Any], indice_fallback: int = 0) -> str:
    """
    Obtiene el ID unívoco de un chunk. Si el objeto no posee una clave 'id'/'chunk_id',
    construye un ID determinista a partir de los metadatos normativos disponibles.
    """
    if isinstance(chunk, dict):
        for clave in ("chunk_id", "id", "doc_id", "uid"):
            if chunk.get(clave):
                return str(chunk[clave])
        norma = chunk.get("norma_id", "NORMA").strip()
        titulo = chunk.get("titulo_bloque", "").strip()
        if norma and titulo:
            return f"{norma}::{titulo}"
        return f"chunk_{indice_fallback}"

    # Si el chunk es un objeto LangChain Document (doc.metadata, doc.page_content)
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        for clave in ("chunk_id", "id", "doc_id", "uid"):
            if chunk.metadata.get(clave):
                return str(chunk.metadata[clave])
        norma = chunk.metadata.get("norma_id", "NORMA")
        titulo = chunk.metadata.get("titulo_bloque", "")
        if norma and titulo:
            return f"{norma}::{titulo}"

    return f"chunk_{indice_fallback}"


def extraer_texto_chunk(chunk: Union[Dict[str, Any], Any]) -> str:
    """Extrae el contenido textual del chunk para comprobaciones por contenido."""
    if isinstance(chunk, dict):
        for clave in ("texto", "texto_embedding", "content", "page_content"):
            valor = chunk.get(clave)
            if valor:
                return str(valor)
        titulo = chunk.get("titulo_bloque", "")
        norma = chunk.get("norma_id", "")
        return f"{titulo} ({norma})".strip()

    if hasattr(chunk, "page_content"):
        return str(chunk.page_content)

    return str(chunk)


def extraer_titulo_bloque(chunk: Union[Dict[str, Any], Any]) -> str:
    """Extrae el título del bloque normativo asociado al chunk."""
    if isinstance(chunk, dict):
        return str(chunk.get("titulo_bloque", "") or chunk.get("title", ""))
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        return str(chunk.metadata.get("titulo_bloque", "") or chunk.metadata.get("title", ""))
    return ""


# =============================================================================
# DATASET DE EVALUACIÓN DE RETRIEVAL (GROUND TRUTH DE RECUPERACIÓN)
# =============================================================================
# NOTA METODOLÓGICA SOBRE 'relevant_chunk_ids' VS 'articulos_esperados':
# -----------------------------------------------------------------------------
# Cada chunk indexado en `corpus_chunks.json` posee un identificador único en el
# campo 'chunk_id' (además de 'texto', 'texto_embedding', 'titulo_bloque' y 'norma_id').
#
# Los valores iniciales listados abajo en `relevant_chunk_ids` (ej. "ET::Artículo 34",
# "ET_art_34") son conjeturas de formato. Para conocer el formato real exacto de
# tus identificadores e inspeccionar los primeros chunks, puedes ejecutar en tu terminal:
#
#   python -c "import json; c=json.load(open('data/embeddings/corpus_chunks.json')); print(c[0]['chunk_id'], c[0].get('titulo_bloque'))"
#
# Una vez conocido el patrón (p. ej. "chunk_001", "ET_art_34_0", etc.), puedes sustituir
# los `relevant_chunk_ids` por los IDs definitivos para máxima rigurosidad.
#
# MIENTRAS TANTO: el matching por `articulos_esperados` (ej. "Artículo 34", "Art. 34")
# actúa como fallback robusto en la función `es_chunk_relevante()`, comprobando la
# presencia del artículo en el título y en el texto del chunk recuperado, por lo que
# este script es 100% operativo de inmediato.

DATASET_RETRIEVAL: List[Dict[str, Any]] = [
    {
        "id_pregunta": 1,
        "query": "¿Cuántas horas semanales puede trabajar un empleado a tiempo completo?",
        "norma_principal": "Estatuto de los Trabajadores (ET)",
        "articulos_esperados": ["Artículo 34", "Art. 34", "artículo 34"],
        # EDITAR: rellena con los IDs reales de tu corpus_chunks.json si los tienes
        "relevant_chunk_ids": ["ET::Artículo 34", "ET_art_34", "chunk_et_34"],
        "descripcion_ground_truth": "ET Art. 34 (Jornada máxima legal de 40 horas semanales de promedio)",
    },
    {
        "id_pregunta": 2,
        "query": "¿Cuánto dura el período de prueba en un contrato indefinido?",
        "norma_principal": "Estatuto de los Trabajadores (ET)",
        "articulos_esperados": ["Artículo 14", "Art. 14", "artículo 14"],
        "relevant_chunk_ids": ["ET::Artículo 14", "ET_art_14", "chunk_et_14"],
        "descripcion_ground_truth": "ET Art. 14 (Período de prueba: límites de 6 meses técnicos titulados y 2 meses demás)",
    },
    {
        "id_pregunta": 3,
        "query": "¿Qué causas justifican un despido disciplinario?",
        "norma_principal": "Estatuto de los Trabajadores (ET)",
        "articulos_esperados": ["Artículo 54", "Art. 54", "artículo 54"],
        "relevant_chunk_ids": ["ET::Artículo 54", "ET_art_54", "chunk_et_54"],
        "descripcion_ground_truth": "ET Art. 54 (Incumplimientos contractuales graves y culpables que justifican despido disciplinario)",
    },
    {
        "id_pregunta": 4,
        "query": "¿Cuándo tiene derecho un trabajador a la prestación por incapacidad temporal?",
        "norma_principal": "Ley General de la Seguridad Social (LGSS)",
        "articulos_esperados": ["Artículo 169", "Artículo 170", "Artículo 172", "Artículo 283", "Art. 169", "artículo 169"],
        "relevant_chunk_ids": ["LGSS::Artículo 169", "LGSS::Artículo 172", "LGSS_art_169"],
        "descripcion_ground_truth": "LGSS Art. 169 y ss. (Requisitos y periodos de cotización previa para subsidio IT)",
    },
    {
        "id_pregunta": 5,
        "query": "¿Cuánto cobra un trabajador en situación de desempleo?",
        "norma_principal": "Ley General de la Seguridad Social (LGSS)",
        "articulos_esperados": ["Artículo 270", "Artículo 271", "Art. 270", "artículo 270"],
        "relevant_chunk_ids": ["LGSS::Artículo 270", "LGSS::Artículo 271", "LGSS_art_270"],
        "descripcion_ground_truth": "LGSS Art. 270 (Cuantía de la prestación contributiva por desempleo: 70% primeros 180 días y 60% resto)",
    },
    {
        "id_pregunta": 6,
        "query": "¿Qué derechos tienen los trabajadores en caso de huelga?",
        "norma_principal": "Real Decreto-ley 17/1977 y Estatuto de los Trabajadores",
        "articulos_esperados": ["Artículo 4", "Artículo 6", "Artículo 28", "Art. 4", "RDL 17/1977"],
        "relevant_chunk_ids": ["ET::Artículo 4", "RDL17_1977::Artículo 6", "ET_art_4"],
        "descripcion_ground_truth": "ET Art. 4.1.e y RDL 17/1977 (Derecho fundamental de huelga y garantías contractuales)",
    },
]


def cargar_dataset_retrieval() -> List[Dict[str, Any]]:
    """Retorna una copia estructurada del dataset de evaluación de retrieval."""
    return [dict(elem) for elem in DATASET_RETRIEVAL]


# =============================================================================
# MATCHING ROBUSTO DE RELEVANCIA
# =============================================================================
def es_chunk_relevante(chunk: Union[Dict[str, Any], Any], item_eval: Dict[str, Any], indice_chunk: int = 0) -> bool:
    """
    Determina si un chunk recuperado satisface el ground truth de la pregunta.

    Aplica una doble verificación en cascada:
      1. Coincidencia exacta por identificador (en `relevant_chunk_ids`).
      2. Coincidencia por presencia de substring de los artículos esperados en
         `titulo_bloque` o en los primeros 300 caracteres de `texto`.
    """
    chunk_id = extraer_id_chunk(chunk, indice_fallback=indice_chunk).strip().lower()
    ids_relevantes = [str(i).strip().lower() for i in item_eval.get("relevant_chunk_ids", [])]

    # 1. Matching por ID exacto
    if any(chunk_id == r_id or chunk_id.endswith(r_id) or r_id.endswith(chunk_id) for r_id in ids_relevantes):
        return True

    # 2. Fallback de robustez: matching por mención de artículo en título o texto
    titulo = extraer_titulo_bloque(chunk).lower()
    texto_inicio = extraer_texto_chunk(chunk)[:300].lower()
    articulos_esperados = [str(art).strip().lower() for art in item_eval.get("articulos_esperados", [])]

    for art in articulos_esperados:
        if art in titulo or art in texto_inicio:
            return True

    return False


# =============================================================================
# EJECUCIÓN DE BÚSQUEDA POR MODO (CLASES REALES: HybridRetriever / BM25Retriever)
# =============================================================================
def ejecutar_busqueda_modo(
    retriever: HybridRetriever,
    query: str,
    modo: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ejecuta la búsqueda llamando directamente a los componentes reales del retriever:

      1. Modo 'denso' (FAISS / VectorStore + Embedder BGE-M3):
         Calcula embedding de la query con `retriever.embedder.encode_query(query)`
         y consulta `retriever.vector_store.buscar(query_embedding, top_k=top_k)`,
         que devuelve nativamente 'score_vector', 'rank_vector' y 'retrieval_idx'
         sin necesidad de asignación manual.

      2. Modo 'bm25' (BM25Retriever / BM25Okapi):
         Consulta directamente `retriever.bm25.buscar(query, top_k=top_k)`.

      3. Modo 'hibrido' (HybridRetriever + Reciprocal Rank Fusion):
         Ejecuta `retriever.buscar(query, top_k=top_k, detalle=True)`.
    """
    modo_norm = modo.strip().lower()

    # MODO 1: RECUPERACIÓN DENSA PURA (FAISS VectorStore)
    # Confirmado: VectorStore.buscar() devuelve nativamente cada chunk con los
    # campos "score_vector" (float), "rank_vector" (int, 1-indexed) y
    # "retrieval_idx" (int) ya incluidos, por lo que no requiere post-procesado.
    if modo_norm in ("denso", "dense", "faiss", "vectorial"):
        query_embedding = retriever.embedder.encode_query(query)
        return retriever.vector_store.buscar(query_embedding, top_k=top_k)

    # MODO 2: RECUPERACIÓN LÉXICA PURA (BM25Retriever)
    if modo_norm in ("bm25", "lexico", "lexical", "sparse"):
        return retriever.bm25.buscar(query, top_k=top_k)

    # MODO 3: RECUPERACIÓN HÍBRIDA RRF (HybridRetriever)
    if modo_norm in ("hibrido", "hybrid", "rrf"):
        return retriever.buscar(query, top_k=top_k, detalle=True)

    raise ValueError(f"Modo de búsqueda desconocido: '{modo}'. Usar 'denso', 'bm25' o 'hibrido'.")


# =============================================================================
# EVALUACIÓN POR MODO Y CÁLCULO DE PRECISION@1 / RECALL@5
# =============================================================================
def evaluar_modo(
    retriever: Any,
    dataset: List[Dict[str, Any]],
    modo: str,
    top_k_max: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ejecuta las consultas del dataset para un modo de recuperación específico
    y registra las posiciones de acierto y chunks recuperados.
    """
    resultados_modo: List[Dict[str, Any]] = []

    print(f"\n[INFO] Evaluando modo: '{modo.upper()}' (Top-k = {top_k_max})...")

    for item in dataset:
        p_id = item["id_pregunta"]
        query = item["query"]
        t_inicio = time.perf_counter()

        try:
            chunks_recuperados = ejecutar_busqueda_modo(
                retriever=retriever,
                query=query,
                modo=modo,
                top_k=top_k_max,
            )
            latencia_ms = round((time.perf_counter() - t_inicio) * 1000, 2)
        except Exception as err:
            print(f"  [ERROR] Pregunta {p_id} ('{query[:40]}...'): fallo al recuperar: {err}")
            resultados_modo.append({
                "id_pregunta": p_id,
                "query": query,
                "modo": modo,
                "error": str(err),
                "precision_at_1": 0.0,
                "recall_at_5": 0.0,
                "hits_at_k": [],
                "posicion_primer_acierto": None,
                "latencia_ms": round((time.perf_counter() - t_inicio) * 1000, 2),
                "chunks_recuperados_resumen": [],
            })
            continue

        # Asegurar límite top_k_max
        chunks_top = list(chunks_recuperados)[:top_k_max]

        # Evaluación posición a posición
        hits: List[bool] = []
        resumen_chunks: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks_top):
            es_hit = es_chunk_relevante(chunk, item, indice_chunk=idx)
            hits.append(es_hit)
            resumen_chunks.append({
                "posicion": idx + 1,
                "id": extraer_id_chunk(chunk, indice_fallback=idx),
                "titulo": extraer_titulo_bloque(chunk),
                "es_relevante": es_hit,
                "preview": extraer_texto_chunk(chunk)[:120].replace("\n", " "),
            })

        # Precision@1: 1.0 si el primer elemento devuelto es relevante, 0.0 en caso contrario
        precision_at_1 = 1.0 if (len(hits) > 0 and hits[0]) else 0.0

        # Recall@5: proporción de artículos relevantes requeridos que han aparecido en el top-5
        # Si el ground truth define 1 artículo principal y apareció en top 5 -> 1.0
        total_aciertos_en_top_k = sum(1 for h in hits if h)
        # Se normaliza entre el mínimo requerido de acierto (al menos 1 relevante) o el tamaño del ground truth
        num_relevantes_esperados = max(1, len(item.get("articulos_esperados", ["art"])))
        # Para preguntas con 1 artículo principal diana, tener 1 hit en top-5 otorga recall=1.0
        recall_at_5 = 1.0 if total_aciertos_en_top_k >= 1 else 0.0

        # Posición del primer acierto (1-indexed) o None
        posicion_primer_acierto = None
        for pos, hit in enumerate(hits, start=1):
            if hit:
                posicion_primer_acierto = pos
                break

        print(
            f"  P{p_id}: P@1={precision_at_1:.0f} | R@5={recall_at_5:.0f} | "
            f"1er acierto en pos={posicion_primer_acierto or 'No'} ({latencia_ms} ms)"
        )

        resultados_modo.append({
            "id_pregunta": p_id,
            "query": query,
            "norma_principal": item.get("norma_principal", ""),
            "modo": modo,
            "precision_at_1": precision_at_1,
            "recall_at_5": recall_at_5,
            "hits_at_k": hits,
            "total_hits_top5": total_aciertos_en_top_k,
            "posicion_primer_acierto": posicion_primer_acierto,
            "latencia_ms": latencia_ms,
            "chunks_recuperados_resumen": resumen_chunks,
            "error": None,
        })

    return resultados_modo


def calcular_metricas(resultados_modo: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula las métricas agregadas (promedios) para un modo de recuperación."""
    validos = [r for r in resultados_modo if r.get("error") is None]
    total_consultas = len(resultados_modo)

    if not validos:
        return {
            "total_consultas": total_consultas,
            "consultas_exitosas": 0,
            "precision_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,  # Mean Reciprocal Rank complementario
            "latencia_media_ms": 0.0,
            "tasa_acierto_top5": 0.0,
        }

    p1_avg = sum(r["precision_at_1"] for r in validos) / len(validos)
    r5_avg = sum(r["recall_at_5"] for r in validos) / len(validos)

    # Cálculo de MRR (Mean Reciprocal Rank: 1/posicion_primer_acierto)
    reciprocal_ranks = []
    for r in validos:
        pos = r.get("posicion_primer_acierto")
        reciprocal_ranks.append(1.0 / pos if pos else 0.0)
    mrr_avg = sum(reciprocal_ranks) / len(reciprocal_ranks)

    latencia_avg = sum(r.get("latencia_ms", 0.0) for r in validos) / len(validos)
    preguntas_con_acierto = sum(1 for r in validos if (r.get("posicion_primer_acierto") is not None))

    return {
        "total_consultas": total_consultas,
        "consultas_exitosas": len(validos),
        "precision_at_1": round(p1_avg, 4),
        "recall_at_5": round(r5_avg, 4),
        "mrr": round(mrr_avg, 4),
        "latencia_media_ms": round(latencia_avg, 2),
        "tasa_acierto_top5": round(preguntas_con_acierto / len(validos), 4),
    }


# =============================================================================
# ORQUESTACIÓN COMPLETA Y COMPARATIVA ENTRE MODOS
# =============================================================================
def ejecutar_evaluacion_completa(
    modos: Optional[List[str]] = None,
    retriever_instancia: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el protocolo de evaluación sobre todos los modos seleccionados:
    'denso', 'bm25' e 'hibrido'. Si un modo falla o no está implementado,
    el script degrada con gracia y continúa con los restantes.
    """
    if modos is None:
        modos = ["denso", "bm25", "hibrido"]

    dataset = cargar_dataset_retrieval()
    retriever = retriever_instancia or instanciar_retriever_real()

    resultados_por_modo: Dict[str, List[Dict[str, Any]]] = {}
    metricas_por_modo: Dict[str, Dict[str, Any]] = {}
    modos_no_disponibles: List[str] = []

    print("=" * 80)
    print("EVALUACIÓN DE RETRIEVAL: PRECISION@1 Y RECALL@5 (ASISTENTE JURÍDICO-LABORAL BOE)")
    print("=" * 80)
    print(f"Dataset: {len(dataset)} preguntas normativas")
    print(f"Modos a evaluar: {', '.join(modos)}")
    print(f"Retriever activo: {type(retriever).__name__} (retriever.hybrid_retriever)")
    print("=" * 80)

    for modo in modos:
        try:
            detalle = evaluar_modo(retriever, dataset, modo=modo, top_k_max=5)
            resultados_por_modo[modo] = detalle
            metricas_por_modo[modo] = calcular_metricas(detalle)
        except NotImplementedError as nie:
            print(f"\n[AVISO] Modo '{modo}' no disponible en el retriever: {nie}")
            modos_no_disponibles.append(modo)
        except Exception as e:
            print(f"\n[ERROR] Fallo inesperado al evaluar modo '{modo}': {e}")
            modos_no_disponibles.append(modo)

    return {
        "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modos_evaluados": list(resultados_por_modo.keys()),
        "modos_no_disponibles": modos_no_disponibles,
        "metricas_agregadas": metricas_por_modo,
        "detalle_por_modo": resultados_por_modo,
        "dataset_utilizado": dataset,
    }


def generar_tabla_comparativa(resultados: Dict[str, Any]) -> pd.DataFrame:
    """Construye un DataFrame comparativo con las métricas agregadas de cada modo."""
    filas = []
    nombres_legibles = {
        "denso": "Densa pura (FAISS + BGE-M3)",
        "bm25": "Léxica pura (BM25Okapi)",
        "hibrido": "Híbrida RRF (Densa + Léxica)",
    }

    for modo, m in resultados.get("metricas_agregadas", {}).items():
        filas.append({
            "Modo de Recuperación": nombres_legibles.get(modo, modo.capitalize()),
            "Precision@1": m["precision_at_1"],
            "Recall@5": m["recall_at_5"],
            "MRR": m["mrr"],
            "Acierto Top-5": f"{m['tasa_acierto_top5'] * 100:.1f}%",
            "Latencia Media (ms)": m["latencia_media_ms"],
            "Consultas Válidas": f"{m['consultas_exitosas']}/{m['total_consultas']}",
        })

    return pd.DataFrame(filas)


# =============================================================================
# EXPORTACIÓN DEFENSIVA DE RESULTADOS (JSON + CSV)
# =============================================================================
def _json_default_serializer(obj: Any) -> Any:
    """Serializador defensivo para tipos NumPy, Pandas, Path y estructuras no estándar."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    # Compatibilidad con escalares numpy (int64, float64, bool_)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def exportar_resultados(
    resultados: Dict[str, Any],
    tabla_comparativa: pd.DataFrame,
    carpeta_salida: Union[str, Path] = "eval/resultados",
) -> Dict[str, Path]:
    """
    Exporta los resultados a la carpeta estándar `eval/resultados/`:
      - `retrieval_comparativa.csv` / `retrieval_comparativa.json`
      - `retrieval_detalle.csv` / `retrieval_detalle.json`
      - `retrieval_evaluation_completo.json`
    """
    ruta_dir = Path(carpeta_salida)
    if not ruta_dir.is_absolute():
        ruta_dir = DIRECTORIO_RAIZ / ruta_dir
    ruta_dir.mkdir(parents=True, exist_ok=True)

    archivos_generados: Dict[str, Path] = {}

    # 1. Exportar Tabla Comparativa Resumida
    ruta_comp_csv = ruta_dir / "retrieval_comparativa.csv"
    tabla_comparativa.to_csv(ruta_comp_csv, index=False, encoding="utf-8-sig")
    archivos_generados["comparativa_csv"] = ruta_comp_csv

    ruta_comp_json = ruta_dir / "retrieval_comparativa.json"
    tabla_comparativa.to_json(ruta_comp_json, orient="records", indent=2, force_ascii=False)
    archivos_generados["comparativa_json"] = ruta_comp_json

    # 2. Exportar Detalle por Pregunta y Modo
    filas_detalle = []
    for modo, items in resultados.get("detalle_por_modo", {}).items():
        for r in items:
            filas_detalle.append({
                "id_pregunta": r["id_pregunta"],
                "modo": modo,
                "query": r["query"],
                "norma_principal": r.get("norma_principal", ""),
                "precision_at_1": r["precision_at_1"],
                "recall_at_5": r["recall_at_5"],
                "posicion_primer_acierto": r["posicion_primer_acierto"],
                "total_hits_top5": r.get("total_hits_top5", 0),
                "latencia_ms": r["latencia_ms"],
                "error": r.get("error"),
            })

    df_detalle = pd.DataFrame(filas_detalle)
    ruta_det_csv = ruta_dir / "retrieval_detalle.csv"
    df_detalle.to_csv(ruta_det_csv, index=False, encoding="utf-8-sig")
    archivos_generados["detalle_csv"] = ruta_det_csv

    ruta_det_json = ruta_dir / "retrieval_detalle.json"
    df_detalle.to_json(ruta_det_json, orient="records", indent=2, force_ascii=False)
    archivos_generados["detalle_json"] = ruta_det_json

    # 3. Exportar Objeto Completo con Previews
    ruta_full_json = ruta_dir / "retrieval_evaluation_completo.json"
    with open(ruta_full_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=_json_default_serializer)
    archivos_generados["completo_json"] = ruta_full_json

    return archivos_generados


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("\nIniciando protocolo de evaluación de retrieval...")

    # Ejecutar evaluación sobre los tres modos del sistema
    resultado_eval = ejecutar_evaluacion_completa(modos=["denso", "bm25", "hibrido"])

    # Generar tabla comparativa
    tabla = generar_tabla_comparativa(resultado_eval)

    # Imprimir resultados por consola
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO DE RECUPERACIÓN (PRECISION@1 / RECALL@5 / MRR)")
    print("=" * 80)
    if not tabla.empty:
        print(tabla.to_string(index=False))
    else:
        print("No se obtuvieron métricas válidas. Revisa los mensajes de error superiores.")
    print("=" * 80)

    # Exportar archivos
    rutas = exportar_resultados(resultado_eval, tabla, carpeta_salida="eval/resultados")
    print("\n[OK] Resultados exportados correctamente:")
    for nombre, path in rutas.items():
        print(f"  - {nombre}: {path}")

    # Recordatorio didáctico sobre Ground Truth
    print("\n" + "-" * 80)
    print("NOTA METODOLÓGICA PARA LA MEMORIA DEL TFM:")
    print("1. Revisa `DATASET_RETRIEVAL` en este script para verificar que los")
    print("   `articulos_esperados` y `relevant_chunk_ids` coinciden exactamente con")
    print("   la numeración de artículos en tu `corpus_chunks.json`.")
    print("2. Las métricas de retrieval (Precision@1, Recall@5) evalúan la calidad del")
    print("   buscador de forma aislada, complementando la evaluación RAGAS de `benchmark.py`.")
    print("-" * 80 + "\n")