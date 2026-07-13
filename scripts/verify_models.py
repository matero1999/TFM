"""
Script de verificación: comprueba que los modelos funcionan
correctamente antes de iniciar el pipeline completo.
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def test_embeddings():
    """Verificar modelo de embeddings."""
    print("\n🔍 TEST 1: Embeddings BGE-M3-Legal-Spanish")
    print("─" * 50)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("wilfredomartel/BGE-M3-Legal-Spanish")

    # Query jurídica de prueba
    query = "¿Cuál es la jornada máxima de trabajo en España?"
    doc = (
        "Artículo 34. Jornada de trabajo. La duración máxima de la "
        "jornada ordinaria de trabajo será de cuarenta horas semanales "
        "de trabajo efectivo de promedio en cómputo anual."
        "[Estatuto de los Trabajadores, BOE-A-2015-11430]"
    )

    q_emb  = model.encode_query([query])
    d_emb  = model.encode_document([doc])
    sim    = model.similarity(q_emb, d_emb).item()

    print(f"  Dimensión embeddings: {q_emb.shape[1]}")
    print(f"  Query:  '{query[:50]}...'")
    print(f"  Doc:    '{doc[:50]}...'")
    print(f"  Similitud coseno: {sim:.4f}")

    assert sim > 0.5, f"Similitud demasiado baja: {sim}"
    print("  ✅ Test embeddings PASSED")
    return True


def test_openai_client():
    """Verificar API de OpenAI."""
    print("\n🔍 TEST 2: OpenAI GPT-4o-mini")
    print("─" * 50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        print("  ⚠️  OPENAI_API_KEY no configurada. Saltando test.")
        return None

    from openai import OpenAI

    client  = OpenAI(api_key=api_key)
    resp    = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Responde solo 'OK' para confirmar conexión."}
        ],
        max_tokens=5,
    )
    print(f"  Respuesta: {resp.choices[0].message.content}")
    print(f"  Tokens usados: {resp.usage.total_tokens}")
    print("  ✅ Test OpenAI PASSED")
    return True


def test_salamandra_disponible():
    """Verificar disponibilidad de Salamandra (sin carga completa)."""
    print("\n🔍 TEST 3: Salamandra-7B (disponibilidad)")
    print("─" * 50)

    try:
        from huggingface_hub import model_info
        info = model_info("BSC-LT/salamandra-7b-instruct")
        print(f"  Modelo: {info.modelId}")
        print(f"  Descargas último mes: {info.downloads:,}")
        print(f"  Licencia: {info.card_data.license if info.card_data else 'Apache 2.0'}")
        print("  ✅ Modelo disponible en HuggingFace")
        print("  ℹ️  Para cargar localmente: python scripts/download_salamandra.py")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFICACIÓN DE MODELOS — TFM ASISTENTE JURÍDICO")
    print("=" * 60)

    resultados = {
        "embeddings":  test_embeddings(),
        "openai":      test_openai_client(),
        "salamandra":  test_salamandra_disponible(),
    }

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("─" * 60)
    for nombre, resultado in resultados.items():
        estado = "✅ OK" if resultado else ("⚠️ SKIP" if resultado is None else "❌ FAIL")
        print(f"  {nombre:15} {estado}")
    print("=" * 60)