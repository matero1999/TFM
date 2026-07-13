# scripts/download_models.py
"""
Descarga y guarda los modelos en la carpeta /models del proyecto.
Ejecutar UNA SOLA VEZ antes de comenzar el desarrollo.
"""

import os
from pathlib import Path

# Ruta local del proyecto
MODELS_DIR = Path("C:/Users/Usuario/TFM/models")
MODELS_DIR.mkdir(exist_ok=True)

# ─── 1. Modelo de Embeddings ─────────────────────────────────────
print("\n📥 Descargando BGE-M3-Legal-Spanish (~570MB)...")
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("wilfredomartel/BGE-M3-Legal-Spanish")
embedding_model.save(str(MODELS_DIR / "bge-m3-legal-spanish"))
print("✅ Embeddings guardado en: models/bge-m3-legal-spanish")

# ─── 2. Modelo LLM: Salamandra 7B ────────────────────────────────
print("\n📥 Descargando Salamandra-7B-Instruct (~15GB)...")
print("⏳ Esto puede tardar varios minutos según su conexión...")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BSC-LT/salamandra-7b-instruct",
    local_dir=str(MODELS_DIR / "salamandra-7b-instruct"),
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # Solo PyTorch
)
print("✅ Salamandra guardado en: models/salamandra-7b-instruct")

# ─── Resumen ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("✅ DESCARGA COMPLETADA")
print(f"📁 Ruta: {MODELS_DIR}")
print("\nModelos disponibles:")
for m in MODELS_DIR.iterdir():
    size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
    print(f"  {m.name:40} {size/1e9:.1f} GB")