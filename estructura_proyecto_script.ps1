# Navegar al directorio del proyecto
cd C:\Users\Usuario\TFM

# ─── Crear estructura completa de carpetas ───────────────────────
$carpetas = @(
    "ingesta",
    "corpus\raw",
    "corpus\processed",
    "embeddings",
    "retriever",
    "llm",
    "api",
    "frontend",
    "eval",
    "notebooks",
    "tests",
    "scripts",
    "models",
    "data\embeddings",
    "data\corpus\raw",
    "data\corpus\processed"
)

foreach ($carpeta in $carpetas) {
    New-Item -ItemType Directory -Path $carpeta -Force | Out-Null
    Write-Host "✅ Creada: $carpeta"
}

# ─── Crear ficheros base vacíos ──────────────────────────────────
$ficheros = @(
    "ingesta\__init__.py",
    "ingesta\boe_client.py",
    "ingesta\boe_parser.py",
    "ingesta\corpus_builder.py",
    "ingesta\run_ingesta.py",
    "embeddings\__init__.py",
    "embeddings\embedding_manager.py",
    "embeddings\generate.py",
    "retriever\__init__.py",
    "retriever\vector_store.py",
    "retriever\bm25_retriever.py",
    "retriever\hybrid_retriever.py",
    "llm\__init__.py",
    "llm\salamandra_client.py",
    "llm\openai_client.py",
    "llm\prompt_templates.py",
    "api\__init__.py",
    "api\main.py",
    "api\routes.py",
    "api\schemas.py",
    "eval\__init__.py",
    "eval\benchmark.py",
    "eval\metrics.py",
    "tests\__init__.py",
    "tests\test_ingesta.py",
    "tests\test_retriever.py",
    "tests\test_llm.py",
    "scripts\setup.sh",
    "scripts\verify_models.py",
    "scripts\download_models.py",
    ".env",
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "docker-compose.yml"
)

foreach ($fichero in $ficheros) {
    New-Item -ItemType File -Path $fichero -Force | Out-Null
    Write-Host "📄 Creado: $fichero"
}

Write-Host ""
Write-Host "🎉 Estructura del proyecto creada correctamente"