#!/bin/bash
# Setup completo del entorno TFM
# Testeado en Ubuntu 22.04 + Python 3.11

echo "=== Setup TFM Asistente Jurídico-Laboral ==="

# 1. Entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# 2. Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. PostgreSQL + pgvector (si no está instalado)
# sudo apt-get install postgresql postgresql-contrib
# sudo -u postgres psql -c "CREATE EXTENSION vector;"

# 4. Descargar modelo de embeddings (caché local)
python -c "
from sentence_transformers import SentenceTransformer
print('Descargando BGE-M3-Legal-Spanish...')
model = SentenceTransformer('wilfredomartel/BGE-M3-Legal-Spanish')
print(f'✅ Modelo cargado. Dimensión: {model.get_sentence_embedding_dimension()}')
"

# 5. Verificar Salamandra (para descarga completa usar HF CLI)
# huggingface-cli download BSC-LT/salamandra-7b-instruct \
#     --local-dir ./models/salamandra-7b

echo ""
echo "✅ Setup completado."
echo "📋 Siguiente paso: cp .env.example .env → rellenar variables"