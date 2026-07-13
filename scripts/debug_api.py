# Guárdelo como scripts\debug_api.py y ejecútelo con:
# python scripts\debug_api.py

import requests
import json

norma_id = "BOE-A-2015-11430"
url = f"https://boe.es/datosabiertos/api/legislacion-consolidada/id/{norma_id}/metadatos"

resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
data = resp.json()

print("── Código de estado API:", data.get("status", {}).get("code"))
print("── Tipo de data:        ", type(data.get("data")))
print("── Contenido (primeros campos):")
print(json.dumps(data.get("data"), indent=2, ensure_ascii=False)[:800])