# 🏛️ Asistente Normativo RAG Multi-Dominio - ASFI / Banco Unión

Pipeline RAG con **scraping concurrente**, **aislamiento vectorial multi-tenant** y
**evaluación de rendimiento paralelo** (Speedup / Eficiencia).

## Arquitectura
- **Backend FastAPI** (`main.py`, puerto 8000): consulta semántica filtrada por colección.
- **Scraper/Ingesta web** (`web_ingest.py`, puerto 8001): descarga concurrente, chunking
  y embeddings por lotes.
- **Frontend Streamlit** (`app_streamlit.py`, puerto 8501).
- **Vector Store**: Supabase + pgvector (768 dims).
- **Modelos**: `models/gemini-embedding-001` (embeddings) y Gemini Flash (generación).

## Requisitos
1. Python 3.11+ o Docker.
2. Cuenta/proyecto Supabase con la extensión `vector`.
3. API Key de Google AI Studio.
4. `.env` a partir de `.env.example`:
   ```
   SUPABASE_URL=https://<proyecto>.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<sb_secret_...>
   GEMINI_API_KEY=<clave>
   ```

## 1. Crear el esquema (aislamiento multi-tenant)
Ejecutar `scripts/setup_database.sql` en **Supabase → SQL Editor**. Crea:
- columna `coleccion_id VARCHAR(50)` con índice **B-Tree**,
- índice vectorial `ivfflat` para distancia coseno,
- RPC **`match_normativa_coleccion`** (PL/pgSQL) que filtra estrictamente por
  `coleccion_id` (cero contaminación cruzada),
- RPC legacy `match_normativa` (sin filtro, sólo compatibilidad).

Alternativa automatizada (requiere `SUPABASE_DB_URL` en `.env`):
```bash
python scripts/apply_schema.py
```

Verificación de aislamiento:
```bash
python scripts/test_aislamiento.py   # PASS = cero contaminación cruzada
```

## 2. Ejecución con Docker
```bash
docker compose up --build
# backend:   http://localhost:8000
# scraper:   http://localhost:8001
# streamlit: http://localhost:8501
```
Base vectorial local opcional (valida el DDL sin tocar Supabase):
```bash
docker compose --profile local-db up --build   # Postgres+pgvector en :55432
```

## 3. Endpoints principales
### Ingesta concurrente — `POST /api/ingestar-web` (puerto 8001)
```json
{
  "coleccion_id": "tributaria_bolivia_2026",
  "urls": ["https://sitio1.com/norma1", "https://sitio2.com/norma2"],
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "concurrency_workers": 4
}
```
Devuelve chunks insertados, URLs fallidas y el desglose temporal (`fases_ms`).
Los embeddings se envían **por lotes** con **backoff exponencial** ante `429`/`RESOURCE_EXHAUSTED`.

### Consulta especializada — `POST /api/consultar` (puerto 8000)
`coleccion_id` es **obligatorio**; la búsqueda vectorial se restringe a esa colección
y el LLM responde con `temperature=0.0` y regla de no-alucinación.
```json
{ "pregunta": "...", "coleccion_id": "tributaria_bolivia_2026", "top_k": 4 }
```

## 4. Prueba de aislamiento (contaminación cruzada)
```bash
python scripts/test_aislamiento.py   # PASS = cero contaminación cruzada
```
Inserta 2 colecciones con embeddings reales y verifica que cada RPC sólo devuelve
su propia colección (y demuestra que la RPC sin filtro sí mezclaría dominios).

## 5. Benchmark de rendimiento paralelo
```bash
python benchmark.py --reps 3 --workers 1,2,4,8          # embeddings API real
python benchmark.py --embed-mode mock --workers 1,2,4,8 # sin consumir cuota API
python benchmark.py --embed-mode cache                  # repetir sin llamar la API
python benchmark.py --skip-db                           # sin escribir en Supabase
python benchmark.py --urls-file mis_urls.json --reps 3
```
Mide las 4 fases (scraping/red, chunking/CPU, API embeddings, inserción Supabase),
calcula `T_p`, `Speedup S_p = T_1/T_p`, `Eficiencia E_p = S_p/p` y **Speedup por
fase** para justificar la Ley de Amdahl. Guarda `resultados/benchmark_<fecha>.json`
y `.csv` listos para el informe.

> La cuota gratuita de `gemini-embedding-001` es de ~1000 peticiones/día. Con 10
> URLs (~211 chunks) una corrida consume gran parte de la cuota; use `cache`/`mock`
> para repetir el experimento sin agotarla.

## 6. Scripts auxiliares
- `scripts/ingest_normativa.py`: ingesta de PDFs a la colección `COLECCION_DEFECTO`
  (por defecto `bancaria_asfi_bolivia`).
- `scripts/query_rag.py`: `python scripts/query_rag.py <coleccion_id> "<pregunta>"`.
- `scripts/apply_schema.py`: aplica el DDL por conexión directa Postgres.
- `scripts/test_aislamiento.py`: prueba de cero contaminación cruzada.
- `scripts/graficar.py`: genera `resultados/speedup.png`, `eficiencia.png`,
  `desglose_fases.png` y `speedup_fases.png` a partir del JSON del benchmark.
- `scripts/generar_informe.py`: genera `INFORME_RENDIMIENTO.pdf` (máx. 5 páginas).

## Estructura
```
main.py                     API de consulta (RPC filtrada por colección)
web_ingest.py               API de ingesta web concurrente
benchmark.py                Benchmark secuencial vs paralelo
app_streamlit.py            Frontend
scripts/setup_database.sql  DDL: tabla, índices y RPC PL/pgSQL
scripts/test_aislamiento.py Prueba de cero contaminación cruzada
scripts/graficar.py         Gráficas de speedup/eficiencia/fases
scripts/generar_informe.py  Genera el informe PDF (5 páginas)
INFORME_RENDIMIENTO.md/.pdf Informe de rendimiento
Dockerfile / docker-compose.yml
```
