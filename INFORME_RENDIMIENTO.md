# Informe de Rendimiento — Pipeline RAG Multi-Dominio con Scraping Concurrente y Aislamiento Vectorial

**Asignatura:** Sistemas Paralelos y Distribuidos
**Stack:** Python 3.11+ · FastAPI · asyncio/httpx · ThreadPoolExecutor · Supabase (PostgreSQL + pgvector) · Gemini `models/gemini-embedding-001`

---

## 1. Arquitectura del flujo concurrente

```
                    POST /api/ingestar-web  (web_ingest.py)
   {coleccion_id, urls[], chunk_size, chunk_overlap, concurrency_workers}
                              │
        ┌─────────────────────┴──────────────────────┐
        │ FASE 1 · Descarga HTTP                     │  asyncio.gather + asyncio.Semaphore(p)
        │   ────────────────────────────────────────▶│  httpx.AsyncClient (I/O-bound)
        │                                            │
        │ FASE 2 · Limpieza HTML + Chunking          │  ThreadPoolExecutor(p)
        │   trafilatura / BeautifulSoup + split      │  (CPU-bound, asyncio.to_thread)
        │                                            │
        │ FASE 3 · Embeddings por lotes              │  gemini-embedding-001 (batch=10)
        │   backoff exponencial ante 429/EXHAUSTED   │  API externa (bottleneck de I/O)
        │                                            │
        │ FASE 4 · Inserción masiva                  │  Supabase / pgvector (upsert por lote)
        │   coleccion_id en cada fila                │
        └─────────────────────┬──────────────────────┘
                              ▼
                  public.normativa_bancaria  (aislamiento lógico por coleccion_id)

   POST /api/consultar {pregunta, coleccion_id, top_k}   (main.py)
        embed(pregunta) ─▶ RPC match_normativa_coleccion(query_embedding, p_coleccion_id, ...)
        ─▶ contexto filtrado ─▶ Gemini (temperature=0.0) ─▶ respuesta + fuentes
```

**Concurrencia:** la descarga usa `asyncio` + `httpx` (Opción A); la limpieza/chunking (CPU) se
despacha a un `ThreadPoolExecutor` (Opción B) para no bloquear el event loop. La ingesta de
vectores se hace por **lotes** con **reintentos exponenciales** ante `429 / RESOURCE_EXHAUSTED`.
No hay condiciones de carrera: la escritura en Supabase ocurre de forma centralizada tras el
`gather`; las URLs caídas se capturan y se reportan en `urls_fallidas`.

---

## 2. Función SQL/RPC de aislamiento (PL/pgSQL)

Columna e índice (Requerimiento 1):

```sql
alter table public.normativa_bancaria add column if not exists coleccion_id varchar(50);
create index if not exists normativa_bancaria_coleccion_id_idx
    on public.normativa_bancaria using btree (coleccion_id);
```

La RPC evalúa la **distancia coseno** (`<=>`) **únicamente** sobre las tuplas de la colección:

```sql
create or replace function public.match_normativa_coleccion(
    query_embedding vector(768),
    p_coleccion_id varchar(50),
    match_threshold float default 0.35,
    match_count int default 4
) returns table ( ... coleccion_id varchar(50), similarity float )
language plpgsql stable as $$
begin
  return query
  select ..., 1 - (n.embedding <=> query_embedding) as similarity
  from public.normativa_bancaria n
  where n.embedding is not null
    and n.coleccion_id = p_coleccion_id                    -- aislamiento determinista
    and 1 - (n.embedding <=> query_embedding) > match_threshold
  order by n.embedding <=> query_embedding
  limit match_count;
end; $$;
```

> El parámetro se llama `p_coleccion_id` para no colisionar con la columna `coleccion_id`
> de `RETURNS TABLE` (evita el error PostgreSQL `42725 parameter name used more than once`).

**Prueba de cero contaminación cruzada** (`scripts/test_aislamiento.py` → **PASS**):
la RPC de la colección A sólo devolvió filas de A; la de B sólo filas de B; una colección
inexistente devolvió 0 filas; y la RPC sin filtro (`match_normativa`) mezcló ambas
colecciones, evidenciando por qué el filtro es obligatorio.

---

## 3. Metodología de medición

- **Corpus:** 10 URLs fijas de tamaño de texto similar (~7.9k–24.9k caracteres).
- **Fases instrumentadas:** (1) scraping/red, (2) chunking/CPU, (3) API embeddings, (4) inserción Supabase.
- **Parámetros:** `chunk_size=1000`, `chunk_overlap=200`, `concurrency_workers p ∈ {1,2,4,8}`.
> `T_1` = 1 worker. `S_p = T_1 / T_p`. `E_p = S_p / p`. Total de chunks por corrida ≈ 211.
>
> Se presentan **dos campañas**: (a) embeddings **API real** (p=1 y p=2, la cuota gratuita
> ~1000 req/día se agotó); (b) barrido completo p∈{1,2,4,8} con `--embed-mode mock`
> (latencia fija por lote) para caracterizar el escalado de las fases paralelizables.

---

## 3.1 Resultados — API real de embeddings

| p | T_p (s) | %red | %chunk | %embeddings | %Supabase | Speedup S_p | Eficiencia E_p |
|---|---------|------|--------|-------------|-----------|-------------|----------------|
| 1 | 156.01  | 1.4% | 0.2%   | 98.4%       | 0.0%*     | 1.000       | 100.0%         |
| 2 | 122.57  | 1.2% | 0.2%   | 98.5%       | 0.0%*     | **1.273**   | 63.6%          |

\* corrida `--skip-db`. Los **embeddings** dominan ~98% del tiempo total.

## 3.2 Resultados — barrido completo (embeddings mock con latencia fija)

| p | T_p (s) | Speedup S_p | Eficiencia E_p | % scraping | % chunking | % embeddings | % Supabase |
|---|---------|-------------|----------------|------------|------------|--------------|------------|
| 1 | 31.23   | 1.000       | 100.0%         | 7.1%       | 1.0%       | 70.7%        | 21.3%      |
| 2 | 28.19   | 1.108       | 55.4%          | 4.7%       | 1.0%       | 78.3%        | 15.9%      |
| 4 | 26.63   | 1.173       | 29.3%          | 3.6%       | 1.5%       | 82.9%        | 12.0%      |
| 8 | 26.39   | 1.183       | 14.8%          | 4.3%       | 1.9%       | 83.6%        | 10.2%      |

### Speedup por fase (amortización del paralelismo)

| p | S_red (scraping) | S_chunking | S_embeddings | S_inserción |
|---|------------------|------------|--------------|-------------|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.66 | 1.06 | 1.00 | 1.48 |
| 4 | **2.29** | 0.78 | 1.00 | 2.08 |
| 8 | 1.97 | 0.60 | 1.00 | 2.47 |

La fase de **red** acelera hasta `p=4` (S≈2.3) y **se degrada en p=8** (contención de red,
límites del servidor y saturación del pool). Los embeddings **no escalan** (S=1.00) y la
inserción mejora por batching.

---

## 4. Análisis — Ley de Amdahl y cuellos de botella

Sea `f` la fracción no paralelizable del pipeline. El mejor speedup posible es
`S(p) = 1 / (f + (1-f)/p)`. En este pipeline:

- **Fracción no paralelizable `f ≈ 0.71–0.84`** (embeddings de API externa + serialización de la inserción).
- Con `f = 0.80`, el límite teórico es `S_max = 1/0.8 = 1.25`, valor que coincide con el
  speedup medido (`1.18–1.27`). Es decir, **la aceleración está limitada por la API de
  embeddings**, cuyo `Rate Limit` (429 / RESOURCE_EXHAUSTED) introduce pausas de backoff
  que no dependen de `p`.
- El **desglose porcentual** confirma el diagnóstico: `% embeddings ≈ 70–98%` del tiempo total.
- Conclusión de ingeniería: para escalar realmente hay que (i) cachear/embeder localmente
  (p. ej. `sentence-transformers`), (ii) aumentar la cuota/paralelizar la API con control de
  tasa, y (iii) particionar el chunking en procesos (`ProcessPoolExecutor`) para liberar GIL.

---

## 5. Conclusiones

1. El **aislamiento vectorial** por `coleccion_id` (índice B-Tree + RPC PL/pgSQL) elimina la
   contaminación cruzada de forma **determinista y verificada**.
2. La **concurrencia** (asyncio + semáforo + thread pool + batching) acelera el scraping
   (S≈2.3 con p=4) y hace el pipeline viable para lotes de URLs.
3. El **techo de escalabilidad** lo impone la **API externa de embeddings** (Ley de Amdahl),
   tal como muestran los datos: `S_p` satura cerca de `1/f`.
4. Recomendaciones: caché de embeddings (implementada), modelo local, y control de tasa
   adaptativo.

---

## 6. Reproducción

```bash
docker compose up --build                      # backend:8000, scraper:8001, frontend:8501
python scripts/test_aislamiento.py             # PASS esperado
python benchmark.py --embed-mode api --workers 1,2,4,8 --reps 3   # cuota API
python benchmark.py --embed-mode cache --workers 1,2,4,8          # repetible sin API
python benchmark.py --embed-mode mock  --workers 1,2,4,8          # sin cuota
```
Resultados crudos: `resultados/benchmark_<fecha>.json` y `.csv`.
