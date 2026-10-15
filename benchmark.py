"""
Benchmark de la ingesta RAG: secuencial (1 worker) vs concurrente (p workers).

Mide el pipeline completo de la guía:
  1. Descarga HTTP          (% scraping/red)
  2. Limpieza + chunking    (% chunking/CPU)
  3. Embeddings (lotes)     (% API embeddings)
  4. Inserción Supabase     (% inserción)

Calcula T_s/T_1, T_p (p in {1,2,4,8}), Speedup S_p = T_1/T_p y Eficiencia
E_p = S_p/p, con repeticiones (media +/- desviación) y guarda los resultados en
resultados/*.json y *.csv listos para el informe.

Uso:
  python benchmark.py                     # 10 URLs fijas, 1 repetición (API real)
  python benchmark.py --reps 3 --workers 1,2,4,8
  python benchmark.py --skip-db           # sin tocar Supabase (fases 1-3)
  python benchmark.py --embed-mode mock   # embeddings locales (sin cuota API)

Modos de embeddings:
  api   -> API real de Gemini (por defecto). Cachea los vectores en resultados/.
  cache -> sólo caché (error si falta algún vector). Repite el benchmark sin API.
  mock  -> vectores deterministas locales con latencia fija por lote, para
           ejecutar el barrido de forma reproducible cuando no hay cuota.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

from web_ingest import clean_html, chunk_text, embed_chunks

load_dotenv()

CACHE_EMBEDDINGS = os.path.join("resultados", "embedding_cache.json")
MOCK_LATENCIA_LOTE = 1.0  # segundos simulados por lote en modo mock
DIM_EMBEDDING = 768

# 10 URLs de tamaño de texto similar (7.9k - 24.9k caracteres, mismo origen)
URLS_DEFECTO = [
    "https://es.wikipedia.org/wiki/Activo_(contabilidad)",
    "https://es.wikipedia.org/wiki/Ley",
    "https://es.wikipedia.org/wiki/Norma_jur%C3%ADdica",
    "https://es.wikipedia.org/wiki/Responsabilidad_civil",
    "https://es.wikipedia.org/wiki/Tasa_de_inter%C3%A9s",
    "https://es.wikipedia.org/wiki/Auditor%C3%ADa",
    "https://es.wikipedia.org/wiki/Delito",
    "https://es.wikipedia.org/wiki/Pol%C3%ADtica_monetaria",
    "https://es.wikipedia.org/wiki/Derecho_comercial",
    "https://es.wikipedia.org/wiki/Patrimonio",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --------------------------------------------------------------------------- #
# Fases instrumentadas                                                         #
# --------------------------------------------------------------------------- #
async def fase_descarga(urls: List[str], workers: int) -> Dict:
    """Fase 1: descarga HTTP concurrente (asyncio + semaphore)."""
    sem = asyncio.Semaphore(workers)
    ok, fallidas = [], []

    async def una(client: httpx.AsyncClient, url: str):
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.get(url, timeout=30, follow_redirects=True)
                r.raise_for_status()
                return url, r.text, None, time.perf_counter() - t
            except Exception as e:
                return url, "", f"{type(e).__name__}: {e}", time.perf_counter() - t

    t0 = time.perf_counter()
    async with httpx.AsyncClient(headers={"User-Agent": "RAG-Benchmark/1.0"}) as client:
        res = await asyncio.gather(*[una(client, u) for u in urls])
    wall = time.perf_counter() - t0

    htmls = []
    for url, html, err, _ in res:
        if err:
            fallidas.append({"url": url, "error": err})
        else:
            htmls.append({"url": url, "html": html})
    return {"wall": wall, "htmls": htmls, "fallidas": fallidas}


def _limpiar_y_partir(html: str) -> List[str]:
    return chunk_text(clean_html(html), CHUNK_SIZE, CHUNK_OVERLAP)


async def fase_chunking(htmls: List[Dict], workers: int) -> Dict:
    """Fase 2: limpieza de HTML + chunking (CPU-bound, thread pool con p hilos)."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = [loop.run_in_executor(pool, _limpiar_y_partir, h["html"]) for h in htmls]
        resultados = await asyncio.gather(*futuros)
    wall = time.perf_counter() - t0

    chunks = []
    for h, parts in zip(htmls, resultados):
        for c in parts:
            chunks.append({"url": h["url"], "contenido": c})
    return {"wall": wall, "chunks": chunks}


def _clave(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _vector_mock(texto: str) -> List[float]:
    """Vector determinista local (no usa API), normalizado."""
    semilla = int(hashlib.sha256(texto.encode("utf-8")).hexdigest()[:8], 16)
    vec = [math.sin(semilla % (i + 1)) for i in range(DIM_EMBEDDING)]
    norma = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norma for x in vec]


def _cargar_cache() -> Dict[str, List[float]]:
    if os.path.exists(CACHE_EMBEDDINGS):
        with open(CACHE_EMBEDDINGS) as f:
            return json.load(f)
    return {}


def _guardar_cache(cache: Dict[str, List[float]]) -> None:
    os.makedirs(os.path.dirname(CACHE_EMBEDDINGS), exist_ok=True)
    with open(CACHE_EMBEDDINGS, "w") as f:
        json.dump(cache, f)


def fase_embeddings(chunks: List[Dict], modo: str = "api") -> Dict:
    """Fase 3: embeddings (API real, caché o mock). Reutiliza embed_chunks real."""
    textos = [c["contenido"] for c in chunks]
    t0 = time.perf_counter()

    if modo == "mock":
        vecs = []
        for i in range(0, len(textos), 10):
            time.sleep(MOCK_LATENCIA_LOTE)
            vecs.extend(_vector_mock(t) for t in textos[i:i + 10])
        return {"wall": time.perf_counter() - t0, "vecs": vecs}

    cache = _cargar_cache()
    faltantes = [t for t in textos if _clave(t) not in cache]
    if modo == "cache" and faltantes:
        raise RuntimeError(
            f"Faltan {len(faltantes)} embeddings en caché. Ejecuta primero con --embed-mode api."
        )
    if modo == "api" and faltantes:
        nuevos = embed_chunks(faltantes)
        for t, v in zip(faltantes, nuevos):
            cache[_clave(t)] = v
        _guardar_cache(cache)

    vecs = [cache[_clave(t)] for t in textos]
    return {"wall": time.perf_counter() - t0, "vecs": vecs}


def fase_insercion(chunks, vecs, coleccion_id: str, skip_db: bool) -> Dict:
    """Fase 4: inserción masiva en Supabase (o simulada con --skip-db)."""
    if skip_db or not vecs:
        return {"wall": 0.0, "insertados": 0, "omitido": True}

    from supabase import create_client
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

    filas = [
        {
            "documento_origen": c["url"],
            "organismo": "Benchmark",
            "tipo_norma": "Web",
            "jerarquia": "Benchmark",
            "articulo_ref": c["url"][:80],
            "contenido": c["contenido"],
            "coleccion_id": coleccion_id,
            "embedding": v,
        }
        for c, v in zip(chunks, vecs)
    ]
    t0 = time.perf_counter()
    (
        client.table("normativa_bancaria")
        .delete()
        .eq("coleccion_id", coleccion_id)
        .execute()
    )
    # inserción por lotes para no superar límites de payload
    for i in range(0, len(filas), 100):
        client.table("normativa_bancaria").insert(filas[i:i + 100]).execute()
    return {"wall": time.perf_counter() - t0, "insertados": len(filas), "omitido": False}


# --------------------------------------------------------------------------- #
# Ejecución de una corrida                                                     #
# --------------------------------------------------------------------------- #
async def correr_una(urls: List[str], workers: int, coleccion_id: str, skip_db: bool,
                     embed_modo: str = "api") -> Dict:
    f1 = await fase_descarga(urls, workers)
    f2 = await fase_chunking(f1["htmls"], workers)

    def _emb():
        return fase_embeddings(f2["chunks"], embed_modo)

    loop = asyncio.get_running_loop()
    f3 = await loop.run_in_executor(None, _emb)
    f4 = await loop.run_in_executor(None, fase_insercion, f2["chunks"], f3["vecs"], coleccion_id, skip_db)

    total = f1["wall"] + f2["wall"] + f3["wall"] + f4["wall"]
    return {
        "workers": workers,
        "total_s": total,
        "fases_s": {
            "scraping_red": f1["wall"],
            "chunking_cpu": f2["wall"],
            "api_embeddings": f3["wall"],
            "insercion_supabase": f4["wall"],
        },
        "chunks": len(f2["chunks"]),
        "urls_ok": len(f1["htmls"]),
        "urls_fallidas": f1["fallidas"],
        "insertados": f4["insertados"],
    }


def cargar_urls(ruta: Optional[str]) -> List[str]:
    if not ruta:
        return URLS_DEFECTO
    with open(ruta) as f:
        datos = json.load(f)
    return datos["urls"] if isinstance(datos, dict) else datos


def resumen(corridas: List[Dict], workers: int) -> Dict:
    totales = [c["total_s"] for c in corridas]
    medias = {k: statistics.mean(c["fases_s"][k] for c in corridas)
              for k in corridas[0]["fases_s"]}
    return {
        "workers": workers,
        "repeticiones": len(corridas),
        "T_s": statistics.mean(totales),
        "desv_s": statistics.stdev(totales) if len(totales) > 1 else 0.0,
        "fases_media_s": medias,
        "chunks": corridas[0]["chunks"],
        "urls_ok": corridas[0]["urls_ok"],
        "urls_fallidas": len(corridas[0]["urls_fallidas"]),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de ingesta RAG secuencial vs concurrente")
    ap.add_argument("--urls-file", help="JSON con la lista de URLs a probar")
    ap.add_argument("--coleccion", default="benchmark_web", help="Colección usada para la fase de inserción")
    ap.add_argument("--workers", default="1,2,4,8", help="Lista de workers, ej: 1,2,4,8")
    ap.add_argument("--reps", type=int, default=1, help="Repeticiones por configuración (recomendado: 3)")
    ap.add_argument("--skip-db", action="store_true", help="No insertar en Supabase (mide fases 1-3)")
    ap.add_argument("--embed-mode", choices=["api", "cache", "mock"], default="api",
                    help="Origen de los embeddings (api real, cache o mock local)")
    ap.add_argument("--salida", default="resultados", help="Directorio de salida")
    args = ap.parse_args()

    urls = cargar_urls(args.urls_file)
    workers_list = [int(w) for w in args.workers.split(",")]
    print(f"[*] URLs: {len(urls)} | workers: {workers_list} | reps: {args.reps} | "
          f"DB: {'OMITIDA' if args.skip_db else args.coleccion} | embeddings: {args.embed_mode}")

    resultados = []
    for w in workers_list:
        corridas = []
        for rep in range(args.reps):
            c = await correr_una(urls, w, args.coleccion, args.skip_db, args.embed_mode)
            corridas.append(c)
            print(f"  [p={w} rep={rep + 1}/{args.reps}] total={c['total_s']:.2f}s "
                  f"(red={c['fases_s']['scraping_red']:.2f}s "
                  f"chunk={c['fases_s']['chunking_cpu']:.2f}s "
                  f"emb={c['fases_s']['api_embeddings']:.2f}s "
                  f"ins={c['fases_s']['insercion_supabase']:.2f}s) "
                  f"chunks={c['chunks']} fallidas={len(c['urls_fallidas'])}")
            if c["urls_fallidas"]:
                for f in c["urls_fallidas"]:
                    print(f"      [!] {f['url']}: {f['error'][:80]}")
        resultados.append(resumen(corridas, w))

    # ---------------- métricas: Speedup y Eficiencia ---------------- #
    t1 = next(r["T_s"] for r in resultados if r["workers"] == 1)
    f1 = next(r["fases_media_s"] for r in resultados if r["workers"] == 1)
    for r in resultados:
        r["Speedup"] = t1 / r["T_s"] if r["T_s"] > 0 else 0.0
        r["Eficiencia"] = r["Speedup"] / r["workers"]
        # speedup por fase (útil para la Ley de Amdahl: sólo red y CPU escalan)
        r["Speedup_fase"] = {
            k: (f1[k] / r["fases_media_s"][k] if r["fases_media_s"][k] > 0 else 0.0)
            for k in r["fases_media_s"]
        }

    # ---------------- reporte consola -------------------------------- #
    print("\n" + "=" * 96)
    print(f"{'p':>3} {'T_p (s)':>10} {'±':>7} {'Speedup':>9} {'Eficiencia':>11} "
          f"{'%red':>7} {'%chunk':>7} {'%emb':>7} {'%ins':>7}")
    print("-" * 96)
    for r in resultados:
        t = r["T_s"]
        pct = {k: 100 * v / t if t else 0 for k, v in r["fases_media_s"].items()}
        print(f"{r['workers']:>3} {t:>10.2f} {r['desv_s']:>7.2f} {r['Speedup']:>9.3f} "
              f"{r['Eficiencia']:>10.1%} {pct['scraping_red']:>6.1f}% "
              f"{pct['chunking_cpu']:>6.1f}% {pct['api_embeddings']:>6.1f}% "
              f"{pct['insercion_supabase']:>6.1f}%")
    print("=" * 96)
    print("Speedup por fase (para justificar la Ley de Amdahl):\n")
    print("| p | S_red (scraping) | S_chunking | S_embeddings | S_insercion |")
    print("|---|------------------|------------|--------------|-------------|")
    for r in resultados:
        sf = r["Speedup_fase"]
        print(f"| {r['workers']} | {sf['scraping_red']:.2f} | {sf['chunking_cpu']:.2f} | "
              f"{sf['api_embeddings']:.2f} | {sf['insercion_supabase']:.2f} |")

    print("\nTabla markdown para el informe:\n")
    print("| p | T_p (s) | Speedup S_p | Eficiencia E_p | % scraping | % chunking | % embeddings | % Supabase |")
    print("|---|---------|-------------|----------------|------------|------------|--------------|------------|")
    for r in resultados:
        t = r["T_s"]
        pct = {k: 100 * v / t if t else 0 for k, v in r["fases_media_s"].items()}
        print(f"| {r['workers']} | {t:.2f} ± {r['desv_s']:.2f} | {r['Speedup']:.3f} | "
              f"{r['Eficiencia']:.1%} | {pct['scraping_red']:.1f}% | {pct['chunking_cpu']:.1f}% | "
              f"{pct['api_embeddings']:.1f}% | {pct['insercion_supabase']:.1f}% |")

    # ---------------- persistencia ----------------------------------- #
    os.makedirs(args.salida, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = {
        "fecha": marca,
        "urls": urls,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "reps": args.reps,
        "skip_db": args.skip_db,
        "embed_mode": args.embed_mode,
        "configuraciones": resultados,
    }
    ruta_json = os.path.join(args.salida, f"benchmark_{marca}.json")
    with open(ruta_json, "w") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    ruta_csv = os.path.join(args.salida, f"benchmark_{marca}.csv")
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workers", "T_p_s", "desv_s", "Speedup", "Eficiencia",
                    "pct_scraping_red", "pct_chunking_cpu", "pct_api_embeddings",
                    "pct_insercion_supabase"])
        for r in resultados:
            t = r["T_s"]
            pct = {k: 100 * v / t if t else 0 for k, v in r["fases_media_s"].items()}
            w.writerow([r["workers"], round(t, 3), round(r["desv_s"], 3),
                        round(r["Speedup"], 4), round(r["Eficiencia"], 4),
                        round(pct["scraping_red"], 2), round(pct["chunking_cpu"], 2),
                        round(pct["api_embeddings"], 2), round(pct["insercion_supabase"], 2)])

    print(f"\n[✓] Resultados guardados en:\n    {ruta_json}\n    {ruta_csv}")


if __name__ == "__main__":
    asyncio.run(main())
