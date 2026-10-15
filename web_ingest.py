import asyncio
import random
import time
import os
from typing import List, Dict, Any, Optional
import httpx
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Faltan variables de entorno requeridas")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_EMBEDDING = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 10
MAX_RETRIES = 6
BACKOFF_BASE = 2.0

app = FastAPI(title="Web Ingest RAG", version="1.1.0")


class IngestRequest(BaseModel):
    coleccion_id: str = Field(min_length=1, max_length=50)
    urls: List[str] = Field(min_length=1)
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    concurrency_workers: int = Field(default=4, ge=1, le=32)

    @field_validator("coleccion_id")
    @classmethod
    def coleccion_no_vacia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("coleccion_id no puede estar vacío")
        return v


async def fetch_url(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Descarga HTTP de una URL. El semáforo limita la concurrencia real."""
    async with semaphore:
        try:
            resp = await client.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            return {"url": url, "html": resp.text, "error": None}
        except Exception as e:
            return {"url": url, "html": "", "error": f"{type(e).__name__}: {e}"}


def clean_html(html: str) -> str:
    if not html:
        return ""
    try:
        text = trafilatura.extract(html, include_links=False, include_comments=False, include_tables=True)
        if not text:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        return " ".join(text.split())
    except Exception:
        return " ".join(html.split())


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 60:
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


async def process_url(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    res = await fetch_url(client, url, semaphore)
    if res["error"]:
        return {"url": url, "chunks": [], "error": res["error"]}
    # limpieza y chunking son CPU-bound: se ejecutan en un hilo del pool para no
    # bloquear el event loop de asyncio (mantén la concurrencia de descargas)
    text, chunks = await asyncio.to_thread(_parse_and_chunk, res["html"], chunk_size, chunk_overlap)
    return {"url": url, "chunks": [{"url": url, "contenido": c} for c in chunks], "error": None, "chars": len(text)}


def _parse_and_chunk(html: str, chunk_size: int, chunk_overlap: int):
    text = clean_html(html)
    return text, chunk_text(text, chunk_size, chunk_overlap)


def _es_rate_limit(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "rate limit" in s.lower()


def embed_chunks(chunks_text: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
    """Embeddings por lotes con backoff exponencial ante 429 / RESOURCE_EXHAUSTED."""
    all_vecs: List[List[float]] = []
    for i in range(0, len(chunks_text), batch_size):
        batch = chunks_text[i:i + batch_size]
        intento = 0
        while True:
            try:
                res = ai_client.models.embed_content(
                    model=MODEL_EMBEDDING,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                all_vecs.extend([e.values for e in res.embeddings])
                time.sleep(0.5)
                break
            except Exception as e:
                if _es_rate_limit(e) and intento < MAX_RETRIES:
                    intento += 1
                    espera = BACKOFF_BASE * (2 ** intento) + random.uniform(0, 1)
                    print(f"[!] 429/RESOURCE_EXHAUSTED en lote {i // batch_size + 1}. "
                          f"Reintento {intento}/{MAX_RETRIES}, esperando {espera:.1f}s...")
                    time.sleep(espera)
                else:
                    raise
    return all_vecs


@app.post("/api/ingestar-web")
async def ingestar_web(req: IngestRequest):
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap debe ser menor que chunk_size")

    t0 = time.perf_counter()
    semaphore = asyncio.Semaphore(req.concurrency_workers)

    async with httpx.AsyncClient(
        headers={"User-Agent": "RAG-Lab/1.0 (pipeline de ingesta concurrente)"},
        limits=httpx.Limits(max_connections=req.concurrency_workers + 4),
    ) as client:
        tasks = [process_url(client, url, semaphore, req.chunk_size, req.chunk_overlap) for url in req.urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_chunks: List[Dict[str, Any]] = []
    urls_fallidas: List[Dict[str, str]] = []
    for url, r in zip(req.urls, results):
        if isinstance(r, Exception):
            urls_fallidas.append({"url": url, "error": f"{type(r).__name__}: {r}"})
        elif isinstance(r, dict):
            if r.get("error"):
                urls_fallidas.append({"url": r["url"], "error": r["error"]})
            else:
                all_chunks.extend(r["chunks"])

    t_scrape = time.perf_counter() - t0

    if not all_chunks:
        return {
            "coleccion_id": req.coleccion_id,
            "chunks_extraidos": 0,
            "insertados": 0,
            "urls_ok": len(req.urls) - len(urls_fallidas),
            "urls_fallidas": urls_fallidas,
            "duracion_ms": int((time.perf_counter() - t0) * 1000),
        }

    # embeddings (bloqueantes respecto al event loop -> en hilo del pool)
    t1 = time.perf_counter()
    textos = [c["contenido"] for c in all_chunks]
    try:
        vecs = await asyncio.to_thread(embed_chunks, textos, EMBED_BATCH_SIZE)
    except Exception as e:
        err = str(e)
        if _es_rate_limit(e):
            raise HTTPException(status_code=429, detail="Cuota de la API de embeddings agotada; reintente más tarde.")
        raise HTTPException(status_code=502, detail=f"Error generando embeddings: {err}")
    t_embed = time.perf_counter() - t1

    filas = []
    for c, v in zip(all_chunks, vecs):
        filas.append({
            "documento_origen": c["url"],
            "organismo": "Web",
            "tipo_norma": "Web",
            "jerarquia": "Web",
            "articulo_ref": c["url"][:80],
            "contenido": c["contenido"],
            "coleccion_id": req.coleccion_id,
            "embedding": v,
        })

    # reingesta idempotente: se eliminan los chunks previos de estas URLs en esta
    # colección para no duplicar vectores
    t2 = time.perf_counter()
    try:
        (
            supabase.table("normativa_bancaria")
            .delete()
            .in_("documento_origen", [c["url"] for c in all_chunks])
            .eq("coleccion_id", req.coleccion_id)
            .execute()
        )
        supabase.table("normativa_bancaria").insert(filas).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error insertando en Supabase: {e}")
    t_insert = time.perf_counter() - t2

    total = time.perf_counter() - t0
    return {
        "coleccion_id": req.coleccion_id,
        "chunks_extraidos": len(all_chunks),
        "insertados": len(filas),
        "urls_ok": len(req.urls) - len(urls_fallidas),
        "urls_fallidas": urls_fallidas,
        "concurrencia": req.concurrency_workers,
        "fases_ms": {
            "scraping_chunking": int(t_scrape * 1000),
            "embeddings": int(t_embed * 1000),
            "insercion_supabase": int(t_insert * 1000),
        },
        "duracion_ms": int(total * 1000),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
