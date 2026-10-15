"""
Prueba de aislamiento multi-tenant (cero contaminación cruzada).

1. Inserta 2 colecciones de prueba con embeddings reales en normativa_bancaria.
2. Consulta la RPC match_normativa_coleccion por colección.
3. Verifica:
   - la colección A sólo devuelve filas de A,
   - la colección B sólo devuelve filas de B,
   - una colección inexistente devuelve 0 filas,
   - la RPC sin filtro (match_normativa) sí mezclaría ambas: demuestra por qué
     el filtro por coleccion_id es obligatorio.

Uso:
  python scripts/test_aislamiento.py            # limpia al finalizar
  python scripts/test_aislamiento.py --keep     # conserva las filas de prueba
"""
import os
import sys
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "models/gemini-embedding-001"

COL_A = "aislamiento_a"
COL_B = "aislamiento_b"

TEXTO_A = (
    "Resolucion administrativa de la Autoridad de Supervision del Sistema Financiero: "
    "las entidades de intermediacion financiera deben mantener la suficiencia patrimonial "
    "minima y reportar indicadores de riesgo crediticio mensualmente."
)
TEXTO_B = (
    "Codigo tributario pluri nacional: el impuesto a las transacciones aplica a toda "
    "persona natural o juridica que realice ventas de bienes muebles e inmuebles, "
    "servicios y suministros en el territorio nacional."
)


def embeber(texto: str):
    res = ai.models.embed_content(
        model=MODEL,
        contents=[texto],
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return res.embeddings[0].values


def rpc_coleccion(vec, coleccion: str, threshold: float = 0.0, count: int = 10):
    r = supabase.rpc("match_normativa_coleccion", {
        "query_embedding": vec,
        "p_coleccion_id": coleccion,
        "match_threshold": threshold,
        "match_count": count,
    }).execute()
    return r.data or []


def rpc_sin_filtro(vec, threshold: float = 0.0, count: int = 10):
    r = supabase.rpc("match_normativa", {
        "query_embedding": vec,
        "match_threshold": threshold,
        "match_count": count,
    }).execute()
    return r.data or []


def main() -> int:
    keep = "--keep" in sys.argv
    sufijo = uuid.uuid4().hex[:8]
    col_a = f"{COL_A}_{sufijo}"
    col_b = f"{COL_B}_{sufijo}"
    ok = True

    print("[*] Generando embeddings de prueba...")
    vec_a = embeber(TEXTO_A)
    vec_b = embeber(TEXTO_B)

    filas = [
        {"documento_origen": "test_aislamiento_A", "organismo": "TEST", "tipo_norma": "TEST",
         "jerarquia": "TEST", "articulo_ref": "A-1", "contenido": TEXTO_A,
         "coleccion_id": col_a, "embedding": vec_a},
        {"documento_origen": "test_aislamiento_B", "organismo": "TEST", "tipo_norma": "TEST",
         "jerarquia": "TEST", "articulo_ref": "B-1", "contenido": TEXTO_B,
         "coleccion_id": col_b, "embedding": vec_b},
    ]
    print(f"[*] Insertando filas de prueba ({col_a} / {col_b})...")
    supabase.table("normativa_bancaria").insert(filas).execute()

    try:
        print("\n[1] RPC coleccion A (esperado: SOLO filas de A)")
        res_a = rpc_coleccion(vec_a, col_a)
        for d in res_a:
            print(f"    {d.get('coleccion_id')} | {d.get('documento_origen')}")
        if not res_a:
            print("    [X] sin resultados"); ok = False
        if any(d.get("coleccion_id") != col_a for d in res_a):
            print("    [X] CONTAMINACION CRUZADA en A"); ok = False
        else:
            print("    [OK] aislamiento de A correcto")

        print("\n[2] RPC coleccion B con embedding de A (esperado: 0 filas de A)")
        res_b = rpc_coleccion(vec_a, col_b)
        for d in res_b:
            print(f"    {d.get('coleccion_id')} | {d.get('documento_origen')}")
        if any(d.get("documento_origen") == "test_aislamiento_A" for d in res_b):
            print("    [X] la coleccion B devolvio filas de A"); ok = False
        elif res_b:
            print("    [OK] B devolvio solo propias (traslape lexico <= umbral)")
        else:
            print("    [OK] B no devolvio filas de A (0 resultados)")

        print("\n[3] RPC coleccion B con embedding de B (esperado: SOLO filas de B)")
        res_b2 = rpc_coleccion(vec_b, col_b)
        for d in res_b2:
            print(f"    {d.get('coleccion_id')} | {d.get('documento_origen')}")
        if not res_b2 or any(d.get("coleccion_id") != col_b for d in res_b2):
            print("    [X] fallo de aislamiento en B"); ok = False
        else:
            print("    [OK] aislamiento de B correcto")

        print("\n[4] RPC coleccion inexistente (esperado: 0 filas)")
        res_v = rpc_coleccion(vec_a, f"no_existe_{sufijo}")
        if res_v:
            print(f"    [X] devolvio {len(res_v)} filas"); ok = False
        else:
            print("    [OK] 0 resultados")

        print("\n[5] RPC SIN filtro (match_normativa) con embedding de A "
              "(demuestra la contaminacion que evitamos)")
        res_mix = rpc_sin_filtro(vec_a)
        colecciones = sorted({d.get("coleccion_id") or "NULL" for d in res_mix})
        print(f"    colecciones mezcladas: {colecciones}")
        if len(colecciones) > 1:
            print("    [OK] sin filtro se mezclan dominios -> por eso el filtro es obligatorio")
        else:
            print("    [i] no se observaron dos colecciones en un solo resultado")

    finally:
        if not keep:
            print("\n[*] Limpiando filas de prueba...")
            supabase.table("normativa_bancaria").delete().in_(
                "coleccion_id", [col_a, col_b]
            ).execute()
        else:
            print(f"\n[*] Filas conservadas: {col_a} / {col_b}")

    print("\n" + ("=" * 60))
    print("RESULTADO: " + ("PASS" if ok else "FAIL"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
