"""
Aplica scripts/setup_database.sql a un Postgres (Supabase) mediante conexión directa.

Requiere la cadena de conexión en .env:
  SUPABASE_DB_URL=postgresql://postgres:<DB_PASSWORD>@db.<ref>.supabase.co:5432/postgres

La DB_PASSWORD está en Supabase -> Settings -> Database.
Alternativa sin password: pegar el contenido de scripts/setup_database.sql en
Supabase -> SQL Editor.

Uso:
  python scripts/apply_schema.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("SUPABASE_DB_URL")
SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_database.sql")


def main() -> int:
    if not URL:
        print("[X] Falta SUPABASE_DB_URL en .env")
        print("    Formato: postgresql://postgres:<DB_PASSWORD>@db.<ref>.supabase.co:5432/postgres")
        print("    O ejecuta scripts/setup_database.sql en Supabase -> SQL Editor.")
        return 1
    try:
        import psycopg
    except ImportError:
        print("[X] Falta psycopg. Instala: pip install 'psycopg[binary]'")
        return 1

    with open(SQL) as f:
        sql = f.read()

    print("[*] Aplicando esquema...")
    with psycopg.connect(URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("[✓] Esquema aplicado (tabla, índices y RPC match_normativa_coleccion).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
