-- Inicialización de la base de datos RAG (Supabase + pgvector, 768 dimensiones)
-- Ejecutar en: Supabase Dashboard -> SQL Editor

create extension if not exists vector;

-- Tabla de normativa indexada
create table if not exists public.normativa_bancaria (
    id uuid primary key default gen_random_uuid(),
    documento_origen text not null,
    organismo text,
    tipo_norma text,
    jerarquia text,
    articulo_ref text,
    contenido text not null,
    coleccion_id varchar(50),
    embedding vector(768),
    created_at timestamptz default now()
);

-- Índices
create index if not exists normativa_bancaria_embedding_idx
    on public.normativa_bancaria
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Requerimiento 1: columna coleccion_id indexada con B-Tree (idempotente para
-- tablas creadas antes de esta práctica)
alter table public.normativa_bancaria
    add column if not exists coleccion_id varchar(50);

create index if not exists normativa_bancaria_coleccion_id_idx
    on public.normativa_bancaria using btree (coleccion_id);

-- Función RPC de búsqueda semántica (compatible con versión anterior)
create or replace function public.match_normativa(
    query_embedding vector(768),
    match_threshold float default 0.35,
    match_count int default 4
)
returns table (
    id uuid,
    documento_origen text,
    articulo_ref text,
    contenido text,
    organismo text,
    tipo_norma text,
    jerarquia text,
    coleccion_id varchar(50),
    similarity float
)
language sql stable
as $$
    select
        n.id,
        n.documento_origen,
        n.articulo_ref,
        n.contenido,
        n.organismo,
        n.tipo_norma,
        n.jerarquia,
        n.coleccion_id,
        1 - (n.embedding <=> query_embedding) as similarity
    from public.normativa_bancaria n
    where n.embedding is not null
      and 1 - (n.embedding <=> query_embedding) > match_threshold
    order by n.embedding <=> query_embedding
    limit match_count;
$$;

-- Función RPC con aislamiento por colección (Requerimiento 1, PL/pgSQL)
-- Nota: el parámetro de entrada se llama p_coleccion_id para no colisionar con la
-- columna coleccion_id de RETURNS TABLE (PostgreSQL error 42725).
create or replace function public.match_normativa_coleccion(
    query_embedding vector(768),
    p_coleccion_id varchar(50),
    match_threshold float default 0.35,
    match_count int default 4
)
returns table (
    id uuid,
    documento_origen text,
    articulo_ref text,
    contenido text,
    organismo text,
    tipo_norma text,
    jerarquia text,
    coleccion_id varchar(50),
    similarity float
)
language plpgsql stable
as $$
begin
    return query
    select
        n.id,
        n.documento_origen,
        n.articulo_ref,
        n.contenido,
        n.organismo,
        n.tipo_norma,
        n.jerarquia,
        n.coleccion_id,
        1 - (n.embedding <=> query_embedding) as similarity
    from public.normativa_bancaria n
    where n.embedding is not null
      and n.coleccion_id = p_coleccion_id
      and 1 - (n.embedding <=> query_embedding) > match_threshold
    order by n.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Migración de datos legados: las filas antiguas (sin colección) pasan a la
-- colección por defecto para que sigan siendo consultables de forma aislada
update public.normativa_bancaria
   set coleccion_id = 'bancaria_asfi_bolivia'
 where coleccion_id is null;

-- Habilitar RLS y permitir lectura/escritura con service_role (se omite si el
-- rol no existe, p. ej. en un Postgres local sin Supabase)
alter table public.normativa_bancaria enable row level security;

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'service_role') then
        if not exists (
            select 1 from pg_policies
            where tablename = 'normativa_bancaria'
              and policyname = 'service role full access on normativa_bancaria'
        ) then
            create policy "service role full access on normativa_bancaria"
                on public.normativa_bancaria
                for all
                to service_role
                using (true)
                with check (true);
        end if;
    end if;
end;
$$;
