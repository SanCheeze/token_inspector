from __future__ import annotations

from pathlib import Path

import asyncpg

_pool: asyncpg.Pool | None = None


async def init_pg(dsn: str):
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=dsn)
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        async with _pool.acquire() as conn:
            await conn.execute(schema)


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("PostgreSQL not initialized")
    return _pool
