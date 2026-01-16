from __future__ import annotations

import asyncpg

from db import get_pool as _get_pool
from db import init_pg as _init_pg


async def init_pg(dsn: str) -> None:
    await _init_pg(dsn)


def get_pool() -> asyncpg.Pool:
    return _get_pool()
