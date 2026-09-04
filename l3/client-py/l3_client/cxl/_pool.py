"""CAMA CXL client pool — round-robin across N connections."""

from __future__ import annotations

import threading

from l3_client.cxl._client import CXLClient
from l3_client.cxl._constants import DEFAULT_CXL_PORT, DEFAULT_DEVDAX_PATH, logger


class CXLClientPool:
    """Pool of N CXLClient connections with round-robin dispatch."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_CXL_PORT,
        password: str = "",
        *,
        pool_size: int = 8,
        devdax_path: str = DEFAULT_DEVDAX_PATH,
        devdax_size: int = 0,
        **kwargs,
    ):
        self._pool: list[CXLClient] = []
        self._idx = 0
        self._lock = threading.Lock()

        for _ in range(pool_size):
            c = CXLClient(host, port, password, devdax_path=devdax_path,
                          devdax_size=devdax_size, **kwargs)
            self._pool.append(c)

        logger.info("[cxl] pool created: size=%d, devdax=%s", pool_size, devdax_path)

    def _next(self) -> CXLClient:
        with self._lock:
            c = self._pool[self._idx % len(self._pool)]
            self._idx += 1
            return c

    def get(self, key: str, sgl=None, size: int = 0):
        return self._next().get(key, sgl, size)

    def set(self, key: str, sgl=None, ttl_ms: int = 0, **kwargs) -> int:
        return self._next().set(key, sgl, ttl_ms, **kwargs)

    def delete(self, key: str) -> int:
        return self._next().delete(key)

    def exists(self, key: str) -> int:
        return self._next().exists(key)

    def mget(self, keys: list[str], sgls: list = None) -> list:
        return self._next().mget(keys, sgls)

    def mset(self, keys: list[str], sgls: list, ttl_ms: int = 0) -> list[int]:
        return self._next().mset(keys, sgls, ttl_ms)

    def mexists(self, keys: list[str]) -> list[int]:
        return self._next().mexists(keys)

    def mdel(self, keys: list[str]) -> list[int]:
        return self._next().mdel(keys)

    def flush(self):
        return self._pool[0].flush()

    def info(self) -> dict:
        return self._pool[0].info()

    def stats(self) -> dict:
        return self._pool[0].stats()

    def report_stats(self, stats: dict) -> None:
        return self._pool[0].report_stats(stats)

    def reg_memory(self, ptr, size, buf=None):
        """No-op: CXL mode does not require memory registration."""
        return 0

    def dereg_memory(self, ptr):
        """No-op."""
        pass

    def close(self):
        for c in self._pool:
            c.close()
        self._pool.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
