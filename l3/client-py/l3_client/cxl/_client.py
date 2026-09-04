"""CAMA CXL client — TCP control plane + devdax direct load."""

from __future__ import annotations

import os
import struct
import threading
import time

from l3_client import protocol
from l3_client.client import CamaClient
from l3_client.cxl._constants import (
    DEFAULT_CXL_PORT,
    DEFAULT_DEVDAX_PATH,
    OP_CXL_REGION_MAP,
    OP_CXL_READ_READY,
    RESP_CXL_REGION_MAP,
    logger,
)


class CXLClient:
    """CAMA client with CXL direct memory access for reads.

    Uses TCP for the control plane (SET, DELETE, EXISTS, etc.) and
    loads values directly from CXL device memory via mmap for GET.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_CXL_PORT,
        password: str = "",
        *,
        devdax_path: str = DEFAULT_DEVDAX_PATH,
        devdax_size: int = 0,
        handshake: bool = True,
        timeout: float | None = None,
        reconnect: bool = False,
    ):
        self._host = host
        self._port = port
        self._devdax_path = devdax_path
        self._lock = threading.Lock()
        self._closed = False

        # TCP control connection (reuse CamaClient for wire protocol)
        self._tcp = CamaClient(host, port, password, handshake=handshake,
                               timeout=timeout, reconnect=reconnect)

        # CXL transport (mmap)
        self._transport = None
        self._region_map: list[dict] = []
        self._devdax_mapped = False

        # C4: TTL-based region map refresh
        self._region_map_ts: float = 0.0
        self._region_map_ttl: float = 5.0

        # Fetch region map and mmap device
        if handshake:
            self._fetch_region_map()
            if devdax_size > 0:
                self._map_device(devdax_size)

    def _fetch_region_map(self):
        """Request the CXL region map from the server."""
        try:
            resp = self._tcp._roundtrip(OP_CXL_REGION_MAP, b"")
            if resp.header.opcode == RESP_CXL_REGION_MAP and len(resp.body) >= 6:
                self._parse_region_map(resp.body)
                self._region_map_ts = time.monotonic()
        except Exception as e:
            logger.debug("[cxl] region map fetch failed: %s", e)

    def _maybe_refresh_region_map(self):
        """Re-fetch the region map if the TTL has expired."""
        if time.monotonic() - self._region_map_ts > self._region_map_ttl:
            self._fetch_region_map()

    def _parse_region_map(self, body: bytes):
        """Parse a CXL region map response."""
        off = 0
        path_len = struct.unpack_from("<H", body, off)[0]
        off += 2
        devdax_path = body[off:off + path_len].decode("utf-8")
        off += path_len
        n = struct.unpack_from("<I", body, off)[0]
        off += 4

        # C5: validate server-supplied devdax path
        if devdax_path:
            real = os.path.realpath(devdax_path)
            if not real.startswith("/dev/dax"):
                logger.warning(
                    "[cxl] server-supplied devdax path %r resolves to %r "
                    "(not under /dev/dax) — ignoring, using constructor arg %r",
                    devdax_path, real, self._devdax_path,
                )
                devdax_path = ""

        entries = []
        for _ in range(n):
            shard_id, class_idx = struct.unpack_from("<HH", body, off)
            off += 4
            offset, size = struct.unpack_from("<QQ", body, off)
            off += 16
            entries.append({
                "shard_id": shard_id,
                "class_idx": class_idx,
                "offset": offset,
                "size": size,
            })

        self._region_map = entries
        if devdax_path and not self._devdax_mapped:
            self._devdax_path = devdax_path
            # Auto-compute size from region map
            if entries:
                max_end = max(e["offset"] + e["size"] for e in entries)
                self._map_device(max_end)

    def _map_device(self, size: int):
        """Open and mmap the devdax device."""
        if self._devdax_mapped:
            return
        try:
            from l3_client._l3_cxl import CXLTransport
            self._transport = CXLTransport()
            self._transport.open(self._devdax_path, size)
            self._devdax_mapped = True
            logger.info("[cxl] mapped %s (%d MB)", self._devdax_path, size // (1024 * 1024))
        except ImportError:
            # M13: ensure no partial transport object
            self._transport = None
            logger.warning("[cxl] _cama_cxl extension not built — CXL reads will fall back to TCP")
        except Exception as e:
            # M13: ensure no partial transport object
            self._transport = None
            logger.warning("[cxl] failed to mmap %s: %s", self._devdax_path, e)

    def get(self, key: str, sgl=None, size: int = 0):
        """GET a value. Uses CXL direct load when possible, TCP fallback otherwise.

        When *sgl* is provided, copies into the SGL and returns 0 (found) or
        -1 (miss), matching the CamaClient/RDMAClient API contract.
        When *sgl* is None, returns raw bytes or None.
        """
        body = protocol.encode_key_body(key.encode() if isinstance(key, str) else key)
        with self._lock:
            # C3: check closed state under lock
            if self._closed:
                raise RuntimeError("CXLClient is closed")
            # C4: refresh region map if TTL expired
            self._maybe_refresh_region_map()
            resp = self._tcp._roundtrip(protocol.OP_GET, body)
            # C3: snapshot transport under lock to prevent use-after-free
            transport = self._transport

        opcode = resp.header.opcode
        resp_body = resp.body

        value: bytes | None = None

        if opcode == OP_CXL_READ_READY and transport and len(resp_body) >= 13:
            if resp_body[0] == 0:
                # Not found
                if sgl is not None:
                    return -1
                return None
            device_offset = struct.unpack_from("<Q", resp_body, 1)[0]
            value_size = struct.unpack_from("<I", resp_body, 9)[0]
            # M14: catch transport.load() exceptions — fall through to miss
            try:
                value = transport.load(device_offset, value_size)
            except Exception as e:
                logger.warning("[cxl] load failed (offset=%d, size=%d): %s",
                               device_offset, value_size, e)
                if sgl is not None:
                    return -1
                return None
        elif opcode == protocol.RESP_VALUE:
            # Inline value response (TCP fallback or small value)
            value, found = protocol.decode_value_response(resp_body)
            if not found:
                if sgl is not None:
                    return -1
                return None
        else:
            if sgl is not None:
                return -1
            return None

        if sgl is not None:
            sgl.from_bytes(value)
            return 0
        return value

    def set(self, key: str, sgl=None, ttl_ms: int = 0, *, value: bytes | None = None) -> int:
        """SET a key-value pair (always via TCP).

        Accepts either an SGL or raw *value* bytes.
        """
        if sgl is not None:
            return self._tcp.set(key, sgl, ttl_ms)
        if value is not None:
            # Build the body manually for raw bytes
            from l3_client.sgl import SGL
            s = SGL(value)
            return self._tcp.set(key, s, ttl_ms)
        raise ValueError("either sgl or value= must be provided")

    def delete(self, key: str) -> int:
        """DELETE a key."""
        return self._tcp.delete(key)

    def exists(self, key: str) -> int:
        """Check if a key exists."""
        return self._tcp.exists(key)

    def mget(self, keys: list[str], sgls: list = None) -> list:
        """Batch GET. Sequential CXL gets for each key."""
        if sgls is not None:
            results = []
            for k, s in zip(keys, sgls):
                results.append(self.get(k, s))
            return results
        return [self.get(k) for k in keys]

    def mset(self, keys: list[str], sgls: list, ttl_ms: int = 0) -> list[int]:
        """Batch SET (TCP)."""
        return self._tcp.mset(keys, sgls, ttl_ms)

    def mexists(self, keys: list[str]) -> list[int]:
        """Batch EXISTS."""
        return self._tcp.mexists(keys)

    def mdel(self, keys: list[str]) -> list[int]:
        """Batch DELETE."""
        return self._tcp.mdel(keys)

    def flush(self):
        """Flush all keys."""
        return self._tcp.flush()

    def info(self) -> dict:
        """Server info."""
        return self._tcp.info()

    def stats(self) -> dict:
        """Server stats."""
        return self._tcp.stats()

    def report_stats(self, stats: dict) -> None:
        """Report client-side stats to the server."""
        stats = dict(stats)
        stats["transport"] = "cxl"
        body = __import__("json").dumps(stats).encode()
        self._tcp._roundtrip(protocol.OP_REPORT_STATS, body)

    # --- RDMA memory registration (no-ops in CXL mode) ---

    def reg_memory(self, ptr, size, buf=None):
        """No-op: CXL mode does not require memory registration."""
        return 0

    def dereg_memory(self, ptr):
        """No-op."""
        pass

    def close(self):
        """Close CXL transport and TCP connection."""
        # C3: acquire lock so concurrent get() sees _closed=True
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._transport:
                self._transport.close()
                self._transport = None
        self._tcp.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"CXLClient({self._host}:{self._port}, devdax={self._devdax_path})"
