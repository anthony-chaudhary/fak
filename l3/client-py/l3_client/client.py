"""CAMA client implementing the PrisKV API contract over TCP.

Usage (drop-in replacement for priskv):
    from l3_client.client import L3Client as PriskvClient
    client = PriskvClient("127.0.0.1", 18000)
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING

from l3_client import protocol
from l3_client.errors import CamaNotReadyError, CamaOOMError, CamaServerOverloadError
from l3_client.handshake import perform_handshake
from l3_client.reconnect import (
    ReconnectCallbackRegistry,
    ReconnectConfig,
    _resolve_reconnect,
    compute_delay,
    is_retriable,
)
from l3_client.sgl import SGL

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class L3Client:
    """PrisKV-compatible client backed by CAMA over TCP.

    Notes on semantic mapping:
    - reg_memory / dereg_memory are no-ops (no RDMA in TCP mode).
    - SGL-based set/get copy through ctypes rather than zero-copy DMA.
    - mset/mget/mexists/mdel use native batch wire-protocol opcodes.
    """

    def __init__(self, addr: str = "127.0.0.1", port: int = 18000, password: str = "", *, handshake: bool = True, timeout: float | None = None, reconnect: bool | ReconnectConfig | None = True):
        """Connect to CAMA server. Password is ignored (no auth).

        Args:
            timeout: Socket timeout in seconds. None = blocking (default).
        """
        # Store connection params for reconnect
        self._addr = addr
        self._port = port
        self._password = password
        self._handshake_enabled = handshake
        self._timeout_value = timeout

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((addr, port))
        if timeout is not None:
            self._sock.settimeout(timeout)
        self._reader = self._sock.makefile("rb", buffering=65536)
        self._lock = threading.Lock()
        self._req_id = 0
        self._server_info: dict | None = None

        # Reconnect state
        self._reconnect_config = _resolve_reconnect(reconnect)
        self._callbacks = ReconnectCallbackRegistry()
        self._alive = True

        if handshake:
            self._server_info = perform_handshake(self._roundtrip, "tcp")

    def close(self) -> None:
        self._reader.close()
        self._sock.close()

    def set_timeout(self, seconds: float | None) -> None:
        """Set socket timeout for all subsequent operations.

        Args:
            seconds: Timeout in seconds. None = blocking.
        """
        self._timeout_value = seconds
        self._sock.settimeout(seconds)

    def __del__(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _roundtrip(self, opcode: int, body: bytes, flags: int = 0) -> protocol.Message:
        """Send a request and read the response (thread-safe).

        If reconnection is enabled, retriable transport errors trigger
        automatic reconnect with exponential backoff.  The reconnect
        happens atomically within the lock scope.
        """
        fire_callbacks = False
        resp = None
        with self._lock:
            try:
                return self._roundtrip_inner(opcode, body, flags)
            except Exception as exc:
                rc = self._reconnect_config
                if rc is None or not rc.enabled or not is_retriable(exc):
                    raise
                # Reconnect loop
                for attempt in range(rc.max_retries):
                    self._alive = False
                    # Close dead socket (best-effort)
                    try:
                        self._reader.close()
                    except Exception:
                        pass
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    delay = compute_delay(attempt, rc)
                    logger.warning(
                        "TCP reconnect attempt %d/%d to %s:%d (delay=%.2fs): %s",
                        attempt + 1, rc.max_retries,
                        self._addr, self._port, delay, exc,
                    )
                    time.sleep(delay)
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        sock.connect((self._addr, self._port))
                        if self._timeout_value is not None:
                            sock.settimeout(self._timeout_value)
                        self._sock = sock
                        self._reader = self._sock.makefile("rb", buffering=65536)
                        self._req_id = 0
                        if self._handshake_enabled:
                            self._server_info = perform_handshake(
                                self._roundtrip_inner, "tcp")
                        self._alive = True
                        # Retry the original operation
                        resp = self._roundtrip_inner(opcode, body, flags)
                        logger.info("TCP reconnect succeeded on attempt %d", attempt + 1)
                        fire_callbacks = True
                        break
                    except Exception as retry_exc:
                        exc = retry_exc
                        if not is_retriable(retry_exc):
                            raise
                else:
                    # Exhausted retries
                    raise exc
        if fire_callbacks:
            self._callbacks.fire_all()
        return resp

    def _roundtrip_inner(self, opcode: int, body: bytes, flags: int = 0) -> protocol.Message:
        """Raw roundtrip without retry logic. Lock must be held by caller."""
        req_id = self._next_req_id()
        protocol.write_message(self._sock, opcode, body, flags=flags, request_id=req_id)
        resp = protocol.read_message_buffered(self._reader)
        if resp.header.opcode == protocol.RESP_OOM:
            raise CamaOOMError.from_body(resp.body)
        if resp.header.opcode == protocol.RESP_NOT_READY:
            raise CamaNotReadyError.from_body(resp.body)
        if resp.header.opcode == protocol.RESP_ERROR:
            body_str = resp.body.decode(errors='replace')
            if "server overloaded" in body_str:
                raise CamaServerOverloadError(body_str)
            raise RuntimeError(f"CAMA error: {body_str}")
        return resp

    # --- Reconnect control ---

    def enable_reconnect(self, config: ReconnectConfig) -> None:
        """Enable automatic reconnection with the given config."""
        self._reconnect_config = config

    def disable_reconnect(self) -> None:
        """Disable automatic reconnection."""
        self._reconnect_config = None

    def set_reconnect_callback(self, name: str, fn) -> None:
        """Register a post-reconnect callback."""
        self._callbacks.register(name, fn)

    # --- Info ---

    def info(self) -> dict:
        """Send OP_INFO and return the parsed JSON server metadata."""
        resp = self._roundtrip(protocol.OP_INFO, b"")
        return json.loads(resp.body)

    def stats(self) -> dict:
        """Fetch server bandwidth and connection statistics."""
        resp = self._roundtrip(protocol.OP_STATS, b"")
        return json.loads(resp.body)

    def report_stats(self, stats: dict) -> None:
        """Report client-side stats to the server for Prometheus exposure."""
        stats = dict(stats)
        stats["transport"] = "tcp"
        body = json.dumps(stats).encode()
        self._roundtrip(protocol.OP_REPORT_STATS, body)

    def rdma_endpoints(self) -> list[dict]:
        """Return RDMA endpoints from the server (each has device, ip, port)."""
        if self._server_info and "rdma_endpoints" in self._server_info:
            return self._server_info["rdma_endpoints"] or []
        info = self.info()
        return info.get("rdma_endpoints") or []

    # --- RDMA memory registration (no-ops in TCP mode) ---

    def reg_memory(self, ptr: int, size: int) -> int:
        """Register memory for RDMA. Returns dummy handle 1 in TCP mode."""
        return 1

    def dereg_memory(self, handle: int) -> None:
        """Deregister RDMA memory. No-op in TCP mode."""
        pass

    # --- Basic string ops (PrisKV compat) ---

    def setstr(self, key: str, value: str) -> int:
        """Store a string key-value pair. Returns 0 on success."""
        body, flags = protocol.encode_kv_body(key.encode(), value.encode())
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def getstr(self, key: str) -> str | None:
        """Retrieve a string value. Returns None if not found."""
        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_GET, body)
        value, found = protocol.decode_value_response(resp.body)
        if not found:
            return None
        return value.decode(errors="replace")

    def exists(self, key: str) -> int:
        """Check if key exists. Returns 1 if found, 0 otherwise."""
        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_TEST, body)
        if len(resp.body) > 0 and resp.body[0] == 1:
            return 1
        return 0

    def delete(self, key: str) -> int:
        """Delete a key. Returns 0 on success."""
        body = protocol.encode_key_body(key.encode())
        self._roundtrip(protocol.OP_DELETE, body)
        return 0

    # --- SGL-based ops (zero-copy in PrisKV, memcpy in TCP mode) ---

    def set(self, key: str, sgl: SGL, ttl_ms: int = 0) -> int:
        """Store value from SGL. Copies from host pointer via TCP. Returns 0 on success."""
        value = sgl.to_bytes()
        body, flags = protocol.encode_kv_body(key.encode(), value, ttl_ms=ttl_ms)
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def get(self, key: str, sgl: SGL, size: int = 0) -> int:
        """Retrieve value into SGL. Copies to host pointer from TCP. Returns 0 on success, -1 if not found."""
        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_GET, body)
        value, found = protocol.decode_value_response(resp.body)
        if not found:
            return -1
        sgl.from_bytes(value)
        return 0

    # --- Batch ops (native wire-protocol batch messages) ---

    def mexists(self, keys: list[str]) -> list[int]:
        """Batch existence check — single MTEST roundtrip.

        Returns list of 1 (found) / 0 (missing) per key.
        """
        if not keys:
            return []
        body = protocol.encode_mget_body([k.encode() for k in keys])
        resp = self._roundtrip(protocol.OP_MTEST, body)
        founds = protocol.decode_mtest_founds(resp.body)
        return [1 if f else 0 for f in founds]

    def _parse_mset_response(self, resp: protocol.Message, count: int) -> list[int]:
        """Parse an MSET response, returning per-key status list."""
        if resp.header.opcode == protocol.RESP_MSET_RESULT:
            return protocol.decode_mset_result(resp.body)
        return [0] * count  # RespOK: all succeeded

    def mset(self, keys: list[str], sgls: list[SGL], ttl_ms: int = 0) -> list[int]:
        """Batch set via native MSET with sub-batch chunking.

        When the full payload exceeds the socket send buffer limit,
        partitions into sub-batches.  Returns list of per-key status
        codes (0=ok, nonzero=error).
        """
        if not keys:
            return []
        values = [sgl.to_bytes() for sgl in sgls]
        encoded_keys = [k.encode() for k in keys]
        body = protocol.encode_mset_body(encoded_keys, values)
        flags = protocol.FLAG_WITH_TTL if ttl_ms > 0 else protocol.FLAG_NONE

        # TCP has no hard send-buffer limit like RDMA, but chunk at 16MB
        # for consistency and to avoid huge single messages
        max_body = getattr(self, '_send_buf_size', None) or 16 * 1024 * 1024
        max_body -= protocol.HEADER_SIZE

        if len(body) <= max_body:
            resp = self._roundtrip(protocol.OP_MSET, body, flags=flags)
            return self._parse_mset_response(resp, len(keys))

        # Chunked: partition into sub-batches
        all_results: list[int] = [0] * len(keys)
        chunk_keys: list[bytes] = []
        chunk_vals: list[bytes] = []
        chunk_indices: list[int] = []
        chunk_size = 4  # 4 bytes for count prefix
        n_chunks = 0
        n_oversized = 0

        for i in range(len(encoded_keys)):
            entry_size = 2 + len(encoded_keys[i]) + 4 + len(values[i])

            if chunk_size + entry_size > max_body and chunk_keys:
                sub_body = protocol.encode_mset_body(chunk_keys, chunk_vals)
                resp = self._roundtrip(protocol.OP_MSET, sub_body, flags=flags)
                chunk_statuses = self._parse_mset_response(resp, len(chunk_keys))
                for j, orig_idx in enumerate(chunk_indices):
                    all_results[orig_idx] = chunk_statuses[j]
                n_chunks += 1
                chunk_keys, chunk_vals, chunk_indices, chunk_size = [], [], [], 4

            if 4 + entry_size > max_body:
                # Single entry exceeds buffer — individual set()
                n_oversized += 1
                all_results[i] = self.set(keys[i], sgls[i], ttl_ms=ttl_ms)
                continue

            chunk_keys.append(encoded_keys[i])
            chunk_vals.append(values[i])
            chunk_indices.append(i)
            chunk_size += entry_size

        if chunk_keys:
            sub_body = protocol.encode_mset_body(chunk_keys, chunk_vals)
            resp = self._roundtrip(protocol.OP_MSET, sub_body, flags=flags)
            chunk_statuses = self._parse_mset_response(resp, len(chunk_keys))
            for j, orig_idx in enumerate(chunk_indices):
                all_results[orig_idx] = chunk_statuses[j]
            n_chunks += 1

        total_batches = n_chunks + n_oversized
        if n_oversized > 0:
            logger.warning(
                "mset: %d/%d entries exceed send buffer (%d MB) — fell back to "
                "individual SET ops. Increase send_buf_size to batch them.",
                n_oversized, len(keys), max_body // (1024 * 1024),
            )
        elif total_batches >= len(keys):
            avg_val = sum(len(v) for v in values) // len(values)
            logger.warning(
                "mset: %d keys chunked into %d batches (~1 key/batch, avg value "
                "%d bytes vs %d MB send buffer). Increase send_buf_size for "
                "better batching.",
                len(keys), total_batches, avg_val, max_body // (1024 * 1024),
            )
        return all_results

    def mget(self, keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]:
        """Batch get via native MGET. Returns list of return codes (0=found, -1=miss)."""
        if not keys:
            return []
        body = protocol.encode_mget_body([k.encode() for k in keys])
        resp = self._roundtrip(protocol.OP_MGET, body)
        values, founds = protocol.decode_multi_value_response(resp.body)
        results = []
        for i, (val, found) in enumerate(zip(values, founds)):
            if not found:
                results.append(-1)
            else:
                sgls[i].from_bytes(val)
                results.append(0)
        return results

    def mdel(self, keys: list[str]) -> int:
        """Batch delete — single MDEL roundtrip. Returns 0 on success."""
        if not keys:
            return 0
        body = protocol.encode_mget_body([k.encode() for k in keys])
        self._roundtrip(protocol.OP_MDEL, body)
        return 0

    # --- Key scan ---

    def keys(self, pattern: str = ".*") -> list[str]:
        """Return keys matching regex pattern.

        Note: This scans all shards and may be slow for large datasets.
        In the TCP shim, returns [] as a safe default since this is only
        used by clear() which is non-critical.
        """
        body = protocol.encode_keys_body(pattern.encode())
        resp = self._roundtrip(protocol.OP_KEYS, body)
        raw_keys = protocol.decode_keys_response(resp.body)
        return [k.decode(errors="replace") for k in raw_keys]

    # --- Maintenance API ---

    def vacuum(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        """Trigger an on-demand vacuum evaluation/rebalance.

        Args:
            force: Bypass health checks and rebalance even healthy shards.
            shard_ids: Target specific shards. None = all shards.
        """
        req: dict = {"action": "vacuum", "force": force}
        if shard_ids is not None:
            req["shard_ids"] = shard_ids
        resp = self._roundtrip(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    def autotune(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        """Trigger on-demand auto-tune detection + rebuild.

        Args:
            force: Force early detection even if warmup hasn't completed.
            shard_ids: Target specific shards. None = all shards.
        """
        req: dict = {"action": "autotune", "force": force}
        if shard_ids is not None:
            req["shard_ids"] = shard_ids
        resp = self._roundtrip(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    def maintenance_status(self) -> dict:
        """Query vacuum and auto-tune status without triggering any action."""
        req = {"action": "status"}
        resp = self._roundtrip(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    # --- Flush ---

    def flush(self) -> int:
        """Flush all data from the cache. Returns 0 on success."""
        self._roundtrip(protocol.OP_FLUSH, b"")
        return 0

    # --- Lease/Pin (PrisKV compat) ---

    def lease(self, key: str, duration_ms: int) -> int:
        """Grant a lease protecting key from eviction. Returns 0 on success."""
        import struct
        body = protocol.encode_key_body(key.encode())
        # Extend body with duration
        body = struct.pack("<H", len(key.encode())) + key.encode() + struct.pack("<q", duration_ms)
        self._roundtrip(protocol.OP_LEASE, body)
        return 0

    def pin(self, key: str) -> int:
        """Pin key (permanent eviction protection). Returns 0 on success."""
        body = protocol.encode_key_body(key.encode())
        self._roundtrip(protocol.OP_PIN, body)
        return 0

    def unpin(self, key: str) -> int:
        """Unpin key. Returns 0 on success."""
        body = protocol.encode_key_body(key.encode())
        self._roundtrip(protocol.OP_UNPIN, body)
        return 0

    # --- Cluster ---

    def cluster_info(self) -> dict:
        """Return cluster membership info. Returns dict with nodes, status, etc."""
        resp = self._roundtrip(protocol.OP_CLUSTER, b"")
        return json.loads(resp.body) if resp.body else {}

    # --- Snapshot/Restore ---

    def snapshot(self, dir: str = "") -> dict:
        """Trigger server-side snapshot. Returns stats dict."""
        body = json.dumps({"dir": dir}).encode() if dir else b""
        resp = self._roundtrip(protocol.OP_SNAPSHOT, body)
        return json.loads(resp.body) if resp.body else {}

    def restore(self, dir: str = "") -> dict:
        """Trigger server-side restore from snapshot. Returns stats dict."""
        body = json.dumps({"dir": dir}).encode() if dir else b""
        resp = self._roundtrip(protocol.OP_RESTORE, body)
        return json.loads(resp.body) if resp.body else {}


class L3ClientPool:
    """Pool of N independent TCP connections. Drop-in for CamaClient.

    Round-robin distributes operations across connections for N-way
    parallelism (each CamaClient already has its own lock/socket).
    """

    def __init__(self, addr: str = "127.0.0.1", port: int = 18000,
                 password: str = "", *, pool_size: int = 8,
                 handshake: bool = True, timeout: float | None = None,
                 reconnect: bool | ReconnectConfig | None = True,
                 **kwargs):
        self._pool_size = pool_size
        self._timeout = timeout

        # Create pool_size independent TCP clients.
        # Handshake only on first client — clients 1..N-1 don't need
        # _server_info since admin ops (info/stats/rdma_endpoints) always
        # route through client[0].  This also applies after reconnect.
        clients = []
        for i in range(pool_size):
            c = CamaClient(addr, port, password,
                           handshake=(handshake and i == 0),
                           timeout=timeout, reconnect=reconnect)
            clients.append(c)
        self._clients = clients
        self._rr_lock = threading.Lock()
        self._rr_idx = 0
        self._server_info = getattr(clients[0], "_server_info", None)
        if pool_size > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._stripe_write_executor = ThreadPoolExecutor(
                max_workers=pool_size, thread_name_prefix="cama-tcp-stripe")
        else:
            self._stripe_write_executor = None

    def _next_client(self) -> CamaClient:
        with self._rr_lock:
            idx = self._rr_idx % self._pool_size
            self._rr_idx += 1
            # If selected client is dead, try to find a live one
            start_idx = idx
            while not self._clients[idx]._alive:
                idx = (idx + 1) % self._pool_size
                if idx == start_idx:
                    break  # all dead, fall through to first (will block on reconnect)
        return self._clients[idx]

    def close(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_timeout(self, seconds: float | None) -> None:
        self._timeout = seconds
        for c in self._clients:
            c.set_timeout(seconds)

    # --- Reconnect control (proxy to all clients) ---

    def enable_reconnect(self, config: ReconnectConfig) -> None:
        for c in self._clients:
            c.enable_reconnect(config)

    def disable_reconnect(self) -> None:
        for c in self._clients:
            c.disable_reconnect()

    def set_reconnect_callback(self, name: str, fn) -> None:
        for c in self._clients:
            c.set_reconnect_callback(name, fn)

    # --- No-op RDMA registration (same as CamaClient) ---

    def reg_memory(self, ptr: int, size: int) -> int:
        return 1

    def dereg_memory(self, handle: int) -> None:
        pass

    # --- Ops delegated via round-robin ---

    def setstr(self, key: str, value: str) -> int:
        return self._next_client().setstr(key, value)

    def getstr(self, key: str) -> str | None:
        return self._next_client().getstr(key)

    def exists(self, key: str) -> int:
        return self._next_client().exists(key)

    def delete(self, key: str) -> int:
        return self._next_client().delete(key)

    def set(self, key: str, sgl: SGL, ttl_ms: int = 0) -> int:
        return self._next_client().set(key, sgl, ttl_ms=ttl_ms)

    def get(self, key: str, sgl: SGL, size: int = 0) -> int:
        return self._next_client().get(key, sgl, size)

    def mexists(self, keys: list[str]) -> list[int]:
        return self._next_client().mexists(keys)

    def mset(self, keys: list[str], sgls: list[SGL], ttl_ms: int = 0) -> list[int]:
        return self._next_client().mset(keys, sgls, ttl_ms=ttl_ms)

    def mset_striped(self, keys: list[str], sgls: list[SGL],
                     ttl_ms: int = 0) -> list[int]:
        """Batch SET striped across pool connections in parallel.

        When pool_size > 1, partitions keys round-robin and writes each
        partition on a separate connection concurrently.
        Returns per-key status list (0=ok, nonzero=error).
        """
        if not keys:
            return []
        if self._pool_size <= 1 or self._stripe_write_executor is None:
            return self._clients[0].mset(keys, sgls, ttl_ms=ttl_ms)

        from concurrent.futures import as_completed

        n = len(keys)
        ps = self._pool_size

        # Phase 1: Partition round-robin
        part_indices: list[list[int]] = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)

        # Redistribute dead-client partitions to alive clients
        alive_clients = [i for i in range(ps) if self._clients[i]._alive]
        if not alive_clients:
            raise RuntimeError("all TCP pool clients are dead")
        for ci in range(ps):
            if not self._clients[ci]._alive and part_indices[ci]:
                for j, idx in enumerate(part_indices[ci]):
                    target = alive_clients[j % len(alive_clients)]
                    part_indices[target].append(idx)
                part_indices[ci] = []

        # Phase 2: Submit per-partition mset
        futures = {}
        for ci, indices in enumerate(part_indices):
            if not indices:
                continue
            p_keys = [keys[j] for j in indices]
            p_sgls = [sgls[j] for j in indices]
            fut = self._stripe_write_executor.submit(
                self._clients[ci].mset, p_keys, p_sgls, ttl_ms)
            futures[fut] = (ci, indices)

        # Phase 3: Collect and merge per-key results
        all_results = [0] * n
        timeout = self._timeout if self._timeout is not None else 30.0
        for fut in as_completed(futures, timeout=timeout):
            ci, indices = futures[fut]
            part_results = fut.result()
            for j, orig_idx in enumerate(indices):
                all_results[orig_idx] = part_results[j]

        return all_results

    def mget(self, keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]:
        return self._next_client().mget(keys, sgls, sizes)

    def mdel(self, keys: list[str]) -> int:
        return self._next_client().mdel(keys)

    def keys(self, pattern: str = ".*") -> list[str]:
        return self._clients[0].keys(pattern)

    # --- Admin ops always on first client ---

    def info(self) -> dict:
        return self._clients[0].info()

    def stats(self) -> dict:
        return self._clients[0].stats()

    def report_stats(self, stats: dict) -> None:
        self._clients[0].report_stats(stats)

    def rdma_endpoints(self) -> list[dict]:
        return self._clients[0].rdma_endpoints()

    def vacuum(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        return self._clients[0].vacuum(force=force, shard_ids=shard_ids)

    def autotune(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        return self._clients[0].autotune(force=force, shard_ids=shard_ids)

    def maintenance_status(self) -> dict:
        return self._clients[0].maintenance_status()

    def flush(self) -> int:
        return self._clients[0].flush()

    def lease(self, key: str, duration_ms: int) -> int:
        return self._clients[0].lease(key, duration_ms)

    def pin(self, key: str) -> int:
        return self._clients[0].pin(key)

    def unpin(self, key: str) -> int:
        return self._clients[0].unpin(key)

    def cluster_info(self) -> dict:
        return self._clients[0].cluster_info()

    def snapshot(self, dir: str = "") -> dict:
        return self._clients[0].snapshot(dir)

    def restore(self, dir: str = "") -> dict:
        return self._clients[0].restore(dir)


# Backward compatibility alias
CamaClient = L3Client

CamaClientPool = L3ClientPool
