"""CAMA RDMA client — single-connection transport.

Uses the pybind11 _cama_rdma extension to perform RDMA Send/Recv for
control messages and RDMA Read for large value retrieval (zero-copy when
the SGL buffer is registered).
"""

from __future__ import annotations

import atexit
import json
import os
import struct
import threading
import time

from l3_client import protocol
from l3_client.errors import CamaNotReadyError, CamaOOMError, CamaServerOverloadError
from l3_client.handshake import perform_handshake
from l3_client.reconnect import (
    ReconnectCallbackRegistry,
    ReconnectConfig,
    _MREntry,
    _resolve_reconnect,
    compute_delay,
    is_retriable,
)
from l3_client.sgl import SGL
from l3_client._l3_rdma import RDMATransport
from l3_client.rdma._constants import (
    _DEFAULT_SEND_BUF_SIZE,
    _wc_status_name,
    logger,
)
from l3_client.rdma._stats import average_get_timings, average_set_timings
from l3_client.rdma._batch_ops import TransportHandle, mget_rdma_flow, mget_rdma_raw_flow


class RDMAClient:
    """PrisKV-compatible client backed by CAMA over RDMA.

    API-identical to CamaClient (client.py) so it can be used as a
    drop-in replacement. The key difference is that GET of large values
    uses RDMA Read instead of TCP, and registered SGL buffers get
    true zero-copy DMA.
    """

    def __init__(self, addr: str = "127.0.0.1", port: int = 18001, password: str = "", *, handshake: bool = True, timeout: float | None = None,
                 send_buf_size: int | None = None, recv_buf_size: int | None = None,
                 debug: bool = False, reconnect: bool | ReconnectConfig | None = True,
                 poll_timeout_ms: int | None = None):
        """Connect to CAMA server over RDMA.

        Args:
            addr: IP address of the server's RDMA interface. Must be the IP of
                an RDMA-capable NIC (e.g. from ``ibdev2netdev`` or the server's
                ``[rdma] resolved listen address:`` log line). Hostnames that
                resolve to a non-RDMA interface will fail with ADDR_ERROR.
            port: RDMA listen port (default 18001).
            password: Ignored (no auth over RDMA).
            timeout: Stored for API compatibility. RDMA transport relies on
                server-side dispatch timeout as the backstop.
            send_buf_size: RDMA Send buffer size in bytes (default 16MB).
                Must match or exceed the server's recv buffer size.
            recv_buf_size: RDMA Recv buffer size in bytes (default 16MB).
                Must match or exceed the server's send buffer size.
            debug: Enable per-operation timing logs. Also enabled by CAMA_DEBUG=1.
        """
        self._debug = debug or os.environ.get("CAMA_DEBUG", "") == "1"
        self._timeout = timeout

        # Store connection params for reconnect
        self._addr = addr
        self._port = port
        self._handshake_enabled = handshake
        kwargs: dict = {}
        if send_buf_size is not None:
            kwargs["send_buf_size"] = send_buf_size
        if recv_buf_size is not None:
            kwargs["recv_buf_size"] = recv_buf_size
        if poll_timeout_ms is not None:
            kwargs["poll_timeout_ms"] = poll_timeout_ms
        self._kwargs = kwargs

        t0 = time.monotonic()
        self._transport = RDMATransport(**kwargs)
        # Read actual buffer size from C++ (single source of truth)
        self._send_buf_size = self._transport.send_buf_size
        self._transport.connect(addr, port)
        if self._debug:
            self._connect_time_ms = (time.monotonic() - t0) * 1000
            logger.info("RDMA connect to %s:%d in %.1fms", addr, port, self._connect_time_ms)

        self._lock = threading.Lock()
        self._req_id = 0
        # Map reg_handle -> _MREntry for registered user buffers.
        # _MREntry stores (lkey, mr_handle, buf_ref, ptr, size) — the ptr/size
        # are needed for MR re-registration after reconnect.
        # buf_ref prevents GC of the buffer during in-flight RDMA Reads.
        self._mr_map: dict[int, _MREntry] = {}
        self._next_handle = 2  # handle 1 is the dummy TCP handle
        self._server_info: dict | None = None
        # RDMA Read reliability counters
        self._rdma_read_retries = 0
        self._rdma_read_failures = 0
        self._counter_lock = threading.Lock()
        # Transfer sub-phase timing accumulators
        self._get_timings: list[dict] = []
        self._set_timings: list[dict] = []

        # Reconnect state
        self._reconnect_config = _resolve_reconnect(reconnect)
        self._callbacks = ReconnectCallbackRegistry()
        self._alive = True

        self._has_mget_rdma = False
        if handshake:
            self._server_info = perform_handshake(self._roundtrip, "rdma")
            caps = self._server_info.get("capabilities", []) if self._server_info else []
            self._has_mget_rdma = "mget_rdma" in caps
            # Warn on client/server buffer size mismatch
            if self._server_info:
                srv_recv_mb = self._server_info.get("rdma_recv_buf_mb")
                client_send_mb = self._send_buf_size // (1024 * 1024) if self._send_buf_size else None
                if srv_recv_mb and client_send_mb and client_send_mb < srv_recv_mb:
                    logger.warning(
                        "client send_buf_size (%d MB) < server rdma_recv_buf_size (%d MB) — "
                        "increase client send_buf_size to match server for optimal batching",
                        client_send_mb, srv_recv_mb,
                    )
                srv_send_mb = self._server_info.get("rdma_send_buf_mb")
                if srv_send_mb:
                    client_recv_mb = (recv_buf_size or self._transport.recv_buf_size) // (1024 * 1024)
                    if client_recv_mb < srv_send_mb:
                        logger.warning(
                            "client recv_buf_size (%d MB) < server rdma_send_buf_size (%d MB) — "
                            "increase client recv_buf_size to match server",
                            client_recv_mb, srv_send_mb,
                        )

        atexit.register(self._atexit_close)

    def set_timeout(self, seconds: float | None) -> None:
        """Set operation timeout (stored, but RDMA transport relies on server-side dispatch timeout).

        Args:
            seconds: Timeout in seconds. None = no timeout.
        """
        self._timeout = seconds
        logger.debug("RDMA set_timeout(%s): RDMA transport relies on server-side dispatch timeout", seconds)

    def close(self) -> None:
        atexit.unregister(self._atexit_close)
        # Acquire lock to prevent racing with in-flight roundtrip()
        with self._lock:
            # Deregister any user MRs still outstanding
            for entry in self._mr_map.values():
                try:
                    self._transport.dereg_mr(entry.mr_handle)
                except Exception:
                    pass
            self._mr_map.clear()  # releases buf_refs, allowing GC of registered buffers
            self._transport.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _atexit_close(self) -> None:
        """Best-effort cleanup at interpreter shutdown."""
        try:
            self.close()
        except Exception:
            pass

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _roundtrip(self, opcode: int, body: bytes, flags: int = 0) -> protocol.Message:
        """Send a request and read the response (thread-safe).

        If reconnection is enabled, retriable transport errors trigger
        automatic reconnect with MR re-registration and exponential backoff.
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
                # Reconnect loop (lock is held)
                for attempt in range(rc.max_retries):
                    self._alive = False
                    delay = compute_delay(attempt, rc)
                    logger.warning(
                        "RDMA reconnect attempt %d/%d to %s:%d (delay=%.2fs): %s",
                        attempt + 1, rc.max_retries,
                        self._addr, self._port, delay, exc,
                    )
                    # Save MR snapshot
                    mr_snapshot = dict(self._mr_map)
                    # Best-effort dereg on old transport
                    for entry in mr_snapshot.values():
                        try:
                            self._transport.dereg_mr(entry.mr_handle)
                        except Exception:
                            pass
                    # Close old transport
                    try:
                        self._transport.close()
                    except Exception:
                        pass
                    time.sleep(delay)
                    try:
                        # Create new transport (C++ object not reusable after close)
                        new_transport = RDMATransport(**self._kwargs)
                        try:
                            new_transport.connect(self._addr, self._port)
                        except Exception:
                            try:
                                new_transport.close()
                            except Exception:
                                pass
                            raise
                        self._transport = new_transport
                        # Re-register all MRs on new PD
                        for handle, entry in mr_snapshot.items():
                            new_lkey, new_mr_handle = self._transport.reg_mr(
                                entry.ptr, entry.size)
                            self._mr_map[handle] = _MREntry(
                                new_lkey, new_mr_handle, entry.buf_ref,
                                entry.ptr, entry.size)
                        self._req_id = 0
                        # Re-handshake
                        if self._handshake_enabled:
                            self._server_info = perform_handshake(
                                self._roundtrip_inner, "rdma")
                            # Refresh capabilities from new server info
                            caps = self._server_info.get("capabilities", []) if self._server_info else []
                            self._has_mget_rdma = "mget_rdma" in caps
                        self._alive = True
                        # Retry the original operation
                        resp = self._roundtrip_inner(opcode, body, flags)
                        logger.info("RDMA reconnect succeeded on attempt %d", attempt + 1)
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
        hdr = protocol._pack_header(opcode, flags, req_id, body)
        raw_resp = self._transport.roundtrip(hdr + body)
        resp = protocol.read_message_from_bytes(raw_resp)
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
        """Report client-side stats to the server for Prometheus exposure.

        Auto-injects RDMA Read reliability counters.
        """
        stats = dict(stats)  # copy to avoid mutating caller's dict
        with self._counter_lock:
            stats["rdma_read_retries"] = self._rdma_read_retries
            stats["rdma_read_failures"] = self._rdma_read_failures
        body = json.dumps(stats).encode()
        self._roundtrip(protocol.OP_REPORT_STATS, body)

    def get_transport_stats(self) -> dict:
        """Return C++ transport timing stats, Python sub-phase averages, and reset counters."""
        try:
            s = self._transport.get_stats()
            self._transport.reset_stats()
            result = {
                "roundtrip_count": s.get("roundtrip_count", 0),
                "rdma_read_count": s.get("rdma_read_count", 0),
                "avg_roundtrip_us": s.get("avg_roundtrip_us", 0.0),
                "avg_rdma_read_us": s.get("avg_rdma_read_us", 0.0),
                "avg_batch_read_us": s.get("avg_batch_read_us", 0.0),
                "batch_read_count": s.get("batch_read_count", 0),
            }
        except Exception:
            result = {}

        # Swap-and-average GET/SET sub-phase timings (under lock to avoid
        # racing with concurrent batch ops that append to these lists)
        with self._counter_lock:
            get_timings = self._get_timings; self._get_timings = []
            set_timings = self._set_timings; self._set_timings = []
            result["rdma_read_retries"] = self._rdma_read_retries
            result["rdma_read_failures"] = self._rdma_read_failures
        result.update(average_get_timings(get_timings))
        result.update(average_set_timings(set_timings))

        return result

    def rdma_endpoints(self) -> list[dict]:
        """Return RDMA endpoints from the server (each has device, ip, port)."""
        if self._server_info and "rdma_endpoints" in self._server_info:
            return self._server_info["rdma_endpoints"] or []
        info = self.info()
        return info.get("rdma_endpoints") or []

    # --- RDMA memory registration ---

    def reg_memory(self, ptr: int, size: int, buf: object = None) -> int:
        """Register a host memory region for RDMA access. Returns a handle.

        Args:
            ptr:  Host memory address (e.g., ctypes.addressof(my_buffer)).
            size: Buffer size in bytes.
            buf:  The Python object that *owns* the memory (e.g., the ctypes
                  buffer, bytearray, or torch tensor backing ``ptr``).

                  **Pass this whenever possible.** RDMA Read is one-sided — the
                  NIC DMA-reads directly from the registered pages while Python
                  continues to execute. If the buffer object is GC'd between the
                  server's RDMA_READ_READY response and the completion poll, the
                  NIC reads freed (or reused) memory, causing
                  IBV_WC_REM_ACCESS_ERR or silent data corruption.

                  Example::

                      buf = ctypes.create_string_buffer(64 * 1024 * 1024)
                      handle = client.reg_memory(ctypes.addressof(buf), len(buf), buf=buf)
                      sgl = SGL(ptr=ctypes.addressof(buf), size=5 * 1024 * 1024,
                                reg_handle=handle)
        """
        lkey, mr_handle = self._transport.reg_mr(ptr, size)
        handle = self._next_handle
        self._next_handle += 1
        self._mr_map[handle] = _MREntry(lkey, mr_handle, buf, ptr, size)
        return handle

    def dereg_memory(self, handle: int) -> None:
        """Deregister a previously registered memory region."""
        entry = self._mr_map.pop(handle, None)
        if entry is not None:
            self._transport.dereg_mr(entry.mr_handle)
            # entry.buf_ref is released here — buffer may now be GC'd

    # --- Basic string ops (PrisKV compat) ---

    def setstr(self, key: str, value: str) -> int:
        """Store a string key-value pair. Returns 0 on success."""
        body, flags = protocol.encode_kv_body(key.encode(), value.encode())
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def getstr(self, key: str) -> str | None:
        """Retrieve a string value. Returns None if not found.

        On RDMA Read WC error, retries the full GET roundtrip once (matches
        get() retry semantics for migration reliability).
        """
        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_GET, body)

        # Handle RDMA Read Ready (large values)
        if resp.header.opcode == protocol.OP_RDMA_READ_READY:
            rkey, remote_addr, length = protocol.decode_rdma_read_ready(resp.body)
            if length == 0:
                return None
            try:
                with self._lock:
                    data = self._transport.rdma_read(rkey, remote_addr, length)
                self._send_read_ack(0)  # success
                return data.decode(errors="replace")
            except RuntimeError:
                # RDMA Read failed — likely stale rkey during migration.
                # Send failure ack, then retry the full GET roundtrip.
                self._send_read_ack(255)
                with self._counter_lock:
                    self._rdma_read_retries += 1

                resp2 = self._roundtrip(protocol.OP_GET, body)
                if resp2.header.opcode == protocol.OP_RDMA_READ_READY:
                    rkey2, addr2, len2 = protocol.decode_rdma_read_ready(resp2.body)
                    if len2 == 0:
                        return None
                    try:
                        with self._lock:
                            data2 = self._transport.rdma_read(rkey2, addr2, len2)
                        self._send_read_ack(0)
                        return data2.decode(errors="replace")
                    except RuntimeError:
                        with self._counter_lock:
                            self._rdma_read_failures += 1
                        self._send_read_ack(255)
                        raise RuntimeError(
                            "RDMA Read failed after retry in getstr()")
                else:
                    value, found = protocol.decode_value_response(resp2.body)
                    if not found:
                        return None
                    return value.decode(errors="replace")

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

    def _send_read_ack(self, wc_status: int) -> None:
        """Send OpReadAck to server (two-phase RDMA Read tracking)."""
        try:
            body = protocol.encode_read_ack(wc_status)
            self._roundtrip(protocol.OP_READ_ACK, body)
        except Exception:
            pass  # best-effort — don't fail the GET on ack failure

    # --- SGL-based ops (zero-copy RDMA Read when buffer is registered) ---

    def set(self, key: str, sgl: SGL, ttl_ms: int = 0) -> int:
        """Store value from SGL via RDMA Send. Returns 0 on success."""
        value = sgl.to_bytes()
        body, flags = protocol.encode_kv_body(key.encode(), value, ttl_ms=ttl_ms)
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def get(self, key: str, sgl: SGL, size: int = 0) -> int:
        """Retrieve value into SGL. Uses RDMA Read for large values.

        If the SGL buffer is registered, does zero-copy RDMA Read directly
        into the buffer. Otherwise falls back to RDMA Read into internal
        buffer + memcpy.

        On RDMA Read WC error, retries the full GET roundtrip once (gets
        fresh rkey from post-swap MRs, or inline if migration-aware).

        Returns 0 on success, -1 if not found.
        """
        if self._debug:
            t0 = time.monotonic()

        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_GET, body)

        if self._debug:
            t_ctrl = time.monotonic()

        # Handle RDMA Read Ready (server has a large value in its slab)
        if resp.header.opcode == protocol.OP_RDMA_READ_READY:
            rkey, remote_addr, length = protocol.decode_rdma_read_ready(resp.body)
            if length == 0:
                return -1

            wc_status = self._do_rdma_read(sgl, rkey, remote_addr, length)

            if wc_status != 0:
                # RDMA Read failed — likely stale rkey during migration.
                # Send failure ack, then retry the full GET roundtrip.
                status_name = _wc_status_name(wc_status)
                logger.warning("RDMA Read failed: %s (status=%d), retrying GET for key=%s",
                               status_name, wc_status, key)
                self._send_read_ack(wc_status)
                with self._counter_lock:
                    self._rdma_read_retries += 1

                # Retry: re-issue OP_GET — server may return inline (migration-aware)
                # or a fresh rkey from post-swap MRs
                resp2 = self._roundtrip(protocol.OP_GET, body)
                if resp2.header.opcode == protocol.OP_RDMA_READ_READY:
                    rkey2, addr2, len2 = protocol.decode_rdma_read_ready(resp2.body)
                    if len2 == 0:
                        return -1
                    wc2 = self._do_rdma_read(sgl, rkey2, addr2, len2)
                    if wc2 != 0:
                        with self._counter_lock:
                            self._rdma_read_failures += 1
                        self._send_read_ack(wc2)
                        raise RuntimeError(
                            f"RDMA Read failed after retry: {_wc_status_name(wc2)} (status={wc2})")
                    self._send_read_ack(0)
                else:
                    # Retry returned inline — decode directly
                    value, found = protocol.decode_value_response(resp2.body)
                    if not found:
                        return -1
                    sgl.from_bytes(value)
            else:
                self._send_read_ack(0)

            if self._debug:
                t_data = time.monotonic()
                logger.debug("GET %s: control=%.2fms data=%.2fms total=%.2fms",
                    key, (t_ctrl - t0) * 1000, (t_data - t_ctrl) * 1000,
                    (t_data - t0) * 1000)
            return 0

        # Inline value response (small values sent directly)
        value, found = protocol.decode_value_response(resp.body)
        if not found:
            return -1
        sgl.from_bytes(value)
        return 0

    def _do_rdma_read(self, sgl: SGL, rkey: int, remote_addr: int, length: int) -> int:
        """Perform RDMA Read into SGL, returning WC status (0=success).

        Uses try_rdma_read_into for registered buffers (returns status instead
        of throwing). For unregistered buffers, uses rdma_read (throws on error,
        caught and mapped to nonzero status).
        """
        mr_entry = self._mr_map.get(sgl.reg_handle)
        with self._lock:
            if mr_entry is not None:
                try:
                    return self._transport.try_rdma_read_into(
                        rkey, remote_addr, length, sgl.ptr, mr_entry.lkey)
                except RuntimeError:
                    return 255  # post_send or poll_cq failure
            else:
                try:
                    data = self._transport.rdma_read(rkey, remote_addr, length)
                    sgl.from_bytes(data)
                    return 0
                except RuntimeError:
                    return 255  # WC error or poll failure

    # --- Size guard ---

    def _fits_send_buf(self, body_size: int) -> bool:
        """Check if body fits in the RDMA send buffer (minus header)."""
        max_body = (getattr(self, '_send_buf_size', None) or _DEFAULT_SEND_BUF_SIZE) - protocol.HEADER_SIZE
        return body_size <= max_body

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

        When the full payload exceeds the RDMA send buffer, partitions
        into sub-batches that each fit, instead of falling back to
        sequential set().  Returns per-key status list (0=ok, nonzero=error).
        """
        if not keys:
            return []
        t0 = time.perf_counter()
        values = [sgl.to_bytes() for sgl in sgls]
        encoded_keys = [k.encode() for k in keys]
        body = protocol.encode_mset_body(encoded_keys, values)
        t_ser = time.perf_counter()

        if self._fits_send_buf(len(body)):
            # Fast path: entire batch fits in one message
            flags = protocol.FLAG_WITH_TTL if ttl_ms > 0 else protocol.FLAG_NONE
            resp = self._roundtrip(protocol.OP_MSET, body, flags=flags)
            t_send = time.perf_counter()
            with self._counter_lock:
                self._set_timings.append({
                    "t_serialize_ms": (t_ser - t0) * 1000,
                    "t_send_ms": (t_send - t_ser) * 1000,
                    "n_keys": len(keys),
                    "total_bytes": sum(len(v) for v in values),
                    "n_sub_batches": 1,
                })
            return self._parse_mset_response(resp, len(keys))

        # Chunked: partition into sub-batches that each fit the send buffer
        max_body = (self._send_buf_size or _DEFAULT_SEND_BUF_SIZE) - protocol.HEADER_SIZE
        flags = protocol.FLAG_WITH_TTL if ttl_ms > 0 else protocol.FLAG_NONE
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

        t_send = time.perf_counter()
        with self._counter_lock:
            self._set_timings.append({
                "t_serialize_ms": (t_ser - t0) * 1000,
                "t_send_ms": (t_send - t_ser) * 1000,
                "n_keys": len(keys),
                "total_bytes": sum(len(v) for v in values),
                "n_sub_batches": n_chunks + n_oversized,
            })

        total_batches = n_chunks + n_oversized
        if n_oversized > 0:
            logger.warning(
                "mset: %d/%d entries exceed send buffer (%d MB) — fell back to "
                "individual SET ops. Increase send_buf_size to batch them.",
                n_oversized, len(keys), max_body // (1024 * 1024),
            )
        elif total_batches >= len(keys):
            # Every key ended up in its own chunk — effectively sequential
            avg_val = sum(len(v) for v in values) // len(values)
            logger.warning(
                "mset: %d keys chunked into %d batches (~1 key/batch, avg value "
                "%d bytes vs %d MB send buffer). Increase send_buf_size for "
                "better batching.",
                len(keys), total_batches, avg_val, max_body // (1024 * 1024),
            )
        return all_results

    def mget(self, keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]:
        """Batch get via individual ops (preserves RDMA Read zero-copy).

        Returns list of return codes (0=found, -1=miss).
        """
        results = []
        for i, k in enumerate(keys):
            rc = self.get(k, sgls[i])
            results.append(rc)
        return results

    def mget_rdma(self, keys: list[str], sgls: list[SGL],
                  sizes: list[int] | None = None) -> list[int]:
        """Batch GET with RDMA Read — single control roundtrip + batch RDMA Reads.

        Falls back to sequential mget() if server doesn't support mget_rdma.
        Returns list of return codes (0=found, -1=miss).
        """
        if not self._has_mget_rdma:
            return self.mget(keys, sgls, sizes)

        if not keys:
            return []

        handle = TransportHandle(
            self._transport, self._lock,
            self._next_req_id,
            lambda reg_handle: self._mr_map.get(reg_handle),
        )
        results, timing_dict, read_count, total_bytes = mget_rdma_flow(
            handle, keys, sgls, sizes)
        if timing_dict is not None:
            self._get_timings.append(timing_dict)
        return results

    def mget_rdma_raw(self, keys: list[str]) -> list[tuple[int, bytes | None]]:
        """Batch GET returning raw bytes via internal read buffer.

        Returns list of (rc, data) per key: (0, bytes) for hit, (-1, None)
        for miss.  Same control flow as mget_rdma but reads into the C++
        transport's internal read_buf_ instead of user-registered SGLs,
        then returns raw bytes for caller-side decompression.
        """
        if not self._has_mget_rdma:
            # Fallback: sequential get() returning raw bytes.
            # Enforce aggregate timeout to prevent unbounded latency
            # when iterating many keys against a slow server.
            deadline = (time.monotonic() + self._timeout) if self._timeout else None
            results: list[tuple[int, bytes | None]] = []
            for k in keys:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"mget_rdma_raw fallback timed out after {self._timeout}s "
                        f"({len(results)}/{len(keys)} keys completed)"
                    )
                body = protocol.encode_key_body(k.encode())
                resp = self._roundtrip(protocol.OP_GET, body)
                if resp.header.opcode == protocol.OP_RDMA_READ_READY:
                    rkey, addr, length = protocol.decode_rdma_read_ready(resp.body)
                    if length == 0:
                        results.append((-1, None))
                    else:
                        with self._lock:
                            raw = self._transport.rdma_read(rkey, addr, length)
                        self._send_read_ack(0)
                        results.append((0, raw))
                elif resp.header.opcode == protocol.RESP_VALUE:
                    value, found = protocol.decode_value_response(resp.body)
                    if found:
                        results.append((0, value))
                    else:
                        results.append((-1, None))
                else:
                    results.append((-1, None))
            return results

        n = len(keys)
        if n == 0:
            return []

        handle = TransportHandle(
            self._transport, self._lock,
            self._next_req_id,
            lambda reg_handle: self._mr_map.get(reg_handle),
        )
        results, timing_dict, found_count = mget_rdma_raw_flow(handle, keys)

        if self._debug and found_count > 0:
            t_ms = timing_dict["t_total_ms"] if timing_dict else 0.0
            logger.debug("mget_rdma_raw: %d keys, %d found in %.1fms",
                         n, found_count, t_ms)

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
        """Return keys matching regex pattern."""
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
