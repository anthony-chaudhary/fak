"""CAMA RDMA client pool — N connections sharing one PD.

Round-robin distributes operations across connections for true N-way
parallelism instead of serializing on a single lock.
"""

from __future__ import annotations

import atexit
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    _DEFAULT_RECV_BUF_SIZE,
    _DEFAULT_READ_BUF_SIZE,
    _wc_status_name,
    logger,
)
from l3_client.rdma._stats import average_get_timings, average_set_timings, compute_nic_balance
from l3_client.rdma._batch_ops import TransportHandle, mget_rdma_flow, mget_rdma_raw_flow


class _PooledConn:
    """A single connection in the pool with its own transport and lock."""
    __slots__ = ("transport", "lock", "req_id", "alive", "endpoint_idx")

    def __init__(self, transport: RDMATransport, lock: threading.Lock,
                 endpoint_idx: int = 0):
        self.transport = transport
        self.lock = lock
        self.req_id = 0
        self.alive = True
        self.endpoint_idx = endpoint_idx

    def next_req_id(self) -> int:
        self.req_id += 1
        return self.req_id


class RDMAClientPool:
    """Pool of N RDMA connections sharing one PD. Drop-in for RDMAClient.

    Round-robin distributes operations across connections for true N-way
    parallelism instead of serializing on a single lock.

    When *endpoints* is provided (list of ``(ip, port)`` tuples), connections
    are distributed round-robin across server NIC endpoints so that striping
    across pool connections actually crosses NIC boundaries (multi-NIC
    saturation).  ``pool_size`` is raised to at least ``len(endpoints)`` to
    guarantee one connection per NIC, but may be larger for higher
    parallelism (e.g. pool_size=8 with 4 NICs → 2 connections per NIC).
    """

    def __init__(self, addr: str = "127.0.0.1", port: int = 18001,
                 password: str = "", *, pool_size: int = 8,
                 endpoints: list[tuple[str, int]] | None = None,
                 handshake: bool = True, timeout: float | None = None,
                 send_buf_size: int | None = None, recv_buf_size: int | None = None,
                 read_buf_size: int | None = None,
                 debug: bool = False, reconnect: bool | ReconnectConfig | None = True,
                 poll_timeout_ms: int | None = None):
        self._debug = debug or os.environ.get("CAMA_DEBUG", "") == "1"
        self._timeout = timeout
        self._send_buf_size = send_buf_size  # overwritten from C++ after first transport init
        self._server_info: dict | None = None
        self._rdma_read_retries = 0
        self._rdma_read_failures = 0
        self._counter_lock = threading.Lock()
        # Transfer sub-phase timing accumulators
        self._get_timings: list[dict] = []
        self._set_timings: list[dict] = []

        # Multi-endpoint support: ensure at least one connection per NIC,
        # but allow higher pool_size for more parallelism (e.g. TP8 + 4 NICs).
        # Cap at MAX_POOL_SIZE to prevent OOM from unbounded endpoint scaling —
        # 32 connections × 32 MB = 1 GB per pool at default 16 MB send+recv buffers
        # (plus 32 MB read buffer for owner connection only).
        MAX_POOL_SIZE = 32
        self._endpoints = endpoints  # None = single-endpoint mode
        if endpoints is not None and len(endpoints) > 1:
            pool_size = max(pool_size, len(endpoints))
        if pool_size > MAX_POOL_SIZE:
            logger.warning(
                "pool_size %d exceeds maximum %d (endpoints=%d) — capping to %d "
                "to prevent OOM during MR registration",
                pool_size, MAX_POOL_SIZE,
                len(endpoints) if endpoints else 0, MAX_POOL_SIZE,
            )
            pool_size = MAX_POOL_SIZE
        self._pool_size = pool_size

        # Store connection params for reconnect
        self._addr = addr
        self._port = port
        self._handshake_enabled = handshake
        kwargs: dict = {}
        if send_buf_size is not None:
            kwargs["send_buf_size"] = send_buf_size
        if recv_buf_size is not None:
            kwargs["recv_buf_size"] = recv_buf_size
        if read_buf_size is not None:
            kwargs["read_buf_size"] = read_buf_size
        if poll_timeout_ms is not None:
            kwargs["poll_timeout_ms"] = poll_timeout_ms
        self._kwargs = kwargs

        # Client-side memory budget warning: each connection allocates
        # send_buf + recv_buf via malloc + ibv_reg_mr. Owner also gets read_buf.
        # Default: 16+16+32 = 64 MB for owner, 16+16 = 32 MB for non-owners.
        _sb = send_buf_size or _DEFAULT_SEND_BUF_SIZE
        _rb = recv_buf_size or _DEFAULT_RECV_BUF_SIZE
        _rdb = read_buf_size or _DEFAULT_READ_BUF_SIZE
        _est_mb = (_sb + _rb + _rdb + (pool_size - 1) * (_sb + _rb)) / (1024 * 1024)
        if _est_mb > 4096:
            logger.warning(
                "RDMA pool will allocate ~%.0f MB for %d connections "
                "(%d MB send + %d MB recv per conn, %d MB read for owner). "
                "Ensure sufficient RAM and memlock ulimit.",
                _est_mb, pool_size,
                _sb // (1024 * 1024), _rb // (1024 * 1024),
                _rdb // (1024 * 1024),
            )

        t0 = time.monotonic()

        # Build endpoint list for each connection slot
        if endpoints and len(endpoints) > 1:
            ep_list = endpoints
        else:
            ep_list = [(addr, port)] * pool_size

        # First transport: owns the PD
        ep0_addr, ep0_port = ep_list[0]
        first_transport = RDMATransport(**kwargs)
        # Read actual buffer size from C++ (single source of truth)
        self._send_buf_size = first_transport.send_buf_size
        first_transport.connect(ep0_addr, ep0_port)
        pd_handle = first_transport.get_pd_handle()
        ctx_handle = first_transport.get_ctx_handle()

        conns = [_PooledConn(first_transport, threading.Lock(), endpoint_idx=0)]

        # Additional transports: try shared PD, fall back to independent PD
        # when connecting to a different server NIC routes through a different
        # client RDMA device.
        self._independent_pd_conns: set[int] = set()
        self._independent_mr_maps: dict[int, dict[int, _MREntry]] = {}
        for i in range(1, pool_size):
            ep_addr, ep_port = ep_list[i % len(ep_list)]
            t = RDMATransport(**kwargs)
            try:
                t.connect_with_shared_pd(ep_addr, ep_port, pd_handle, ctx_handle,
                                         skip_read_buf=True)
            except RuntimeError:
                # Different client RDMA device — fall back to independent PD
                logger.debug(
                    "Pool conn %d: %s:%d routes through a different local HCA "
                    "than conn 0 — cannot share PD, using independent PD "
                    "(expected in multi-NIC topologies)",
                    i, ep_addr, ep_port,
                )
                t = RDMATransport(**kwargs)
                t.connect(ep_addr, ep_port)
                self._independent_pd_conns.add(i)
                self._independent_mr_maps[i] = {}
            conns.append(_PooledConn(t, threading.Lock(),
                                     endpoint_idx=i % len(ep_list)))

        self._conns = conns
        self._rr_lock = threading.Lock()
        self._rr_idx = 0

        # MR map (pool-level, shared across all connections with shared PD)
        self._mr_map: dict[int, _MREntry] = {}
        self._mr_lock = threading.RLock()
        self._next_handle = 2

        # Reconnect state
        self._reconnect_config = _resolve_reconnect(reconnect)
        self._callbacks = ReconnectCallbackRegistry()
        self._rebuild_lock = threading.Lock()
        self._rebuilding_event = threading.Event()
        self._rebuilding_event.set()  # set = NOT rebuilding (ready)
        self._degraded = False  # True after exhausting all rebuild retries

        # Stripe executor for parallel mget_rdma across NICs
        if pool_size > 1:
            self._stripe_executor = ThreadPoolExecutor(
                max_workers=pool_size, thread_name_prefix="cama-stripe")
        else:
            self._stripe_executor = None

        # Per-NIC striping metrics (reads)
        self._stripe_calls = 0
        self._stripe_nics_used = 0
        self._stripe_errors = 0
        self._per_nic_reads: list[int] = [0] * pool_size
        self._per_nic_bytes: list[int] = [0] * pool_size

        # Per-NIC write striping metrics
        self._stripe_write_calls = 0
        self._stripe_write_errors = 0
        self._per_nic_writes: list[int] = [0] * pool_size
        self._per_nic_write_bytes: list[int] = [0] * pool_size

        # Lifetime accumulators — survive pool rebuilds so Prometheus counters
        # reported via report_stats() never decrease on reconnect.
        self._lifetime_stripe_calls = 0
        self._lifetime_stripe_nics_used = 0
        self._lifetime_stripe_errors = 0
        self._lifetime_per_nic_reads: list[int] = [0] * pool_size
        self._lifetime_per_nic_bytes: list[int] = [0] * pool_size
        self._lifetime_stripe_write_calls = 0
        self._lifetime_stripe_write_errors = 0
        self._lifetime_per_nic_writes: list[int] = [0] * pool_size
        self._lifetime_per_nic_write_bytes: list[int] = [0] * pool_size
        self._pool_rebuilds = 0

        if self._debug:
            elapsed = (time.monotonic() - t0) * 1000
            if endpoints and len(endpoints) > 1:
                logger.info("RDMA pool (%d conns, %d endpoints) connect in %.1fms",
                            pool_size, len(endpoints), elapsed)
            else:
                logger.info("RDMA pool (%d conns) connect to %s:%d in %.1fms",
                            pool_size, addr, port, elapsed)

        self._has_mget_rdma = False
        if handshake:
            # Handshake on first connection only
            self._server_info = perform_handshake(self._roundtrip_on_conn0, "rdma")
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
                    client_recv_mb = (recv_buf_size or first_transport.recv_buf_size) // (1024 * 1024)
                    if client_recv_mb < srv_send_mb:
                        logger.warning(
                            "client recv_buf_size (%d MB) < server rdma_send_buf_size (%d MB) — "
                            "increase client recv_buf_size to match server",
                            client_recv_mb, srv_send_mb,
                        )

        atexit.register(self._atexit_close)

    def _roundtrip_on_conn0(self, opcode: int, body: bytes, flags: int = 0) -> protocol.Message:
        """Roundtrip on conn[0] specifically (for handshake/admin ops).

        If conn[0] (PD owner) fails and reconnect is enabled, triggers
        a full pool rebuild.
        """
        if self._degraded:
            raise RuntimeError(
                "RDMA pool is degraded (server unreachable after all retries). "
                "Close and recreate the pool, or restart the process.")
        if not self._rebuilding_event.is_set():
            raise RuntimeError("pool rebuilding, retry later")
        saved_exc = None
        conn = self._conns[0]
        with conn.lock:
            try:
                return self._roundtrip_on_conn_inner(conn, opcode, body, flags)
            except Exception as exc:
                rc = self._reconnect_config
                if rc is None or not rc.enabled or not is_retriable(exc):
                    raise
                saved_exc = exc
                # conn[0] is PD owner — need full rebuild
                # Release conn lock before rebuild (rebuild acquires all locks)
        # Full rebuild (outside conn lock)
        return self._full_rebuild_and_retry(opcode, body, flags, saved_exc)

    def _roundtrip_on_conn_inner(self, conn: _PooledConn, opcode: int,
                                  body: bytes, flags: int = 0) -> protocol.Message:
        """Raw roundtrip on a specific connection. Lock must be held."""
        req_id = conn.next_req_id()
        hdr = protocol._pack_header(opcode, flags, req_id, body)
        raw_resp = conn.transport.roundtrip(hdr + body)
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

    def _conn_with_read_buf(self, conn: _PooledConn) -> _PooledConn:
        """Return conn if it has an internal read buffer, else fall back to owner (conn 0)."""
        if conn.transport.has_read_buf():
            return conn
        return self._conns[0]

    def _next_conn(self) -> _PooledConn:
        """Round-robin connection selection, skipping dead connections."""
        with self._rr_lock:
            idx = self._rr_idx % self._pool_size
            self._rr_idx += 1
            start_idx = idx
            while not self._conns[idx].alive:
                idx = (idx + 1) % self._pool_size
                if idx == start_idx:
                    break
            if not self._conns[idx].alive:
                logger.error("RDMA pool: all %d connections dead. Call enable_reconnect() to attempt recovery.", self._pool_size)
                raise RuntimeError("all RDMA pool connections are dead")
        return self._conns[idx]

    def _next_conn_with_index(self) -> tuple[_PooledConn, int]:
        """Round-robin returning both connection and its index."""
        with self._rr_lock:
            idx = self._rr_idx % self._pool_size
            self._rr_idx += 1
            start_idx = idx
            while not self._conns[idx].alive:
                idx = (idx + 1) % self._pool_size
                if idx == start_idx:
                    break
            if not self._conns[idx].alive:
                logger.error("RDMA pool: all %d connections dead. Call enable_reconnect() to attempt recovery.", self._pool_size)
                raise RuntimeError("all RDMA pool connections are dead")
        return self._conns[idx], idx

    def _reconnect_non_owner(self, conn: _PooledConn, exc: Exception) -> None:
        """Reconnect a non-PD-owner connection (cheap: PD/MRs survive).

        Lock on conn must be held by caller.
        """
        rc = self._reconnect_config
        for attempt in range(rc.max_retries):
            # Bail out early if a full rebuild has started — it will
            # recreate ALL connections including this one.
            if not self._rebuilding_event.is_set():
                raise exc
            conn.alive = False
            delay = compute_delay(attempt, rc)
            logger.warning(
                "RDMA pool non-owner reconnect attempt %d/%d (delay=%.2fs): %s",
                attempt + 1, rc.max_retries, delay, exc,
            )
            try:
                conn.transport.close()
            except Exception:
                pass
            # Release lock during sleep — conn.alive=False prevents other
            # threads from picking this conn via round-robin.
            conn.lock.release()
            try:
                time.sleep(delay)
            finally:
                conn.lock.acquire()
            try:
                # Read PD/ctx handles from conn[0] under its lock.
                # Use non-blocking acquire: caller holds conn[N].lock,
                # and full rebuild acquires [0..N-1] in order — blocking
                # on conn[0] while holding conn[N] would deadlock.
                if not self._rebuilding_event.is_set():
                    raise exc
                conn0 = self._conns[0]
                if not conn0.lock.acquire(blocking=False):
                    raise exc  # conn[0] busy (rebuild or other op)
                try:
                    pd_handle = conn0.transport.get_pd_handle()
                    ctx_handle = conn0.transport.get_ctx_handle()
                finally:
                    conn0.lock.release()
                # Determine endpoint for this connection
                ep_addr, ep_port = self._addr, self._port
                if self._endpoints and len(self._endpoints) > 1:
                    ep_idx = conn.endpoint_idx
                    ep_addr, ep_port = self._endpoints[ep_idx]

                new_transport = RDMATransport(**self._kwargs)
                try:
                    new_transport.connect_with_shared_pd(
                        ep_addr, ep_port, pd_handle, ctx_handle,
                        skip_read_buf=True)
                except RuntimeError:
                    try:
                        new_transport.close()
                    except Exception:
                        pass
                    # Different client device — fall back to independent PD
                    new_transport = RDMATransport(**self._kwargs)
                    try:
                        new_transport.connect(ep_addr, ep_port)
                    except Exception:
                        try:
                            new_transport.close()
                        except Exception:
                            pass
                        raise
                    conn_idx = self._conns.index(conn)
                    self._independent_pd_conns.add(conn_idx)
                    if conn_idx not in self._independent_mr_maps:
                        self._independent_mr_maps[conn_idx] = {}
                    # Register existing MRs on new independent PD.
                    # Snapshot under _mr_lock to avoid dict-changed-size races
                    # with concurrent reg_memory()/dereg_memory() calls.
                    with self._mr_lock:
                        mr_snap = list(self._mr_map.items())
                    for handle, entry in mr_snap:
                        il, im = new_transport.reg_mr(entry.ptr, entry.size)
                        self._independent_mr_maps[conn_idx][handle] = _MREntry(
                            il, im, entry.buf_ref, entry.ptr, entry.size)
                conn.transport = new_transport
                conn.req_id = 0
                conn.alive = True
                logger.info("RDMA pool non-owner reconnect succeeded on attempt %d",
                            attempt + 1)
                return
            except Exception as retry_exc:
                exc = retry_exc
                if not is_retriable(retry_exc):
                    raise
        raise exc

    def _full_rebuild_and_retry(self, opcode: int, body: bytes, flags: int,
                                 original_exc: Exception) -> protocol.Message:
        """Full pool rebuild when PD owner (conn[0]) dies.

        Acquires _rebuild_lock, drains all conn locks, rebuilds all
        transports, re-registers MRs, and retries the operation.
        """
        rc = self._reconnect_config
        if not self._rebuild_lock.acquire(blocking=False):
            # Another thread is already rebuilding — wait for it
            with self._rebuild_lock:
                # Rebuild done by other thread — retry directly
                conn = self._conns[0]
                with conn.lock:
                    return self._roundtrip_on_conn_inner(conn, opcode, body, flags)

        try:
            self._rebuilding_event.clear()

            for attempt in range(rc.max_retries):
                delay = compute_delay(attempt, rc)
                logger.warning(
                    "RDMA pool FULL REBUILD attempt %d/%d (delay=%.2fs): %s",
                    attempt + 1, rc.max_retries, delay, original_exc,
                )

                # Acquire all conn locks in order [0..N-1] with timeout.
                # If a lock is stuck (thread blocked in C++ poll), force-close
                # the transport to trigger connected_=false + CQ flush, then retry.
                _LOCK_TIMEOUT = 5.0
                acquired = []
                for c in self._conns:
                    if not c.lock.acquire(timeout=_LOCK_TIMEOUT):
                        logger.warning(
                            "rebuild: lock timeout after %.1fs, force-closing transport",
                            _LOCK_TIMEOUT,
                        )
                        try:
                            c.transport.close()
                        except Exception:
                            pass
                        if not c.lock.acquire(timeout=_LOCK_TIMEOUT):
                            logger.error(
                                "rebuild: lock still held after force-close, skipping conn"
                            )
                            c.alive = False
                            continue
                    acquired.append(c)
                try:
                    # Mark all dead
                    for c in self._conns:
                        c.alive = False

                    # Save MR snapshot under _mr_lock to avoid dict-changed-size
                    # races with concurrent reg_memory()/dereg_memory() calls.
                    with self._mr_lock:
                        mr_snapshot = dict(self._mr_map)

                    # Best-effort dereg MRs on old PD owner
                    for entry in mr_snapshot.values():
                        try:
                            self._conns[0].transport.dereg_mr(entry.mr_handle)
                        except Exception:
                            pass

                    # Close all transports (non-owners first, then owner)
                    for c in reversed(self._conns):
                        try:
                            c.transport.close()
                        except Exception:
                            pass

                    time.sleep(delay)

                    try:
                        # Build endpoint list (same as __init__)
                        if self._endpoints and len(self._endpoints) > 1:
                            ep_list = self._endpoints
                        else:
                            ep_list = [(self._addr, self._port)] * self._pool_size

                        # Create new PD owner — explicitly close on failure to
                        # avoid GC-triggered rdma_disconnect on partial CM ID
                        # (segfault risk: see rdma_transport.cpp close_impl).
                        ep0_addr, ep0_port = ep_list[0]
                        first_transport = RDMATransport(**self._kwargs)
                        try:
                            first_transport.connect(ep0_addr, ep0_port)
                        except Exception:
                            try:
                                first_transport.close()
                            except Exception:
                                pass
                            raise
                        pd_handle = first_transport.get_pd_handle()
                        ctx_handle = first_transport.get_ctx_handle()
                        self._conns[0].transport = first_transport
                        self._conns[0].req_id = 0

                        # Create N-1 secondary transports
                        # Track newly created transports so we can close them
                        # on partial failure (avoid leaking RDMA resources).
                        new_transports = [first_transport]
                        with self._mr_lock:
                            self._independent_pd_conns.clear()
                            self._independent_mr_maps.clear()
                        for i in range(1, self._pool_size):
                            ep_addr, ep_port = ep_list[i % len(ep_list)]
                            t = RDMATransport(**self._kwargs)
                            try:
                                t.connect_with_shared_pd(
                                    ep_addr, ep_port, pd_handle, ctx_handle,
                                    skip_read_buf=True)
                            except RuntimeError:
                                try:
                                    t.close()
                                except Exception:
                                    pass
                                t = RDMATransport(**self._kwargs)
                                try:
                                    t.connect(ep_addr, ep_port)
                                except Exception:
                                    try:
                                        t.close()
                                    except Exception:
                                        pass
                                    # Close all already-created transports
                                    for nt in new_transports:
                                        try:
                                            nt.close()
                                        except Exception:
                                            pass
                                    raise
                                with self._mr_lock:
                                    self._independent_pd_conns.add(i)
                                    self._independent_mr_maps[i] = {}
                            new_transports.append(t)
                            self._conns[i].transport = t
                            self._conns[i].req_id = 0

                        # Re-register all MRs on new PD
                        with self._mr_lock:
                            for handle, entry in mr_snapshot.items():
                                new_lkey, new_mr_handle = first_transport.reg_mr(
                                    entry.ptr, entry.size)
                                self._mr_map[handle] = _MREntry(
                                    new_lkey, new_mr_handle, entry.buf_ref,
                                    entry.ptr, entry.size)
                                # Re-register on independent-PD conns
                                for ci, ind_map in self._independent_mr_maps.items():
                                    il, im = self._conns[ci].transport.reg_mr(
                                        entry.ptr, entry.size)
                                    ind_map[handle] = _MREntry(
                                        il, im, entry.buf_ref, entry.ptr, entry.size)

                        # Re-handshake on conn[0]
                        if self._handshake_enabled:
                            self._server_info = perform_handshake(
                                lambda op, b, f=0: self._roundtrip_on_conn_inner(
                                    self._conns[0], op, b, f),
                                "rdma")
                            # Refresh capabilities from new server info
                            caps = self._server_info.get("capabilities", []) if self._server_info else []
                            self._has_mget_rdma = "mget_rdma" in caps

                        # Mark all alive
                        for c in self._conns:
                            c.alive = True

                        # Snapshot current stripe metrics into lifetime accumulators
                        # before resetting, so Prometheus counters never decrease.
                        with self._counter_lock:
                            self._lifetime_stripe_calls += self._stripe_calls
                            self._lifetime_stripe_nics_used += self._stripe_nics_used
                            self._lifetime_stripe_errors += self._stripe_errors
                            for i in range(min(len(self._per_nic_reads), len(self._lifetime_per_nic_reads))):
                                self._lifetime_per_nic_reads[i] += self._per_nic_reads[i]
                                self._lifetime_per_nic_bytes[i] += self._per_nic_bytes[i]
                            self._lifetime_stripe_write_calls += self._stripe_write_calls
                            self._lifetime_stripe_write_errors += self._stripe_write_errors
                            for i in range(min(len(self._per_nic_writes), len(self._lifetime_per_nic_writes))):
                                self._lifetime_per_nic_writes[i] += self._per_nic_writes[i]
                                self._lifetime_per_nic_write_bytes[i] += self._per_nic_write_bytes[i]
                            self._pool_rebuilds += 1
                            # Stripe metrics intentionally reset after rebuild to prevent stale
                            # per-NIC counters from old topology.
                            # Now reset epoch counters
                            self._stripe_calls = 0
                            self._stripe_nics_used = 0
                            self._stripe_errors = 0
                            self._per_nic_reads = [0] * self._pool_size
                            self._per_nic_bytes = [0] * self._pool_size
                            self._stripe_write_calls = 0
                            self._stripe_write_errors = 0
                            self._per_nic_writes = [0] * self._pool_size
                            self._per_nic_write_bytes = [0] * self._pool_size

                        self._rebuilding_event.set()

                        # Retry the original operation
                        resp = self._roundtrip_on_conn_inner(
                            self._conns[0], opcode, body, flags)
                        logger.info("RDMA pool full rebuild succeeded on attempt %d",
                                    attempt + 1)
                    except Exception as retry_exc:
                        original_exc = retry_exc
                        if not is_retriable(retry_exc):
                            self._rebuilding_event.set()
                            raise
                        continue  # try next attempt
                    else:
                        break  # success
                finally:
                    # Release all conn locks in reverse order
                    for c in reversed(acquired):
                        c.lock.release()
            else:
                # Exhausted retries — enter degraded mode so subsequent ops
                # get a clean RuntimeError instead of triggering more rebuilds
                # (or segfaulting from RDMA driver corruption).
                self._degraded = True
                self._rebuilding_event.set()
                logger.error(
                    "RDMA pool DEGRADED after %d failed rebuild attempts. "
                    "ALL operations will fail. Recovery: call enable_reconnect() "
                    "or recreate pool. Server: %s:%d",
                    rc.max_retries, self._addr, self._port,
                )
                raise original_exc
        finally:
            self._rebuild_lock.release()

        self._callbacks.fire_all()
        return resp

    def _roundtrip(self, opcode: int, body: bytes, flags: int = 0) -> protocol.Message:
        """Send a request and read the response on a round-robin connection.

        Handles reconnection for both non-owner (cheap) and owner (full
        rebuild) connection failures.
        """
        if self._degraded:
            raise RuntimeError(
                "RDMA pool is degraded (server unreachable after all retries). "
                "Close and recreate the pool, or restart the process.")
        if not self._rebuilding_event.is_set():
            raise RuntimeError("pool rebuilding, retry later")
        conn, conn_idx = self._next_conn_with_index()
        saved_exc = None
        fire_callbacks = False
        with conn.lock:
            try:
                return self._roundtrip_on_conn_inner(conn, opcode, body, flags)
            except Exception as exc:
                rc = self._reconnect_config
                if rc is None or not rc.enabled or not is_retriable(exc):
                    raise
                saved_exc = exc
                if conn_idx == 0:
                    # PD owner died — need full rebuild (release lock first)
                    pass
                else:
                    # Non-owner: cheap reconnect (PD/MRs survive)
                    try:
                        self._reconnect_non_owner(conn, exc)
                        resp = self._roundtrip_on_conn_inner(conn, opcode, body, flags)
                        fire_callbacks = True
                    except Exception as inner_exc:
                        if not is_retriable(inner_exc):
                            raise
                        saved_exc = inner_exc
                        # Non-owner reconnect failed — PD might be dead too
                        # Fall through to full rebuild

        if fire_callbacks:
            self._callbacks.fire_all()
            return resp

        # Full rebuild (outside conn lock)
        return self._full_rebuild_and_retry(opcode, body, flags, saved_exc)

    # --- Reconnect control ---

    @property
    def is_degraded(self) -> bool:
        """True if the pool has exhausted all rebuild retries."""
        return self._degraded

    def enable_reconnect(self, config: ReconnectConfig) -> None:
        """Enable automatic reconnection with the given config."""
        self._reconnect_config = config
        # Reset degraded state so the pool can attempt reconnection again
        self._degraded = False

    def disable_reconnect(self) -> None:
        """Disable automatic reconnection."""
        self._reconnect_config = None

    def set_reconnect_callback(self, name: str, fn) -> None:
        """Register a post-reconnect callback."""
        self._callbacks.register(name, fn)

    def set_timeout(self, seconds: float | None) -> None:
        self._timeout = seconds

    def close(self) -> None:
        atexit.unregister(self._atexit_close)
        # Shut down stripe executor — wait for in-flight striped ops to
        # finish before closing transports underneath them.
        if self._stripe_executor is not None:
            self._stripe_executor.shutdown(wait=True)
            self._stripe_executor = None

        # Deregister MRs via first transport (PD owner)
        first = self._conns[0]
        with first.lock:
            with self._mr_lock:
                for entry in self._mr_map.values():
                    try:
                        first.transport.dereg_mr(entry.mr_handle)
                    except Exception:
                        pass
                self._mr_map.clear()

                # Deregister MRs on independent-PD connections
                for conn_idx, ind_map in self._independent_mr_maps.items():
                    for entry in ind_map.values():
                        try:
                            self._conns[conn_idx].transport.dereg_mr(entry.mr_handle)
                        except Exception:
                            pass
                    ind_map.clear()

        # Close all transports (non-owners first, then owner)
        for conn in reversed(self._conns):
            got = conn.lock.acquire(timeout=2.0)
            try:
                conn.transport.close()
            except Exception:
                pass
            finally:
                if got:
                    conn.lock.release()

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

    # --- RDMA memory registration (shared PD) ---

    def reg_memory(self, ptr: int, size: int, buf: object = None) -> int:
        """Register on first transport's PD. lkey valid for all shared-PD connections.

        For independent-PD connections, registers separately on each.
        """
        with self._mr_lock:
            lkey, mr_handle = self._conns[0].transport.reg_mr(ptr, size)
            handle = self._next_handle
            self._next_handle += 1
            self._mr_map[handle] = _MREntry(lkey, mr_handle, buf, ptr, size)
            # Register on independent-PD connections too
            for conn_idx, ind_map in self._independent_mr_maps.items():
                ind_lkey, ind_mr = self._conns[conn_idx].transport.reg_mr(ptr, size)
                ind_map[handle] = _MREntry(ind_lkey, ind_mr, buf, ptr, size)
            return handle

    def dereg_memory(self, handle: int) -> None:
        with self._mr_lock:
            entry = self._mr_map.pop(handle, None)
            if entry is not None:
                self._conns[0].transport.dereg_mr(entry.mr_handle)
            # Deregister from independent-PD connections
            for conn_idx, ind_map in self._independent_mr_maps.items():
                ind_entry = ind_map.pop(handle, None)
                if ind_entry is not None:
                    try:
                        self._conns[conn_idx].transport.dereg_mr(ind_entry.mr_handle)
                    except Exception:
                        pass

    # --- Info (admin ops always on conn[0]) ---

    def info(self) -> dict:
        resp = self._roundtrip_on_conn0(protocol.OP_INFO, b"")
        return json.loads(resp.body)

    def stats(self) -> dict:
        resp = self._roundtrip_on_conn0(protocol.OP_STATS, b"")
        return json.loads(resp.body)

    def report_stats(self, stats: dict) -> None:
        stats = dict(stats)
        with self._counter_lock:
            stats["rdma_read_retries"] = self._rdma_read_retries
            stats["rdma_read_failures"] = self._rdma_read_failures
            # Report lifetime + current epoch for monotonic counters
            total_stripe_calls = self._lifetime_stripe_calls + self._stripe_calls
            total_stripe_nics = self._lifetime_stripe_nics_used + self._stripe_nics_used
            total_stripe_errors = self._lifetime_stripe_errors + self._stripe_errors
            stats["stripe_calls"] = total_stripe_calls
            stats["stripe_errors"] = total_stripe_errors
            stripe_avg = (total_stripe_nics / total_stripe_calls
                          if total_stripe_calls > 0 else 0.0)
            stats["stripe_avg_nics"] = round(stripe_avg, 2)
            total_write_calls = self._lifetime_stripe_write_calls + self._stripe_write_calls
            total_write_errors = self._lifetime_stripe_write_errors + self._stripe_write_errors
            stats["stripe_write_calls"] = total_write_calls
            stats["stripe_write_errors"] = total_write_errors
            for i, (reads, byt) in enumerate(
                    zip(self._per_nic_reads, self._per_nic_bytes)):
                lt_reads = self._lifetime_per_nic_reads[i] if i < len(self._lifetime_per_nic_reads) else 0
                lt_bytes = self._lifetime_per_nic_bytes[i] if i < len(self._lifetime_per_nic_bytes) else 0
                stats[f"nic_{i}_reads"] = lt_reads + reads
                stats[f"nic_{i}_bytes_gb"] = round((lt_bytes + byt) / (1 << 30), 4)
            for i, (writes, wbyt) in enumerate(
                    zip(self._per_nic_writes, self._per_nic_write_bytes)):
                lt_writes = self._lifetime_per_nic_writes[i] if i < len(self._lifetime_per_nic_writes) else 0
                lt_wbytes = self._lifetime_per_nic_write_bytes[i] if i < len(self._lifetime_per_nic_write_bytes) else 0
                stats[f"nic_{i}_writes"] = lt_writes + writes
                stats[f"nic_{i}_write_bytes_gb"] = round((lt_wbytes + wbyt) / (1 << 30), 4)
            stats["pool_rebuilds"] = self._pool_rebuilds
            # NIC balance (ops-based, link-rate-independent)
            nic_ops = []
            for i in range(len(self._per_nic_reads)):
                lt_r = self._lifetime_per_nic_reads[i] if i < len(self._lifetime_per_nic_reads) else 0
                lt_w = self._lifetime_per_nic_writes[i] if i < len(self._lifetime_per_nic_writes) else 0
                nic_ops.append(lt_r + self._per_nic_reads[i] + lt_w + self._per_nic_writes[i])
            stats["nic_balance_pct"] = compute_nic_balance(nic_ops)
        body = json.dumps(stats).encode()
        self._roundtrip_on_conn0(protocol.OP_REPORT_STATS, body)

    def get_transport_stats(self) -> dict:
        """Return aggregated C++ transport timing stats across all pool connections and reset counters."""
        total_rt_count = 0
        total_rd_count = 0
        total_rt_ns = 0
        total_rd_ns = 0
        total_batch_read_ns = 0
        total_batch_read_count = 0
        for conn in self._conns:
            try:
                s = conn.transport.get_stats()
                rt_count = s.get("roundtrip_count", 0)
                rd_count = s.get("rdma_read_count", 0)
                total_rt_count += rt_count
                total_rd_count += rd_count
                total_rt_ns += s.get("avg_roundtrip_us", 0) * rt_count * 1000
                total_rd_ns += s.get("avg_rdma_read_us", 0) * rd_count * 1000
                br_count = s.get("batch_read_count", 0)
                total_batch_read_count += br_count
                total_batch_read_ns += s.get("avg_batch_read_us", 0) * br_count * 1000
                conn.transport.reset_stats()
            except Exception:
                pass

        with self._counter_lock:
            stripe_calls = self._stripe_calls
            stripe_errors = self._stripe_errors
            stripe_avg_nics = (self._stripe_nics_used / stripe_calls
                               if stripe_calls > 0 else 0.0)
            per_nic_reads = list(self._per_nic_reads)
            per_nic_bytes = list(self._per_nic_bytes)
            stripe_write_calls = self._stripe_write_calls
            stripe_write_errors = self._stripe_write_errors
            per_nic_writes = list(self._per_nic_writes)
            per_nic_write_bytes = list(self._per_nic_write_bytes)
            # Swap-and-average sub-phase timings
            get_timings = self._get_timings; self._get_timings = []
            set_timings = self._set_timings; self._set_timings = []
            rdma_read_retries = self._rdma_read_retries
            rdma_read_failures = self._rdma_read_failures

        result = {
            "roundtrip_count": total_rt_count,
            "rdma_read_count": total_rd_count,
            "avg_roundtrip_us": (total_rt_ns / total_rt_count / 1000) if total_rt_count > 0 else 0.0,
            "avg_rdma_read_us": (total_rd_ns / total_rd_count / 1000) if total_rd_count > 0 else 0.0,
            "avg_batch_read_us": (total_batch_read_ns / total_batch_read_count / 1000) if total_batch_read_count > 0 else 0.0,
            "batch_read_count": total_batch_read_count,
            "stripe_calls": stripe_calls,
            "stripe_errors": stripe_errors,
            "stripe_avg_nics": round(stripe_avg_nics, 2),
            "per_nic_reads": per_nic_reads,
            "per_nic_bytes_gb": [b / (1 << 30) for b in per_nic_bytes],
            "stripe_write_calls": stripe_write_calls,
            "stripe_write_errors": stripe_write_errors,
            "per_nic_writes": per_nic_writes,
            "per_nic_write_bytes_gb": [b / (1 << 30) for b in per_nic_write_bytes],
            "rdma_read_retries": rdma_read_retries,
            "rdma_read_failures": rdma_read_failures,
        }

        # --- Saturation metrics (additive) ---
        nic_ops = [r + w for r, w in zip(per_nic_reads, per_nic_writes)]
        result["nic_balance_pct"] = compute_nic_balance(nic_ops)

        # GET sub-phase averages
        result.update(average_get_timings(get_timings))

        # SET sub-phase averages
        result.update(average_set_timings(set_timings))

        return result

    def rdma_endpoints(self) -> list[dict]:
        if self._server_info and "rdma_endpoints" in self._server_info:
            return self._server_info["rdma_endpoints"] or []
        info = self.info()
        return info.get("rdma_endpoints") or []

    # --- String ops ---

    def setstr(self, key: str, value: str) -> int:
        body, flags = protocol.encode_kv_body(key.encode(), value.encode())
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def getstr(self, key: str) -> str | None:
        body = protocol.encode_key_body(key.encode())
        conn = self._next_conn()
        with conn.lock:
            req_id = conn.next_req_id()
            hdr = protocol._pack_header(protocol.OP_GET, 0, req_id, body)
            raw_resp = conn.transport.roundtrip(hdr + body)
        resp = protocol.read_message_from_bytes(raw_resp)
        if resp.header.opcode == protocol.RESP_OOM:
            raise CamaOOMError.from_body(resp.body)
        if resp.header.opcode == protocol.RESP_ERROR:
            body_str = resp.body.decode(errors='replace')
            if "server overloaded" in body_str:
                raise CamaServerOverloadError(body_str)
            raise RuntimeError(f"CAMA error: {body_str}")

        if resp.header.opcode == protocol.OP_RDMA_READ_READY:
            rkey, remote_addr, length = protocol.decode_rdma_read_ready(resp.body)
            if length == 0:
                return None
            try:
                rd_conn = self._conn_with_read_buf(conn)
                with rd_conn.lock:
                    data = rd_conn.transport.rdma_read(rkey, remote_addr, length)
                self._send_read_ack_on(conn, 0)
                return data.decode(errors="replace")
            except RuntimeError:
                self._send_read_ack_on(conn, 255)
                with self._counter_lock:
                    self._rdma_read_retries += 1
                return None  # simplified retry for string ops

        value, found = protocol.decode_value_response(resp.body)
        if not found:
            return None
        return value.decode(errors="replace")

    def exists(self, key: str) -> int:
        body = protocol.encode_key_body(key.encode())
        resp = self._roundtrip(protocol.OP_TEST, body)
        if len(resp.body) > 0 and resp.body[0] == 1:
            return 1
        return 0

    def delete(self, key: str) -> int:
        body = protocol.encode_key_body(key.encode())
        self._roundtrip(protocol.OP_DELETE, body)
        return 0

    def _send_read_ack_on(self, conn: _PooledConn, wc_status: int) -> None:
        """Send ReadAck on a specific connection."""
        try:
            ack_body = protocol.encode_read_ack(wc_status)
            with conn.lock:
                req_id = conn.next_req_id()
                hdr = protocol._pack_header(protocol.OP_READ_ACK, 0, req_id, ack_body)
                conn.transport.roundtrip(hdr + ack_body)
        except Exception:
            pass

    # --- SGL ops ---

    def set(self, key: str, sgl: SGL, ttl_ms: int = 0) -> int:
        value = sgl.to_bytes()
        body, flags = protocol.encode_kv_body(key.encode(), value, ttl_ms=ttl_ms)
        self._roundtrip(protocol.OP_SET, body, flags=flags)
        return 0

    def get(self, key: str, sgl: SGL, size: int = 0) -> int:
        """Retrieve value into SGL. Holds ONE connection for entire GET flow.

        Critical: the GET->RDMA Read->ReadAck sequence must use the same
        connection to avoid interleaving with other operations.
        """
        body = protocol.encode_key_body(key.encode())
        conn, conn_idx = self._next_conn_with_index()

        # Phase 1: OP_GET roundtrip
        with conn.lock:
            req_id = conn.next_req_id()
            hdr = protocol._pack_header(protocol.OP_GET, 0, req_id, body)
            raw_resp = conn.transport.roundtrip(hdr + body)
        resp = protocol.read_message_from_bytes(raw_resp)
        if resp.header.opcode == protocol.RESP_OOM:
            raise CamaOOMError.from_body(resp.body)
        if resp.header.opcode == protocol.RESP_ERROR:
            body_str = resp.body.decode(errors='replace')
            if "server overloaded" in body_str:
                raise CamaServerOverloadError(body_str)
            raise RuntimeError(f"CAMA error: {body_str}")

        if resp.header.opcode == protocol.OP_RDMA_READ_READY:
            rkey, remote_addr, length = protocol.decode_rdma_read_ready(resp.body)
            if length == 0:
                return -1

            # Phase 2: RDMA Read (same connection)
            wc_status = self._do_rdma_read_on(conn, conn_idx, sgl, rkey, remote_addr, length)

            if wc_status != 0:
                status_name = _wc_status_name(wc_status)
                logger.warning("RDMA Read failed: %s (status=%d), retrying GET for key=%s",
                               status_name, wc_status, key)
                # Phase 2b: ReadAck (failure) + retry
                self._send_read_ack_on(conn, wc_status)
                with self._counter_lock:
                    self._rdma_read_retries += 1

                # Retry on same connection
                with conn.lock:
                    req_id = conn.next_req_id()
                    hdr = protocol._pack_header(protocol.OP_GET, 0, req_id, body)
                    raw_resp = conn.transport.roundtrip(hdr + body)
                resp2 = protocol.read_message_from_bytes(raw_resp)
                if resp2.header.opcode == protocol.OP_RDMA_READ_READY:
                    rkey2, addr2, len2 = protocol.decode_rdma_read_ready(resp2.body)
                    if len2 == 0:
                        return -1
                    wc2 = self._do_rdma_read_on(conn, conn_idx, sgl, rkey2, addr2, len2)
                    if wc2 != 0:
                        with self._counter_lock:
                            self._rdma_read_failures += 1
                        self._send_read_ack_on(conn, wc2)
                        raise RuntimeError(
                            f"RDMA Read failed after retry: {_wc_status_name(wc2)} (status={wc2})")
                    self._send_read_ack_on(conn, 0)
                else:
                    value, found = protocol.decode_value_response(resp2.body)
                    if not found:
                        return -1
                    sgl.from_bytes(value)
            else:
                # Phase 3: ReadAck (success)
                self._send_read_ack_on(conn, 0)
            return 0

        # Inline value
        value, found = protocol.decode_value_response(resp.body)
        if not found:
            return -1
        sgl.from_bytes(value)
        return 0

    def _do_rdma_read_on(self, conn: _PooledConn, conn_idx: int,
                         sgl: SGL,
                         rkey: int, remote_addr: int, length: int) -> int:
        """RDMA Read on a specific connection, returning WC status."""
        mr_entry = self._get_mr_entry(conn_idx, sgl.reg_handle)
        with conn.lock:
            if mr_entry is not None:
                try:
                    return conn.transport.try_rdma_read_into(
                        rkey, remote_addr, length, sgl.ptr, mr_entry.lkey)
                except RuntimeError:
                    return 255
            else:
                rd_conn = self._conn_with_read_buf(conn)
                if rd_conn is conn:
                    try:
                        data = conn.transport.rdma_read(rkey, remote_addr, length)
                        sgl.from_bytes(data)
                        return 0
                    except RuntimeError:
                        return 255
                else:
                    # Release current conn lock, acquire owner conn lock
                    conn.lock.release()
                    try:
                        with rd_conn.lock:
                            data = rd_conn.transport.rdma_read(rkey, remote_addr, length)
                        sgl.from_bytes(data)
                        return 0
                    except RuntimeError:
                        return 255
                    finally:
                        conn.lock.acquire()

    # --- Size guard ---

    def _fits_send_buf(self, body_size: int) -> bool:
        max_body = (self._send_buf_size or _DEFAULT_SEND_BUF_SIZE) - protocol.HEADER_SIZE
        return body_size <= max_body

    # --- Batch ops ---

    def mexists(self, keys: list[str]) -> list[int]:
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
        """Batch set with sub-batch chunking (same as RDMAClient.mset)."""
        if not keys:
            return []
        t0 = time.perf_counter()
        values = [sgl.to_bytes() for sgl in sgls]
        encoded_keys = [k.encode() for k in keys]
        body = protocol.encode_mset_body(encoded_keys, values)
        t_ser = time.perf_counter()

        if self._fits_send_buf(len(body)):
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

        max_body = (self._send_buf_size or _DEFAULT_SEND_BUF_SIZE) - protocol.HEADER_SIZE
        flags = protocol.FLAG_WITH_TTL if ttl_ms > 0 else protocol.FLAG_NONE
        all_results: list[int] = [0] * len(keys)
        chunk_keys: list[bytes] = []
        chunk_vals: list[bytes] = []
        chunk_indices: list[int] = []
        chunk_size = 4
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
            avg_val = sum(len(v) for v in values) // len(values)
            logger.warning(
                "mset: %d keys chunked into %d batches (~1 key/batch, avg value "
                "%d bytes vs %d MB send buffer). Increase send_buf_size for "
                "better batching.",
                len(keys), total_batches, avg_val, max_body // (1024 * 1024),
            )
        return all_results

    # --- Striped mset ---

    def _mset_on_conn(self, conn: _PooledConn, conn_idx: int,
                      keys: list[str], sgls: list[SGL],
                      ttl_ms: int = 0) -> tuple[list[int], int]:
        """Run mset with sub-batch chunking on a single connection.

        Returns (per-key status list, n_sub_batches) where n_sub_batches is
        the number of wire-level MSET round-trips actually sent.
        """
        values = [sgl.to_bytes() for sgl in sgls]
        encoded_keys = [k.encode() for k in keys]
        total_bytes = sum(len(v) for v in values)
        body = protocol.encode_mset_body(encoded_keys, values)
        flags = protocol.FLAG_WITH_TTL if ttl_ms > 0 else protocol.FLAG_NONE
        n_sub_batches = 0

        if self._fits_send_buf(len(body)):
            with conn.lock:
                resp = self._roundtrip_on_conn_inner(conn, protocol.OP_MSET, body, flags)
            all_results = self._parse_mset_response(resp, len(keys))
            n_sub_batches = 1
        else:
            max_body = (self._send_buf_size or _DEFAULT_SEND_BUF_SIZE) - protocol.HEADER_SIZE
            all_results: list[int] = [0] * len(keys)
            chunk_keys: list[bytes] = []
            chunk_vals: list[bytes] = []
            chunk_indices: list[int] = []
            chunk_size = 4

            for i in range(len(encoded_keys)):
                entry_size = 2 + len(encoded_keys[i]) + 4 + len(values[i])
                if chunk_size + entry_size > max_body and chunk_keys:
                    sub_body = protocol.encode_mset_body(chunk_keys, chunk_vals)
                    with conn.lock:
                        resp = self._roundtrip_on_conn_inner(conn, protocol.OP_MSET, sub_body, flags)
                    chunk_statuses = self._parse_mset_response(resp, len(chunk_keys))
                    for j, orig_idx in enumerate(chunk_indices):
                        all_results[orig_idx] = chunk_statuses[j]
                    n_sub_batches += 1
                    chunk_keys, chunk_vals, chunk_indices, chunk_size = [], [], [], 4
                if 4 + entry_size > max_body:
                    # Oversized single entry — send individually
                    single_body = protocol.encode_mset_body(
                        [encoded_keys[i]], [values[i]])
                    with conn.lock:
                        resp = self._roundtrip_on_conn_inner(conn, protocol.OP_MSET, single_body, flags)
                    all_results[i] = self._parse_mset_response(resp, 1)[0]
                    n_sub_batches += 1
                    continue
                chunk_keys.append(encoded_keys[i])
                chunk_vals.append(values[i])
                chunk_indices.append(i)
                chunk_size += entry_size

            if chunk_keys:
                sub_body = protocol.encode_mset_body(chunk_keys, chunk_vals)
                with conn.lock:
                    resp = self._roundtrip_on_conn_inner(conn, protocol.OP_MSET, sub_body, flags)
                chunk_statuses = self._parse_mset_response(resp, len(chunk_keys))
                for j, orig_idx in enumerate(chunk_indices):
                    all_results[orig_idx] = chunk_statuses[j]
                n_sub_batches += 1

        with self._counter_lock:
            self._per_nic_writes[conn_idx] += len(keys)
            self._per_nic_write_bytes[conn_idx] += total_bytes

        return all_results, n_sub_batches

    def mset_striped(self, keys: list[str], sgls: list[SGL],
                     ttl_ms: int = 0) -> list[int]:
        """Batch SET striped across pool connections in parallel.

        When pool_size > 1, partitions keys round-robin and writes each
        partition on a separate connection concurrently.
        Returns per-key status list (0=ok, nonzero=error).
        """
        if not keys:
            return []

        # Single-connection fast path
        if self._pool_size == 1 or self._stripe_executor is None:
            return self.mset(keys, sgls, ttl_ms=ttl_ms)

        if self._degraded:
            raise RuntimeError(
                "RDMA pool is degraded (server unreachable after all retries). "
                "Close and recreate the pool, or restart the process.")
        if not self._rebuilding_event.is_set():
            raise RuntimeError("pool rebuilding, retry later")

        t0 = time.perf_counter()
        n = len(keys)
        ps = self._pool_size
        all_results: list[int] = [0] * n

        # Phase 1: Partition keys round-robin
        part_indices: list[list[int]] = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)

        # Phase 2: Submit per-partition mset_on_conn
        futures = {}
        for conn_idx, indices in enumerate(part_indices):
            if not indices:
                continue
            if not self._conns[conn_idx].alive:
                for j in indices:
                    all_results[j] = -1
                logger.warning("mset_striped: conn %d dead, %d keys failed", conn_idx, len(indices))
                continue
            p_keys = [keys[j] for j in indices]
            p_sgls = [sgls[j] for j in indices]
            fut = self._stripe_executor.submit(
                self._mset_on_conn,
                self._conns[conn_idx], conn_idx,
                p_keys, p_sgls, ttl_ms,
            )
            futures[fut] = (conn_idx, indices)

        with self._counter_lock:
            self._stripe_write_calls += 1

        # Phase 3: Collect results
        total_sub_batches = 0
        stripe_timeout = self._timeout if self._timeout is not None else 30.0
        try:
            for fut in as_completed(futures, timeout=stripe_timeout):
                conn_idx, indices = futures[fut]
                try:
                    part_results, part_sub_batches = fut.result()
                    total_sub_batches += part_sub_batches
                    for j, orig_idx in enumerate(indices):
                        all_results[orig_idx] = part_results[j]
                except Exception as exc:
                    logger.warning("Striped mset failed on conn %d: %s", conn_idx, exc)
                    with self._counter_lock:
                        self._stripe_write_errors += 1
                    raise
        except TimeoutError:
            logger.warning("Striped mset timed out after %.1fs", stripe_timeout)
            with self._counter_lock:
                self._stripe_write_errors += 1
            raise RuntimeError(f"mset_striped timed out after {stripe_timeout:.1f}s")

        t_total = time.perf_counter()
        # Estimate total bytes from SGL sizes (avoids redundant to_bytes() copy)
        total_bytes = sum(sgl.size for sgl in sgls)
        with self._counter_lock:
            self._set_timings.append({
                "t_serialize_ms": 0.0,  # serialization done per-conn
                "t_send_ms": (t_total - t0) * 1000,
                "n_keys": n,
                "total_bytes": total_bytes,
                "n_sub_batches": total_sub_batches,
            })

        return all_results

    def mget(self, keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]:
        results = []
        for i, k in enumerate(keys):
            rc = self.get(k, sgls[i])
            results.append(rc)
        return results

    def _get_mr_entry(self, conn_idx: int, reg_handle: int) -> _MREntry | None:
        """Look up MR entry for a connection, handling independent-PD conns."""
        with self._mr_lock:
            if conn_idx in self._independent_pd_conns:
                return self._independent_mr_maps[conn_idx].get(reg_handle)
            return self._mr_map.get(reg_handle)

    def mget_rdma(self, keys: list[str], sgls: list[SGL],
                  sizes: list[int] | None = None) -> list[int]:
        """Batch GET with RDMA Read — striped across NICs when pool_size > 1.

        Falls back to sequential mget() if server doesn't support mget_rdma.
        Returns list of return codes (0=found, -1=miss).
        """
        if not self._has_mget_rdma:
            return self.mget(keys, sgls, sizes)

        n = len(keys)
        if n == 0:
            return []

        # Wait for any in-progress rebuild
        if self._degraded:
            raise RuntimeError(
                "RDMA pool is degraded (server unreachable after all retries). "
                "Close and recreate the pool, or restart the process.")
        if not self._rebuilding_event.is_set():
            raise RuntimeError("pool rebuilding, retry later")

        # Single-connection fast path (no threading overhead)
        if self._pool_size == 1 or self._stripe_executor is None:
            return self._mget_rdma_on_conn(
                self._conns[0], 0, keys, sgls, sizes)

        return self._mget_rdma_striped(keys, sgls, sizes)

    def _mget_rdma_on_conn(self, conn: _PooledConn, conn_idx: int,
                            keys: list[str], sgls: list[SGL],
                            sizes: list[int] | None = None) -> list[int]:
        """Run the full mget_rdma flow on a single connection.

        Returns list of return codes aligned to the input keys/sgls.
        """
        if not keys:
            return []

        handle = TransportHandle(
            conn.transport, conn.lock,
            conn.next_req_id,
            lambda reg_handle, ci=conn_idx: self._get_mr_entry(ci, reg_handle),
        )
        results, timing_dict, read_count, total_bytes = mget_rdma_flow(
            handle, keys, sgls, sizes)

        if timing_dict is not None:
            with self._counter_lock:
                self._per_nic_reads[conn_idx] += read_count
                self._per_nic_bytes[conn_idx] += total_bytes
                self._get_timings.append(timing_dict)

        return results

    def _mget_rdma_striped(self, keys: list[str], sgls: list[SGL],
                            sizes: list[int] | None = None) -> list[int]:
        """Stripe mget_rdma across pool connections in parallel.

        Phase 1: Partition keys round-robin across connections.
        Phase 2: Run _mget_rdma_on_conn in parallel (one thread per active conn).
        Phase 3: Merge per-partition results back to original key order.
        """
        n = len(keys)
        ps = self._pool_size

        # Phase 1: Partition
        # partitions[conn_idx] = list of (original_idx, key, sgl, size)
        part_indices: list[list[int]] = [[] for _ in range(ps)]
        for i in range(n):
            conn_idx = i % ps
            part_indices[conn_idx].append(i)

        # Phase 2: Submit parallel tasks (skip empty partitions)
        futures = {}
        active_nics = 0
        for conn_idx, indices in enumerate(part_indices):
            if not indices:
                continue
            if not self._conns[conn_idx].alive:
                continue
            active_nics += 1
            p_keys = [keys[j] for j in indices]
            p_sgls = [sgls[j] for j in indices]
            p_sizes = [sizes[j] for j in indices] if sizes else None
            fut = self._stripe_executor.submit(
                self._mget_rdma_on_conn,
                self._conns[conn_idx], conn_idx,
                p_keys, p_sgls, p_sizes,
            )
            futures[fut] = (conn_idx, indices)

        # Update stripe metrics
        with self._counter_lock:
            self._stripe_calls += 1
            self._stripe_nics_used += active_nics

        # Phase 3: Collect and merge
        results = [-1] * n
        stripe_timeout = self._timeout if self._timeout is not None else 30.0
        try:
            for fut in as_completed(futures, timeout=stripe_timeout):
                conn_idx, indices = futures[fut]
                try:
                    part_results = fut.result()
                    for j, orig_idx in enumerate(indices):
                        results[orig_idx] = part_results[j]
                except Exception as exc:
                    logger.warning("Striped mget_rdma failed on conn %d: %s", conn_idx, exc)
                    with self._counter_lock:
                        self._stripe_errors += 1
                    # Leave results as -1 for this partition's keys
        except TimeoutError:
            logger.warning("Striped mget_rdma timed out after %.1fs, %d/%d futures incomplete",
                           stripe_timeout, sum(1 for f in futures if not f.done()), len(futures))
            with self._counter_lock:
                self._stripe_errors += 1

        return results

    def _mget_rdma_raw_on_conn(self, conn: _PooledConn, conn_idx: int,
                               keys: list[str]) -> list[tuple[int, bytes | None]]:
        """Run mget_rdma_raw_flow on a single connection.

        Returns list of (rc, data) per key aligned to input keys.
        """
        if not keys:
            return []

        handle = TransportHandle(
            conn.transport, conn.lock,
            conn.next_req_id,
            lambda reg_handle, ci=conn_idx: self._get_mr_entry(ci, reg_handle),
        )
        results, timing_dict, found_count = mget_rdma_raw_flow(handle, keys)

        if timing_dict is not None:
            with self._counter_lock:
                self._per_nic_reads[conn_idx] += found_count
                self._per_nic_bytes[conn_idx] += timing_dict.get("total_bytes", 0)
                self._get_timings.append(timing_dict)

        return results

    def _mget_rdma_raw_striped(self, keys: list[str]) -> list[tuple[int, bytes | None]]:
        """Stripe mget_rdma_raw across pool connections in parallel.

        Same pattern as _mget_rdma_striped: round-robin partition, parallel
        submit, merge results back to original order.
        """
        n = len(keys)
        ps = self._pool_size

        # Phase 1: Partition keys round-robin
        part_indices: list[list[int]] = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)

        # Phase 2: Submit parallel tasks
        futures = {}
        active_nics = 0
        for conn_idx, indices in enumerate(part_indices):
            if not indices:
                continue
            if not self._conns[conn_idx].alive:
                continue
            active_nics += 1
            p_keys = [keys[j] for j in indices]
            fut = self._stripe_executor.submit(
                self._mget_rdma_raw_on_conn,
                self._conns[conn_idx], conn_idx, p_keys,
            )
            futures[fut] = (conn_idx, indices)

        with self._counter_lock:
            self._stripe_calls += 1
            self._stripe_nics_used += active_nics

        # Phase 3: Collect and merge
        results: list[tuple[int, bytes | None]] = [(-1, None)] * n
        stripe_timeout = self._timeout if self._timeout is not None else 30.0
        try:
            for fut in as_completed(futures, timeout=stripe_timeout):
                conn_idx, indices = futures[fut]
                try:
                    part_results = fut.result()
                    for j, orig_idx in enumerate(indices):
                        results[orig_idx] = part_results[j]
                except Exception as exc:
                    logger.warning("Striped mget_rdma_raw failed on conn %d: %s", conn_idx, exc)
                    with self._counter_lock:
                        self._stripe_errors += 1
        except TimeoutError:
            logger.warning("Striped mget_rdma_raw timed out after %.1fs, %d/%d futures incomplete",
                           stripe_timeout, sum(1 for f in futures if not f.done()), len(futures))
            with self._counter_lock:
                self._stripe_errors += 1

        return results

    def mget_rdma_raw(self, keys: list[str]) -> list[tuple[int, bytes | None]]:
        """Batch GET returning raw bytes via internal read buffer.

        Stripes across all pool connections when pool_size > 1.
        Returns list of (rc, data) per key: (0, bytes) for hit, (-1, None)
        for miss.
        """
        if not self._has_mget_rdma:
            # Fallback: sequential get() returning raw bytes, round-robin across conns
            results: list[tuple[int, bytes | None]] = []
            for k in keys:
                conn, conn_idx = self._next_conn_with_index()
                body = protocol.encode_key_body(k.encode())
                with conn.lock:
                    req_id = conn.next_req_id()
                    hdr = protocol._pack_header(protocol.OP_GET, 0, req_id, body)
                    raw_resp = conn.transport.roundtrip(hdr + body)
                resp = protocol.read_message_from_bytes(raw_resp)
                if resp.header.opcode == protocol.OP_RDMA_READ_READY:
                    rkey, addr, length = protocol.decode_rdma_read_ready(resp.body)
                    if length == 0:
                        results.append((-1, None))
                    else:
                        rd_conn = self._conn_with_read_buf(conn)
                        with rd_conn.lock:
                            raw = rd_conn.transport.rdma_read(rkey, addr, length)
                        self._send_read_ack_on(conn, 0)
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

        if self._degraded:
            raise RuntimeError(
                "RDMA pool is degraded (server unreachable after all retries). "
                "Close and recreate the pool, or restart the process.")
        if not self._rebuilding_event.is_set():
            raise RuntimeError("pool rebuilding, retry later")

        # Single-connection fast path
        if self._pool_size == 1 or self._stripe_executor is None:
            return self._mget_rdma_raw_on_conn(self._conns[0], 0, keys)

        return self._mget_rdma_raw_striped(keys)

    def mdel(self, keys: list[str]) -> int:
        if not keys:
            return 0
        body = protocol.encode_mget_body([k.encode() for k in keys])
        self._roundtrip(protocol.OP_MDEL, body)
        return 0

    # --- Key scan ---

    def keys(self, pattern: str = ".*") -> list[str]:
        body = protocol.encode_keys_body(pattern.encode())
        resp = self._roundtrip_on_conn0(protocol.OP_KEYS, body)
        raw_keys = protocol.decode_keys_response(resp.body)
        return [k.decode(errors="replace") for k in raw_keys]

    # --- Maintenance ---

    def vacuum(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        req: dict = {"action": "vacuum", "force": force}
        if shard_ids is not None:
            req["shard_ids"] = shard_ids
        resp = self._roundtrip_on_conn0(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    def autotune(self, *, force: bool = False, shard_ids: list[int] | None = None) -> dict:
        req: dict = {"action": "autotune", "force": force}
        if shard_ids is not None:
            req["shard_ids"] = shard_ids
        resp = self._roundtrip_on_conn0(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    def maintenance_status(self) -> dict:
        req = {"action": "status"}
        resp = self._roundtrip_on_conn0(protocol.OP_MAINTENANCE, json.dumps(req).encode())
        return json.loads(resp.body)

    def flush(self) -> int:
        self._roundtrip_on_conn0(protocol.OP_FLUSH, b"")
        return 0

    def lease(self, key: str, duration_ms: int) -> int:
        body = struct.pack("<H", len(key.encode())) + key.encode() + struct.pack("<q", duration_ms)
        self._roundtrip_on_conn0(protocol.OP_LEASE, body)
        return 0

    def pin(self, key: str) -> int:
        body = protocol.encode_key_body(key.encode())
        self._roundtrip_on_conn0(protocol.OP_PIN, body)
        return 0

    def unpin(self, key: str) -> int:
        body = protocol.encode_key_body(key.encode())
        self._roundtrip_on_conn0(protocol.OP_UNPIN, body)
        return 0
