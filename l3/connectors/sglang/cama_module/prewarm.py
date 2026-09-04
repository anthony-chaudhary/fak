"""Background connection pre-warming for CAMA/PrisKV storage backend.

Establishes RDMA/TCP connection pools in the background during model loading,
so that CamaStorage.__init__ can adopt a ready connection instead of blocking.

The prewarm now performs the *full* connection setup that CamaStorage would do:
  1. Create initial pool/client
  2. Discover RDMA endpoints
  3. Recreate with multi-NIC striping (if applicable)
  4. Keep the connection alive with periodic pings

This means CamaStorage.__init__ can skip step [3/7] entirely when a pre-warmed
connection is available, saving the expensive RDMA MR registration + shared-PD
negotiation that now takes significantly longer with 16 MB buffers.

Usage:
    # In preflight (after connectivity probe succeeds):
    start_cama_prewarm(addr, port, password, pool_size, send_buf_size)

    # In CamaStorage.__init__ (before normal connect):
    conn = claim_prewarmed_connection(addr, port, pool_size, send_buf_size)
"""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[prewarm]"

# Module-level state for wait-for-prewarm synchronization.
# When start_cama_prewarm() fires, it creates an Event that is set once
# the daemon finishes.  claim_prewarmed_connection() can wait on it so
# that fast model loads don't race past the prewarm thread.
_prewarm_ready: Optional[threading.Event] = None
_prewarm_start_time: Optional[float] = None
_prewarm_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


@dataclass
class PrewarmResult:
    """Holds a pre-warmed connection and metadata."""

    provider_name: str
    connection: Any  # The pool or client object
    fingerprint: str  # Config fingerprint for match verification
    created_at: float = field(default_factory=time.monotonic)
    error: Optional[Exception] = None
    endpoints: Optional[list] = None  # Discovered RDMA endpoints (dicts)
    _keepalive_stop: Optional[threading.Event] = field(default=None, repr=False)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.created_at

    def close(self) -> None:
        # Stop keepalive first so it doesn't fire on a closing connection.
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
            self._keepalive_stop = None
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                logger.debug(
                    "%s error closing connection for %s",
                    _LOG_PREFIX, self.provider_name, exc_info=True,
                )
            self.connection = None


class PrewarmProvider(ABC):
    """Abstract provider that knows how to create a pre-warmed connection."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create_connection(self) -> PrewarmResult:
        ...


# ---------------------------------------------------------------------------
# Registry — process-global, thread-safe store/claim
# ---------------------------------------------------------------------------

# 15 minutes — large-model loading (70B+) routinely exceeds the old 5 min TTL,
# causing the reaper to kill the connection before CamaStorage.__init__ runs.
_PREWARM_TTL_S = 900.0
_REAPER_INTERVAL_S = 60.0

# Keepalive pings prevent idle-connection timeouts on both the RDMA CM channel
# and the TCP side while the model is still loading.
_KEEPALIVE_INTERVAL_S = 60.0


class PrewarmRegistry:
    """Thread-safe singleton that stores one PrewarmResult per provider name."""

    _instance: Optional["PrewarmRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._store: dict[str, PrewarmResult] = {}
        self._lock = threading.Lock()
        self._reaper_started = False

    @classmethod
    def get(cls) -> "PrewarmRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, result: PrewarmResult) -> None:
        with self._lock:
            old = self._store.pop(result.provider_name, None)
            if old is not None:
                old.close()
                logger.info(
                    "%s closed unclaimed connection for %s (age=%.1fs)",
                    _LOG_PREFIX, result.provider_name, old.age_s,
                )
            self._store[result.provider_name] = result
            self._ensure_reaper()

    def claim(
        self,
        provider_name: str,
        match_fn: Optional[Callable[[PrewarmResult], bool]] = None,
    ) -> Optional[PrewarmResult]:
        """Atomically claim a pre-warmed result.

        Returns the full PrewarmResult (connection + endpoints), or None if:
        - No entry exists for this provider
        - Entry has an error
        - Entry has expired (>TTL)
        - match_fn returns False (fingerprint mismatch)
        """
        with self._lock:
            result = self._store.pop(provider_name, None)

        if result is None:
            return None

        if result.error is not None:
            logger.warning(
                "%s discarding failed pre-warm for %s: %s",
                _LOG_PREFIX, provider_name, result.error,
            )
            result.close()
            return None

        if result.age_s > _PREWARM_TTL_S:
            logger.info(
                "%s pre-warmed connection expired for %s (age=%.1fs > %.0fs TTL)",
                _LOG_PREFIX, provider_name, result.age_s, _PREWARM_TTL_S,
            )
            result.close()
            return None

        if match_fn is not None and not match_fn(result):
            logger.warning(
                "%s config mismatch for %s, discarding pre-warmed connection",
                _LOG_PREFIX, provider_name,
            )
            result.close()
            return None

        # Stop keepalive — caller now owns the connection.
        if result._keepalive_stop is not None:
            result._keepalive_stop.set()
            result._keepalive_stop = None

        logger.info(
            "%s claimed pre-warmed connection for %s (age=%.2fs, endpoints=%d)",
            _LOG_PREFIX, provider_name, result.age_s,
            len(result.endpoints) if result.endpoints else 0,
        )
        return result

    def _ensure_reaper(self) -> None:
        """Start the reaper daemon thread if not already running."""
        if self._reaper_started:
            return
        self._reaper_started = True
        t = threading.Thread(target=self._reaper_loop, daemon=True, name="cama-prewarm-reaper")
        t.start()

    def _reaper_loop(self) -> None:
        """Periodically close expired, unclaimed connections."""
        while True:
            time.sleep(_REAPER_INTERVAL_S)
            try:
                with self._lock:
                    expired = [
                        name for name, r in self._store.items()
                        if r.age_s > _PREWARM_TTL_S
                    ]
                    for name in expired:
                        result = self._store.pop(name)
                        result.close()
                        logger.debug(
                            "%s reaped expired connection for %s (age=%.1fs)",
                            _LOG_PREFIX, name, result.age_s,
                        )
                    if not self._store:
                        self._reaper_started = False
                        return  # Exit thread when registry is empty
            except Exception:
                logger.error("%s reaper loop error", _LOG_PREFIX, exc_info=True)


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------

def _start_keepalive(conn: Any, stop_event: threading.Event) -> None:
    """Ping the connection periodically to keep it alive during model loading."""

    def _loop() -> None:
        while not stop_event.wait(_KEEPALIVE_INTERVAL_S):
            try:
                conn.exists("__cama_keepalive__")
            except Exception as exc:
                logger.warning(
                    "%s keepalive ping FAILED: %s — connection is likely dead. "
                    "CamaStorage may need to reconnect on init. Check server availability.",
                    _LOG_PREFIX, exc)
                return  # Connection is broken — stop pinging.

    t = threading.Thread(target=_loop, daemon=True, name="cama-prewarm-keepalive")
    t.start()


# ---------------------------------------------------------------------------
# CAMA provider
# ---------------------------------------------------------------------------

def _cama_config_fingerprint(
    addr: str, port: int, pool_size: int, send_buf_size: int,
) -> str:
    """Deterministic fingerprint for connection config matching."""
    raw = f"{addr}:{port}:pool={pool_size}:sbuf={send_buf_size}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class CamaPrewarmProvider(PrewarmProvider):
    """Creates a CAMA connection pool/client exactly as cama_storage.py does.

    Performs the full three-phase setup so that CamaStorage.__init__ can
    adopt the connection without any additional reconnection:
      Phase 1: Initial pool/client connection
      Phase 2: RDMA endpoint discovery
      Phase 3: Multi-NIC pool recreation (if endpoints > 1 and nic_striping)
    """

    def __init__(
        self,
        addr: str,
        port: int,
        password: str,
        pool_size: int = 8,
        send_buf_size: int = 0,
        nic_striping: bool = True,
        model_page_bytes: int = 0,
    ) -> None:
        self._addr = addr
        self._port = port
        self._password = password
        self._pool_size = pool_size
        self._send_buf_size = send_buf_size
        self._nic_striping = nic_striping
        self._model_page_bytes = model_page_bytes

    @property
    def name(self) -> str:
        return "cama"

    def create_connection(self) -> PrewarmResult:
        fingerprint = _cama_config_fingerprint(
            self._addr, self._port, self._pool_size, self._send_buf_size,
        )
        try:
            start = time.monotonic()

            try:
                from sglang.srt.mem_cache.storage.cama.cama_storage import _make_connection
            except ImportError:
                from cama_module.cama_storage import _make_connection

            # -- Phase 1: initial connection --
            logger.debug(
                "%s [1/3] connecting to %s:%d (pool_size=%d, send_buf=%d)...",
                _LOG_PREFIX, self._addr, self._port,
                self._pool_size, self._send_buf_size,
            )
            conn = _make_connection(
                self._addr, self._port, self._password,
                self._pool_size, send_buf_size=self._send_buf_size,
            )
            transport = type(conn).__name__
            phase1_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "%s [1/3] initial pool ready in %.0fms (transport=%s)",
                _LOG_PREFIX, phase1_ms, transport,
            )

            # -- Phase 2: discover RDMA endpoints --
            endpoints: list[dict] = []
            if hasattr(conn, "rdma_endpoints"):
                try:
                    phase2_start = time.monotonic()
                    endpoints = conn.rdma_endpoints()
                    phase2_ms = (time.monotonic() - phase2_start) * 1000
                    logger.debug(
                        "%s [2/3] discovered %d RDMA endpoint(s) in %.0fms%s",
                        _LOG_PREFIX, len(endpoints), phase2_ms,
                        (" — " + ", ".join(
                            f"{ep.get('device', '?')}@{ep['ip']}:{ep['port']}"
                            for ep in endpoints
                        )) if endpoints else "",
                    )
                except Exception as exc:
                    logger.warning(
                        "%s [2/3] endpoint discovery failed: %s", _LOG_PREFIX, exc,
                    )
            else:
                logger.debug(
                    "%s [2/3] TCP transport, no RDMA endpoint discovery",
                    _LOG_PREFIX,
                )

            # -- Phase 3: recreate pool with multi-NIC striping --
            if len(endpoints) > 1 and self._nic_striping:
                phase3_start = time.monotonic()
                ep_list = [(ep["ip"], int(ep["port"])) for ep in endpoints]
                target_ip, target_port = ep_list[0]
                buf_desc = (
                    f"{self._send_buf_size / (1 << 20):.0f} MB"
                    if self._send_buf_size > 0 else "16 MB (default)"
                )
                logger.debug(
                    "%s [3/3] recreating pool with %d NIC endpoints for striping "
                    "(ibv_reg_mr for %s send+recv buffers per connection)...",
                    _LOG_PREFIX, len(ep_list), buf_desc,
                )
                conn.close()
                conn = _make_connection(
                    target_ip, target_port, self._password,
                    self._pool_size,
                    send_buf_size=self._send_buf_size,
                    endpoints=ep_list,
                )
                phase3_ms = (time.monotonic() - phase3_start) * 1000
                logger.debug(
                    "%s [3/3] multi-NIC pool ready in %.0fms (%d endpoints)",
                    _LOG_PREFIX, phase3_ms, len(ep_list),
                )
            else:
                reason = (
                    "single endpoint" if len(endpoints) <= 1
                    else "nic_striping disabled"
                )
                logger.debug(
                    "%s [3/3] skipped multi-NIC recreation (%s)", _LOG_PREFIX, reason,
                )

            # -- Start keepalive pings --
            stop_event = threading.Event()
            _start_keepalive(conn, stop_event)

            # -- Send early page-size hint (before model loading) --
            if self._model_page_bytes > 0:
                try:
                    conn.report_stats({"model_page_bytes": self._model_page_bytes})
                    logger.debug(
                        "%s sent early model_page_bytes hint: %d",
                        _LOG_PREFIX, self._model_page_bytes,
                    )
                except Exception as exc:
                    logger.debug(
                        "%s failed to send early page-size hint: %s",
                        _LOG_PREFIX, exc,
                    )

            total_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "%s fully pre-warmed in %.0fms "
                "(transport=%s, pool_size=%d, endpoints=%d, "
                "keepalive=%ds, TTL=%ds)",
                _LOG_PREFIX, total_ms, transport, self._pool_size,
                len(endpoints), _KEEPALIVE_INTERVAL_S, int(_PREWARM_TTL_S),
            )
            return PrewarmResult(
                provider_name=self.name,
                connection=conn,
                fingerprint=fingerprint,
                endpoints=endpoints,
                _keepalive_stop=stop_event,
            )

        except Exception as exc:
            logger.warning("%s connection failed: %s", _LOG_PREFIX, exc)
            return PrewarmResult(
                provider_name=self.name,
                connection=None,
                fingerprint=fingerprint,
                error=exc,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_cama_prewarm(
    addr: str,
    port: int,
    password: str = "",
    pool_size: int = 8,
    send_buf_size: int = 0,
    nic_striping: bool = True,
    model_page_bytes: int = 0,
) -> None:
    """Spawn a daemon thread that pre-warms a CAMA connection pool.

    Performs the full connection setup including RDMA endpoint discovery
    and multi-NIC pool creation, so CamaStorage.__init__ can skip these
    expensive steps entirely.

    If *model_page_bytes* > 0, an early page-size hint is sent to the server
    during prewarm so slab allocators can be optimised before model loading.

    Safe to call from preflight — any failure is logged as a warning and
    does not affect the caller.
    """
    global _prewarm_ready, _prewarm_start_time

    provider = CamaPrewarmProvider(
        addr, port, password, pool_size, send_buf_size,
        nic_striping=nic_striping,
        model_page_bytes=model_page_bytes,
    )

    ready_event = threading.Event()
    with _prewarm_state_lock:
        _prewarm_ready = ready_event
        _prewarm_start_time = time.monotonic()

    def _run() -> None:
        try:
            result = provider.create_connection()
            PrewarmRegistry.get().store(result)
        finally:
            ready_event.set()

    buf_desc = f"{send_buf_size / (1 << 20):.0f} MB" if send_buf_size > 0 else "16 MB (default)"
    logger.debug(
        "%s background pre-warm started (addr=%s:%d, pool_size=%d, "
        "send_buf=%s, nic_striping=%s)",
        _LOG_PREFIX, addr, port, pool_size, buf_desc, nic_striping,
    )
    t = threading.Thread(target=_run, daemon=True, name="cama-prewarm")
    t.start()


def claim_prewarmed_connection(
    addr: str,
    port: int,
    pool_size: int = 8,
    send_buf_size: int = 0,
    wait_timeout: float = 30.0,
) -> Optional[PrewarmResult]:
    """Claim a pre-warmed result if one exists and config matches.

    If a prewarm daemon is still running, waits up to *wait_timeout*
    seconds for it to finish before falling back.  This prevents the
    common race where CamaStorage.__init__ starts before the prewarm
    thread completes (e.g. fast/cached model loads), which would waste
    the prewarm and duplicate the expensive MR registration.

    Returns a PrewarmResult with .connection and .endpoints, or None.
    """
    with _prewarm_state_lock:
        event = _prewarm_ready
        started_at = _prewarm_start_time

    if event is not None and not event.is_set():
        elapsed = time.monotonic() - started_at if started_at else 0
        logger.debug(
            "%s prewarm still in progress (%.1fs elapsed) — "
            "waiting up to %.0fs for background connection to finish "
            "(avoids duplicate MR registration)...",
            _LOG_PREFIX, elapsed, wait_timeout,
        )
        finished = event.wait(timeout=wait_timeout)
        if finished:
            wait_ms = (time.monotonic() - started_at - elapsed) * 1000 if started_at else 0
            logger.debug(
                "%s prewarm completed (waited %.0fms), claiming connection",
                _LOG_PREFIX, wait_ms,
            )
        else:
            total = time.monotonic() - started_at if started_at else wait_timeout
            logger.warning(
                "%s prewarm did NOT finish within %.0fs (total %.1fs elapsed) — "
                "falling back to fresh connection. This is unusual; check server "
                "connectivity or RDMA device availability.",
                _LOG_PREFIX, wait_timeout, total,
            )
            # Jitter before fallback to prevent thundering herd
            import random
            time.sleep(random.uniform(0, 2.0))
    elif event is None:
        logger.debug(
            "%s no prewarm was started (preflight may have been skipped)",
            _LOG_PREFIX,
        )

    expected_fp = _cama_config_fingerprint(addr, port, pool_size, send_buf_size)
    return PrewarmRegistry.get().claim(
        "cama",
        match_fn=lambda r: r.fingerprint == expected_fp,
    )
