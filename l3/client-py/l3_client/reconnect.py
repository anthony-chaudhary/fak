"""Reconnection engine for cama-client TCP and RDMA transports.

Provides:
- ReconnectConfig: backoff and retry parameters
- _MREntry: enhanced MR tracking with ptr/size for re-registration
- is_retriable(): exception classifier
- compute_delay(): exponential backoff with jitter
- ReconnectCallbackRegistry: post-reconnect hook system
"""

from __future__ import annotations

import errno
import logging
import random
import socket
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- errno values that indicate a retriable transport error ----
_RETRIABLE_ERRNOS = frozenset({
    errno.EPIPE,
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ENOTCONN,
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
})

# ---- RDMA error substrings that indicate a retriable transport error ----
_RETRIABLE_RDMA_SUBSTRINGS = (
    "not connected",
    "WR_FLUSH_ERR",
    "RETRY_EXC_ERR",
    "RESP_TIMEOUT_ERR",
    "FATAL_ERR",
    "connection",
)

# ---- C++ transport throw sites (std::runtime_error) ----
_RETRIABLE_CPP_SUBSTRINGS = (
    "ibv_post_send failed",
    "ibv_post_recv failed",
    "ibv_poll_cq error",
    "RDMA CM event",
    "rdma_resolve",
    "rdma_getaddrinfo",
    "WC error",
    "poll timeout",
    "transport closed during poll",
)


@dataclass
class ReconnectConfig:
    """Reconnection parameters for backoff and retry.

    Default budget: 0.5+1+2+4+8+16+30+30+30+30 ≈ 152s (covers ~2.5min downtime).
    Previous default (max_retries=5) only covered ~15s which was insufficient
    for typical server restarts (20-60s).
    """

    enabled: bool = True
    max_retries: int = 10
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: float = 0.1  # +-10% random jitter on delay


def _resolve_reconnect(
    reconnect: bool | ReconnectConfig | None,
) -> ReconnectConfig | None:
    """Resolve a reconnect kwarg to a ReconnectConfig or None.

    True  -> default ReconnectConfig
    ReconnectConfig -> use as-is
    False/None -> disabled (None)
    """
    if reconnect is True:
        return ReconnectConfig()
    if isinstance(reconnect, ReconnectConfig):
        return reconnect
    return None


class _MREntry:
    """Enhanced MR tracking that preserves ptr/size for re-registration.

    After a reconnect the RDMA PD is replaced, so all MRs must be
    re-registered.  The original ``(lkey, mr_handle, buf_ref)`` tuple
    doesn't store the registration parameters; _MREntry does.
    """

    __slots__ = ("lkey", "mr_handle", "buf_ref", "ptr", "size")

    def __init__(self, lkey: int, mr_handle: int, buf_ref: object,
                 ptr: int, size: int):
        self.lkey = lkey
        self.mr_handle = mr_handle
        self.buf_ref = buf_ref
        self.ptr = ptr
        self.size = size


def is_retriable(exc: BaseException) -> bool:
    """Return True if *exc* represents a transient transport error.

    **Not retriable**: ``RuntimeError("CAMA error: ...")`` — these are
    server-side application errors with a specific prefix added in
    ``_roundtrip()``.  Also not retriable: ``ValueError``,
    ``AssertionError``, ``MemoryError``.
    """
    # Application-level errors are never retriable
    from l3_client.errors import CamaNotReadyError, CamaServerOverloadError
    if isinstance(exc, (ValueError, AssertionError, MemoryError, CamaServerOverloadError, CamaNotReadyError)):
        return False

    # TCP retriable errors
    if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                        ConnectionRefusedError, ConnectionAbortedError)):
        return True
    if isinstance(exc, socket.timeout):
        return True

    # OSError with retriable errno
    if isinstance(exc, OSError) and exc.errno in _RETRIABLE_ERRNOS:
        return True

    # RuntimeError: distinguish CAMA server errors from transport errors
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        # Server-side application errors have this prefix
        if msg.startswith("CAMA error: "):
            return False
        # Pool rebuild in progress — caller should retry after rebuild completes
        if "pool rebuilding" in msg:
            return True
        # RDMA transport errors (C++ throw std::runtime_error sites)
        for sub in _RETRIABLE_RDMA_SUBSTRINGS:
            if sub in msg:
                return True
        # Known C++ transport throw sites
        for sub in _RETRIABLE_CPP_SUBSTRINGS:
            if sub in msg:
                return True
        # Unknown RuntimeError — not safe to assume retriable
        return False

    return False


def compute_delay(attempt: int, config: ReconnectConfig) -> float:
    """Exponential backoff: ``min(base * 2^attempt, max) +/- jitter``.

    Default sequence: 0.5, 1, 2, 4, 8, 16, 30, 30, 30, 30 (total ~152s worst case).
    """
    delay = min(config.base_delay_s * (2 ** attempt), config.max_delay_s)
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    return max(0.0, delay)


class ReconnectCallbackRegistry:
    """Post-reconnect hooks.  Fired after transport is replaced and MRs
    re-registered."""

    def __init__(self):
        self._callbacks: dict[str, callable] = {}

    def register(self, name: str, fn) -> None:
        """Register a callback under *name* (replaces if exists)."""
        self._callbacks[name] = fn

    def fire_all(self) -> None:
        """Call every registered callback, logging and swallowing errors."""
        fail_count = 0
        total = len(self._callbacks)
        for name, fn in list(self._callbacks.items()):
            try:
                fn()
            except Exception:
                fail_count += 1
                logger.exception("Reconnect callback %r failed", name)
        if fail_count > 0:
            logger.warning("Reconnect: %d/%d callbacks failed — post-reconnect state may be stale.", fail_count, total)
