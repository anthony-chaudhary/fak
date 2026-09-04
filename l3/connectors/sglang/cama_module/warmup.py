"""Data-driven warmup system for CAMA connector.

Prevents the cold-start death spiral where adaptive batch sizing halves
throughput due to TLB faults, MR registration, and cache misses during
the first minutes of operation.

Two phases: COLD -> STEADY (data-driven, no timers until first data arrives).

INIT (no phase running): Steady-state values returned — no throttling while
    idle (model loading, RDMA setup, etc.).
COLD: Activated on first ``record_pages()`` call.  Uses aggressive coalescer
    params.  Spawns background poller that checks server readiness.
STEADY: Activated when server confirms all shards rebuilt OR safety timeout.

The old 3-phase time-based system (AGGRESSIVE 30s -> RAMP 60s -> STEADY) was
fundamentally broken: the timer started at init, but data doesn't flow until
model loading completes (10-20+ minutes for large models).  By the time pages
arrive, warmup has already expired.
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WarmupPhase(enum.Enum):
    COLD = "cold"
    STEADY = "steady"
    # Deprecated aliases for backward compatibility
    AGGRESSIVE = "cold"
    RAMP = "steady"


@dataclass
class WarmupConfig:
    enabled: bool = True
    # Cold-phase params (used during COLD)
    cold_batch_size: int = 4096
    cold_jitter_ms: float = 0.0
    cold_deadline_ms: float = 2.0
    # Minimum batch size floor (replaces adaptive sizing blockade)
    min_batch_size: int = 256
    # Server polling
    server_poll_interval_s: float = 2.0
    server_poll_timeout_s: float = 300.0  # 5 min safety valve
    # TP scaling
    tp_size: int = 8

    # --- Deprecated fields (mapped to new names if new names absent) ---
    min_seconds: float = 15.0
    max_aggressive_seconds: float = 30.0
    pages_threshold: int = 5000
    ramp_seconds: float = 60.0
    aggressive_batch_size: int = 0  # 0 = not set (use cold_batch_size)
    aggressive_jitter_ms: float = -1.0  # -1 = not set
    aggressive_deadline_ms: float = -1.0  # -1 = not set


def _apply_compat_mapping(cfg: WarmupConfig) -> WarmupConfig:
    """Map deprecated config keys to new names with deprecation warning."""
    mapped = False
    if cfg.aggressive_batch_size > 0 and cfg.cold_batch_size == 4096:
        cfg.cold_batch_size = cfg.aggressive_batch_size
        mapped = True
    if cfg.aggressive_jitter_ms >= 0 and cfg.cold_jitter_ms == 0.0:
        cfg.cold_jitter_ms = cfg.aggressive_jitter_ms
        mapped = True
    if cfg.aggressive_deadline_ms >= 0 and cfg.cold_deadline_ms == 2.0:
        cfg.cold_deadline_ms = cfg.aggressive_deadline_ms
        mapped = True
    if mapped:
        warnings.warn(
            "WarmupConfig: aggressive_* fields are deprecated, use cold_* instead. "
            "The old fields were mapped automatically.",
            DeprecationWarning,
            stacklevel=3,
        )
    return cfg


class WarmupController:
    """Thread-safe data-driven warmup state machine.

    Unlike the old time-based system, no timer starts until the first
    ``record_pages()`` call (meaning actual data is flowing).  A background
    poller then checks server readiness via ``maintenance_status()`` and
    transitions to STEADY once all shards confirm rebuild.
    """

    def __init__(
        self,
        config: WarmupConfig,
        steady_batch_size: int,
        steady_jitter_ms: float,
        steady_deadline_ms: float,
        local_rank: int = 0,
    ):
        config = _apply_compat_mapping(config)
        self._config = config
        self._steady_batch_size = steady_batch_size
        self._steady_jitter_ms = steady_jitter_ms
        self._steady_deadline_ms = steady_deadline_ms
        self._local_rank = local_rank
        self._log_level = logging.INFO if local_rank == 0 else logging.DEBUG

        self._lock = threading.Lock()
        self._pages_written = 0
        self._first_data_time: Optional[float] = None

        # TP-scaled jitter: reduce jitter for small TP sizes (less contention)
        self._tp_jitter_scale = max(0.0, min(1.0, (config.tp_size - 1) / 4.0))

        # Connection factory for poller thread (set via set_conn_factory)
        self._conn_factory: Optional[Callable] = None

        # Poller thread state
        self._poller_thread: Optional[threading.Thread] = None
        self._poller_stop = threading.Event()

        if config.enabled:
            # Start in INIT (no phase) — return steady-state values
            self._phase: Optional[WarmupPhase] = None
            logger.log(
                self._log_level,
                "[warmup] data-driven mode: waiting for first data before "
                "activating COLD phase (cold_batch=%d, cold_jitter=%.0fms, "
                "cold_deadline=%.0fms, poll_interval=%.0fs, poll_timeout=%.0fs, "
                "min_batch=%d, tp=%d)",
                config.cold_batch_size, config.cold_jitter_ms,
                config.cold_deadline_ms, config.server_poll_interval_s,
                config.server_poll_timeout_s, config.min_batch_size,
                config.tp_size,
            )
        else:
            self._phase = WarmupPhase.STEADY
            logger.log(self._log_level, "[warmup] disabled — all parameters at steady-state values")

    # --- Public API ---

    def set_conn_factory(self, fn: Callable) -> None:
        """Store callable that returns a connection (for poller thread)."""
        self._conn_factory = fn

    def record_pages(self, count: int) -> None:
        with self._lock:
            self._pages_written += count
            if self._first_data_time is None and count > 0 and self._phase is None:
                self._first_data_time = time.monotonic()
                self._phase = WarmupPhase.COLD
                logger.log(
                    self._log_level,
                    "[warmup] first data arrived (%d pages) — COLD phase activated. "
                    "Params: batch=%d, jitter=%.0fms, deadline=%.0fms. "
                    "Spawning server readiness poller.",
                    count, self._config.cold_batch_size,
                    self._config.cold_jitter_ms, self._config.cold_deadline_ms,
                )
                self._spawn_poller()

    def update_steady_batch_size(self, new_size: int) -> None:
        """Update steady-state batch size (called by adaptive sizing)."""
        with self._lock:
            if new_size > 0 and new_size != self._steady_batch_size:
                self._steady_batch_size = new_size

    def update_steady_deadline_ms(self, new_ms: float) -> None:
        """Update steady-state coalesce deadline (called by adaptive sizing)."""
        with self._lock:
            if new_ms > 0 and new_ms != self._steady_deadline_ms:
                self._steady_deadline_ms = new_ms

    def effective_batch_size(self) -> int:
        with self._lock:
            return self._effective_batch_size()

    def effective_jitter_ms(self) -> float:
        with self._lock:
            return self._effective_jitter_ms()

    def effective_deadline_ms(self) -> float:
        with self._lock:
            return self._effective_deadline_ms()

    def allow_adaptive_sizing(self) -> bool:
        """Always returns True — death-spiral protection is now via min_batch_size floor."""
        return True

    def is_cache_cold(self) -> bool:
        """Returns True when in COLD phase (cache is empty, skip dedup)."""
        with self._lock:
            return self._phase is WarmupPhase.COLD

    def phase(self) -> WarmupPhase:
        with self._lock:
            if self._phase is None:
                return WarmupPhase.STEADY  # INIT returns steady-state
            return self._phase

    def phase_info(self) -> dict:
        with self._lock:
            phase_name = self._phase.value if self._phase is not None else "init"
            elapsed = (
                time.monotonic() - self._first_data_time
                if self._first_data_time is not None else 0.0
            )
            return {
                "phase": phase_name,
                "elapsed_s": elapsed,
                "pages_written": self._pages_written,
                "effective_batch_size": self._effective_batch_size(),
                "effective_jitter_ms": self._effective_jitter_ms(),
                "effective_deadline_ms": self._effective_deadline_ms(),
                "tp_size": self._config.tp_size,
                "tp_jitter_scale": self._tp_jitter_scale,
            }

    def reset(self) -> None:
        """Reset to INIT state. Called on reconnect (server may have restarted)."""
        with self._lock:
            # Stop poller if running
            self._poller_stop.set()
            if self._poller_thread is not None:
                self._poller_thread = None
            self._poller_stop = threading.Event()
            # Reset state
            self._first_data_time = None
            self._pages_written = 0
            if self._config.enabled:
                self._phase = None
                logger.log(
                    self._log_level,
                    "[warmup] reset to INIT (reconnect or manual reset). "
                    "Waiting for next data before re-entering COLD.",
                )
            else:
                self._phase = WarmupPhase.STEADY

    # --- Internal (must be called with _lock held) ---

    def _effective_batch_size(self) -> int:
        if self._phase is WarmupPhase.COLD:
            return self._config.cold_batch_size
        return self._steady_batch_size

    def _effective_jitter_ms(self) -> float:
        if self._phase is WarmupPhase.COLD:
            return self._config.cold_jitter_ms * self._tp_jitter_scale
        return self._steady_jitter_ms * self._tp_jitter_scale

    def _effective_deadline_ms(self) -> float:
        if self._phase is WarmupPhase.COLD:
            return self._config.cold_deadline_ms
        return self._steady_deadline_ms

    def _spawn_poller(self) -> None:
        """Spawn background thread to poll server readiness. Lock must be held."""
        if self._conn_factory is None:
            logger.log(
                self._log_level,
                "[warmup] no conn_factory set — will use timeout fallback "
                "(%.0fs) to exit COLD phase",
                self._config.server_poll_timeout_s,
            )
            # Schedule a timeout-based fallback
            self._poller_thread = threading.Thread(
                target=self._timeout_fallback,
                daemon=True,
                name="cama-warmup-timeout",
            )
            self._poller_thread.start()
            return

        self._poller_thread = threading.Thread(
            target=self._poll_server_ready,
            daemon=True,
            name="cama-warmup-poller",
        )
        self._poller_thread.start()

    def _poll_server_ready(self) -> None:
        """Background thread: poll maintenance_status() until all shards ready."""
        # Capture stop event locally so reset() replacing self._poller_stop
        # with a new Event doesn't cause this thread to miss the stop signal.
        stop = self._poller_stop
        poll_start = time.monotonic()
        interval = self._config.server_poll_interval_s
        timeout = self._config.server_poll_timeout_s
        ready_statuses = {"rebuilt", "detected", "disabled"}

        while not stop.is_set():
            elapsed = time.monotonic() - poll_start

            # Safety valve: timeout
            if elapsed >= timeout:
                with self._lock:
                    if self._phase is WarmupPhase.COLD:
                        self._phase = WarmupPhase.STEADY
                        logger.warning(
                            "[warmup] COLD -> STEADY (timeout after %.1fs, "
                            "pages=%d). Server may not have confirmed slab "
                            "rebuild — check maintenance_status().",
                            elapsed, self._pages_written,
                        )
                return

            # Poll server
            try:
                conn = self._conn_factory()
                if conn is None:
                    raise RuntimeError("conn_factory returned None")
                status = conn.maintenance_status()
                shards = status.get("shards", [])
                if shards:
                    all_ready = all(
                        s.get("detection", {}).get("status", "") in ready_statuses
                        for s in shards
                    )
                    if all_ready:
                        with self._lock:
                            if self._phase is WarmupPhase.COLD:
                                data_elapsed = (
                                    time.monotonic() - self._first_data_time
                                    if self._first_data_time else elapsed
                                )
                                self._phase = WarmupPhase.STEADY
                                shard_statuses = [
                                    s.get("detection", {}).get("status", "?")
                                    for s in shards
                                ]
                                logger.log(
                                    self._log_level,
                                    "[warmup] COLD -> STEADY (server confirmed, "
                                    "%.1fs after first data, pages=%d, shards=%s). "
                                    "Effective values: batch=%d, jitter=%.0fms, "
                                    "deadline=%.0fms",
                                    data_elapsed, self._pages_written,
                                    shard_statuses,
                                    self._steady_batch_size,
                                    self._steady_jitter_ms,
                                    self._steady_deadline_ms,
                                )
                        return
            except Exception as exc:
                logger.debug("[warmup] poller: maintenance_status() failed: %s", exc)

            # Jittered sleep
            jitter = random.uniform(0.8, 1.2) * interval
            stop.wait(jitter)

    def _timeout_fallback(self) -> None:
        """Fallback when no conn_factory: transition after timeout."""
        self._poller_stop.wait(self._config.server_poll_timeout_s)
        with self._lock:
            if self._phase is WarmupPhase.COLD:
                self._phase = WarmupPhase.STEADY
                logger.log(
                    self._log_level,
                    "[warmup] COLD -> STEADY (timeout fallback, no conn_factory). "
                    "pages=%d",
                    self._pages_written,
                )
