import ctypes
import json
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.cama.profiling import nvtx_range, tag_wrapper

SETUP_TIMEOUT = 600  # 10 minutes
HEALTH_CHECK_INTERVAL = 3  # seconds
MAX_HEALTH_ATTEMPTS = SETUP_TIMEOUT // HEALTH_CHECK_INTERVAL

logger = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


# NOTE: on master/newer branches this class lives in
# sglang.srt.observability.metrics_collector — update import if upgrading
from sglang.srt.metrics.collector import StorageMetrics


# ---------------------------------------------------------------------------
# Backend detection — runs once at module import time.
# RC maps are static; verified conventions are in cama_client/rc.py (CAMA)
# and documented inline below (PrisKV).
# ---------------------------------------------------------------------------
try:
    try:
        import l3_client as _kv_mod
        from l3_client import PriskvClient as _PriskvClient
        from l3_client import rc as _RC
        from l3_client.errors import CamaServerOverloadError as _CamaServerOverloadError
        from l3_client.errors import CamaNotReadyError as _CamaNotReadyError
    except ImportError:
        import cama_client as _kv_mod
        from cama_client import PriskvClient as _PriskvClient
        from cama_client import rc as _RC        # CAMA: EXISTS_FOUND=1, EXISTS_MISSING=0
        from cama_client.errors import CamaServerOverloadError as _CamaServerOverloadError
        from cama_client.errors import CamaNotReadyError as _CamaNotReadyError
    _BACKEND = "cama-" + ("rdma" if "RDMA" in _PriskvClient.__name__ else "tcp")
    _USING_CAMA_GO = True
except ImportError:
    try:
        import priskv as _kv_mod
        from priskv.priskv_client import PriskvClient as _PriskvClient
        _USING_CAMA_GO = False
        _BACKEND = "priskv"
        # Unreachable: exception types only used in CAMA Go paths
        _CamaServerOverloadError = RuntimeError
        _CamaNotReadyError = RuntimeError
        class _RC:
            # PrisKV convention (opposite of CAMA): 0=found, non-zero=missing
            EXISTS_FOUND   = 0
            EXISTS_MISSING = 1   # any non-zero; 1 used for comparison
            SET_OK   =  0
            GET_OK   =  0
            GET_MISS = -1
            DELETE_OK = 0
    except ImportError as e:
        raise ImportError(
            "Install 'cama-client' (pip install cama-client) "
            "or 'priskv' to use the CAMA storage backend."
        ) from e

logger.debug(
    "CAMA storage backend: %s | EXISTS_FOUND=%d  GET_OK=%d  GET_MISS=%d",
    _BACKEND, _RC.EXISTS_FOUND, _RC.GET_OK, _RC.GET_MISS,
)

# Sentinel for exists() errors — safe read-only constant instead of
# monkey-patching _RC.EXISTS_ERROR at import time.
_EXISTS_ERROR = getattr(_RC, 'EXISTS_ERROR', -99)


# ---------------------------------------------------------------------------
# Section A: Configuration
# ---------------------------------------------------------------------------


@dataclass
class CamaConfig:
    """Configuration for a single CamaStorage instance (one per GPU rank).

    All thread-pool and timeout settings apply **per rank** — each rank
    creates its own CamaStorage with its own PriskvClient connection,
    RDMA buffer registration, and I/O thread pool.  There is no sharing
    across ranks.
    """

    remote_addr: str
    remote_port: int
    password: str
    use_mput_mget: bool  # Enable native batch wire-protocol ops (mexists, mset, mdel)
    check_server: bool

    #: Per-batch I/O timeout in seconds.  Applies to ``as_completed()``
    #: calls in ``_get_batch_zero_copy``, ``_put_batch_zero_copy``, and
    #: ``_batch_exist``.  If any key operation in a batch has not
    #: completed within this window the batch returns partial results
    #: (timed-out keys report failure).  Also passed to
    #: ``conn.set_timeout()`` if the client supports it.
    op_timeout_s: float = 10.0

    #: Number of threads in ``CamaStorage._io_pool`` that execute
    #: individual key operations (get/set/exists) concurrently within a
    #: single batch call.  These threads share one RDMA connection, so
    #: diminishing returns past ~16.  Increasing this helps when
    #: individual key latency is high (e.g. cross-rack RDMA) but batch
    #: sizes are large.  Also shared between prefetch and backup paths.
    io_workers: int = 8

    #: Number of RDMA/TCP connections per rank.  Values >1 create a
    #: connection pool for N-way parallelism across io_workers.
    #: 8 matches common TP sizes (1 conn per TP worker).  With 4 NICs
    #: this gives 2 conns/NIC — full parallelism with no lock contention.
    #: Per non-owner conn: ~32MB (16MB send + 16MB recv, shared PD
    #: eliminates read buffer).  Total for 8: ~256MB.
    pool_size: int = 8

    #: Maximum body size for a single MSET message.  0 = use client
    #: default (16 MB).  Passed as ``send_buf_size`` to the client.
    send_buf_size: int = 0

    #: Enable automatic reconnection after transport failures.
    reconnect_enabled: bool = True

    #: Maximum number of reconnect attempts before giving up.
    #: Default 10 covers ~152s of downtime (0.5+1+2+4+8+16+30+30+30+30).
    reconnect_max_retries: int = 10

    #: Base delay for exponential backoff between reconnect attempts.
    reconnect_base_delay_s: float = 0.5

    #: Maximum delay cap for exponential backoff.
    reconnect_max_delay_s: float = 30.0

    #: Number of retry attempts for warmup validation.  A transient server
    #: error (e.g. shard dispatch timeout during hugepage allocation) will
    #: be retried with exponential backoff instead of crashing the scheduler.
    warmup_retries: int = 3

    #: Enable multi-NIC striped RDMA reads.  When True and multiple RDMA
    #: endpoints are discovered, the pool connects to ALL server NICs and
    #: stripes ``mget_rdma`` across them in parallel for N× bandwidth.
    nic_striping: bool = True

    #: Compression codec. ``""`` = disabled (raw bytes, zero-copy RDMA Read).
    #: ``"int8"`` = INT8 symmetric quantization (~2x, lossy).
    #: ``"shuffle_zstd"`` = byte-shuffle + zstd (~1.3x, lossless).
    #: ``"int8+shuffle_zstd"`` = chained (~2.6x).
    #: Changing codec requires FLUSH (existing values use old encoding).
    codec: str = ""

    #: Zstd compression level (1-22). Only used when codec includes shuffle_zstd.
    codec_zstd_level: int = 3

    #: Write-dedup mode. ``"auto"`` enables dedup initially but auto-
    #: disables after ``dedup_auto_window`` consecutive low-hit batches,
    #: then periodically probes to re-enable if the workload shifts.
    #: ``"always"`` keeps dedup on permanently.  ``"never"`` skips it.
    dedup_mode: str = "auto"

    #: Hit-rate threshold below which auto mode considers a batch "low-hit".
    dedup_auto_threshold: float = 0.05

    #: Number of consecutive low-hit batches before auto mode disables dedup.
    dedup_auto_window: int = 2

    #: Cost-aware disable: if exists_ms / transfer_ms ratio exceeds this for
    #: ``dedup_auto_window`` consecutive batches, auto-disable regardless of hit rate.
    dedup_cost_ratio_threshold: float = 0.5

    #: When dedup is auto-disabled, run a probe batch with dedup ON every
    #: this many batches to detect workload shifts.  0 = no probing (legacy
    #: permanent disable).
    dedup_probe_interval: int = 20

    #: Number of consecutive probe batches that must exceed the hit-rate
    #: threshold before dedup is re-enabled.
    dedup_probe_window: int = 2

    #: Model page size in bytes.  When > 0, prewarm sends this as a
    #: page-size hint to the server *before* model loading completes,
    #: allowing early slab optimisation on ``auto`` preset servers.
    #: 0 = don't send early hint (compute from register_mem_pool_host).
    model_page_bytes: int = 0

    # -- Field spec: single source of truth for config keys, env vars, defaults, and casts --
    _FIELD_SPEC = (
        ("remote_addr",             "SGLANG_CAMA_REMOTE_ADDR",             "127.0.0.1", None),
        ("remote_port",             "SGLANG_CAMA_REMOTE_PORT",             18001,        int),
        ("password",                "SGLANG_CAMA_PASSWORD",                "",           None),
        ("use_mput_mget",           "SGLANG_CAMA_USE_MPUT_MGET",          True,         None),
        ("check_server",            "SGLANG_CAMA_CHECK_SERVER",            False,        None),
        ("op_timeout_s",            "SGLANG_CAMA_OP_TIMEOUT_S",           10.0,         float),
        ("io_workers",              "SGLANG_CAMA_IO_WORKERS",              8,            int),
        ("pool_size",               "SGLANG_CAMA_POOL_SIZE",               8,            int),
        ("send_buf_size",           "SGLANG_CAMA_SEND_BUF_SIZE",           0,            int),
        ("warmup_retries",          "SGLANG_CAMA_WARMUP_RETRIES",          3,            int),
        ("reconnect_enabled",       "SGLANG_CAMA_RECONNECT_ENABLED",       True,         None),
        ("reconnect_max_retries",   "SGLANG_CAMA_RECONNECT_MAX_RETRIES",   10,           int),
        ("reconnect_base_delay_s",  "SGLANG_CAMA_RECONNECT_BASE_DELAY_S",  0.5,          float),
        ("reconnect_max_delay_s",   "SGLANG_CAMA_RECONNECT_MAX_DELAY_S",   30.0,         float),
        ("nic_striping",            "SGLANG_CAMA_NIC_STRIPING",            True,         None),
        ("codec",                   "SGLANG_CAMA_CODEC",                   "",           None),
        ("codec_zstd_level",        "SGLANG_CAMA_CODEC_ZSTD_LEVEL",        3,            int),
        ("dedup_mode",              "SGLANG_CAMA_DEDUP_MODE",              "auto",       None),
        ("dedup_auto_threshold",    "SGLANG_CAMA_DEDUP_AUTO_THRESHOLD",    0.05,         float),
        ("dedup_auto_window",       "SGLANG_CAMA_DEDUP_AUTO_WINDOW",       2,            int),
        ("dedup_cost_ratio_threshold", "SGLANG_CAMA_DEDUP_COST_RATIO_THRESHOLD", 0.5, float),
        ("dedup_probe_interval",    "SGLANG_CAMA_DEDUP_PROBE_INTERVAL",    20,           int),
        ("dedup_probe_window",      "SGLANG_CAMA_DEDUP_PROBE_WINDOW",      2,            int),
        ("model_page_bytes",        "SGLANG_CAMA_MODEL_PAGE_BYTES",        0,            int),
    )

    # -- Safe env accessors (resilient to older environ patches) --

    @staticmethod
    def _env_get(attr: str, default):
        """Get an envs attribute, falling back to *default* if the attribute
        was not registered in the host sglang's ``Envs`` class (i.e. the
        environ.py patch is an older version)."""
        field = getattr(envs, attr, None)
        if field is None:
            return default
        return field.get()

    @staticmethod
    def _env_default(attr: str, default):
        """Return the declared default for an envs attribute, or *default*
        if the attribute is missing."""
        field = getattr(envs, attr, None)
        if field is None:
            return default
        return field.default

    @classmethod
    def _from_dict(cls, source: dict, fallback) -> "CamaConfig":
        """Build a CamaConfig from *source* dict, using *fallback(env_var, default)*
        for any key not present in *source*."""
        kwargs = {}
        for key, env_var, default, cast in cls._FIELD_SPEC:
            raw = source.get(key, fallback(env_var, default))
            kwargs[key] = cast(raw) if cast else raw
        return cls(**kwargs)

    @classmethod
    def from_file(cls) -> "CamaConfig":
        """Load config from a JSON file specified by SGLANG_CAMA_CONFIG_PATH."""
        if not envs.SGLANG_CAMA_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_CAMA_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_CAMA_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {e}")

        if "remote_addr" not in config:
            raise ValueError("'remote_addr' is required in config file")

        return cls._from_dict(config, cls._env_default)

    @classmethod
    def load_from_env(cls) -> "CamaConfig":
        """Load config from individual SGLANG_CAMA_* environment variables."""
        return cls._from_dict({}, cls._env_get)

    @classmethod
    def load_from_extra_config(cls, extra_config: dict) -> "CamaConfig":
        """Load config from the extra_config dictionary (e.g. Kubernetes/runtime attach)."""
        if "remote_addr" not in extra_config:
            raise ValueError("'remote_addr' is required in extra_config")

        return cls._from_dict(extra_config, cls._env_default)

    @classmethod
    def resolve(cls, extra_config: dict = None) -> "CamaConfig":
        """Load config using triple-source priority: extra_config > file > env."""
        if extra_config is not None and extra_config.get("remote_addr") is not None:
            return cls.load_from_extra_config(extra_config)
        if hasattr(envs, "SGLANG_CAMA_CONFIG_PATH") and envs.SGLANG_CAMA_CONFIG_PATH.is_set():
            return cls.from_file()
        return cls.load_from_env()


# ---------------------------------------------------------------------------
# Connection factory — used by CamaStorage.__init__ and prewarm.py
# ---------------------------------------------------------------------------


def _make_connection(addr, port, password, pool_size,
                     send_buf_size=0, reconnect=None, endpoints=None):
    """Create a CAMA client connection (pool or single).

    Args:
        reconnect: None = omit kwarg (prewarm), True/ReconnectConfig = enable,
                   False = explicitly disable.
        endpoints: list of (ip, port) for NIC-striped pools.
    """
    conn_kwargs = {}
    if send_buf_size > 0:
        conn_kwargs["send_buf_size"] = send_buf_size
        conn_kwargs["recv_buf_size"] = send_buf_size
    if reconnect is not None:
        conn_kwargs["reconnect"] = reconnect

    if pool_size > 1 or endpoints is not None:
        try:
            from l3_client import create_pool
        except ImportError:
            from cama_client import create_pool
        if endpoints is not None:
            conn_kwargs["endpoints"] = endpoints
            pool_size = max(pool_size, len(endpoints))
        # Cap applied here as well (client _pool.py also caps, but belt-and-suspenders
        # prevents unbounded endpoint counts from even reaching the client).
        _MAX_POOL = 32
        if pool_size > _MAX_POOL:
            logger.warning(
                "_make_connection: pool_size %d exceeds max %d (endpoints=%d) — capping",
                pool_size, _MAX_POOL, len(endpoints) if endpoints else 0,
            )
            pool_size = _MAX_POOL
        return create_pool(addr, port, password, pool_size=pool_size, **conn_kwargs)

    return _PriskvClient(addr, port, password, **conn_kwargs)


# ---------------------------------------------------------------------------
# Section B–I: CamaStorage
# ---------------------------------------------------------------------------


class _BackpressureGuard:
    """Exponential backoff for server overload errors.

    Tracks consecutive overload errors and sleeps before retries
    to avoid hammering a struggling server.  Thread-safe.
    """

    def __init__(self, base_delay: float = 0.05, max_delay: float = 2.0):
        self._lock = threading.Lock()
        self._consecutive = 0
        self._base_delay = base_delay
        self._max_delay = max_delay

    def on_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def on_overload(self) -> None:
        with self._lock:
            self._consecutive += 1
            n = self._consecutive
        delay = min(self._base_delay * (2 ** (n - 1)), self._max_delay)
        delay *= 0.5 + random.random()  # jitter [0.5x, 1.5x]
        logger.warning("Server overload backpressure: sleeping %.3fs (consecutive=%d)", delay, n)
        time.sleep(delay)


class CamaStorage(HiCacheStorage):

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool: HostKVCache = None,
    ):
        init_start = time.perf_counter()

        # -- Resolve rank EARLY so all init logs can be rank-gated --
        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1
        self._sub_keys_per_page = 1 if self.is_mla_backend else 2

        # Helper: INFO on rank 0, DEBUG on all others — keeps full detail
        # in debug logs but stops N×TP identical messages in production.
        def _r0(msg, *args, level=logging.INFO, **kwargs):
            effective = level if self.local_rank == 0 else logging.DEBUG
            logger.log(effective, msg, *args, **kwargs)

        self._backend = _BACKEND
        self._priskv = _kv_mod  # keep module reference for SGL construction
        self._client_version = getattr(_kv_mod, "__version__", "unknown")

        # SGLang compatibility check (silent — banner shows version)
        try:
            try:
                from l3_client._version import check_sglang_compatibility
                check_sglang_compatibility()
            except (ImportError, AttributeError):
                try:
                    from cama_client._version import check_sglang_compatibility
                    check_sglang_compatibility()
                except (ImportError, AttributeError):
                    pass
        except ImportError:
            pass

        try:
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )

            # -- [1/7] Config loading --
            phase_start = time.perf_counter()
            self.config = CamaConfig.resolve(extra_config)
            _r0("[1/7] Config loaded (%.2fs)", time.perf_counter() - phase_start)

            # -- [2/7] Connect --
            phase_start = time.perf_counter()

            # Try to adopt a pre-warmed connection from preflight.
            _prewarm_result = None
            try:
                try:
                    from sglang.srt.mem_cache.storage.cama.prewarm import claim_prewarmed_connection
                except ImportError:
                    from cama_module.prewarm import claim_prewarmed_connection
                _prewarm_result = claim_prewarmed_connection(
                    self.config.remote_addr,
                    self.config.remote_port,
                    self.config.pool_size,
                    self.config.send_buf_size,
                )
            except Exception:
                pass  # fall through to normal init

            # Build reconnect config once for all connection sites
            if self.config.reconnect_enabled:
                try:
                    from l3_client.reconnect import ReconnectConfig
                except ImportError:
                    from cama_client.reconnect import ReconnectConfig
                reconnect_cfg = ReconnectConfig(
                    max_retries=self.config.reconnect_max_retries,
                    base_delay_s=self.config.reconnect_base_delay_s,
                    max_delay_s=self.config.reconnect_max_delay_s,
                )
            else:
                reconnect_cfg = False

            # Track whether prewarm already did multi-NIC setup
            _prewarmed_endpoints = None

            if _prewarm_result is not None:
                self.conn = _prewarm_result.connection
                _prewarmed_endpoints = _prewarm_result.endpoints
                # Apply user-configured reconnect settings (prewarm used defaults)
                try:
                    if self.config.reconnect_enabled:
                        if hasattr(self.conn, "enable_reconnect"):
                            self.conn.enable_reconnect(reconnect_cfg)
                    else:
                        if hasattr(self.conn, "disable_reconnect"):
                            self.conn.disable_reconnect()
                except Exception as exc:
                    logger.warning("Failed to apply reconnect config to pre-warmed conn: %s", exc)
                _r0("[2/7] Connected (%.2fs, pre-warmed)", time.perf_counter() - phase_start)
            else:
                self.conn = _make_connection(
                    self.config.remote_addr, self.config.remote_port,
                    self.config.password, self.config.pool_size,
                    send_buf_size=self.config.send_buf_size,
                    reconnect=reconnect_cfg,
                )
                _r0("[2/7] Connected (%.2fs)", time.perf_counter() - phase_start)

            # Apply operation timeout (detail logged in settings banner)
            if self.config.op_timeout_s and hasattr(self.conn, "set_timeout"):
                self.conn.set_timeout(self.config.op_timeout_s)

            # Capture server info for settings banner
            self._server_info = getattr(self.conn, "_server_info", None)

            # -- [3/7] Multi-NIC: discover RDMA endpoints and reconnect if needed --
            phase_start = time.perf_counter()

            # If prewarm already did endpoint discovery + multi-NIC setup, check
            # whether the pool already has the correct endpoints and skip the
            # expensive close/recreate cycle (MR registration with 16 MB buffers).
            _prewarm_did_multinic = (
                _prewarmed_endpoints is not None
                and len(_prewarmed_endpoints) > 1
                and hasattr(self.conn, '_endpoints')
                and self.conn._endpoints is not None
                and len(self.conn._endpoints) > 1
            )

            self._rdma_eps = []  # populated below, used by settings banner

            if _prewarm_did_multinic:
                # Prewarm already created the multi-NIC pool — just update config
                # to match the first endpoint's address.
                rdma_eps = _prewarmed_endpoints
                target_ip = rdma_eps[0]["ip"]
                target_port = int(rdma_eps[0]["port"])
                self.config.remote_addr = target_ip
                self.config.remote_port = target_port
                if self.config.op_timeout_s and hasattr(self.conn, "set_timeout"):
                    self.conn.set_timeout(self.config.op_timeout_s)
                self._rdma_eps = rdma_eps
                _r0("[3/7] NIC discovery (%.2fs, %d endpoints, pre-warmed)",
                    time.perf_counter() - phase_start, len(rdma_eps))
            else:
                try:
                    rdma_eps = (
                        self.conn.rdma_endpoints()
                        if hasattr(self.conn, "rdma_endpoints")
                        else []
                    )
                    if len(rdma_eps) > 1 and self.config.nic_striping:
                        # NIC striping: pass ALL endpoints to the pool so that
                        # connections are distributed across server NICs.
                        endpoints = [(ep["ip"], int(ep["port"])) for ep in rdma_eps]
                        target_ip, target_port = endpoints[0]
                        if (
                            target_ip != self.config.remote_addr
                            or target_port != self.config.remote_port
                            or not (hasattr(self.conn, '_endpoints')
                                    and self.conn._endpoints is not None
                                    and len(self.conn._endpoints) > 1)
                        ):
                            # Close old pool BEFORE creating new one to avoid
                            # double-allocation OOM — both pools alive simultaneously
                            # pins 2× the RDMA buffers (up to 1+ GB per rank).
                            old_conn = self.conn
                            self.conn = None  # prevent accidental use during transition
                            old_conn.close()
                            try:
                                self.conn = _make_connection(
                                    target_ip, target_port, self.config.password,
                                    self.config.pool_size,
                                    send_buf_size=self.config.send_buf_size,
                                    reconnect=reconnect_cfg,
                                    endpoints=endpoints,
                                )
                            except Exception:
                                # Re-create a minimal connection for fallback
                                self.conn = _make_connection(
                                    self.config.remote_addr, self.config.remote_port,
                                    self.config.password, self.config.pool_size,
                                    send_buf_size=self.config.send_buf_size,
                                    reconnect=reconnect_cfg,
                                )
                                raise
                            if self.config.op_timeout_s and hasattr(self.conn, "set_timeout"):
                                self.conn.set_timeout(self.config.op_timeout_s)
                            if self.config.reconnect_enabled and hasattr(self.conn, "set_reconnect_callback"):
                                self.conn.set_reconnect_callback("cama_storage", self._on_reconnect)
                            self.config.remote_addr = target_ip
                            self.config.remote_port = target_port
                        self._rdma_eps = rdma_eps
                    elif len(rdma_eps) > 1:
                        # Striping disabled — legacy single-NIC per rank
                        target = rdma_eps[self.local_rank % len(rdma_eps)]
                        target_ip = target["ip"]
                        target_port = target["port"]
                        if (
                            target_ip != self.config.remote_addr
                            or target_port != self.config.remote_port
                        ):
                            # Close old before creating new (same OOM prevention as striping path)
                            old_conn = self.conn
                            self.conn = None
                            old_conn.close()
                            try:
                                self.conn = _make_connection(
                                    target_ip, target_port, self.config.password,
                                    self.config.pool_size,
                                    send_buf_size=self.config.send_buf_size,
                                    reconnect=reconnect_cfg,
                                )
                            except Exception:
                                self.conn = _make_connection(
                                    self.config.remote_addr, self.config.remote_port,
                                    self.config.password, self.config.pool_size,
                                    send_buf_size=self.config.send_buf_size,
                                    reconnect=reconnect_cfg,
                                )
                                raise
                            if self.config.op_timeout_s and hasattr(self.conn, "set_timeout"):
                                self.conn.set_timeout(self.config.op_timeout_s)
                            self.config.remote_addr = target_ip
                            self.config.remote_port = target_port
                        self._rdma_eps = rdma_eps
                    else:
                        self._rdma_eps = rdma_eps
                except Exception as exc:
                    if self.conn is None:
                        # Both multi-NIC and fallback connect failed — old conn
                        # was already closed, nothing left to work with.
                        raise RuntimeError(
                            "NIC discovery failed and fallback reconnect also "
                            f"failed — no usable connection: {exc}"
                        ) from exc
                    logger.warning("NIC discovery failed, keeping original connection: %s", exc)
                _r0("[3/7] NIC discovery (%.2fs, %d endpoints)",
                    time.perf_counter() - phase_start, len(self._rdma_eps))

            # -- [4/7] Health check (optional) --
            phase_start = time.perf_counter()
            if self.config.check_server:
                self._check_server()
                _r0("[4/7] Health check (%.2fs)", time.perf_counter() - phase_start)
            else:
                _r0("[4/7] Health check skipped")

            # -- [5/7] Warmup: validate full round-trip --
            phase_start = time.perf_counter()
            self._retry_transient(
                self._warmup,
                max_retries=self.config.warmup_retries,
            )
            _r0("[5/7] Warmup validated (%.2fs, 6/6 checks passed)", time.perf_counter() - phase_start)

            # -- Key suffixes (identical to mooncake logic) --
            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

            # -- Extra backend tag --
            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                _r0("Using extra_backend_tag: %s", self.extra_backend_tag)

            # -- Metrics --
            self.gb_per_page = None
            self._bytes_per_page = 0
            self.prefetch_pgs: List[int] = []
            self.backup_pgs: List[int] = []
            self.prefetch_bandwidth: List[float] = []
            self.backup_bandwidth: List[float] = []
            # Concurrent I/O metrics
            self._io_batch_sizes: List[int] = []
            self._io_latencies_ms: List[float] = []

            # Phase timing accumulators (per-batch, cleared each get_stats interval)
            self._phase_preprocess_ms: List[float] = []
            self._phase_exists_ms: List[float] = []      # batch_set dedup path only
            self._phase_transfer_ms: List[float] = []    # mget_rdma / mset time
            self._phase_postprocess_ms: List[float] = []

            # In-flight operation gauges
            self._inflight_gets = 0
            self._inflight_sets = 0

            # Latency histogram (cumulative within each reporting interval)
            _LATENCY_BUCKETS_MS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0)
            self._LATENCY_BUCKETS_MS = _LATENCY_BUCKETS_MS
            self._hist_counts = [0] * (len(_LATENCY_BUCKETS_MS) + 1)  # +1 for +Inf
            self._hist_sum = 0.0
            self._hist_total = 0
            self._telemetry_lock = threading.Lock()

            # Interval tracking for enriched I/O stats
            self._io_max_latency_ms: float = 0.0       # reset each interval
            self._last_stats_time: float = time.monotonic()
            self._interval_pages_set: int = 0           # reset each interval
            self._interval_pages_get: int = 0           # reset each interval
            self._total_pages_set: int = 0              # cumulative (never reset)
            self._total_pages_get: int = 0              # cumulative (never reset)

            # Server health tracking
            self._last_server_evictions: int = 0
            self._last_server_stats_time: float = time.monotonic()

            # -- SGLang-level metrics (updated by HiRadixCache before get_stats) --
            self._sglang_metrics: dict = {}

            # -- Cumulative error counters (never reset) --
            self._counter_lock = threading.Lock()
            self._get_errors = 0       # conn.get() raised or timed out
            self._get_successes = 0    # conn.get() returned GET_OK
            self._set_errors = 0       # conn.set() raised or timed out
            self._set_successes = 0    # conn.set() returned SET_OK
            self._exists_errors = 0    # conn.exists() raised or timed out
            self._exists_timeouts = 0  # batch timed out before completion
            self._reconnect_count = 0  # number of successful reconnections
            # Error log throttling (rank 0 only, at most every 10s)
            self._last_error_log_time: float = time.monotonic()
            self._last_err_get = 0
            self._last_err_set = 0
            self._last_err_exists = 0
            self._last_err_exists_to = 0

            # -- Adaptive dedup state --
            self._dedup_enabled = self.config.dedup_mode in ("always", "auto")
            self._dedup_low_hit_streak = 0
            self._dedup_auto_disabled = False
            self._dedup_cost_streak = 0  # consecutive batches where exists cost > threshold
            self._dedup_batches_since_disable = 0  # batches elapsed since auto-disable
            self._dedup_probe_hit_streak = 0  # consecutive probes above threshold
            self._dedup_probes_total = 0  # lifetime probe count
            self._dedup_reenables_total = 0  # lifetime re-enable count
            self._warmup_phase_ref = None  # callable returning bool (is_cache_cold)
            self._warmup_reset_ref = None  # callable to reset warmup state

            # -- RDMA registration handle (set in register_mem_pool_host) --
            self._reg_buf = None

            # -- Compression codec --
            self._codec = None
            if self.config.codec:
                from cama_module.codec import get_codec, register_chain, ShuffleZstdCodec, ChainCodec
                codec_name = self.config.codec.strip().lower()
                if "+" in codec_name:
                    parts = [p.strip() for p in codec_name.split("+")]
                    self._codec = register_chain(codec_name, *parts)
                else:
                    self._codec = get_codec(codec_name)
                # Apply non-default zstd level via a LOCAL instance (never mutate global singleton)
                if self.config.codec_zstd_level != 3:
                    local_zstd = ShuffleZstdCodec(level=self.config.codec_zstd_level)
                    if isinstance(self._codec, ShuffleZstdCodec):
                        self._codec = local_zstd
                    elif hasattr(self._codec, '_codecs'):  # ChainCodec
                        new_codecs = [
                            local_zstd if isinstance(c, ShuffleZstdCodec) else c
                            for c in self._codec._codecs
                        ]
                        self._codec = ChainCodec(
                            self._codec.name,
                            new_codecs,
                            self._codec.codec_id,
                        )
            # -- [6/7] Thread pool + reconnect --
            phase_start = time.perf_counter()
            self._io_pool = ThreadPoolExecutor(
                max_workers=self.config.io_workers,
                thread_name_prefix="cama_io",
            )
            self._backpressure = _BackpressureGuard()
            # Background executor for report_stats — prevents scheduler stalls
            # when the RDMA transport is stuck (e.g. dead remote).
            self._stats_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cama-stats"
            )
            if self.config.reconnect_enabled and hasattr(self.conn, "set_reconnect_callback"):
                self.conn.set_reconnect_callback("cama_storage", self._on_reconnect)
            _r0("[6/7] Thread pool ready (%.2fs, %d workers)",
                time.perf_counter() - phase_start, self.config.io_workers)

            # -- Settings banner: single source of truth for all config --
            if self.local_rank == 0:
                self._log_settings_banner(time.perf_counter() - init_start)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            self._cleanup_partial_init()
            raise
        except Exception as exc:
            logger.error("An error occurred during Cama setup: %s", exc)
            self._cleanup_partial_init()
            raise

    # -------------------------------------------------------------------
    # Partial-init cleanup
    # -------------------------------------------------------------------

    def _cleanup_partial_init(self) -> None:
        """Release resources acquired during a failed __init__.

        Called from __init__'s except block so that connections, thread pools,
        and RDMA registrations don't leak when a later phase raises.
        """
        for attr in ("_stats_executor", "_io_pool"):
            pool = getattr(self, attr, None)
            if pool is not None:
                try:
                    pool.shutdown(wait=False)
                except Exception:
                    pass
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self.conn = None

    # -------------------------------------------------------------------
    # Health check & warmup
    # -------------------------------------------------------------------

    def _get_metrics_url(self) -> Optional[str]:
        """Build URL for the server's /ready endpoint from handshake info."""
        info = getattr(self, "_server_info", None)
        if info and info.get("metrics_addr"):
            addr = info["metrics_addr"]
            # metrics_addr is typically ":9090" — need to prepend host
            if addr.startswith(":"):
                addr = self.config.remote_addr + addr
            return f"http://{addr}/ready"
        # Fallback: assume default metrics port
        return f"http://{self.config.remote_addr}:9090/ready"

    def _fetch_server_progress(self, url: str) -> Optional[str]:
        """Best-effort HTTP GET to /ready, returns human-readable progress or None."""
        try:
            import urllib.request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ready":
                    return "ready"
                parts = []
                if "phase" in data:
                    parts.append(f"phase={data['phase']}")
                if "percent" in data:
                    parts.append(f"{data['percent']:.0f}%")
                if "mem_reserved_gb" in data and "mem_total_gb" in data:
                    parts.append(f"{data['mem_reserved_gb']:.0f}/{data['mem_total_gb']:.0f} GB")
                if "eta_seconds" in data:
                    parts.append(f"ETA {data['eta_seconds']:.0f}s")
                return ", ".join(parts) if parts else None
        except Exception:
            return None

    def _check_server(self):
        """Poll PrisKV until it responds, up to SETUP_TIMEOUT seconds."""
        start_time = time.perf_counter()
        metrics_url = self._get_metrics_url()
        attempt = 0
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            attempt += 1
            try:
                self.conn.exists("__cama_health__")
                if self.local_rank == 0:
                    logger.info("PrisKV server is reachable.")
                return
            except Exception:
                elapsed = time.perf_counter() - start_time
                progress = self._fetch_server_progress(metrics_url) if metrics_url else None
                if self.local_rank == 0:
                    if progress:
                        logger.info(
                            "Waiting for server [attempt %d/%d, %.1fs elapsed] — %s",
                            attempt, MAX_HEALTH_ATTEMPTS, elapsed, progress,
                        )
                    else:
                        logger.info(
                            "Waiting for server [attempt %d/%d, %.1fs elapsed]",
                            attempt, MAX_HEALTH_ATTEMPTS, elapsed,
                        )
                time.sleep(HEALTH_CHECK_INTERVAL)
        raise RuntimeError(
            f"PrisKV server not reachable after {SETUP_TIMEOUT}s at "
            f"{self.config.remote_addr}:{self.config.remote_port}"
        )

    @staticmethod
    def _retry_transient(fn, max_retries, base_delay=1.0, max_delay=10.0):
        """Retry *fn* on transient ``RuntimeError`` with exponential backoff.

        ``AssertionError`` (data-integrity failures) is NOT retried — only
        server-side transient errors such as *shard dispatch timeout* are
        eligible.  Returns whatever *fn* returns on success.
        """
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except RuntimeError as exc:
                last_exc = exc
                if attempt == max_retries:
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay *= random.uniform(0.5, 1.5)  # jitter to prevent thundering herd
                logger.warning(
                    "Transient error on attempt %d/%d, retrying in %.1fs: %s",
                    attempt, max_retries, delay, exc,
                )
                time.sleep(delay)
        raise last_exc  # unreachable, satisfies type checkers

    def _warmup(self):
        """Validate full RDMA round-trip: register -> set(SGL) -> exists -> get(SGL) -> compare -> delete."""
        _wlog = logging.INFO if self.local_rank == 0 else logging.DEBUG
        warmup_start = time.perf_counter()
        key = "cama_warmup_" + uuid.uuid4().hex

        # 1. String round-trip (basic connectivity)
        step_start = time.perf_counter()
        ret = self.conn.setstr(key + "_str", "ok")
        assert ret == 0, f"Warmup setstr failed with code {ret}"
        got = self.conn.getstr(key + "_str")
        assert got == "ok", f"Warmup getstr mismatch: expected 'ok', got '{got}'"
        self.conn.delete(key + "_str")
        logger.log(_wlog, "  warmup [1/6] string round-trip... OK (%.3fs)", time.perf_counter() - step_start)

        # 2. SGL/RDMA buffer registration
        step_start = time.perf_counter()
        import numpy as np

        pattern = np.arange(256, dtype=np.float32)  # 1 KB deterministic pattern
        send_buf = pattern.copy()
        recv_buf = np.zeros_like(pattern)

        send_reg = self.conn.reg_memory(send_buf.ctypes.data, send_buf.nbytes)
        recv_reg = self.conn.reg_memory(recv_buf.ctypes.data, recv_buf.nbytes)
        assert send_reg != 0, "Warmup: send buffer RDMA registration failed"
        assert recv_reg != 0, "Warmup: recv buffer RDMA registration failed"
        logger.log(_wlog, "  warmup [2/6] RDMA buffer registration... OK (%.3fs)", time.perf_counter() - step_start)

        # 3. SGL set → exists → get
        step_start = time.perf_counter()
        send_sgl = self._priskv.SGL(send_buf.ctypes.data, send_buf.nbytes, send_reg)
        recv_sgl = self._priskv.SGL(recv_buf.ctypes.data, recv_buf.nbytes, recv_reg)

        ret = self.conn.set(key, send_sgl)
        assert ret == 0, f"Warmup SGL set failed with code {ret}"

        ret = self.conn.exists(key)
        assert ret == _RC.EXISTS_FOUND, (
            f"Warmup SGL exists failed: got {ret}, expected _RC.EXISTS_FOUND={_RC.EXISTS_FOUND} "
            f"(backend={_BACKEND})"
        )

        ret = self.conn.get(key, recv_sgl, recv_buf.nbytes)
        assert ret == 0, f"Warmup SGL get failed with code {ret}"
        logger.log(_wlog, "  warmup [3/6] SGL set -> exists -> get... OK (%.3fs)", time.perf_counter() - step_start)

        # 4. Data integrity check
        step_start = time.perf_counter()
        assert np.array_equal(send_buf, recv_buf), (
            f"Warmup RDMA data mismatch! First 4 sent: {send_buf[:4]}, received: {recv_buf[:4]}"
        )

        # Verify exists returns EXISTS_MISSING for missing key (catch pybind11 vector bug)
        missing_ret = self.conn.exists(key + "_nonexistent")
        assert missing_ret == _RC.EXISTS_MISSING, (
            f"Warmup: exists() returned unexpected value for non-existent key: "
            f"got {missing_ret}, expected _RC.EXISTS_MISSING={_RC.EXISTS_MISSING} "
            f"(backend={_BACKEND})"
        )
        logger.log(_wlog, "  warmup [4/6] data integrity check... OK (%.3fs)", time.perf_counter() - step_start)

        # 5. Batch wire-path validation (mexists/mset/mdel)
        step_start = time.perf_counter()
        batch_k1 = key + "_batch_0"
        batch_k2 = key + "_batch_1"
        send_sgl2 = self._priskv.SGL(send_buf.ctypes.data, send_buf.nbytes, send_reg)
        send_sgl3 = self._priskv.SGL(send_buf.ctypes.data, send_buf.nbytes, send_reg)
        results = self.conn.mset([batch_k1, batch_k2], [send_sgl2, send_sgl3])
        assert all(r == 0 for r in results), f"Warmup batch mset failed: {results}"
        batch_exists = self.conn.mexists([batch_k1, batch_k2, key + "_nonexistent"])
        assert batch_exists[0] == _RC.EXISTS_FOUND, (
            f"Warmup batch mexists[0] failed: got {batch_exists[0]}, expected {_RC.EXISTS_FOUND}"
        )
        assert batch_exists[1] == _RC.EXISTS_FOUND, (
            f"Warmup batch mexists[1] failed: got {batch_exists[1]}, expected {_RC.EXISTS_FOUND}"
        )
        assert batch_exists[2] == _RC.EXISTS_MISSING, (
            f"Warmup batch mexists[2] should be missing: got {batch_exists[2]}"
        )
        self.conn.mdel([batch_k1, batch_k2])
        logger.log(_wlog, "  warmup [5/6] batch wire-path validation... OK (%.3fs)", time.perf_counter() - step_start)

        # 6. Cleanup
        step_start = time.perf_counter()
        self.conn.delete(key)
        self.conn.dereg_memory(send_reg)
        self.conn.dereg_memory(recv_reg)
        logger.log(_wlog, "  warmup [6/6] cleanup... OK (%.3fs)", time.perf_counter() - step_start)

        logger.log(_wlog, "  warmup complete (%.3fs total)", time.perf_counter() - warmup_start)

    # -------------------------------------------------------------------
    # Settings banner — printed once at startup
    # -------------------------------------------------------------------

    def _log_settings_banner(self, init_elapsed: float = 0.0):
        """Log a comprehensive summary of all performance-relevant settings.

        This is the single source of truth for startup config — individual
        phases log only timing, not config detail.
        """
        cfg = self.config
        codec_desc = self._codec.name if self._codec else "disabled"
        if self._codec and self._codec.is_lossy:
            codec_desc += " (lossy)"

        # Use cached NIC info from phase 3 (avoids redundant rdma_endpoints() call)
        rdma_eps = getattr(self, "_rdma_eps", [])
        nic_count = len(rdma_eps)

        if cfg.nic_striping and nic_count > 1:
            nic_names = ", ".join(ep.get("device", "?") for ep in rdma_eps)
            nic_striping_status = f"{nic_count} NICs ({nic_names})"
        elif cfg.nic_striping:
            nic_striping_status = "ON (single NIC)"
        else:
            nic_striping_status = "OFF"

        reconnect_status = (
            f"ON (max={cfg.reconnect_max_retries}, "
            f"base={cfg.reconnect_base_delay_s}s, cap={cfg.reconnect_max_delay_s}s)"
            if cfg.reconnect_enabled else "OFF"
        )

        send_buf_desc = (
            _fmt_bytes(cfg.send_buf_size) if cfg.send_buf_size > 0
            else "default (16 MB)"
        )

        server_info = getattr(self, "_server_info", None) or {}
        server_ver = server_info.get("server_version", "unknown")

        lines = [
            "",
            "\u2550" * 60,
            f"  CAMA Connector v{self._client_version}",
            "\u2550" * 60,
            f"  server         : {cfg.remote_addr}:{cfg.remote_port} (v{server_ver})",
            f"  backend        : {self._backend}",
            f"  pool_size      : {cfg.pool_size}",
            f"  io_workers     : {cfg.io_workers}",
            f"  op_timeout     : {cfg.op_timeout_s}s",
            f"  send_buf_size  : {send_buf_desc}",
            "\u2500" * 60,
            f"  nic_striping   : {nic_striping_status}",
            f"  batch_ops      : {'ON' if cfg.use_mput_mget else 'OFF'}",
            f"  codec          : {codec_desc}",
            f"  dedup          : {cfg.dedup_mode}",
            f"  reconnect      : {reconnect_status}",
            "\u2550" * 60,
            f"  Ready in {init_elapsed:.2f}s",
        ]
        logger.info("\n".join(lines))

    # -------------------------------------------------------------------
    # Post-reconnect state refresh
    # -------------------------------------------------------------------

    def _on_reconnect(self):
        """Fired by client after transport is replaced and MRs are re-registered.

        Refreshes stale connector state:
        - _server_info (metrics_addr may change)
        - dedup state (server may have been restarted/flushed)
        - reconnect counter
        """
        # 1. Refresh server info
        self._server_info = getattr(self.conn, "_server_info", None)

        # 2. _reg_buf handle is still valid — the client's _mr_map
        #    re-registered it with the same handle and updated the lkey
        #    internally via _MREntry.  No action needed at connector level.

        # 3. Reset dedup state (server may have been restarted/flushed)
        self.reset_dedup_state()

        # 4. Increment reconnect counter
        with self._counter_lock:
            self._reconnect_count += 1

        # 5. Reset warmup (server may have restarted with cold shards)
        if self._warmup_reset_ref is not None:
            try:
                self._warmup_reset_ref()
            except Exception:
                pass

        # 6. Re-send page size hint (server may have lost it on restart)
        if self._bytes_per_page > 0:
            try:
                self.conn.report_stats({"model_page_bytes": self._bytes_per_page})
            except Exception:
                pass

        logger.warning("CAMA reconnected (total: %d)", self._reconnect_count)

    # -------------------------------------------------------------------
    # Section C: register_mem_pool_host — RDMA buffer registration
    # -------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        logger.debug(
            "register_mem_pool_host: layout=%s", self.mem_pool_host.layout,
        )
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
        ], (
            "Cama storage backend only supports page_first, page_first_direct, "
            "or page_head layout, got %s" % self.mem_pool_host.layout
        )

        buffer = self.mem_pool_host.kv_buffer
        try:
            buffer_ptr = buffer.data_ptr()
            buffer_size = buffer.numel() * buffer.element_size()
            _rlvl = logging.INFO if self.local_rank == 0 else logging.DEBUG
            logger.log(
                _rlvl,
                "[7/7] RDMA buffer: registering %s...",
                _fmt_bytes(buffer_size),
            )
            reg_start = time.perf_counter()
            self._reg_buf = self.conn.reg_memory(buffer_ptr, buffer_size)
            reg_elapsed = time.perf_counter() - reg_start
            if self._reg_buf == 0:
                raise RuntimeError(
                    "PrisKV reg_memory returned 0 — RDMA buffer registration failed"
                )
            logger.log(
                _rlvl,
                "[7/7] RDMA buffer: registered %s (%.2fs)",
                _fmt_bytes(buffer_size),
                reg_elapsed,
            )
        except TypeError as err:
            logger.error("Failed to register buffer with PrisKV: %s", err)
            raise TypeError("Cama PrisKV Register Buffer Error.") from err

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)
        self._bytes_per_page = int(bytes_per_page)

        # Warn if config model_page_bytes was set but differs from computed value.
        # The computed value (from the actual model) is authoritative.
        if (
            hasattr(self, "config")
            and self.config is not None
            and self.config.model_page_bytes > 0
            and self.config.model_page_bytes != self._bytes_per_page
        ):
            logger.warning(
                "model_page_bytes config (%d) differs from computed page size (%d) "
                "— computed value takes precedence",
                self.config.model_page_bytes,
                self._bytes_per_page,
            )

        # Eagerly send page-size hint so the server can build optimized slab
        # classes *before* the first batch of writes arrives.  Without this,
        # the server relies on auto-detect (warmup_ops SETs), which means the
        # first writes allocate from a sub-optimal class and trigger a
        # ZeroLatencyBalance migration under load.
        if self._bytes_per_page > 0:
            try:
                self.conn.report_stats({"model_page_bytes": self._bytes_per_page})
                logger.log(
                    _rlvl,
                    "Sent model_page_bytes hint to server: %d bytes",
                    self._bytes_per_page,
                )
            except Exception as exc:
                logger.debug("Failed to send page-size hint: %s", exc)

    # -------------------------------------------------------------------
    # Section D: Key naming helpers
    # -------------------------------------------------------------------

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _apply_tag(self, keys):
        """Prefix keys with extra_backend_tag if set."""
        if self.extra_backend_tag is not None:
            return [f"{self.extra_backend_tag}_{key}" for key in keys]
        return keys

    # -------------------------------------------------------------------
    # Section E: Zero-copy transfer primitives
    # -------------------------------------------------------------------

    @nvtx_range("_put_batch_zero_copy", "cama.PrisKV")
    def _put_batch_zero_copy(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int],
    ) -> List[int]:
        with tag_wrapper({"op": "rdma_write"}):
            if self._codec is not None:
                from cama_module.codec import wrap_value, _CompressedSGL
                sgls = []
                for ptr, size in zip(buffer_ptrs, buffer_sizes):
                    raw_sgl = self._priskv.SGL(ptr, size, self._reg_buf)
                    raw = raw_sgl.to_bytes()
                    compressed = wrap_value(self._codec, raw)
                    sgls.append(_CompressedSGL(compressed))
            else:
                sgls = [
                    self._priskv.SGL(ptr, size, self._reg_buf)
                    for ptr, size in zip(buffer_ptrs, buffer_sizes)
                ]

            if self.config.use_mput_mget:
                # Native batch: prefer striped mset when pool supports it
                t0 = time.perf_counter()
                try:
                    if hasattr(self.conn, "mset_striped"):
                        results = self.conn.mset_striped(key_strs, sgls)
                    else:
                        results = self.conn.mset(key_strs, sgls)
                    self._backpressure.on_success()
                except (_CamaServerOverloadError, _CamaNotReadyError):
                    self._backpressure.on_overload()
                    try:
                        if hasattr(self.conn, "mset_striped"):
                            results = self.conn.mset_striped(key_strs, sgls)
                        else:
                            results = self.conn.mset(key_strs, sgls)
                        self._backpressure.on_success()
                    except (_CamaServerOverloadError, _CamaNotReadyError):
                        logger.error("_put_batch_zero_copy: server overloaded after retry, returning all-failures")
                        results = [-1] * len(key_strs)
                except Exception as e:
                    logger.error("_put_batch_zero_copy: native mset failed: %s, falling back", e)
                    results = self._put_batch_zero_copy_fallback(key_strs, sgls)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                ok = sum(1 for r in results if r == 0)
                with self._counter_lock:
                    self._set_successes += ok
                    self._set_errors += len(key_strs) - ok
                logger.debug(
                    "_put_batch_zero_copy: %d keys, %d ok, %d failed, %.1fms (native batch)",
                    len(key_strs), ok, len(key_strs) - ok, elapsed_ms,
                )
                return results

            # Fallback: individual ops via thread pool
            return self._put_batch_zero_copy_fallback(key_strs, sgls)

    def _put_batch_zero_copy_fallback(
        self, key_strs: List[str], sgls
    ) -> List[int]:
        """Individual set() calls via ThreadPoolExecutor (fallback path)."""
        def _do_set(key, sgl):
            try:
                return self.conn.set(key, sgl)
            except Exception as e:
                logger.error("_put_batch_zero_copy: set(%s) failed: %s", key[:64], e)
                return -1

        t0 = time.perf_counter()
        futures = {
            self._io_pool.submit(_do_set, k, s): idx
            for idx, (k, s) in enumerate(zip(key_strs, sgls))
        }
        results = [-1] * len(key_strs)
        try:
            for future in as_completed(futures, timeout=self.config.op_timeout_s):
                idx = futures[future]
                results[idx] = future.result()
        except TimeoutError:
            for f in futures:
                if not f.done():
                    f.cancel()
            timed_out = [key_strs[futures[f]] for f in futures if not f.done()]
            logger.error(
                "_put_batch_zero_copy: %d/%d keys timed out after %.1fs: %s",
                len(timed_out), len(key_strs), self.config.op_timeout_s,
                [k[:48] for k in timed_out[:5]],
            )
            completed = len(key_strs) - len(timed_out)
            logger.warning(
                "PARTIAL RESULTS: %d/%d keys completed, %d timed out. "
                "Timed-out keys are indistinguishable from cache misses.",
                completed, len(key_strs), len(timed_out),
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = sum(1 for r in results if r == 0)
        with self._counter_lock:
            self._set_successes += ok
            self._set_errors += len(key_strs) - ok
        logger.debug(
            "_put_batch_zero_copy: %d keys, %d ok, %d failed, %.1fms (workers=%d)",
            len(key_strs), ok, len(key_strs) - ok,
            elapsed_ms, self.config.io_workers,
        )
        return results

    def _record_get_metrics(self, results, key_count, t0, label):
        """Record GET success/error counters and debug log."""
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = sum(1 for r in results if r == _RC.GET_OK)
        with self._counter_lock:
            self._get_successes += ok
            self._get_errors += key_count - ok
        logger.debug(
            "_get_batch_zero_copy: %d keys, %d ok, %d failed, %.1fms (%s)",
            key_count, ok, key_count - ok, elapsed_ms, label,
        )

    @nvtx_range("_get_batch_zero_copy", "cama.PrisKV")
    def _get_batch_zero_copy(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int],
    ) -> List[int]:
        with tag_wrapper({"op": "rdma_read"}):
            if self._codec is not None:
                # Try batch RDMA path with internal buffer + decompress
                if hasattr(self.conn, "mget_rdma_raw"):
                    from cama_module.codec import unwrap_value
                    t0 = time.perf_counter()
                    try:
                        raw_results = self.conn.mget_rdma_raw(key_strs)
                        self._backpressure.on_success()
                    except (_CamaServerOverloadError, _CamaNotReadyError):
                        self._backpressure.on_overload()
                        try:
                            raw_results = self.conn.mget_rdma_raw(key_strs)
                            self._backpressure.on_success()
                        except (_CamaServerOverloadError, _CamaNotReadyError):
                            logger.error("_get_batch_zero_copy: server overloaded after retry")
                            self._record_get_metrics([_RC.GET_MISS] * len(key_strs), len(key_strs), t0, "overloaded")
                            return [_RC.GET_MISS] * len(key_strs)
                    except Exception as e:
                        logger.error("_get_batch_zero_copy: mget_rdma_raw failed: %s, returning misses", e)
                        self._record_get_metrics([_RC.GET_MISS] * len(key_strs), len(key_strs), t0, "mget_rdma_raw error")
                        return [_RC.GET_MISS] * len(key_strs)
                    results = []
                    for i, (rc, data) in enumerate(raw_results):
                        if rc == 0 and data is not None:
                            decompressed = unwrap_value(data)
                            n = min(len(decompressed), buffer_sizes[i])
                            ctypes.memmove(buffer_ptrs[i], decompressed, n)
                            results.append(_RC.GET_OK)
                        else:
                            results.append(_RC.GET_MISS)
                    self._record_get_metrics(results, len(key_strs), t0, "compressed mget_rdma")
                    return results

                # Fallback: _DecodeSGL + sequential get() (old path)
                from cama_module.codec import _DecodeSGL
                sgls = [
                    _DecodeSGL(ptr, size)
                    for ptr, size in zip(buffer_ptrs, buffer_sizes)
                ]
                return self._get_batch_zero_copy_fallback(
                    key_strs, sgls, buffer_sizes,
                )

            sgls = [
                self._priskv.SGL(ptr, size, self._reg_buf)
                for ptr, size in zip(buffer_ptrs, buffer_sizes)
            ]

            # Batch RDMA path: single control roundtrip + batch RDMA Reads
            if hasattr(self.conn, "mget_rdma"):
                t0 = time.perf_counter()
                try:
                    results = self.conn.mget_rdma(key_strs, sgls, buffer_sizes)
                    self._backpressure.on_success()
                except (_CamaServerOverloadError, _CamaNotReadyError):
                    self._backpressure.on_overload()
                    try:
                        results = self.conn.mget_rdma(key_strs, sgls, buffer_sizes)
                        self._backpressure.on_success()
                    except (_CamaServerOverloadError, _CamaNotReadyError):
                        logger.error("_get_batch_zero_copy: server overloaded after retry")
                        results = [_RC.GET_MISS] * len(key_strs)
                except Exception as e:
                    logger.error("_get_batch_zero_copy: mget_rdma failed: %s, falling back", e)
                    results = self._get_batch_zero_copy_fallback(key_strs, sgls, buffer_sizes)
                self._record_get_metrics(results, len(key_strs), t0, "mget_rdma")
                return results

            # Fallback: individual get() via ThreadPoolExecutor
            return self._get_batch_zero_copy_fallback(key_strs, sgls, buffer_sizes)

    def _get_batch_zero_copy_fallback(
        self,
        key_strs: List[str],
        sgls,
        buffer_sizes: List[int],
    ) -> List[int]:
        """Individual get() calls via ThreadPoolExecutor (fallback path)."""
        def _do_get(key, sgl, size):
            try:
                return self.conn.get(key, sgl, size)
            except Exception as e:
                logger.error("_get_batch_zero_copy: get(%s) failed: %s", key[:64], e)
                return -1

        t0 = time.perf_counter()
        futures = {
            self._io_pool.submit(_do_get, k, s, sz): idx
            for idx, (k, s, sz) in enumerate(zip(key_strs, sgls, buffer_sizes))
        }
        results = [-1] * len(key_strs)
        try:
            for future in as_completed(futures, timeout=self.config.op_timeout_s):
                idx = futures[future]
                results[idx] = future.result()
        except TimeoutError:
            for f in futures:
                if not f.done():
                    f.cancel()
            timed_out = [key_strs[futures[f]] for f in futures if not f.done()]
            logger.error(
                "_get_batch_zero_copy: %d/%d keys timed out after %.1fs: %s",
                len(timed_out), len(key_strs), self.config.op_timeout_s,
                [k[:48] for k in timed_out[:5]],
            )
            completed = len(key_strs) - len(timed_out)
            logger.warning(
                "PARTIAL RESULTS: %d/%d keys completed, %d timed out. "
                "Timed-out keys are indistinguishable from cache misses.",
                completed, len(key_strs), len(timed_out),
            )
        self._record_get_metrics(results, len(key_strs), t0, f"workers={self.config.io_workers}")
        return results

    @nvtx_range("_batch_exist", "cama.PrisKV")
    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        with tag_wrapper({"op": "mexists"}):
            if self.config.use_mput_mget:
                # Native batch: single MTEST roundtrip
                t0 = time.perf_counter()
                try:
                    results = self.conn.mexists(key_strs)
                    self._backpressure.on_success()
                except (_CamaServerOverloadError, _CamaNotReadyError):
                    self._backpressure.on_overload()
                    try:
                        results = self.conn.mexists(key_strs)
                        self._backpressure.on_success()
                    except (_CamaServerOverloadError, _CamaNotReadyError):
                        logger.error("_batch_exist: server overloaded after retry, returning all-missing")
                        with self._counter_lock:
                            self._exists_errors += len(key_strs)
                        return [_RC.EXISTS_MISSING] * len(key_strs)
                except Exception as e:
                    logger.error("_batch_exist: native mexists failed: %s, falling back", e)
                    with self._counter_lock:
                        self._exists_errors += len(key_strs)
                    return self._batch_exist_fallback(key_strs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                found = sum(1 for r in results if r == _RC.EXISTS_FOUND)
                logger.debug(
                    "_batch_exist: %d keys, %d found, %d missing, %.1fms (native batch)",
                    len(key_strs), found, len(key_strs) - found, elapsed_ms,
                )
                return results

            # Fallback: individual ops via thread pool
            return self._batch_exist_fallback(key_strs)

    def _batch_exist_fallback(self, key_strs: List[str]) -> List[int]:
        """Individual exists() calls via ThreadPoolExecutor (fallback path)."""
        def _do_exists(key):
            try:
                return self.conn.exists(key)
            except Exception as e:
                logger.error("_batch_exist: exists(%s) failed: %s", key[:64], e)
                with self._counter_lock:
                    self._exists_errors += 1
                return _EXISTS_ERROR

        t0 = time.perf_counter()
        futures = {
            self._io_pool.submit(_do_exists, k): idx
            for idx, k in enumerate(key_strs)
        }
        results = [_EXISTS_ERROR] * len(key_strs)
        try:
            for future in as_completed(futures, timeout=self.config.op_timeout_s):
                idx = futures[future]
                results[idx] = future.result()
        except TimeoutError:
            timed_out = [key_strs[futures[f]] for f in futures if not f.done()]
            for f in futures:
                if not f.done():
                    f.cancel()
            with self._counter_lock:
                self._exists_timeouts += 1
                self._exists_errors += len(timed_out)
            logger.error(
                "_batch_exist: %d/%d keys timed out after %.1fs: %s",
                len(timed_out), len(key_strs), self.config.op_timeout_s,
                [k[:48] for k in timed_out[:5]],
            )
            completed = len(key_strs) - len(timed_out)
            logger.warning(
                "PARTIAL RESULTS: %d/%d keys completed, %d timed out. "
                "Timed-out keys are indistinguishable from cache misses.",
                completed, len(key_strs), len(timed_out),
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        found = sum(1 for r in results if r == _RC.EXISTS_FOUND)
        errors = sum(1 for r in results if r == _EXISTS_ERROR)
        logger.debug(
            "_batch_exist: %d keys, %d found, %d missing, %d errors, %.1fms (workers=%d)",
            len(key_strs), found, len(key_strs) - found - errors, errors,
            elapsed_ms, self.config.io_workers,
        )
        return results

    # -------------------------------------------------------------------
    # Section F: Result postprocessing
    # -------------------------------------------------------------------

    def _batch_postprocess(self, results: List[int], is_set_operate=False) -> List[bool]:
        """Convert per-sub-key integer results to per-page booleans.

        CAMA/PrisKV: SET_OK=0 and GET_OK=0 for success (unlike mooncake where
        get success = bytes > 0).
        """
        ok = _RC.SET_OK if is_set_operate else _RC.GET_OK
        if self.is_mla_backend:
            # 1 result per page (fused KV)
            return [r == ok for r in results]
        else:
            # 2 results (K, V) per page
            kv_pairs = zip(results[::2], results[1::2])
            return [(k == ok and v == ok) for k, v in kv_pairs]

    # -------------------------------------------------------------------
    # Section G: V1 API (primary interface SGLang calls)
    # -------------------------------------------------------------------

    def _percentile_bucket(self, p: float) -> str:
        """Return '<=Xms' for the bucket containing the p-th percentile."""
        if self._hist_total == 0:
            return "n/a"
        target = self._hist_total * p
        running = 0
        for i, count in enumerate(self._hist_counts):
            running += count
            if running >= target:
                if i < len(self._LATENCY_BUCKETS_MS):
                    return f"<={self._LATENCY_BUCKETS_MS[i]:.0f}ms"
                return ">5000ms"
        return ">5000ms"

    def _record_histogram_unlocked(self, elapsed_ms: float) -> None:
        """Record a latency sample — caller MUST hold _telemetry_lock."""
        self._hist_sum += elapsed_ms
        self._hist_total += 1
        for i, boundary in enumerate(self._LATENCY_BUCKETS_MS):
            if elapsed_ms <= boundary:
                self._hist_counts[i] += 1
                return
        self._hist_counts[-1] += 1  # +Inf bucket

    def _record_histogram(self, elapsed_ms: float) -> None:
        """Record a latency sample into the histogram buckets.

        Acquires _telemetry_lock to prevent hist_sum/total corruption
        when get_stats() snapshots and resets concurrently.
        """
        with self._telemetry_lock:
            self._record_histogram_unlocked(elapsed_ms)

    @nvtx_range("batch_get_v1", "cama.PrisKV")
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        with tag_wrapper({"op": "prefetch"}):
            with self._counter_lock:
                self._inflight_gets += 1
            try:
                t_start = time.perf_counter()
                keys = self._apply_tag(keys)
                key_strs, ptrs, sizes = self._batch_preprocess(keys, host_indices)
                t_pre = time.perf_counter()

                results = self._get_batch_zero_copy(key_strs, ptrs, sizes)
                t_xfer = time.perf_counter()

                get_ok = sum(1 for r in results if r == 0)
                ret = self._batch_postprocess(results, is_set_operate=False)
                t_post = time.perf_counter()

                elapsed_ms = (t_post - t_start) * 1000
                with self._telemetry_lock:
                    self._io_batch_sizes.append(len(key_strs))
                    self._io_latencies_ms.append(elapsed_ms)
                    self._phase_preprocess_ms.append((t_pre - t_start) * 1000)
                    self._phase_transfer_ms.append((t_xfer - t_pre) * 1000)
                    self._phase_postprocess_ms.append((t_post - t_xfer) * 1000)
                    self._record_histogram_unlocked(elapsed_ms)
                    if elapsed_ms > self._io_max_latency_ms:
                        self._io_max_latency_ms = elapsed_ms
                n_pages = len(key_strs) // self._sub_keys_per_page
                with self._counter_lock:
                    self._interval_pages_get += n_pages
                    self._total_pages_get += n_pages

                logger.debug(
                    "batch_get_v1: %d sub-keys, %d hit, %d miss (%.1fms total, "
                    "pre=%.1fms xfer=%.1fms post=%.1fms)",
                    len(results), get_ok, len(results) - get_ok, elapsed_ms,
                    (t_pre - t_start) * 1000, (t_xfer - t_pre) * 1000,
                    (t_post - t_xfer) * 1000,
                )
                return ret
            except Exception as e:
                logger.error("batch_get_v1: unhandled exception: %s", e, exc_info=True)
                with self._counter_lock:
                    self._get_errors += len(keys) * self._sub_keys_per_page
                return [False] * len(keys)
            finally:
                with self._counter_lock:
                    self._inflight_gets -= 1

    def _batch_set_exists_dedup(
        self, key_strs: List[str], ptrs: List[int], sizes: List[int]
    ) -> tuple:
        """Dedup path: check existence, write only missing keys.

        Returns (merged_results, hit_rate, exists_ms, transfer_ms).
        """
        t_exists_start = time.perf_counter()
        exist_results = self._batch_exist(key_strs)
        t_exists_end = time.perf_counter()

        set_keys = []
        set_ptrs = []
        set_sizes = []
        set_indices = []
        merged_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_results[i] == _RC.EXISTS_FOUND:
                merged_results[i] = _RC.SET_OK
            else:
                set_keys.append(key_strs[i])
                set_ptrs.append(ptrs[i])
                set_sizes.append(sizes[i])
                set_indices.append(i)

        deduped_count = len(key_strs) - len(set_keys)
        hit_rate = deduped_count / len(key_strs) if key_strs else 0.0
        logger.debug(
            "batch_set_v1: %d sub-keys, %d already exist (%.0f%% deduped), %d to write",
            len(key_strs), deduped_count, hit_rate * 100, len(set_keys),
        )

        t_xfer_start = time.perf_counter()
        if len(set_keys) > 0:
            put_results = self._put_batch_zero_copy(set_keys, set_ptrs, set_sizes)
            write_ok = sum(1 for r in put_results if r == 0)
            write_fail = len(put_results) - write_ok
            logger.debug(
                "batch_set_v1: wrote %d sub-keys: %d ok, %d failed",
                len(put_results), write_ok, write_fail,
            )
            for i in range(len(set_indices)):
                merged_results[set_indices[i]] = put_results[i]
        t_xfer_end = time.perf_counter()

        exists_ms = (t_exists_end - t_exists_start) * 1000
        transfer_ms = (t_xfer_end - t_xfer_start) * 1000
        return merged_results, hit_rate, exists_ms, transfer_ms

    def reset_dedup_state(self) -> None:
        """Reset adaptive dedup state (e.g. after server flush)."""
        self._dedup_low_hit_streak = 0
        self._dedup_auto_disabled = False
        self._dedup_cost_streak = 0
        self._dedup_batches_since_disable = 0
        self._dedup_probe_hit_streak = 0
        self._dedup_enabled = self.config.dedup_mode in ("always", "auto")
        logger.info("Dedup state reset: enabled=%s, mode=%s",
                     self._dedup_enabled, self.config.dedup_mode)

    def set_warmup_cold_check(self, ref) -> None:
        """Set a callable that returns True when cache is cold (skip dedup)."""
        self._warmup_phase_ref = ref

    def set_warmup_phase_ref(self, ref) -> None:
        """Deprecated: use set_warmup_cold_check instead."""
        self._warmup_phase_ref = ref

    def set_warmup_reset_ref(self, ref) -> None:
        """Set a callable to reset warmup state (called on reconnect)."""
        self._warmup_reset_ref = ref

    @nvtx_range("batch_set_v1", "cama.PrisKV")
    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        with tag_wrapper({"op": "backup"}):
            with self._counter_lock:
                self._inflight_sets += 1
            try:
                t_start = time.perf_counter()
                keys = self._apply_tag(keys)
                key_strs, ptrs, sizes = self._batch_preprocess(keys, host_indices)
                t_pre = time.perf_counter()

                # Check if caller asked to skip dedup
                skip_dedup = (
                    extra_info is not None
                    and getattr(extra_info, "extra_info", None) is not None
                    and extra_info.extra_info.get("skip_dedup", False)
                )

                # Skip dedup during COLD warmup (cache is empty)
                cache_cold = False
                if self._warmup_phase_ref is not None:
                    try:
                        cache_cold = bool(self._warmup_phase_ref())
                    except Exception:
                        pass

                # Determine whether to run dedup (including probe logic)
                is_probe = False
                should_dedup = (
                    not skip_dedup
                    and not cache_cold
                    and self._dedup_enabled
                    and self.config.dedup_mode != "never"
                )
                if should_dedup and self._dedup_auto_disabled:
                    # Dedup is auto-disabled — check if this batch is a probe
                    probe_interval = self.config.dedup_probe_interval
                    if probe_interval > 0:
                        with self._counter_lock:
                            self._dedup_batches_since_disable += 1
                            if self._dedup_batches_since_disable >= probe_interval:
                                self._dedup_batches_since_disable = 0
                                self._dedup_probes_total += 1
                                is_probe = True
                    if not is_probe:
                        should_dedup = False

                if skip_dedup or not should_dedup:
                    reason = (
                        "skip_dedup" if skip_dedup
                        else "cache_cold" if cache_cold
                        else f"dedup_mode={self.config.dedup_mode}" if self.config.dedup_mode == "never"
                        else "auto_disabled"
                    )
                    logger.debug(
                        "batch_set_v1: %s, writing all %d sub-keys",
                        reason, len(key_strs),
                    )
                    t_xfer_start = time.perf_counter()
                    merged_results = self._put_batch_zero_copy(key_strs, ptrs, sizes)
                    t_io_done = time.perf_counter()
                    with self._telemetry_lock:
                        self._phase_transfer_ms.append((t_io_done - t_xfer_start) * 1000)
                else:
                    if is_probe:
                        logger.debug(
                            "batch_set_v1: dedup PROBE batch (%d keys, probe #%d)",
                            len(key_strs), self._dedup_probes_total,
                        )
                    merged_results, hit_rate, exists_ms, transfer_ms = self._batch_set_exists_dedup(
                        key_strs, ptrs, sizes
                    )
                    t_io_done = time.perf_counter()
                    with self._telemetry_lock:
                        self._phase_exists_ms.append(exists_ms)
                        self._phase_transfer_ms.append(transfer_ms)

                    # Adaptive dedup evaluation runs between t_io_done and t_post;
                    # overhead captured in postprocess phase timing.
                    if self.config.dedup_mode == "auto":
                        with self._counter_lock:
                            if is_probe:
                                # Evaluate probe result for re-enable
                                if hit_rate >= self.config.dedup_auto_threshold:
                                    self._dedup_probe_hit_streak += 1
                                    if self._dedup_probe_hit_streak >= self.config.dedup_probe_window:
                                        self._dedup_auto_disabled = False
                                        self._dedup_low_hit_streak = 0
                                        self._dedup_cost_streak = 0
                                        self._dedup_probe_hit_streak = 0
                                        self._dedup_batches_since_disable = 0
                                        self._dedup_reenables_total += 1
                                        logger.warning(
                                            "batch_set_v1: dedup RE-ENABLED after %d consecutive "
                                            "probe batches above threshold (hit_rate=%.1f%%, "
                                            "re-enable #%d).",
                                            self.config.dedup_probe_window,
                                            hit_rate * 100,
                                            self._dedup_reenables_total,
                                        )
                                    else:
                                        logger.info(
                                            "batch_set_v1: dedup probe hit (%.1f%%), "
                                            "streak %d/%d for re-enable.",
                                            hit_rate * 100,
                                            self._dedup_probe_hit_streak,
                                            self.config.dedup_probe_window,
                                        )
                                else:
                                    self._dedup_probe_hit_streak = 0
                                    logger.debug(
                                        "batch_set_v1: dedup probe miss (%.1f%%), "
                                        "staying disabled.",
                                        hit_rate * 100,
                                    )
                            else:
                                # Normal dedup evaluation (not a probe)
                                if hit_rate < self.config.dedup_auto_threshold:
                                    self._dedup_low_hit_streak += 1
                                    if self._dedup_low_hit_streak == self.config.dedup_auto_window - 1:
                                        logger.warning(
                                            "batch_set_v1: dedup auto-disable imminent — "
                                            "%d/%d consecutive low-hit batches.",
                                            self._dedup_low_hit_streak,
                                            self.config.dedup_auto_window,
                                        )
                                    if self._dedup_low_hit_streak >= self.config.dedup_auto_window:
                                        self._dedup_auto_disabled = True
                                        self._dedup_batches_since_disable = 0
                                        self._dedup_probe_hit_streak = 0
                                        if self.config.dedup_probe_interval > 0:
                                            logger.warning(
                                                "batch_set_v1: adaptive dedup AUTO-DISABLED "
                                                "after %d consecutive low-hit batches "
                                                "(threshold=%.1f%%). Will probe every %d "
                                                "batches to re-evaluate.",
                                                self._dedup_low_hit_streak,
                                                self.config.dedup_auto_threshold * 100,
                                                self.config.dedup_probe_interval,
                                            )
                                        else:
                                            logger.warning(
                                                "batch_set_v1: adaptive dedup AUTO-DISABLED "
                                                "after %d consecutive low-hit batches "
                                                "(threshold=%.1f%%). "
                                                "Call reset_dedup_state() to re-enable, "
                                                "or set dedup_mode='always'.",
                                                self._dedup_low_hit_streak,
                                                self.config.dedup_auto_threshold * 100,
                                            )
                                else:
                                    self._dedup_low_hit_streak = 0
                                    self._dedup_cost_streak = 0

                                # Cost-aware disable: if exists overhead dominates transfer
                                if not self._dedup_auto_disabled and transfer_ms > 0:
                                    cost_ratio = exists_ms / transfer_ms
                                    if cost_ratio > self.config.dedup_cost_ratio_threshold:
                                        self._dedup_cost_streak += 1
                                        if self._dedup_cost_streak >= self.config.dedup_auto_window:
                                            self._dedup_auto_disabled = True
                                            self._dedup_batches_since_disable = 0
                                            self._dedup_probe_hit_streak = 0
                                            if self.config.dedup_probe_interval > 0:
                                                logger.warning(
                                                    "batch_set_v1: dedup AUTO-DISABLED "
                                                    "(cost-aware) — exists_ms/transfer_ms "
                                                    "ratio > %.2f for %d consecutive batches. "
                                                    "Will probe every %d batches.",
                                                    self.config.dedup_cost_ratio_threshold,
                                                    self._dedup_cost_streak,
                                                    self.config.dedup_probe_interval,
                                                )
                                            else:
                                                logger.warning(
                                                    "batch_set_v1: dedup AUTO-DISABLED "
                                                    "(cost-aware) — exists_ms/transfer_ms "
                                                    "ratio > %.2f for %d consecutive batches.",
                                                    self.config.dedup_cost_ratio_threshold,
                                                    self._dedup_cost_streak,
                                                )
                                    else:
                                        self._dedup_cost_streak = 0

                ret = self._batch_postprocess(merged_results, is_set_operate=True)
                t_post = time.perf_counter()

                elapsed_ms = (t_post - t_start) * 1000
                with self._telemetry_lock:
                    self._io_batch_sizes.append(len(key_strs))
                    self._io_latencies_ms.append(elapsed_ms)
                    self._phase_preprocess_ms.append((t_pre - t_start) * 1000)
                    self._phase_postprocess_ms.append((t_post - t_io_done) * 1000)
                    self._record_histogram_unlocked(elapsed_ms)
                    if elapsed_ms > self._io_max_latency_ms:
                        self._io_max_latency_ms = elapsed_ms
                n_pages = len(key_strs) // self._sub_keys_per_page
                with self._counter_lock:
                    self._interval_pages_set += n_pages
                    self._total_pages_set += n_pages
                logger.debug("batch_set_v1: total I/O %.1fms", elapsed_ms)

                return ret
            except Exception as e:
                logger.error("batch_set_v1: unhandled exception: %s", e, exc_info=True)
                with self._counter_lock:
                    self._set_errors += len(keys) * self._sub_keys_per_page
                return [False] * len(keys)
            finally:
                with self._counter_lock:
                    self._inflight_sets -= 1

    # -------------------------------------------------------------------
    # Section H: Legacy API (required by abstract base class)
    # -------------------------------------------------------------------

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy([key], [target_location], [target_sizes])
        success = get_result[0] == 0
        logger.debug("get: key=%.64s, success=%s", key, success)
        return success

    @nvtx_range("batch_get", "cama.PrisKV")
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        with tag_wrapper({"op": "prefetch"}):
            assert len(keys) == len(target_locations) == len(target_sizes)
            if len(keys) == 0:
                return 0

            start_time = time.perf_counter()
            get_result = self._get_batch_zero_copy(keys, target_locations, target_sizes)
            end_time = time.perf_counter()

            if self.is_mla_backend:
                key_multiplier = 1
            else:
                key_multiplier = 2

            if self.gb_per_page is not None:
                self.prefetch_pgs.append(len(keys))
                elapsed = end_time - start_time
                if elapsed > 0:
                    self.prefetch_bandwidth.append(
                        len(keys) / elapsed * self.gb_per_page
                    )

            success_pages = len(keys) // key_multiplier
            for i in range(len(keys)):
                if get_result[i] != 0:
                    success_pages = i // key_multiplier
                    break
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(
                "batch_get: success=%d/%d pages, elapsed=%.1fms",
                success_pages, len(keys) // key_multiplier, elapsed_ms,
            )
            return success_pages

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist([key])
        if exist_result[0] == _RC.EXISTS_FOUND:
            logger.debug("set: key=%.64s, already_exists=True", key)
            return True
        put_result = self._put_batch_zero_copy([key], [target_location], [target_sizes])
        success = put_result[0] == 0
        logger.debug("set: key=%.64s, success=%s", key, success)
        return success

    @nvtx_range("batch_set", "cama.PrisKV")
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        with tag_wrapper({"op": "backup"}):
            assert target_locations is not None and target_sizes is not None
            assert len(keys) == len(target_locations) == len(target_sizes)

            if len(keys) == 0:
                return False

            for i in range(len(keys)):
                if keys[i] is None or target_locations[i] is None or target_sizes[i] is None:
                    return False

            exist_result = self._batch_exist(keys)
            set_keys = []
            set_locations = []
            set_sizes = []
            set_indices = []
            for i in range(len(keys)):
                if exist_result[i] != _RC.EXISTS_FOUND:
                    # EXISTS_MISSING or EXISTS_ERROR — write the key
                    set_keys.append(keys[i])
                    set_locations.append(target_locations[i])
                    set_sizes.append(target_sizes[i])
                    set_indices.append(i)

            start_time = time.perf_counter()
            put_result = self._put_batch_zero_copy(set_keys, set_locations, set_sizes)
            end_time = time.perf_counter()

            if self.gb_per_page is not None:
                self.backup_pgs.append(len(keys))
                elapsed = end_time - start_time
                if elapsed > 0:
                    self.backup_bandwidth.append(
                        len(keys) / elapsed * self.gb_per_page
                    )

            for i in range(len(set_indices)):
                if put_result[i] == 0:
                    exist_result[set_indices[i]] = 0

            success_count = 0
            for i in range(len(keys)):
                if exist_result[i] != 0:
                    break
                success_count += 1
            elapsed_ms = (end_time - start_time) * 1000
            logger.debug(
                "batch_set: success=%d/%d pages, elapsed=%.1fms",
                success_count, len(keys), elapsed_ms,
            )
            return success_count == len(keys)

    # -------------------------------------------------------------------
    # Section I: Existence, cleanup, metrics
    # -------------------------------------------------------------------

    def exists(self, key) -> bool:
        try:
            ret = self.conn.exists(key)
        except Exception as e:
            logger.warning("exists: key=%.64s failed: %s", key, e)
            return False
        found = ret == _RC.EXISTS_FOUND
        logger.debug("exists: key=%.64s, found=%s", key, found)
        return found

    @nvtx_range("batch_exists", "cama.PrisKV")
    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        with tag_wrapper({"op": "exists"}):
            if self.extra_backend_tag is not None:
                keys = [f"{self.extra_backend_tag}_{key}" for key in keys]

            if self.is_mla_backend:
                query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
                key_multiplier = 1
            else:
                query_keys = []
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

            if not query_keys:
                return 0

            # Short-circuit: if the very first sub-key is missing, skip the
            # full parallel batch check (saves N-1 RPCs when hit rate is ~0).
            try:
                if self.conn.exists(query_keys[0]) == _RC.EXISTS_MISSING:
                    logger.debug(
                        "batch_exists: short-circuit miss on first key, skipped %d RPCs",
                        len(query_keys) - 1,
                    )
                    return 0
            except Exception as exc:
                logger.error(
                    "batch_exists: backend error on first key (%s): %s — treating as 0 hits",
                    query_keys[0][:64] if query_keys else "?", exc,
                )
                with self._counter_lock:
                    self._exists_errors += 1
                return 0

            # First key exists — check the rest in parallel
            exist_result = self._batch_exist(query_keys)
            hit_count = 0
            for i in range(len(query_keys)):
                if exist_result[i] != _RC.EXISTS_FOUND:
                    # EXISTS_MISSING or EXISTS_ERROR — stop counting hits
                    hit_count = i // key_multiplier
                    logger.debug(
                        "batch_exists: %d/%d pages hit (first non-hit at sub-key %d)",
                        hit_count, len(keys), i,
                    )
                    return hit_count
            hit_count = len(query_keys) // key_multiplier
            logger.debug("batch_exists: all %d/%d pages hit", hit_count, len(keys))
            return hit_count

    def close(self):
        if hasattr(self, "_stats_executor") and self._stats_executor is not None:
            self._stats_executor.shutdown(wait=False)
        if hasattr(self, "_io_pool") and self._io_pool is not None:
            self._io_pool.shutdown(wait=False)
            logger.info("Shut down I/O thread pool.")
        if self._reg_buf is not None and self._reg_buf != 0:
            try:
                self.conn.dereg_memory(self._reg_buf)
                logger.info("Deregistered RDMA buffer handle %d", self._reg_buf)
            except Exception as e:
                logger.warning("Failed to deregister RDMA buffer: %s", e)
            self._reg_buf = None
        try:
            self.conn.close()
            logger.info("Closed PrisKV connection.")
        except Exception as e:
            logger.warning("Failed to close PrisKV connection: %s", e)

    def clear(self) -> None:
        try:
            all_keys = self.conn.keys("*")
            if all_keys:
                self.conn.mdel(all_keys)
                logger.info("Cleared %d keys from PrisKV.", len(all_keys))
        except Exception as e:
            logger.warning("Failed to clear PrisKV store: %s", e)

    def update_sglang_metrics(self, metrics: dict) -> None:
        """Merge SGLang-level metrics for inclusion in the next report_stats() call.

        Multiple callers (backup thread, prefetch thread, scheduler) may push
        their own metric keys independently.  Metrics are *merged* (dict.update)
        rather than replaced so that each caller's keys coexist.

        Common keys: cache_hit_rate, token_usage, num_running_reqs,
                     evictable_ratio, gen_throughput, backup_*, prefetch_*.
        """
        if metrics is not None:
            self._sglang_metrics.update(metrics)

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()

        # Concurrent I/O metrics — swap-and-operate under _telemetry_lock
        # to prevent races with concurrent batch_get_v1/batch_set_v1 appends.
        with self._telemetry_lock:
            io_batch_sizes = self._io_batch_sizes; self._io_batch_sizes = []
            io_latencies_ms = self._io_latencies_ms; self._io_latencies_ms = []
            phase_pre = self._phase_preprocess_ms; self._phase_preprocess_ms = []
            phase_xfer = self._phase_transfer_ms; self._phase_transfer_ms = []
            phase_post = self._phase_postprocess_ms; self._phase_postprocess_ms = []
            phase_exists = self._phase_exists_ms; self._phase_exists_ms = []
            max_lat = self._io_max_latency_ms; self._io_max_latency_ms = 0.0
            # Snapshot and reset histogram
            hist_counts = self._hist_counts
            hist_sum = self._hist_sum
            hist_total = self._hist_total
            self._hist_counts = [0] * (len(self._LATENCY_BUCKETS_MS) + 1)
            self._hist_sum = 0.0
            self._hist_total = 0

        if io_batch_sizes:
            avg_batch = sum(io_batch_sizes) / len(io_batch_sizes)
            avg_lat = sum(io_latencies_ms) / len(io_latencies_ms)
            storage_metrics.io_workers = self.config.io_workers
            storage_metrics.avg_io_batch_size = avg_batch
            storage_metrics.avg_io_latency_ms = avg_lat
            storage_metrics.io_calls = len(io_batch_sizes)

            # Compute phase averages from local copies (outside lock)
            avg_pre = sum(phase_pre) / len(phase_pre) if phase_pre else 0.0
            avg_xfer = sum(phase_xfer) / len(phase_xfer) if phase_xfer else 0.0
            avg_post = sum(phase_post) / len(phase_post) if phase_post else 0.0
            avg_exists = sum(phase_exists) / len(phase_exists) if phase_exists else 0.0

            # --- Saturation metrics (additive) ---
            coalesce_max = getattr(self, 'storage_batch_size', 2048)
            if coalesce_max > 0:
                storage_metrics.coalesce_fill_pct = round(avg_batch / coalesce_max * 100, 2)
            else:
                storage_metrics.coalesce_fill_pct = 0.0
            if avg_lat > 0:
                storage_metrics.transfer_utilization_pct = round(avg_xfer / avg_lat * 100, 2)
            else:
                storage_metrics.transfer_utilization_pct = 0.0

            # Compute percentiles from snapshotted histogram
            def _percentile_from_snapshot(p):
                if hist_total == 0:
                    return "n/a"
                target = hist_total * p
                running = 0
                for i, count in enumerate(hist_counts):
                    running += count
                    if running >= target:
                        if i < len(self._LATENCY_BUCKETS_MS):
                            return f"<={self._LATENCY_BUCKETS_MS[i]:.0f}ms"
                        return ">5000ms"
                return ">5000ms"

            p50 = _percentile_from_snapshot(0.50)
            p99 = _percentile_from_snapshot(0.99)

            model_tag = "MLA" if self.is_mla_backend else "MHA"
            avg_batch_pages = avg_batch / self._sub_keys_per_page

            now = time.monotonic()
            elapsed_s = now - self._last_stats_time
            interval_pages = self._interval_pages_get + self._interval_pages_set
            if elapsed_s > 0 and self.gb_per_page and self.gb_per_page > 0:
                throughput_gbps = (interval_pages * self.gb_per_page) / elapsed_s
            else:
                throughput_gbps = 0.0
            self._last_stats_time = now

            logger.info(
                "[%s] CAMA I/O stats: %d calls, avg_batch=%.1f (%d pg), "
                "avg=%.1fms, max=%.1fms, %.2f GB/s, workers=%d | "
                "inflight: get=%d set=%d | phases: pre=%.1fms xfer=%.1fms post=%.1fms exists=%.1fms | "
                "p50=%s p99=%s | fill=%.0f%% xfer_util=%.0f%%",
                model_tag, len(io_batch_sizes), avg_batch, int(avg_batch_pages),
                avg_lat, max_lat, throughput_gbps, self.config.io_workers,
                self._inflight_gets, self._inflight_sets,
                avg_pre, avg_xfer, avg_post, avg_exists, p50, p99,
                getattr(storage_metrics, "coalesce_fill_pct", 0.0),
                getattr(storage_metrics, "transfer_utilization_pct", 0.0),
            )
            self._interval_pages_get = 0
            self._interval_pages_set = 0

        # --- Server health summary (rank 0 only, at most every 10s) ---
        srv_now = time.monotonic()
        srv_elapsed = srv_now - self._last_server_stats_time
        if self.local_rank == 0 and srv_elapsed >= 10.0:
            try:
                srv = self.conn.stats()
                totals = srv.get("totals", {})

                srv_evictions = totals.get("evictions", 0)
                evict_per_sec = (srv_evictions - self._last_server_evictions) / srv_elapsed if srv_elapsed > 0 else 0.0
                self._last_server_evictions = srv_evictions
                self._last_server_stats_time = srv_now

                entries = totals.get("entries", 0)
                if entries >= 1_000_000:
                    entries_str = f"{entries / 1_000_000:.1f}M"
                elif entries >= 1_000:
                    entries_str = f"{entries / 1_000:.1f}K"
                else:
                    entries_str = str(entries)

                active_gb = totals.get("active_gb", 0.0)
                max_mem_gb = totals.get("max_memory_gb", 0)

                logger.info(
                    "CAMA server: util=%.1f%%, evictions=%.0f/s, evict_fail=%d, "
                    "inflight=%d, entries=%s, mem=%.1f/%d GB",
                    totals.get("key_utilization_percent", 0.0),
                    evict_per_sec,
                    totals.get("evictions_failed", 0),
                    totals.get("inflight_ops", 0),
                    entries_str,
                    active_gb,
                    max_mem_gb,
                )
            except Exception as exc:
                logger.debug("Server health query failed: %s", exc)

        # Cumulative error counters (never reset — monotonically increasing)
        storage_metrics.get_errors = self._get_errors
        storage_metrics.get_successes = self._get_successes
        storage_metrics.set_errors = self._set_errors
        storage_metrics.set_successes = self._set_successes
        storage_metrics.exists_errors = self._exists_errors
        storage_metrics.exists_timeouts = self._exists_timeouts

        # --- Error counter summary (rank 0 only, at most every 10s) ---
        err_now = time.monotonic()
        err_elapsed = err_now - self._last_error_log_time
        if self.local_rank == 0 and err_elapsed >= 10.0:
            d_get = self._get_errors - self._last_err_get
            d_set = self._set_errors - self._last_err_set
            d_exists = self._exists_errors - self._last_err_exists
            d_exists_to = self._exists_timeouts - self._last_err_exists_to
            if d_get > 0 or d_set > 0 or d_exists > 0:
                logger.warning(
                    "CAMA errors (last %.0fs): get_errors=+%d, set_errors=+%d, "
                    "exists_errors=+%d, exists_timeouts=+%d  "
                    "[cumulative: get_err=%d, get_ok=%d, set_err=%d, set_ok=%d, "
                    "exists_err=%d, exists_to=%d]",
                    err_elapsed,
                    d_get, d_set, d_exists, d_exists_to,
                    self._get_errors, self._get_successes,
                    self._set_errors, self._set_successes,
                    self._exists_errors, self._exists_timeouts,
                )
            self._last_error_log_time = err_now
            self._last_err_get = self._get_errors
            self._last_err_set = self._set_errors
            self._last_err_exists = self._exists_errors
            self._last_err_exists_to = self._exists_timeouts

        # Reconnect metrics
        storage_metrics.reconnect_count = self._reconnect_count

        # Dedup metrics
        storage_metrics.dedup_mode = self.config.dedup_mode
        with self._counter_lock:
            storage_metrics.dedup_auto_disabled = self._dedup_auto_disabled
            storage_metrics.dedup_low_hit_streak = self._dedup_low_hit_streak
            dedup_cost_streak = self._dedup_cost_streak
            dedup_probes_total = self._dedup_probes_total
            dedup_reenables_total = self._dedup_reenables_total
            dedup_batches_since_disable = self._dedup_batches_since_disable
            dedup_probe_hit_streak = self._dedup_probe_hit_streak
        storage_metrics.pool_size = self.config.pool_size

        # Phase timing averages (from local copies swapped above)
        if phase_pre:
            storage_metrics.avg_preprocess_ms = sum(phase_pre) / len(phase_pre)
            storage_metrics.avg_transfer_ms = sum(phase_xfer) / len(phase_xfer) if phase_xfer else 0.0
            storage_metrics.avg_postprocess_ms = sum(phase_post) / len(phase_post) if phase_post else 0.0
        if phase_exists:
            storage_metrics.avg_exists_ms = sum(phase_exists) / len(phase_exists)

        # Report stats to server for Prometheus exposure (fire-and-forget)
        try:
            stats_dict = {
                "get_errors": self._get_errors,
                "get_successes": self._get_successes,
                "set_errors": self._set_errors,
                "set_successes": self._set_successes,
                "exists_errors": self._exists_errors,
                "exists_timeouts": self._exists_timeouts,
                "reconnect_count": self._reconnect_count,
                "io_workers": self.config.io_workers,
                "pool_size": self.config.pool_size,
                "dedup_mode": self.config.dedup_mode,
                "dedup_auto_disabled": storage_metrics.dedup_auto_disabled,
                "dedup_low_hit_streak": storage_metrics.dedup_low_hit_streak,
                "dedup_cost_streak": dedup_cost_streak,
                "dedup_probes_total": dedup_probes_total,
                "dedup_reenables_total": dedup_reenables_total,
                "dedup_batches_since_disable": dedup_batches_since_disable,
                "dedup_probe_hit_streak": dedup_probe_hit_streak,
            }
            # Include SGLang-level metrics (cache hit rate, token usage, etc.)
            stats_dict.update(self._sglang_metrics)
            # Derive prefetch bandwidth (GB/s) from this interval's samples
            if storage_metrics.prefetch_bandwidth:
                stats_dict["prefetch_bandwidth_gbps"] = (
                    sum(storage_metrics.prefetch_bandwidth)
                    / len(storage_metrics.prefetch_bandwidth)
                )
            if storage_metrics.backup_bandwidth:
                stats_dict["backup_bandwidth_gbps"] = (
                    sum(storage_metrics.backup_bandwidth)
                    / len(storage_metrics.backup_bandwidth)
                )
            stats_dict["io_max_latency_ms"] = max_lat
            if hasattr(storage_metrics, "avg_io_batch_size"):
                stats_dict["avg_io_batch_size"] = storage_metrics.avg_io_batch_size
                stats_dict["avg_io_latency_ms"] = storage_metrics.avg_io_latency_ms
                stats_dict["io_calls"] = storage_metrics.io_calls
            stats_dict["total_pages_set"] = self._total_pages_set
            stats_dict["total_pages_get"] = self._total_pages_get
            if self._bytes_per_page > 0:
                stats_dict["model_page_bytes"] = self._bytes_per_page
            if self._codec is not None:
                stats_dict["codec"] = self._codec.name
                stats_dict["codec_lossy"] = self._codec.is_lossy

            # Saturation metrics
            stats_dict["coalesce_fill_pct"] = getattr(storage_metrics, "coalesce_fill_pct", 0.0)
            stats_dict["transfer_utilization_pct"] = getattr(storage_metrics, "transfer_utilization_pct", 0.0)

            # C++ transport timing (control roundtrip vs RDMA Read split)
            if hasattr(self.conn, 'get_transport_stats'):
                try:
                    transport_stats = self.conn.get_transport_stats()
                    stats_dict["avg_roundtrip_us"] = transport_stats.get("avg_roundtrip_us", 0.0)
                    stats_dict["avg_rdma_read_us"] = transport_stats.get("avg_rdma_read_us", 0.0)
                    stats_dict["roundtrip_count"] = transport_stats.get("roundtrip_count", 0)
                    stats_dict["rdma_read_count"] = transport_stats.get("rdma_read_count", 0)
                    # GET sub-phase timing
                    for k in ("get_avg_ctrl_ms", "get_avg_meta_ms", "get_avg_read_ms",
                              "get_avg_ack_ms", "get_batch_count", "get_avg_bytes"):
                        stats_dict[k] = transport_stats.get(k, 0.0)
                    # SET sub-phase timing
                    for k in ("set_avg_serialize_ms", "set_avg_send_ms",
                              "set_batch_count", "set_avg_bytes"):
                        stats_dict[k] = transport_stats.get(k, 0.0)
                    # C++ batch read
                    stats_dict["avg_batch_read_us"] = transport_stats.get("avg_batch_read_us", 0.0)
                    stats_dict["batch_read_count"] = transport_stats.get("batch_read_count", 0)
                    # Sub-phase logging
                    if transport_stats.get("get_batch_count", 0) > 0:
                        logger.info(
                            "  GET sub-phases: ctrl=%.1fms meta=%.1fms read=%.1fms ack=%.1fms "
                            "(avg %.0f keys, %.1f MB/batch)",
                            transport_stats.get("get_avg_ctrl_ms", 0),
                            transport_stats.get("get_avg_meta_ms", 0),
                            transport_stats.get("get_avg_read_ms", 0),
                            transport_stats.get("get_avg_ack_ms", 0),
                            transport_stats.get("get_avg_keys", 0),
                            transport_stats.get("get_avg_bytes", 0) / (1024 * 1024),
                        )
                    if transport_stats.get("set_batch_count", 0) > 0:
                        logger.info(
                            "  SET sub-phases: serialize=%.1fms send=%.1fms "
                            "(avg %.0f keys, %.1f MB/batch)",
                            transport_stats.get("set_avg_serialize_ms", 0),
                            transport_stats.get("set_avg_send_ms", 0),
                            transport_stats.get("set_avg_keys", 0),
                            transport_stats.get("set_avg_bytes", 0) / (1024 * 1024),
                        )
                except Exception:
                    pass

            # In-flight operation gauges
            stats_dict["inflight_gets"] = self._inflight_gets
            stats_dict["inflight_sets"] = self._inflight_sets

            # Phase timing averages (always send; 0.0 when idle to avoid stale values)
            stats_dict["avg_preprocess_ms"] = getattr(storage_metrics, "avg_preprocess_ms", 0.0)
            stats_dict["avg_transfer_ms"] = getattr(storage_metrics, "avg_transfer_ms", 0.0)
            stats_dict["avg_postprocess_ms"] = getattr(storage_metrics, "avg_postprocess_ms", 0.0)
            stats_dict["avg_exists_ms"] = getattr(storage_metrics, "avg_exists_ms", 0.0)

            # Latency histogram (from snapshot swapped under _telemetry_lock above)
            if hist_total > 0:
                cumulative = []
                running = 0
                for c in hist_counts:
                    running += c
                    cumulative.append(running)
                stats_dict["io_latency_hist_buckets"] = list(self._LATENCY_BUCKETS_MS)
                stats_dict["io_latency_hist_counts"] = cumulative
                stats_dict["io_latency_hist_sum"] = hist_sum
                stats_dict["io_latency_hist_total"] = hist_total

            # Warmup phase (from external warmup state machine)
            if self._warmup_phase_ref is not None:
                try:
                    cold = bool(self._warmup_phase_ref())
                    stats_dict["warmup_phase"] = "cold" if cold else "steady"
                except Exception:
                    pass

            fut = self._stats_executor.submit(self.conn.report_stats, stats_dict)
            try:
                fut.result(timeout=3.0)
            except TimeoutError:
                logger.warning("report_stats timed out after 3s, skipping")
            except Exception as e:
                logger.debug("report_stats failed: %s", e)
        except Exception as e:
            logger.debug("report_stats failed: %s", e)

        return storage_metrics
