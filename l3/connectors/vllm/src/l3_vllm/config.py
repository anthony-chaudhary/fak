"""Configuration parsing for l3-vllm-connector.

Reads from ``vllm_config.kv_transfer_config.kv_connector_extra_config`` (a
dict). All keys have safe defaults so a minimal config like
``{"remote_addr":"localhost","remote_port":18001}`` works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class L3ConnectorConfig:
    # Server connection
    remote_addr: str = "localhost"
    remote_port: int = 18001
    password: str | None = None

    # Transport
    use_rdma: bool = True              # falls back to TCP if no RDMA devices
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 5.0

    # Key namespace — see plan section "Disagg-prefill forward compatibility".
    engine_namespace: str = "vllm_l3"

    # Async / batching
    async_load_min_tokens: int = 32    # below this, do blocking inline mget
    signal_per_layer: bool = False     # if True, signal each layer event as
                                       # its data arrives (requires per-layer
                                       # mget; default fires all together)
    save_chunk_size: int = 256         # max keys per mset call

    # Scheduler-side
    bloom_capacity: int = 10_000_000   # 10M block hashes at ~1% FP
    bloom_fpr: float = 0.01
    lru_capacity: int = 100_000        # confirmed-present block hashes
    bg_sweeper_interval_s: float = 0.05

    # Circuit breaker
    cb_failure_threshold: int = 10
    cb_overload_rate_per_sec: float = 3.0
    cb_probe_interval_s: float = 5.0
    cb_close_after_successes: int = 100

    # Block / MLA detection
    is_mla: bool | None = None         # None = auto-detect from model config

    # Misc
    enable_codec: bool = False         # vendored codec for compression
    codec_name: str = "none"

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any] | None) -> "CamaConnectorConfig":
        if not extra:
            extra = {}
        kwargs: dict[str, Any] = {}
        for f in cls.__dataclass_fields__.values():
            if f.name in extra:
                kwargs[f.name] = extra[f.name]
        unknown = set(extra) - set(cls.__dataclass_fields__)
        if unknown:
            logger.warning(
                "l3-vllm-connector: ignoring unknown extra_config keys: %s",
                sorted(unknown),
            )
        return cls(**kwargs)


# Backward compatibility alias
CamaConnectorConfig = L3ConnectorConfig
