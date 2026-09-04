"""Conditional Pyroscope + NVTX profiling for the CAMA vLLM connector.

When ``CAMA_VLLM_PROFILING_ENABLED=1`` is set, this module configures the
Pyroscope continuous profiler and exposes real ``tag_wrapper`` / ``nvtx_range``
helpers.  When disabled (the default), both exports are zero-cost no-ops so
that ``pyroscope`` is never even imported.

Originally modelled on ``aibrix_kvcache/profiling.py``; de-SGLanged for vLLM.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_NVTX_COLORS = ["green", "blue", "yellow", "purple", "rapids", "red"]


@functools.lru_cache
def _nvtx_get_color(name: str):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _env_get(name: str, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


if _env_get("CAMA_VLLM_PROFILING_ENABLED", False):
    try:
        import pyroscope  # type: ignore

        _service = os.environ.get("CAMA_VLLM_PROFILING_SERVICE_NAME", "l3-vllm-connector")
        _server = os.environ.get("CAMA_VLLM_PROFILING_SERVER_ADDRESS", "http://0.0.0.0:4040")
        pyroscope.configure(
            application_name=_service,
            server_address=_server,
            sample_rate=100,
            oncpu=True,
            gil_only=False,
            tags={"backend": "cama", "frontend": "vllm"},
        )
        tag_wrapper = pyroscope.tag_wrapper
        logger.info(
            "Pyroscope profiling enabled for cama-vllm (service=%s, server=%s)",
            _service,
            _server,
        )
    except ImportError:
        logger.warning(
            "CAMA_VLLM_PROFILING_ENABLED=1 but pyroscope is not installed; "
            "profiling will be a no-op."
        )

        @contextmanager
        def tag_wrapper(tags):
            yield

    try:
        import nvtx  # type: ignore

        @contextmanager
        def _nvtx_range_context(msg: str, domain: str):
            nvtx.push_range(message=msg, domain=domain, color=_nvtx_get_color(msg))
            try:
                yield
            finally:
                nvtx.pop_range()

        def nvtx_range(msg: str, domain: str):
            def decorator(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with _nvtx_range_context(msg, domain):
                        return await func(*args, **kwargs)

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with _nvtx_range_context(msg, domain):
                        return func(*args, **kwargs)

                return (
                    async_wrapper
                    if asyncio.iscoroutinefunction(func)
                    else sync_wrapper
                )

            return decorator

    except ImportError:

        def nvtx_range(msg: str, domain: str):
            def decorator(func):
                return func

            return decorator

else:

    @contextmanager
    def tag_wrapper(tags):
        yield

    def nvtx_range(msg: str, domain: str):
        def decorator(func):
            return func

        return decorator
