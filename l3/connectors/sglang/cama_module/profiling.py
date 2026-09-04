"""Conditional Pyroscope + NVTX profiling for CAMA storage.

When ``SGLANG_CAMA_PROFILING_ENABLED=1`` is set, this module configures the
Pyroscope continuous profiler and exposes real ``tag_wrapper`` / ``nvtx_range``
helpers.  When disabled (the default), both exports are zero-cost no-ops so
that ``pyroscope`` is never even imported.

Originally modelled on ``aibrix_kvcache/profiling.py``.
"""

import asyncio
import functools
import hashlib
import logging
from contextlib import contextmanager

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NVTX_COLORS = ["green", "blue", "yellow", "purple", "rapids", "red"]


@functools.lru_cache
def _nvtx_get_color(name: str):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


@contextmanager
def _nvtx_range_context(msg: str, domain: str):
    nvtx.push_range(message=msg, domain=domain, color=_nvtx_get_color(msg))
    try:
        yield
    finally:
        nvtx.pop_range()


def _env_get(attr, default):
    field = getattr(envs, attr, None)
    return field.get() if field is not None else default


if _env_get("SGLANG_CAMA_PROFILING_ENABLED", False):
    # ================ CPU profiling ================
    try:
        import pyroscope

        _prof_service = _env_get("SGLANG_CAMA_PROFILING_SERVICE_NAME", "cama-connector")
        _prof_server = _env_get("SGLANG_CAMA_PROFILING_SERVER_ADDRESS", "http://0.0.0.0:4040")
        pyroscope.configure(
            application_name=_prof_service,
            server_address=_prof_server,
            sample_rate=100,
            oncpu=True,
            # gil_only=False captures native C/C++ frames (pybind11, RDMA, PrisKV)
            # that execute outside the GIL.
            gil_only=False,
            tags={"backend": "cama"},
        )
        tag_wrapper = pyroscope.tag_wrapper
        logger.info(
            "Pyroscope profiling enabled for CAMA (service=%s, server=%s)",
            _prof_service,
            _prof_server,
        )
    except ImportError:
        logger.warning(
            "SGLANG_CAMA_PROFILING_ENABLED=1 but pyroscope is not installed; "
            "profiling will be a no-op."
        )

        @contextmanager
        def tag_wrapper(tags):
            yield

    # ================ NVTX profiling ================
    try:
        import nvtx

        def nvtx_range(msg: str, domain: str):
            """Decorator for NVTX profiling. Supports sync and async functions."""

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
