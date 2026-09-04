"""L3 Python client — RDMA-first with TCP fallback (CAMA / L3 KV cache)."""

import logging
import os
import warnings

from l3_client._version import __version__
from l3_client import rc  # noqa: F401 — explicit import for __all__ export
from l3_client.errors import (  # noqa: F401
    CamaNotReadyError,
    CamaOOMError,
    CamaServerOverloadError,
    L3Error,
    L3NotReadyError,
    L3OOMError,
    L3ServerOverloadError,
)
from l3_client.sgl import SGL

_logger = logging.getLogger(__name__)


def _get_rdma_extension():
    """Try importing _l3_rdma, fallback to _cama_rdma."""
    try:
        import l3_client._l3_rdma as ext
        return ext
    except ImportError:
        pass
    try:
        import l3_client._cama_rdma as ext
        return ext
    except ImportError:
        return None


def _get_cxl_extension():
    """Try importing _l3_cxl, fallback to _cama_cxl."""
    try:
        import l3_client._l3_cxl as ext
        return ext
    except ImportError:
        pass
    try:
        import l3_client._cama_cxl as ext
        return ext
    except ImportError:
        return None


def _check_rdma_version():
    """Warn if the compiled C++ extension doesn't match the Python package version."""
    ext = _get_rdma_extension()
    if ext is None:
        return
    ext_version = getattr(ext, "__version__", None)
    if ext_version and ext_version != __version__:
        warnings.warn(
            f"RDMA extension version mismatch: _l3_rdma.so was compiled for "
            f"{ext_version!r} but l3-client is {__version__!r}. "
            f"Rebuild with: pip install --no-build-isolation --force-reinstall -e .",
            stacklevel=2,
        )


def _check_cxl_version():
    """Warn if the compiled CXL extension doesn't match the Python package version."""
    ext = _get_cxl_extension()
    if ext is None:
        return
    ext_version = getattr(ext, "__version__", None)
    if ext_version and ext_version != __version__:
        warnings.warn(
            f"CXL extension version mismatch: _l3_cxl.so was compiled for "
            f"{ext_version!r} but l3-client is {__version__!r}. "
            f"Rebuild with: pip install --no-build-isolation --force-reinstall -e .",
            stacklevel=2,
        )


def _is_cxl_available() -> bool:
    """Check if CXL transport is active (extension built + devices present)."""
    ext = _get_cxl_extension()
    if ext is None:
        return False
    try:
        return ext.is_available()
    except Exception:
        return False


def _is_rdma_available() -> bool:
    """Check if RDMA transport is active (extension built + devices present)."""
    ext = _get_rdma_extension()
    if ext is None:
        return False
    try:
        return ext.is_available()
    except Exception:
        return False


def _create_client_class():
    """Select the best available transport.

    Priority: CXL > RDMA > TCP.

    CXL is opt-in (CAMA_USE_CXL=1). RDMA is the default
    (SGLANG_CAMA_USE_RDMA=1). Falls back to TCP if:
    - The C++ extension is not built (ImportError)
    - No RDMA/CXL devices are available
    - SGLANG_CAMA_USE_RDMA=0

    The selected class is logged/warned so silent fallback is visible.
    Check which transport is active: print(PriskvClient)
    """
    use_cxl = os.environ.get("CAMA_USE_CXL", "0")
    if use_cxl == "1":
        ext = _get_cxl_extension()
        if ext is not None:
            try:
                if ext.is_available():
                    from l3_client.cxl._client import CXLClient

                    _logger.debug(
                        "[l3] transport: CXLClient — CXL direct memory access active."
                    )
                    return CXLClient
                warnings.warn(
                    "[l3] transport: CXL requested (CAMA_USE_CXL=1) but no CXL "
                    "devices found. Falling back to RDMA/TCP.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                pass
        else:
            warnings.warn(
                "[l3] transport: CXL requested (CAMA_USE_CXL=1) but _l3_cxl "
                "extension not built. Falling back to RDMA/TCP.",
                RuntimeWarning,
                stacklevel=2,
            )

    use_rdma = os.environ.get("SGLANG_CAMA_USE_RDMA", "1")
    if use_rdma == "1":
        ext = _get_rdma_extension()
        if ext is not None:
            try:
                if ext.is_available():
                    from l3_client.rdma_client import RDMAClient

                    _logger.debug(
                        "[l3] transport: RDMAClient — RDMA zero-copy active. "
                        "Verify with: python -c \"from l3_client import PriskvClient; print(PriskvClient)\""
                    )
                    return RDMAClient

                warnings.warn(
                    "[l3] transport: TCP fallback (L3Client) — RDMA was requested "
                    "(SGLANG_CAMA_USE_RDMA=1) but no RDMA devices were found "
                    "(ibv_get_device_list returned 0). "
                    "Impact: double-copy through kernel instead of zero-copy DMA for every "
                    "KV page transfer. Check 'ibv_devices' and RDMA driver. "
                    "Disable this warning by setting SGLANG_CAMA_USE_RDMA=0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                pass
        else:
            warnings.warn(
                "[l3] transport: TCP fallback (L3Client) — RDMA was requested "
                "(SGLANG_CAMA_USE_RDMA=1) but the _l3_rdma extension is not built. "
                "Impact: double-copy through kernel instead of zero-copy DMA. "
                "Rebuild: pip install --no-build-isolation --force-reinstall -e \".[rdma]\". "
                "Disable this warning by setting SGLANG_CAMA_USE_RDMA=0.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        _logger.debug("[l3] transport: L3Client (TCP) — SGLANG_CAMA_USE_RDMA=0")

    from l3_client.client import L3Client

    return L3Client


_check_rdma_version()
_check_cxl_version()

from l3_client.client import CamaClient, CamaClientPool, L3Client, L3ClientPool
from l3_client.reconnect import ReconnectConfig

PriskvClient = _create_client_class()


def discover_rdma_endpoints(tcp_addr: str = "127.0.0.1", tcp_port: int = 18000) -> list:
    """Connect via TCP, return server's RDMA endpoints, disconnect.

    Each endpoint is a dict with keys: device, ip, port.
    """
    c = L3Client(tcp_addr, tcp_port, handshake=True, reconnect=False)
    try:
        return c.rdma_endpoints()
    finally:
        c.close()


def create_pool(addr: str, port: int, password: str = "", *, pool_size: int = 8,
                endpoints: list[tuple[str, int]] | None = None,
                reconnect: bool | ReconnectConfig | None = True, **kwargs):
    """Create a connection pool (CXL, RDMA, or TCP depending on availability).

    Returns a CXLClientPool, RDMAClientPool, or L3ClientPool (CamaClientPool) — all are
    API-compatible drop-in replacements for PriskvClient.

    When *endpoints* is provided (list of ``(ip, port)`` tuples), RDMA pool
    connections are striped across multiple server NIC endpoints for
    multi-NIC bandwidth saturation.
    """
    if pool_size <= 1 and not endpoints:
        # No pooling — return a regular single client
        return PriskvClient(addr, port, password, reconnect=reconnect, **kwargs)

    use_cxl = os.environ.get("CAMA_USE_CXL", "0")
    if use_cxl == "1" and _is_cxl_available():
        from l3_client.cxl._pool import CXLClientPool
        _logger.debug("[l3] creating CXL pool (size=%d) to %s:%d", pool_size, addr, port)
        return CXLClientPool(addr, port, password, pool_size=pool_size,
                             reconnect=reconnect, **kwargs)

    if _is_rdma_available():
        from l3_client.rdma_client import RDMAClientPool
        if endpoints and len(endpoints) > 1:
            _logger.debug("[l3] creating RDMA pool (size=%d, %d endpoints) striped",
                          pool_size, len(endpoints))
        else:
            _logger.debug("[l3] creating RDMA pool (size=%d) to %s:%d", pool_size, addr, port)
        return RDMAClientPool(addr, port, password, pool_size=pool_size,
                              endpoints=endpoints, reconnect=reconnect, **kwargs)

    _logger.debug("[l3] creating TCP pool (size=%d) to %s:%d", pool_size, addr, port)
    return L3ClientPool(addr, port, password, pool_size=pool_size, reconnect=reconnect, **kwargs)


__all__ = [
    "L3Client",
    "CamaClient",
    "L3ClientPool",
    "CamaClientPool",
    "PriskvClient",
    "SGL",
    "__version__",
    "rc",
    "ReconnectConfig",
    "discover_rdma_endpoints",
    "create_pool",
    "L3Error",
    "CamaNotReadyError",
    "L3NotReadyError",
    "CamaOOMError",
    "L3OOMError",
    "CamaServerOverloadError",
    "L3ServerOverloadError",
]
