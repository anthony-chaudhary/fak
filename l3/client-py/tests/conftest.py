"""Shared fixtures, markers, and skip conditions for the CAMA test suite.

Tier definitions:
  Tier 0 — Pure Python, no C++ extension, no server.
  Tier 1 — Requires compiled _l3_rdma.so but no server.
  Tier 2 — Requires running CAMA server over TCP (CAMA_TEST_SERVER=addr:port).
  Tier 3 — Requires RDMA server on DGX (CAMA_TEST_RDMA_SERVER=addr:port).
"""

import ctypes
import os

import pytest


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _can_import_rdma_ext():
    try:
        import l3_client._l3_rdma  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _rdma_hw_available():
    try:
        from l3_client._l3_rdma import is_available
        return is_available()
    except Exception:
        return False


requires_rdma_ext = pytest.mark.skipif(
    not _can_import_rdma_ext(),
    reason="_l3_rdma extension not available (Linux + RDMA headers required)",
)

requires_rdma_hw = pytest.mark.skipif(
    not _rdma_hw_available(),
    reason="No RDMA hardware detected (is_available() returned False)",
)

requires_server = pytest.mark.skipif(
    not os.environ.get("CAMA_TEST_SERVER"),
    reason="Set CAMA_TEST_SERVER=addr:port to run TCP integration tests",
)

requires_rdma_server = pytest.mark.skipif(
    not os.environ.get("CAMA_TEST_RDMA_SERVER"),
    reason="Set CAMA_TEST_RDMA_SERVER=addr:port to run RDMA integration tests",
)


# ---------------------------------------------------------------------------
# Server address fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tcp_server():
    """Parse CAMA_TEST_SERVER into (addr, port)."""
    raw = os.environ.get("CAMA_TEST_SERVER", "")
    if not raw:
        pytest.skip("CAMA_TEST_SERVER not set")
    addr, port_s = raw.rsplit(":", 1)
    return addr, int(port_s)


@pytest.fixture(scope="session")
def rdma_server():
    """Parse CAMA_TEST_RDMA_SERVER into (addr, port)."""
    raw = os.environ.get("CAMA_TEST_RDMA_SERVER", "")
    if not raw:
        pytest.skip("CAMA_TEST_RDMA_SERVER not set")
    addr, port_s = raw.rsplit(":", 1)
    return addr, int(port_s)


# ---------------------------------------------------------------------------
# Buffer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sgl_buffer():
    """1KB ctypes buffer + SGL wrapper."""
    from l3_client.sgl import SGL
    buf = ctypes.create_string_buffer(1024)
    ptr = ctypes.addressof(buf)
    return buf, SGL(ptr=ptr, size=1024)


@pytest.fixture
def large_sgl_buffer():
    """5MB ctypes buffer + SGL wrapper."""
    from l3_client.sgl import SGL
    size = 5 * 1024 * 1024
    buf = ctypes.create_string_buffer(size)
    ptr = ctypes.addressof(buf)
    return buf, SGL(ptr=ptr, size=size)
