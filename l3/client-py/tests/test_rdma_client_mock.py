"""Tier 1 — RDMAClient with mocked transport.

Injects a fake _l3_rdma module into sys.modules so rdma_client.py
can be imported on any platform, then tests client logic with MagicMock.
"""

import importlib
import logging
import struct
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from l3_client.protocol import (
    _pack_header,
    RESP_OK,
    RESP_VALUE,
    OP_RDMA_READ_READY,
    FLAG_NONE,
)


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------

def _make_ok_response(req_id=1):
    body = b""
    hdr = _pack_header(RESP_OK, FLAG_NONE, req_id, body)
    return hdr + body


def _make_value_response(value: bytes, req_id=1):
    resp_body = b"\x01" + struct.pack("<I", len(value)) + value
    hdr = _pack_header(RESP_VALUE, FLAG_NONE, req_id, resp_body)
    return hdr + resp_body


def _make_rdma_read_ready_response(rkey, remote_addr, length, req_id=1):
    resp_body = b"\x01" + struct.pack("<I", rkey) + struct.pack("<Q", remote_addr) + struct.pack("<I", length)
    hdr = _pack_header(OP_RDMA_READ_READY, FLAG_NONE, req_id, resp_body)
    return hdr + resp_body


def _make_not_found_response(req_id=1):
    resp_body = b"\x00"
    hdr = _pack_header(RESP_VALUE, FLAG_NONE, req_id, resp_body)
    return hdr + resp_body


# ---------------------------------------------------------------------------
# Fixture: inject fake _l3_rdma and get a fresh RDMAClient class each time
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_transport():
    """MagicMock mimicking RDMATransport."""
    t = MagicMock()
    t.reg_mr.return_value = (100, 200)
    t.get_stats.return_value = {
        "roundtrip_count": 0,
        "rdma_read_count": 0,
        "avg_roundtrip_us": 0.0,
        "avg_rdma_read_us": 0.0,
    }
    return t


@pytest.fixture
def rdma_env(mock_transport):
    """Inject fake _l3_rdma into sys.modules, force-reimport rdma_client.

    Yields (RDMAClient_class, mock_transport).
    Cleans up sys.modules on teardown.
    """
    # Build a fake extension module
    fake_ext = types.ModuleType("l3_client._l3_rdma")
    fake_ext.RDMATransport = MagicMock(return_value=mock_transport)
    fake_ext.GIL_RELEASED = True
    fake_ext.DEFAULT_SEND_BUF_SIZE = 16 * 1024 * 1024
    fake_ext.is_available = lambda: False

    saved = {}
    pop_keys = [
        "l3_client._l3_rdma",
        "l3_client.rdma._constants",
        "l3_client.rdma._batch_ops",
        "l3_client.rdma._client",
        "l3_client.rdma._pool",
        "l3_client.rdma_client",
    ]
    for key in pop_keys:
        saved[key] = sys.modules.pop(key, None)

    sys.modules["l3_client._l3_rdma"] = fake_ext

    # Force reimport of subpackage modules then facade
    import l3_client.rdma._constants
    importlib.reload(l3_client.rdma._constants)
    import l3_client.rdma._batch_ops
    importlib.reload(l3_client.rdma._batch_ops)
    import l3_client.rdma._client
    importlib.reload(l3_client.rdma._client)
    import l3_client.rdma._pool
    importlib.reload(l3_client.rdma._pool)
    import l3_client.rdma_client as rdma_mod
    importlib.reload(rdma_mod)
    RDMAClient = rdma_mod.RDMAClient

    yield RDMAClient, mock_transport

    # Restore
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


@pytest.fixture
def make_client(rdma_env):
    """Factory that creates an RDMAClient with mocked transport."""
    RDMAClient, mock_transport = rdma_env

    def _make(debug=False, env_debug=False):
        env_patch = {"CAMA_DEBUG": "1"} if env_debug else {}
        with patch.dict("os.environ", env_patch, clear=False):
            client = RDMAClient(
                "127.0.0.1", port=18001,
                handshake=False, debug=debug,
            )
        return client
    return _make


# ======================================================================
# Debug mode
# ======================================================================

class TestDebugMode:
    def test_debug_via_constructor(self, make_client):
        client = make_client(debug=True)
        assert client._debug is True
        client.close()

    def test_debug_via_env_var(self, make_client):
        client = make_client(env_debug=True)
        assert client._debug is True
        client.close()

    def test_debug_off_by_default(self, make_client):
        client = make_client()
        assert client._debug is False
        client.close()


# ======================================================================
# Debug logging
# ======================================================================

class TestDebugLogging:
    def test_connect_log(self, rdma_env, caplog):
        """Debug mode logs 'RDMA connect' on init."""
        RDMAClient, _ = rdma_env
        with caplog.at_level(logging.INFO, logger="l3_client.rdma_client"):
            client = RDMAClient(
                "127.0.0.1", port=18001,
                handshake=False, debug=True,
            )
        assert any("RDMA connect" in r.message for r in caplog.records)
        client.close()


# ======================================================================
# Memory registration (mr_map management)
# ======================================================================

class TestMemoryRegistration:
    def test_reg_memory_returns_handle_gte_2(self, make_client, rdma_env):
        client = make_client()
        handle = client.reg_memory(0xDEAD, 4096)
        assert handle >= 2
        client.close()

    def test_dereg_memory_removes_from_map(self, make_client, rdma_env):
        _, mock_transport = rdma_env
        client = make_client()
        handle = client.reg_memory(0xDEAD, 4096)
        assert handle in client._mr_map
        client.dereg_memory(handle)
        assert handle not in client._mr_map
        mock_transport.dereg_mr.assert_called_once()
        client.close()

    def test_buf_ref_stored_as_gc_ref(self, make_client):
        client = make_client()
        sentinel = object()
        handle = client.reg_memory(0xDEAD, 4096, buf=sentinel)
        entry = client._mr_map[handle]
        assert entry.buf_ref is sentinel
        client.close()

    def test_close_deregs_all_outstanding(self, make_client, rdma_env):
        _, mock_transport = rdma_env
        client = make_client()
        client.reg_memory(0x1000, 1024)
        client.reg_memory(0x2000, 2048)
        assert len(client._mr_map) == 2
        client.close()
        assert len(client._mr_map) == 0
        assert mock_transport.dereg_mr.call_count == 2

    def test_dereg_nonexistent_handle_is_noop(self, make_client):
        client = make_client()
        client.dereg_memory(999)  # should not raise
        client.close()


# ======================================================================
# Request ID sequencing
# ======================================================================

class TestRequestIdSequencing:
    def test_monotonic_increment(self, make_client):
        client = make_client()
        ids = [client._next_req_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]
        client.close()

    def test_starts_at_one(self, make_client):
        client = make_client()
        assert client._next_req_id() == 1
        client.close()
