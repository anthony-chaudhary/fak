"""Tier 1 — Debug logging output verification.

Injects a fake _l3_rdma module so rdma_client.py can be imported
on any platform, then verifies debug timing logs.
"""

import ctypes
import importlib
import logging
import struct
import sys
import types
from unittest.mock import MagicMock

import pytest

from l3_client.protocol import (
    _pack_header,
    OP_RDMA_READ_READY,
    RESP_VALUE,
    FLAG_NONE,
)
from l3_client.sgl import SGL


def _make_rdma_read_ready_raw(rkey, remote_addr, length, req_id=1):
    resp_body = b"\x01" + struct.pack("<I", rkey) + struct.pack("<Q", remote_addr) + struct.pack("<I", length)
    hdr = _pack_header(OP_RDMA_READ_READY, FLAG_NONE, req_id, resp_body)
    return hdr + resp_body


@pytest.fixture
def rdma_env():
    """Inject fake _l3_rdma, force-reimport rdma_client."""
    mock_transport = MagicMock()
    mock_transport.get_stats.return_value = {
        "roundtrip_count": 0, "rdma_read_count": 0,
        "avg_roundtrip_us": 0.0, "avg_rdma_read_us": 0.0,
    }
    mock_transport.reg_mr.return_value = (100, 200)

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

    yield rdma_mod.RDMAClient, mock_transport

    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


class TestDebugTimingLogs:
    def test_get_rdma_read_ready_logs_control_and_data(self, rdma_env, caplog):
        """get() with debug=True and RDMA_READ_READY logs control= and data= timing."""
        RDMAClient, mock_transport = rdma_env

        mock_transport.roundtrip.return_value = _make_rdma_read_ready_raw(
            rkey=42, remote_addr=0x1000, length=1024,
        )
        mock_transport.rdma_read.return_value = b"\x00" * 1024

        client = RDMAClient(
            "127.0.0.1", port=18001,
            handshake=False, debug=True,
        )

        buf = ctypes.create_string_buffer(1024)
        sgl = SGL(ptr=ctypes.addressof(buf), size=1024)

        with caplog.at_level(logging.DEBUG, logger="l3_client.rdma_client"):
            rc = client.get("testkey", sgl)

        assert rc == 0
        timing_logs = [r for r in caplog.records if "control=" in r.message and "data=" in r.message]
        assert len(timing_logs) >= 1, (
            f"Expected timing log with 'control=' and 'data=', got: "
            f"{[r.message for r in caplog.records]}"
        )
        client.close()

    def test_get_inline_value_no_timing_log(self, rdma_env, caplog):
        """get() with inline value response should NOT log data= timing."""
        RDMAClient, mock_transport = rdma_env

        value = b"smallvalue"
        resp_body = b"\x01" + struct.pack("<I", len(value)) + value
        hdr = _pack_header(RESP_VALUE, FLAG_NONE, 1, resp_body)
        mock_transport.roundtrip.return_value = hdr + resp_body

        client = RDMAClient(
            "127.0.0.1", port=18001,
            handshake=False, debug=True,
        )

        buf = ctypes.create_string_buffer(1024)
        sgl = SGL(ptr=ctypes.addressof(buf), size=1024)

        with caplog.at_level(logging.DEBUG, logger="l3_client.rdma_client"):
            rc = client.get("testkey", sgl)

        assert rc == 0
        timing_logs = [r for r in caplog.records if "data=" in r.message]
        assert len(timing_logs) == 0
        client.close()
