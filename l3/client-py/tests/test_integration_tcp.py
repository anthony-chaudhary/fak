"""Tier 2 — Full round-trip over TCP.

Requires running CAMA server. Set CAMA_TEST_SERVER=addr:port.
"""

import ctypes
import os
import uuid

import pytest

from l3_client.client import CamaClient
from l3_client.sgl import SGL

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAMA_TEST_SERVER"),
    reason="Set CAMA_TEST_SERVER=addr:port to run TCP integration tests",
)


@pytest.fixture(scope="module")
def client():
    raw = os.environ["CAMA_TEST_SERVER"]
    addr, port_s = raw.rsplit(":", 1)
    c = CamaClient(addr, port=int(port_s))
    yield c
    c.close()


def _unique_key():
    return f"test_{uuid.uuid4().hex[:12]}"


class TestTCPStringOps:
    def test_setstr_getstr_roundtrip(self, client):
        key = _unique_key()
        client.setstr(key, "hello world")
        val = client.getstr(key)
        assert val == "hello world"

    def test_exists_found(self, client):
        key = _unique_key()
        client.setstr(key, "v")
        assert client.exists(key) == 1

    def test_exists_not_found(self, client):
        key = _unique_key()
        assert client.exists(key) == 0

    def test_delete_removes_key(self, client):
        key = _unique_key()
        client.setstr(key, "to_delete")
        assert client.exists(key) == 1
        client.delete(key)
        assert client.exists(key) == 0


class TestTCPSglOps:
    def test_set_get_1kb(self, client):
        key = _unique_key()
        data = os.urandom(1024)
        buf = ctypes.create_string_buffer(data, 1024)
        ptr = ctypes.addressof(buf)
        sgl = SGL(ptr=ptr, size=1024)

        client.set(key, sgl)

        # Read back into fresh buffer
        rbuf = ctypes.create_string_buffer(1024)
        rsgl = SGL(ptr=ctypes.addressof(rbuf), size=1024)
        rc = client.get(key, rsgl)
        assert rc == 0
        assert rbuf.raw == data


class TestTCPServerOps:
    def test_info_returns_dict(self, client):
        info = client.info()
        assert isinstance(info, dict)

    def test_flush(self, client):
        key = _unique_key()
        client.setstr(key, "will_be_flushed")
        client.flush()
        assert client.exists(key) == 0
