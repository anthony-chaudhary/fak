"""Tier 3 — Full round-trip over RDMA.

Requires RDMA server on DGX. Set CAMA_TEST_RDMA_SERVER=addr:port.
"""

import ctypes
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAMA_TEST_RDMA_SERVER"),
    reason="Set CAMA_TEST_RDMA_SERVER=addr:port to run RDMA integration tests",
)


@pytest.fixture(scope="module")
def client():
    from l3_client.rdma_client import RDMAClient
    raw = os.environ["CAMA_TEST_RDMA_SERVER"]
    addr, port_s = raw.rsplit(":", 1)
    c = RDMAClient(addr, port=int(port_s))
    yield c
    c.close()


@pytest.fixture(scope="module")
def registered_buffer():
    """4KB registered buffer for zero-copy tests."""
    from l3_client.rdma_client import RDMAClient
    from l3_client.sgl import SGL

    raw = os.environ["CAMA_TEST_RDMA_SERVER"]
    addr, port_s = raw.rsplit(":", 1)
    c = RDMAClient(addr, port=int(port_s))

    size = 4096
    buf = ctypes.create_string_buffer(size)
    handle = c.reg_memory(ctypes.addressof(buf), size, buf=buf)
    sgl = SGL(ptr=ctypes.addressof(buf), size=size, reg_handle=handle)

    yield c, buf, sgl, handle

    c.dereg_memory(handle)
    c.close()


def _unique_key():
    return f"rdma_test_{uuid.uuid4().hex[:12]}"


class TestRDMAStringOps:
    def test_setstr_getstr_roundtrip(self, client):
        key = _unique_key()
        client.setstr(key, "rdma hello")
        val = client.getstr(key)
        assert val == "rdma hello"


class TestRDMASglOps:
    def test_registered_buffer_roundtrip(self, registered_buffer):
        """SGL roundtrip with registered 4KB buffer, byte-verified."""
        client, buf, sgl, handle = registered_buffer
        from l3_client.sgl import SGL

        key = _unique_key()
        data = os.urandom(4096)

        # Write
        wbuf = ctypes.create_string_buffer(data, 4096)
        wsgl = SGL(ptr=ctypes.addressof(wbuf), size=4096, reg_handle=handle)
        rc = client.set(key, wsgl)
        assert rc == 0

        # Read back
        rbuf = ctypes.create_string_buffer(4096)
        rhandle = client.reg_memory(ctypes.addressof(rbuf), 4096, buf=rbuf)
        rsgl = SGL(ptr=ctypes.addressof(rbuf), size=4096, reg_handle=rhandle)
        rc = client.get(key, rsgl)
        assert rc == 0
        assert rbuf.raw == data
        client.dereg_memory(rhandle)

    def test_large_value_rdma_read(self, client):
        """5MB value triggers OP_RDMA_READ_READY path, verified byte-for-byte."""
        from l3_client.sgl import SGL

        key = _unique_key()
        size = 5 * 1024 * 1024
        data = os.urandom(size)

        # Write via SGL
        wbuf = ctypes.create_string_buffer(data, size)
        wptr = ctypes.addressof(wbuf)
        whandle = client.reg_memory(wptr, size, buf=wbuf)
        wsgl = SGL(ptr=wptr, size=size, reg_handle=whandle)
        rc = client.set(key, wsgl)
        assert rc == 0

        # Read back
        rbuf = ctypes.create_string_buffer(size)
        rptr = ctypes.addressof(rbuf)
        rhandle = client.reg_memory(rptr, size, buf=rbuf)
        rsgl = SGL(ptr=rptr, size=size, reg_handle=rhandle)
        rc = client.get(key, rsgl)
        assert rc == 0
        assert rbuf.raw == data

        client.dereg_memory(whandle)
        client.dereg_memory(rhandle)


class TestRDMAStats:
    def test_stats_populated_after_ops(self, client):
        """Stats show roundtrip_count >= 2 after SET + GET."""
        from l3_client.sgl import SGL

        key = _unique_key()
        buf = ctypes.create_string_buffer(b"statstest", 64)
        sgl = SGL(ptr=ctypes.addressof(buf), size=9)

        client._transport.reset_stats()
        client.set(key, sgl)
        client.get(key, sgl)

        stats = client._transport.get_stats()
        assert stats["roundtrip_count"] >= 2


class TestRDMAMissingKey:
    def test_getstr_missing_returns_none(self, client):
        key = _unique_key()
        assert client.getstr(key) is None

    def test_get_missing_returns_minus_one(self, client):
        from l3_client.sgl import SGL

        key = _unique_key()
        buf = ctypes.create_string_buffer(64)
        sgl = SGL(ptr=ctypes.addressof(buf), size=64)
        rc = client.get(key, sgl)
        assert rc == -1


class TestRDMAUnregisteredFallback:
    def test_unregistered_sgl_uses_memcpy_fallback(self, client):
        """5MB GET with reg_handle=1 (unregistered) uses internal buffer + memcpy."""
        from l3_client.sgl import SGL

        key = _unique_key()
        size = 5 * 1024 * 1024
        data = os.urandom(size)

        # Write with registered buffer
        wbuf = ctypes.create_string_buffer(data, size)
        wptr = ctypes.addressof(wbuf)
        whandle = client.reg_memory(wptr, size, buf=wbuf)
        wsgl = SGL(ptr=wptr, size=size, reg_handle=whandle)
        rc = client.set(key, wsgl)
        assert rc == 0

        # Read with unregistered SGL (reg_handle=1, the TCP dummy)
        rbuf = ctypes.create_string_buffer(size)
        rsgl = SGL(ptr=ctypes.addressof(rbuf), size=size, reg_handle=1)
        rc = client.get(key, rsgl)
        assert rc == 0
        assert rbuf.raw == data

        client.dereg_memory(whandle)
