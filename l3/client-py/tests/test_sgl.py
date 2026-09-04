"""Tier 0 — SGL ctypes wrapper tests.

Pure Python, runs anywhere.
"""

import ctypes

from l3_client.sgl import SGL


class TestSGLToBytes:
    def test_reads_correct_data(self):
        data = b"hello world!"
        buf = ctypes.create_string_buffer(data, len(data))
        sgl = SGL(ptr=ctypes.addressof(buf), size=len(data))
        assert sgl.to_bytes() == data

    def test_reads_binary_data(self):
        data = bytes(range(256))
        buf = ctypes.create_string_buffer(data, len(data))
        sgl = SGL(ptr=ctypes.addressof(buf), size=len(data))
        assert sgl.to_bytes() == data


class TestSGLFromBytes:
    def test_writes_into_buffer(self):
        buf = ctypes.create_string_buffer(16)
        sgl = SGL(ptr=ctypes.addressof(buf), size=16)
        sgl.from_bytes(b"0123456789abcdef")
        assert buf.raw == b"0123456789abcdef"

    def test_truncates_to_size(self):
        buf = ctypes.create_string_buffer(4)
        sgl = SGL(ptr=ctypes.addressof(buf), size=4)
        sgl.from_bytes(b"longer than four bytes")
        assert buf.raw == b"long"


class TestSGLRoundtrip:
    def test_from_bytes_then_to_bytes(self):
        data = b"roundtrip data \x00\xff"
        buf = ctypes.create_string_buffer(len(data))
        sgl = SGL(ptr=ctypes.addressof(buf), size=len(data))
        sgl.from_bytes(data)
        assert sgl.to_bytes() == data


class TestSGLRegHandle:
    def test_default_is_1(self):
        buf = ctypes.create_string_buffer(8)
        sgl = SGL(ptr=ctypes.addressof(buf), size=8)
        assert sgl.reg_handle == 1

    def test_custom_value_preserved(self):
        buf = ctypes.create_string_buffer(8)
        sgl = SGL(ptr=ctypes.addressof(buf), size=8, reg_handle=42)
        assert sgl.reg_handle == 42
