"""Tier 0 — Wire protocol encode/decode tests.

Validates every encode/decode path in l3_client/protocol.py.
Pure Python, runs anywhere.
"""

import struct
from unittest.mock import MagicMock

import pytest

from l3_client.protocol import (
    HEADER_SIZE,
    MAGIC,
    VERSION,
    FLAG_NONE,
    FLAG_WITH_TTL,
    OP_GET,
    OP_SET,
    OP_DELETE,
    OP_TEST,
    OP_LEASE,
    OP_PIN,
    OP_UNPIN,
    OP_MGET,
    OP_MSET,
    OP_MTEST,
    OP_MDEL,
    OP_INFO,
    OP_KEYS,
    OP_HANDSHAKE,
    OP_FLUSH,
    OP_STATS,
    OP_REPORT_STATS,
    OP_MAINTENANCE,
    OP_RDMA_READ_READY,
    OP_READ_ACK,
    RESP_OK,
    RESP_ERROR,
    RESP_VALUE,
    RESP_MULTI_VALUE,
    Header,
    Message,
    _pack_header,
    encode_header,
    encode_key_body,
    encode_kv_body,
    encode_mget_body,
    encode_mset_body,
    encode_keys_body,
    decode_value_response,
    decode_multi_value_response,
    decode_keys_response,
    decode_rdma_read_ready,
    encode_read_ack,
    read_message_from_bytes,
    write_message,
)


# ======================================================================
# _pack_header
# ======================================================================

class TestPackHeader:
    def test_length_is_13(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 1, b"")
        assert len(hdr) == HEADER_SIZE == 13

    def test_magic_bytes(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 1, b"")
        assert hdr[0] == 0xBE
        assert hdr[1] == 0xEF

    def test_version_byte(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 1, b"")
        assert hdr[2] == VERSION

    def test_opcode(self):
        hdr = _pack_header(OP_SET, FLAG_NONE, 1, b"")
        assert hdr[3] == OP_SET

    def test_flags(self):
        hdr = _pack_header(OP_SET, FLAG_WITH_TTL, 1, b"")
        assert hdr[4] == FLAG_WITH_TTL

    def test_request_id(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 42, b"")
        req_id = struct.unpack_from("<I", hdr, 5)[0]
        assert req_id == 42

    def test_body_len(self):
        body = b"hello"
        hdr = _pack_header(OP_GET, FLAG_NONE, 1, body)
        body_len = struct.unpack_from("<I", hdr, 9)[0]
        assert body_len == len(body)

    def test_body_len_zero(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 0, b"")
        body_len = struct.unpack_from("<I", hdr, 9)[0]
        assert body_len == 0

    def test_large_request_id(self):
        hdr = _pack_header(OP_GET, FLAG_NONE, 0xFFFFFFFF, b"")
        req_id = struct.unpack_from("<I", hdr, 5)[0]
        assert req_id == 0xFFFFFFFF


# ======================================================================
# encode_header (struct-based alternative)
# ======================================================================

class TestEncodeHeader:
    # NOTE: encode_header() has a pre-existing bug — uses "<2s" format which
    # expects bytes, but passes int 0xEFBE. The function is unused in the
    # codebase (_pack_header is used instead). These tests document the bug.

    @pytest.mark.xfail(reason="encode_header has struct format bug: '<2s' expects bytes, gets int")
    def test_length_is_13(self):
        hdr = encode_header(OP_GET, FLAG_NONE, 1, 0)
        assert len(hdr) == HEADER_SIZE

    @pytest.mark.xfail(reason="encode_header has struct format bug: '<2s' expects bytes, gets int")
    def test_magic_and_version(self):
        hdr = encode_header(OP_GET, FLAG_NONE, 1, 0)
        assert hdr[0] == 0xBE
        assert hdr[1] == 0xEF
        assert hdr[2] == VERSION

    @pytest.mark.xfail(reason="encode_header has struct format bug: '<2s' expects bytes, gets int")
    def test_opcode_and_flags(self):
        hdr = encode_header(OP_DELETE, FLAG_WITH_TTL, 1, 0)
        assert hdr[3] == OP_DELETE
        assert hdr[4] == FLAG_WITH_TTL


# ======================================================================
# encode_key_body
# ======================================================================

class TestEncodeKeyBody:
    def test_format(self):
        body = encode_key_body(b"mykey")
        key_len = struct.unpack_from("<H", body, 0)[0]
        assert key_len == 5
        assert body[2:] == b"mykey"

    def test_empty_key(self):
        body = encode_key_body(b"")
        key_len = struct.unpack_from("<H", body, 0)[0]
        assert key_len == 0
        assert len(body) == 2

    def test_binary_key(self):
        key = bytes(range(256))
        body = encode_key_body(key)
        key_len = struct.unpack_from("<H", body, 0)[0]
        assert key_len == 256
        assert body[2:] == key


# ======================================================================
# encode_kv_body
# ======================================================================

class TestEncodeKvBody:
    def test_no_ttl(self):
        body, flags = encode_kv_body(b"k", b"val")
        assert flags == FLAG_NONE
        key_len = struct.unpack_from("<H", body, 0)[0]
        assert key_len == 1
        assert body[2:3] == b"k"
        val_len = struct.unpack_from("<I", body, 3)[0]
        assert val_len == 3
        assert body[7:10] == b"val"

    def test_with_ttl(self):
        body, flags = encode_kv_body(b"k", b"v", ttl_ms=5000)
        assert flags == FLAG_WITH_TTL
        # Last 8 bytes should be the TTL
        ttl = struct.unpack_from("<Q", body, len(body) - 8)[0]
        assert ttl == 5000

    def test_zero_ttl_no_flag(self):
        body, flags = encode_kv_body(b"k", b"v", ttl_ms=0)
        assert flags == FLAG_NONE

    def test_empty_value(self):
        body, flags = encode_kv_body(b"k", b"")
        val_len = struct.unpack_from("<I", body, 3)[0]
        assert val_len == 0

    def test_large_value(self):
        big_val = b"X" * 100_000
        body, flags = encode_kv_body(b"k", big_val)
        val_len = struct.unpack_from("<I", body, 3)[0]
        assert val_len == 100_000


# ======================================================================
# decode_value_response
# ======================================================================

class TestDecodeValueResponse:
    def test_found(self):
        # found=1, valueLen=3, "abc"
        body = b"\x01" + struct.pack("<I", 3) + b"abc"
        val, found = decode_value_response(body)
        assert found is True
        assert val == b"abc"

    def test_not_found(self):
        body = b"\x00"
        val, found = decode_value_response(body)
        assert found is False
        assert val is None

    def test_empty_body(self):
        val, found = decode_value_response(b"")
        assert found is False
        assert val is None

    def test_empty_value(self):
        body = b"\x01" + struct.pack("<I", 0)
        val, found = decode_value_response(body)
        assert found is True
        assert val == b""

    def test_large_value(self):
        data = b"X" * 10_000
        body = b"\x01" + struct.pack("<I", len(data)) + data
        val, found = decode_value_response(body)
        assert found is True
        assert val == data


# ======================================================================
# encode_mget_body / encode_mset_body
# ======================================================================

class TestMultiEncode:
    def test_mget_body(self):
        body = encode_mget_body([b"a", b"bb", b"ccc"])
        count = struct.unpack_from("<I", body, 0)[0]
        assert count == 3

    def test_mget_empty(self):
        body = encode_mget_body([])
        count = struct.unpack_from("<I", body, 0)[0]
        assert count == 0

    def test_mset_body(self):
        body = encode_mset_body([b"k1", b"k2"], [b"v1", b"v2"])
        count = struct.unpack_from("<I", body, 0)[0]
        assert count == 2

    def test_mset_roundtrip_structure(self):
        keys = [b"alpha", b"beta"]
        vals = [b"111", b"22"]
        body = encode_mset_body(keys, vals)
        count = struct.unpack_from("<I", body, 0)[0]
        assert count == 2
        off = 4
        for k, v in zip(keys, vals):
            kl = struct.unpack_from("<H", body, off)[0]
            off += 2
            assert body[off:off + kl] == k
            off += kl
            vl = struct.unpack_from("<I", body, off)[0]
            off += 4
            assert body[off:off + vl] == v
            off += vl


# ======================================================================
# decode_multi_value_response
# ======================================================================

class TestDecodeMultiValueResponse:
    def _make_multi_body(self, entries):
        """Build a multi-value response body from list of (found, value_or_None)."""
        parts = [struct.pack("<I", len(entries))]
        for found, val in entries:
            if found:
                parts.append(b"\x01")
                parts.append(struct.pack("<I", len(val)))
                parts.append(val)
            else:
                parts.append(b"\x00")
        return b"".join(parts)

    def test_all_found(self):
        body = self._make_multi_body([(True, b"a"), (True, b"bb")])
        vals, founds = decode_multi_value_response(body)
        assert vals == [b"a", b"bb"]
        assert founds == [True, True]

    def test_all_missing(self):
        body = self._make_multi_body([(False, None), (False, None)])
        vals, founds = decode_multi_value_response(body)
        assert vals == [None, None]
        assert founds == [False, False]

    def test_mixed(self):
        body = self._make_multi_body([(True, b"x"), (False, None), (True, b"zz")])
        vals, founds = decode_multi_value_response(body)
        assert vals == [b"x", None, b"zz"]
        assert founds == [True, False, True]

    def test_empty(self):
        body = struct.pack("<I", 0)
        vals, founds = decode_multi_value_response(body)
        assert vals == []
        assert founds == []


# ======================================================================
# encode_keys_body / decode_keys_response
# ======================================================================

class TestKeysEncoding:
    def test_encode(self):
        body = encode_keys_body(b".*")
        kl = struct.unpack_from("<H", body, 0)[0]
        assert kl == 2
        assert body[2:] == b".*"

    def test_decode_empty(self):
        assert decode_keys_response(b"") == []
        assert decode_keys_response(b"\x00\x00\x00") == []  # < 4 bytes

    def test_decode_zero_count(self):
        body = struct.pack("<I", 0)
        assert decode_keys_response(body) == []

    def test_decode_multiple(self):
        keys = [b"key1", b"key2", b"key3"]
        parts = [struct.pack("<I", len(keys))]
        for k in keys:
            parts.append(struct.pack("<H", len(k)) + k)
        body = b"".join(parts)
        result = decode_keys_response(body)
        assert result == keys


# ======================================================================
# decode_rdma_read_ready
# ======================================================================

class TestDecodeRdmaReadReady:
    def test_found(self):
        # found=1, rkey(4), remote_addr(8), length(4)
        body = b"\x01" + struct.pack("<I", 42) + struct.pack("<Q", 0xDEADBEEF) + struct.pack("<I", 5_000_000)
        rkey, addr, length = decode_rdma_read_ready(body)
        assert rkey == 42
        assert addr == 0xDEADBEEF
        assert length == 5_000_000

    def test_not_found_zero(self):
        body = b"\x00"
        rkey, addr, length = decode_rdma_read_ready(body)
        assert rkey == 0
        assert addr == 0
        assert length == 0

    def test_not_found_empty(self):
        rkey, addr, length = decode_rdma_read_ready(b"")
        assert rkey == 0
        assert addr == 0
        assert length == 0

    def test_large_address(self):
        body = b"\x01" + struct.pack("<I", 1) + struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) + struct.pack("<I", 1)
        rkey, addr, length = decode_rdma_read_ready(body)
        assert addr == 0xFFFFFFFFFFFFFFFF


# ======================================================================
# encode_read_ack
# ======================================================================

class TestEncodeReadAck:
    def test_success(self):
        body = encode_read_ack(0)
        assert body == b"\x00"
        assert len(body) == 1

    def test_failure(self):
        body = encode_read_ack(10)  # REM_ACCESS_ERR
        assert body == b"\x0a"
        assert len(body) == 1

    def test_max_status(self):
        body = encode_read_ack(255)
        assert body == b"\xff"

    def test_opcode_value(self):
        assert OP_READ_ACK == 0x33


# ======================================================================
# read_message_from_bytes
# ======================================================================

class TestReadMessageFromBytes:
    def _make_raw(self, opcode, flags, req_id, body):
        hdr = _pack_header(opcode, flags, req_id, body)
        return hdr + body

    def test_roundtrip(self):
        body = b"payload"
        raw = self._make_raw(RESP_OK, FLAG_NONE, 7, body)
        msg = read_message_from_bytes(raw)
        assert msg.header.opcode == RESP_OK
        assert msg.header.flags == FLAG_NONE
        assert msg.header.request_id == 7
        assert msg.header.body_len == len(body)
        assert msg.body == body

    def test_empty_body(self):
        raw = self._make_raw(RESP_OK, FLAG_NONE, 1, b"")
        msg = read_message_from_bytes(raw)
        assert msg.body == b""
        assert msg.header.body_len == 0

    def test_bad_magic(self):
        raw = bytearray(self._make_raw(RESP_OK, FLAG_NONE, 1, b""))
        raw[0] = 0xFF
        with pytest.raises(ValueError, match="bad magic"):
            read_message_from_bytes(bytes(raw))

    def test_bad_magic_second_byte(self):
        raw = bytearray(self._make_raw(RESP_OK, FLAG_NONE, 1, b""))
        raw[1] = 0x00
        with pytest.raises(ValueError, match="bad magic"):
            read_message_from_bytes(bytes(raw))

    def test_bad_version(self):
        raw = bytearray(self._make_raw(RESP_OK, FLAG_NONE, 1, b""))
        raw[2] = 0xFF
        with pytest.raises(ValueError, match="unsupported version"):
            read_message_from_bytes(bytes(raw))

    def test_truncated_header(self):
        raw = self._make_raw(RESP_OK, FLAG_NONE, 1, b"x")
        with pytest.raises(ValueError, match="buffer too short"):
            read_message_from_bytes(raw[:5])

    def test_preserves_flags(self):
        raw = self._make_raw(RESP_VALUE, FLAG_WITH_TTL, 99, b"data")
        msg = read_message_from_bytes(raw)
        assert msg.header.flags == FLAG_WITH_TTL
        assert msg.header.request_id == 99

    def test_extra_trailing_bytes_ignored(self):
        body = b"abc"
        raw = self._make_raw(RESP_OK, FLAG_NONE, 1, body) + b"trailing"
        msg = read_message_from_bytes(raw)
        assert msg.body == body


# ======================================================================
# Constants sanity
# ======================================================================

class TestConstants:
    def test_header_size(self):
        assert HEADER_SIZE == 13

    def test_magic(self):
        assert MAGIC == b"\xbe\xef"

    def test_opcodes_distinct(self):
        opcodes = [
            OP_GET, OP_SET, OP_DELETE, OP_TEST, OP_LEASE, OP_PIN, OP_UNPIN,
            OP_MGET, OP_MSET, OP_MTEST, OP_MDEL,
            OP_INFO, OP_KEYS, OP_HANDSHAKE, OP_FLUSH, OP_STATS,
            OP_REPORT_STATS, OP_MAINTENANCE,
            OP_RDMA_READ_READY, OP_READ_ACK,
        ]
        assert len(opcodes) == len(set(opcodes)), f"duplicate opcodes: {len(opcodes)} vs {len(set(opcodes))}"

    def test_response_codes_distinct(self):
        codes = [RESP_OK, RESP_ERROR, RESP_VALUE, RESP_MULTI_VALUE]
        assert len(codes) == len(set(codes))


# ======================================================================
# write_message sendall coalescing
# ======================================================================

class TestWriteMessage:
    """Verify write_message uses one sendall for small messages, two for large."""

    def test_empty_body_one_sendall(self):
        sock = MagicMock()
        write_message(sock, OP_GET, b"")
        assert sock.sendall.call_count == 1
        # Single call should contain only the header
        sent = sock.sendall.call_args_list[0][0][0]
        assert len(sent) == HEADER_SIZE

    def test_small_body_one_sendall(self):
        sock = MagicMock()
        body = b"x" * 100
        write_message(sock, OP_SET, body)
        assert sock.sendall.call_count == 1
        sent = sock.sendall.call_args_list[0][0][0]
        assert len(sent) == HEADER_SIZE + len(body)

    def test_threshold_body_one_sendall(self):
        sock = MagicMock()
        body = b"x" * 4096  # exactly at threshold
        write_message(sock, OP_SET, body)
        assert sock.sendall.call_count == 1

    def test_large_body_two_sendalls(self):
        sock = MagicMock()
        body = b"x" * 4097  # just above threshold
        write_message(sock, OP_SET, body)
        assert sock.sendall.call_count == 2
        hdr_sent = sock.sendall.call_args_list[0][0][0]
        body_sent = sock.sendall.call_args_list[1][0][0]
        assert len(hdr_sent) == HEADER_SIZE
        assert body_sent == body

    def test_large_body_preserves_content(self):
        sock = MagicMock()
        body = b"Y" * 10000
        write_message(sock, OP_SET, body, flags=FLAG_WITH_TTL, request_id=42)
        hdr_sent = sock.sendall.call_args_list[0][0][0]
        body_sent = sock.sendall.call_args_list[1][0][0]
        # Verify header fields
        assert hdr_sent[3] == OP_SET
        assert hdr_sent[4] == FLAG_WITH_TTL
        req_id = struct.unpack_from("<I", hdr_sent, 5)[0]
        assert req_id == 42
        body_len = struct.unpack_from("<I", hdr_sent, 9)[0]
        assert body_len == 10000
        assert body_sent == body
