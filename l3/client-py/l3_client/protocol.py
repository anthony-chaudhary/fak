"""CAMA binary wire protocol encoder/decoder.

Matches the Go implementation in internal/transport/protocol/protocol.go.
All integers are little-endian.
"""

import struct
from dataclasses import dataclass

# Magic bytes
MAGIC = b"\xbe\xef"
VERSION = 0x01
HEADER_SIZE = 13  # 2 magic + 1 version + 1 opcode + 1 flags + 4 reqID + 4 bodyLen

# OpCodes
OP_GET = 0x01
OP_SET = 0x02
OP_DELETE = 0x03
OP_TEST = 0x04
OP_LEASE = 0x05
OP_PIN = 0x06
OP_UNPIN = 0x07

OP_MGET = 0x10
OP_MSET = 0x11
OP_MTEST = 0x12
OP_MDEL = 0x13

OP_INFO = 0x20
OP_KEYS = 0x22
OP_HANDSHAKE = 0x23
OP_FLUSH = 0x24
OP_STATS = 0x25
OP_REPORT_STATS = 0x26
OP_MAINTENANCE = 0x27

OP_SNAPSHOT = 0x28
OP_RESTORE = 0x29
OP_CLUSTER = 0x21

OP_RDMA_READ_READY = 0x32
OP_READ_ACK = 0x33  # client→server: RDMA Read completion ack (1-byte WC status)
OP_MGET_RDMA = 0x34  # client→server: batch GET requesting RDMA metadata
OP_MGET_READ_READY = 0x35  # server→client: batch RDMA coordinates
OP_BATCH_READ_ACK = 0x36  # client→server: batch WC statuses

OP_CXL_REGION_MAP = 0x37  # client→server: request CXL region map
OP_CXL_READ_READY = 0x38  # server→client: CXL device offset for GET

# Response codes
RESP_OK = 0xF0
RESP_ERROR = 0xF1
RESP_VALUE = 0xF2
RESP_MULTI_VALUE = 0xF3
RESP_MSET_RESULT = 0xF4  # per-key MSET status bytes (partial failure)
RESP_CXL_REGION_MAP = 0xF5  # server→client: CXL region map response
RESP_OOM = 0xF6  # server memory pressure — SET rejected with diagnostics
RESP_NOT_READY = 0xF7  # server starting — shard not allocated yet

# Flags
FLAG_NONE = 0x00
FLAG_WITH_TTL = 0x01


@dataclass
class Header:
    opcode: int
    flags: int
    request_id: int
    body_len: int


@dataclass
class Message:
    header: Header
    body: bytes


def encode_header(opcode: int, flags: int, request_id: int, body_len: int) -> bytes:
    return struct.pack("<2sBBBI", 0xEFBE, VERSION, opcode, flags, request_id) + struct.pack("<I", body_len)


def _pack_header(opcode: int, flags: int, request_id: int, body: bytes) -> bytes:
    hdr = bytearray(HEADER_SIZE)
    hdr[0] = 0xBE
    hdr[1] = 0xEF
    hdr[2] = VERSION
    hdr[3] = opcode
    hdr[4] = flags
    struct.pack_into("<I", hdr, 5, request_id)
    struct.pack_into("<I", hdr, 9, len(body))
    return bytes(hdr)


def write_message(sock, opcode: int, body: bytes, flags: int = FLAG_NONE, request_id: int = 0) -> None:
    """Write a complete protocol message to a socket."""
    hdr = _pack_header(opcode, flags, request_id, body)
    if body and len(body) > 4096:
        sock.sendall(hdr)
        sock.sendall(body)
    else:
        sock.sendall(hdr + body)


def read_message(sock) -> Message:
    """Read a complete protocol message from a socket."""
    hdr_data = _recv_exact(sock, HEADER_SIZE)
    if hdr_data[0] != 0xBE or hdr_data[1] != 0xEF:
        raise ValueError(f"bad magic: {hdr_data[0]:02x}{hdr_data[1]:02x}")
    if hdr_data[2] != VERSION:
        raise ValueError(f"unsupported version: {hdr_data[2]}")

    opcode = hdr_data[3]
    flags = hdr_data[4]
    request_id = struct.unpack_from("<I", hdr_data, 5)[0]
    body_len = struct.unpack_from("<I", hdr_data, 9)[0]

    body = _recv_exact(sock, body_len) if body_len > 0 else b""

    return Message(
        header=Header(opcode=opcode, flags=flags, request_id=request_id, body_len=body_len),
        body=body,
    )


def _recv_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


def _read_exact(reader, n: int) -> bytes:
    """Read exactly n bytes from a file-like object (.read() interface)."""
    data = reader.read(n)
    if not data:
        raise ConnectionError("connection closed while reading")
    if len(data) < n:
        buf = bytearray(data)
        while len(buf) < n:
            chunk = reader.read(n - len(buf))
            if not chunk:
                raise ConnectionError("connection closed while reading")
            buf.extend(chunk)
        return bytes(buf)
    return data


def read_message_buffered(reader) -> Message:
    """Read a protocol message from a BufferedReader (file-like .read())."""
    hdr_data = _read_exact(reader, HEADER_SIZE)
    if hdr_data[0] != 0xBE or hdr_data[1] != 0xEF:
        raise ValueError(f"bad magic: {hdr_data[0]:02x}{hdr_data[1]:02x}")
    if hdr_data[2] != VERSION:
        raise ValueError(f"unsupported version: {hdr_data[2]}")
    opcode = hdr_data[3]
    flags = hdr_data[4]
    request_id = struct.unpack_from("<I", hdr_data, 5)[0]
    body_len = struct.unpack_from("<I", hdr_data, 9)[0]
    body = _read_exact(reader, body_len) if body_len > 0 else b""
    return Message(
        header=Header(opcode=opcode, flags=flags, request_id=request_id, body_len=body_len),
        body=body,
    )


# --- Body encoding helpers ---

def encode_key_body(key: bytes) -> bytes:
    """keyLen(2) + key"""
    return struct.pack("<H", len(key)) + key


def encode_kv_body(key: bytes, value: bytes, ttl_ms: int = 0) -> tuple[bytes, int]:
    """keyLen(2) + key + valueLen(4) + value [+ ttlMs(8)]
    Returns (body, flags).
    """
    flags = FLAG_NONE
    body = struct.pack("<H", len(key)) + key + struct.pack("<I", len(value)) + value
    if ttl_ms > 0:
        flags = FLAG_WITH_TTL
        body += struct.pack("<Q", ttl_ms)
    return body, flags


def decode_value_response(body: bytes) -> tuple[bytes | None, bool]:
    """Decode a RespValue body. Returns (value, found)."""
    if len(body) < 1 or body[0] == 0:
        return None, False
    val_len = struct.unpack_from("<I", body, 1)[0]
    return body[5 : 5 + val_len], True


def encode_mget_body(keys: list[bytes]) -> bytes:
    """count(4) + [keyLen(2) + key]*"""
    parts = [struct.pack("<I", len(keys))]
    for k in keys:
        parts.append(struct.pack("<H", len(k)) + k)
    return b"".join(parts)


def encode_mset_body(keys: list[bytes], values: list[bytes]) -> bytes:
    """count(4) + [keyLen(2) + key + valueLen(4) + value]*"""
    parts = [struct.pack("<I", len(keys))]
    for k, v in zip(keys, values):
        parts.append(struct.pack("<H", len(k)) + k + struct.pack("<I", len(v)) + v)
    return b"".join(parts)


def decode_multi_value_response(body: bytes) -> tuple[list[bytes | None], list[bool]]:
    """Decode a RespMultiValue body."""
    count = struct.unpack_from("<I", body, 0)[0]
    off = 4
    values: list[bytes | None] = []
    founds: list[bool] = []
    try:
        for _ in range(count):
            found = body[off] != 0
            off += 1
            if found:
                val_len = struct.unpack_from("<I", body, off)[0]
                off += 4
                values.append(body[off : off + val_len])
                off += val_len
                founds.append(True)
            else:
                values.append(None)
                founds.append(False)
    except (IndexError, struct.error) as e:
        raise ValueError(
            f"decode_multi_value_response: body truncated at offset {off}/{len(body)} "
            f"while parsing entry {len(values)+1}/{count}") from e
    if off != len(body):
        import logging
        logging.getLogger(__name__).warning(
            "decode_multi_value_response: %d trailing bytes after parsing %d entries "
            "(body=%d bytes, consumed=%d)", len(body) - off, count, len(body), off)
    return values, founds


def encode_mdel_body(keys: list[bytes]) -> bytes:
    """Encode MDEL body — same wire format as MGET (count + keys)."""
    return encode_mget_body(keys)


def decode_mtest_founds(body: bytes) -> list[bool]:
    """Decode an MTEST response body into a list of booleans.

    MTEST uses the same RespMultiValue encoding as MGET, but only the
    ``found`` flags are meaningful (values are always nil).
    """
    _, founds = decode_multi_value_response(body)
    return founds


def encode_keys_body(pattern: bytes) -> bytes:
    """patternLen(2) + pattern"""
    return struct.pack("<H", len(pattern)) + pattern


def decode_keys_response(body: bytes) -> list[bytes]:
    """Decode KEYS response — same format as MGet keys."""
    if len(body) < 4:
        return []
    count = struct.unpack_from("<I", body, 0)[0]
    off = 4
    keys: list[bytes] = []
    for _ in range(count):
        key_len = struct.unpack_from("<H", body, off)[0]
        off += 2
        keys.append(body[off : off + key_len])
        off += key_len
    return keys


def read_message_from_bytes(data: bytes) -> Message:
    """Deserialize a protocol message from a raw byte buffer.

    Used by RDMA transport where recv delivers raw bytes instead of a socket stream.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"buffer too short for header: {len(data)} < {HEADER_SIZE}")
    if data[0] != 0xBE or data[1] != 0xEF:
        raise ValueError(f"bad magic: {data[0]:02x}{data[1]:02x}")
    if data[2] != VERSION:
        raise ValueError(f"unsupported version: {data[2]}")

    opcode = data[3]
    flags = data[4]
    request_id = struct.unpack_from("<I", data, 5)[0]
    body_len = struct.unpack_from("<I", data, 9)[0]

    body = data[HEADER_SIZE : HEADER_SIZE + body_len] if body_len > 0 else b""

    return Message(
        header=Header(opcode=opcode, flags=flags, request_id=request_id, body_len=body_len),
        body=body,
    )


def encode_read_ack(wc_status: int) -> bytes:
    """Encode an OpReadAck body (1-byte WC status: 0=success, nonzero=failure)."""
    return struct.pack("B", wc_status)


def decode_rdma_read_ready(body: bytes) -> tuple[int, int, int]:
    """Decode an OpRDMAReadReady body.

    Returns (rkey, remote_addr, length).
    Body format: found(1) + rkey(4) + remote_addr(8) + length(4) = 17 bytes.
    """
    if len(body) < 1 or body[0] == 0:
        return 0, 0, 0  # not found
    if len(body) < 17:
        raise ValueError(
            f"truncated RDMA_READ_READY body: expected 17 bytes, got {len(body)}"
        )
    rkey = struct.unpack_from("<I", body, 1)[0]
    remote_addr = struct.unpack_from("<Q", body, 5)[0]
    length = struct.unpack_from("<I", body, 13)[0]
    return rkey, remote_addr, length


def decode_mget_read_ready(body: bytes) -> list[tuple[bool, int, int, int]]:
    """Decode an OpMGetReadReady body.

    Returns list of (found, rkey, remote_addr, length) tuples.
    Body format: count(4) + [found(1) + rkey(4) + remote_addr(8) + length(4)] × count
    """
    if len(body) < 4:
        raise ValueError(f"mget_read_ready body too short: {len(body)}")
    count = struct.unpack_from("<I", body, 0)[0]
    if len(body) < 4 + count * 17:
        raise ValueError(
            f"mget_read_ready body truncated: need {4 + count * 17}, got {len(body)}"
        )
    entries = []
    off = 4
    for _ in range(count):
        found = body[off] != 0
        off += 1
        rkey = struct.unpack_from("<I", body, off)[0]
        off += 4
        remote_addr = struct.unpack_from("<Q", body, off)[0]
        off += 8
        length = struct.unpack_from("<I", body, off)[0]
        off += 4
        entries.append((found, rkey, remote_addr, length))
    return entries


def encode_batch_read_ack(wc_statuses: list[int]) -> bytes:
    """Encode an OpBatchReadAck body.

    Format: count(4) + [wcStatus(1)] × count
    """
    count = len(wc_statuses)
    body = struct.pack("<I", count) + bytes(wc_statuses)
    return body


def decode_batch_read_ack(body: bytes) -> list[int]:
    """Decode an OpBatchReadAck body. Returns list of WC statuses."""
    if len(body) < 4:
        raise ValueError(f"batch_read_ack body too short: {len(body)}")
    count = struct.unpack_from("<I", body, 0)[0]
    if len(body) < 4 + count:
        raise ValueError(
            f"batch_read_ack body truncated: need {4 + count}, got {len(body)}"
        )
    return list(body[4 : 4 + count])


def decode_mset_result(body: bytes) -> list[int]:
    """Decode per-key MSET status bytes. Returns list of 0 (ok) or nonzero (error)."""
    if len(body) < 4:
        raise ValueError(f"mset_result body too short: {len(body)}")
    count = struct.unpack_from("<I", body, 0)[0]
    if len(body) < 4 + count:
        raise ValueError(
            f"mset_result body truncated: need {4 + count}, got {len(body)}"
        )
    return list(body[4 : 4 + count])
