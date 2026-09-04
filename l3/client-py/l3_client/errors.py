"""CAMA client exception hierarchy."""

import struct


class CamaOOMError(RuntimeError):
    """Server rejected SET due to memory pressure (OOM).

    Raised when the server returns RespOOM (0xF6). The client should back off
    and retry after a delay, or reduce write rate.

    Attributes:
        utilization_pct: Slab utilization percentage (0-100).
        allocated_bytes: Bytes currently allocated in the shard.
        total_bytes: Total slab capacity in the shard.
        server_message: Diagnostic message from the server.
    """

    def __init__(
        self,
        utilization_pct: int,
        allocated_bytes: int,
        total_bytes: int,
        server_message: str,
    ):
        self.utilization_pct = utilization_pct
        self.allocated_bytes = allocated_bytes
        self.total_bytes = total_bytes
        self.server_message = server_message
        super().__init__(
            f"CAMA OOM: server at {utilization_pct}% capacity "
            f"({allocated_bytes}/{total_bytes} bytes) — {server_message}"
        )

    @classmethod
    def from_body(cls, body: bytes) -> "CamaOOMError":
        """Decode a RespOOM body into a CamaOOMError."""
        if len(body) < 19:
            return cls(0, 0, 0, "malformed OOM response")
        util_pct = body[0]
        allocated = struct.unpack_from("<Q", body, 1)[0]
        total = struct.unpack_from("<Q", body, 9)[0]
        msg_len = struct.unpack_from("<H", body, 17)[0]
        msg = body[19 : 19 + msg_len].decode(errors="replace") if msg_len > 0 else ""
        return cls(util_pct, allocated, total, msg)


class CamaNotReadyError(RuntimeError):
    """Server still starting — shard not allocated yet.

    Raised when the server returns RespNotReady (0xF7). The connection is
    healthy — no reconnection needed. The client should retry after a short
    delay (1-2s).

    Attributes:
        shards_ready: Number of shards allocated so far.
        shards_total: Total number of shards.
        server_message: Diagnostic message from the server.
    """

    def __init__(self, shards_ready: int, shards_total: int, server_message: str):
        self.shards_ready = shards_ready
        self.shards_total = shards_total
        self.server_message = server_message
        super().__init__(
            f"CAMA not ready: {shards_ready}/{shards_total} shards allocated — {server_message}"
        )

    @classmethod
    def from_body(cls, body: bytes) -> "CamaNotReadyError":
        """Decode a RespNotReady body into a CamaNotReadyError."""
        if len(body) < 10:
            return cls(0, 0, "malformed not-ready response")
        shards_ready = struct.unpack_from("<I", body, 0)[0]
        shards_total = struct.unpack_from("<I", body, 4)[0]
        msg_len = struct.unpack_from("<H", body, 8)[0]
        msg = body[10 : 10 + msg_len].decode(errors="replace") if msg_len > 0 else ""
        return cls(shards_ready, shards_total, msg)


class CamaServerOverloadError(RuntimeError):
    """Server rejected operation due to dispatch queue pressure.

    Raised when the server returns a RESP_ERROR whose body contains
    "server overloaded".  This is distinct from a transport error:

    - The connection is healthy (no QP teardown needed).
    - ``is_retriable()`` returns **False** (no reconnection).
    - The connector should catch this, back off briefly, and retry
      the batch in-place without reconnecting.

    Attributes:
        server_message: The raw error string from the server.
    """

    def __init__(self, server_message: str):
        self.server_message = server_message
        super().__init__(f"CAMA server overloaded: {server_message}")


# L3 error aliases
class L3Error(RuntimeError):
    pass

CamaError = L3Error
L3NotReadyError = CamaNotReadyError
L3OOMError = CamaOOMError
L3ServerOverloadError = CamaServerOverloadError
