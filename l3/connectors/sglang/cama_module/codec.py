"""Modular compression codec for CAMA connector values.

Provides a pluggable codec architecture for compressing KV cache tensor values
before storing them in the CAMA server.  The server stores opaque bytes, so
compression is purely connector-side with zero server changes.

Built-in codecs:
  - ``none``           — identity (codec_id=0x00)
  - ``int8``           — INT8 symmetric per-tensor quantization (~2x, lossy)
  - ``shuffle_zstd``   — byte-shuffle + zstd (~1.3x, lossless)
  - ``ChainCodec``     — compose codecs left-to-right (e.g. int8+shuffle_zstd)

Custom codecs: subclass :class:`Codec`, call :func:`register_codec`.
"""

import ctypes
import logging
import struct
from abc import ABC, abstractmethod
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-value header (8 bytes, prepended to every compressed value)
# ---------------------------------------------------------------------------

CODEC_MAGIC = b"\xCA\xCA"
HEADER_SIZE = 8
HEADER_FMT = "<2sBBI"  # magic(2) + codec_id(1) + reserved(1) + original_size(4)


# ---------------------------------------------------------------------------
# Codec ABC
# ---------------------------------------------------------------------------


class Codec(ABC):
    """Abstract base for all value codecs."""

    @property
    @abstractmethod
    def codec_id(self) -> int:
        """Unique uint8 identifier (0x00–0x7F for built-ins, 0x80+ for chains)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable codec name (used in config strings)."""

    @abstractmethod
    def encode(self, data: bytes) -> bytes:
        """Encode raw tensor bytes -> compressed payload (no header)."""

    @abstractmethod
    def decode(self, payload: bytes, original_size: int) -> bytes:
        """Decode compressed payload -> raw tensor bytes."""

    @property
    def is_lossy(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict = {}   # name -> Codec instance
_ID_MAP: dict = {}     # codec_id -> Codec instance


def register_codec(codec: Codec) -> None:
    """Register a codec instance by name and id."""
    _REGISTRY[codec.name] = codec
    _ID_MAP[codec.codec_id] = codec


def get_codec(name: str) -> Codec:
    """Look up a codec by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown codec '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_codec_by_id(codec_id: int) -> Codec:
    """Look up a codec by its numeric id."""
    if codec_id not in _ID_MAP:
        raise ValueError(
            f"Unknown codec_id 0x{codec_id:02X}. "
            f"Registered: {[f'0x{k:02X}' for k in _ID_MAP]}"
        )
    return _ID_MAP[codec_id]


# ---------------------------------------------------------------------------
# Header encode / decode
# ---------------------------------------------------------------------------


def wrap_value(codec: Codec, raw: bytes) -> bytes:
    """Encode + prepend 8-byte header."""
    payload = codec.encode(raw)
    header = struct.pack(HEADER_FMT, CODEC_MAGIC, codec.codec_id, 0, len(raw))
    return header + payload


def unwrap_value(data: bytes) -> bytes:
    """Auto-detect header, decode if present, passthrough if not."""
    if len(data) >= HEADER_SIZE and data[:2] == CODEC_MAGIC:
        codec_id, _, original_size = struct.unpack_from("<BBI", data, 2)
        codec = get_codec_by_id(codec_id)
        return codec.decode(data[HEADER_SIZE:], original_size)
    return data  # legacy uncompressed


# ---------------------------------------------------------------------------
# Built-in codecs
# ---------------------------------------------------------------------------


class NoneCodec(Codec):
    """Identity codec — no compression."""

    @property
    def codec_id(self) -> int:
        return 0x00

    @property
    def name(self) -> str:
        return "none"

    def encode(self, data: bytes) -> bytes:
        return data

    def decode(self, payload: bytes, original_size: int) -> bytes:
        return payload


class Int8Codec(Codec):
    """INT8 symmetric per-tensor quantization (~2x compression, lossy).

    Layout of encoded payload:
      - 4 bytes: float32 scale (little-endian)
      - N bytes: int8 quantized values
    """

    @property
    def codec_id(self) -> int:
        return 0x01

    @property
    def name(self) -> str:
        return "int8"

    @property
    def is_lossy(self) -> bool:
        return True

    def encode(self, data: bytes) -> bytes:
        fp16 = np.frombuffer(data, dtype=np.float16)
        absmax = np.abs(fp16).max()
        if absmax == 0:
            scale = np.float32(1.0)
        else:
            scale = np.float32(absmax / 127.0)
        quantized = np.round(fp16.astype(np.float32) / float(scale)).astype(np.int8)
        return scale.tobytes() + quantized.tobytes()

    def decode(self, payload: bytes, original_size: int) -> bytes:
        scale = np.frombuffer(payload[:4], dtype=np.float32)[0]
        quantized = np.frombuffer(payload[4:], dtype=np.int8)
        fp16 = (quantized.astype(np.float32) * float(scale)).astype(np.float16)
        return fp16.tobytes()


class ShuffleZstdCodec(Codec):
    """Byte-shuffle + zstd compression (~1.3x on structured tensor data, lossless).

    Byte-shuffle groups the MSBs and LSBs of 2-byte elements together,
    improving zstd's compression ratio on FP16/BF16 data.
    """

    def __init__(self, level: int = 3):
        self._level = level

    @property
    def codec_id(self) -> int:
        return 0x02

    @property
    def name(self) -> str:
        return "shuffle_zstd"

    def encode(self, data: bytes) -> bytes:
        import zstandard
        shuffled = _byte_shuffle(data, element_size=2)
        return zstandard.compress(shuffled, self._level)

    def decode(self, payload: bytes, original_size: int) -> bytes:
        import zstandard
        shuffled = zstandard.decompress(payload, max_output_size=original_size)
        return _byte_unshuffle(shuffled, element_size=2)


def _byte_shuffle(data: bytes, element_size: int = 2) -> bytes:
    """Group bytes by position within each element (e.g. all MSBs, then all LSBs)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)
    # Pad to multiple of element_size
    pad = (element_size - n % element_size) % element_size
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    reshaped = arr.reshape(-1, element_size)
    # Transpose: group by byte position across all elements
    shuffled = reshaped.T.ravel()
    return shuffled.tobytes()


def _byte_unshuffle(data: bytes, element_size: int = 2) -> bytes:
    """Reverse of _byte_shuffle."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_elements = len(arr) // element_size
    reshaped = arr.reshape(element_size, n_elements)
    unshuffled = reshaped.T.ravel()
    return unshuffled.tobytes()


# ---------------------------------------------------------------------------
# Chain codec
# ---------------------------------------------------------------------------

def _derive_chain_id(name: str, *codec_ids: int) -> int:
    """Derive a deterministic chain codec_id from name and constituent codec IDs.

    Returns a value in 0x80..0xFF. Same inputs always produce the same output.
    """
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    for cid in codec_ids:
        h = (h * 31 + cid) & 0xFFFFFFFF
    return 0x80 + (h % 128)


class ChainCodec(Codec):
    """Compose multiple codecs: encode left-to-right, decode right-to-left.

    Encoded payload includes a sub-header recording intermediate input sizes
    so that each codec in the reversed decode chain receives the correct
    ``original_size`` (not the final tensor size).

    Sub-header format: ``count(1B) + sizes(4B × count)``.
    """

    def __init__(self, name: str, codecs: List[Codec], codec_id: int):
        self._name = name
        self._codecs = codecs
        self._codec_id = codec_id

    @property
    def codec_id(self) -> int:
        return self._codec_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_lossy(self) -> bool:
        return any(c.is_lossy for c in self._codecs)

    def encode(self, data: bytes) -> bytes:
        n = len(self._codecs)
        sizes = []
        result = data
        for c in self._codecs:
            sizes.append(len(result))
            result = c.encode(result)
        # Sub-header: count(1B) + sizes(4B each)
        header = struct.pack("<B", n) + b"".join(
            struct.pack("<I", s) for s in sizes
        )
        return header + result

    def decode(self, payload: bytes, original_size: int) -> bytes:
        # Parse sub-header
        n = payload[0]
        offset = 1
        sizes = []
        for _ in range(n):
            sizes.append(struct.unpack_from("<I", payload, offset)[0])
            offset += 4
        result = payload[offset:]
        for i, c in enumerate(reversed(self._codecs)):
            j = n - 1 - i
            result = c.decode(result, sizes[j])
        return result


def register_chain(name: str, *codec_names: str) -> ChainCodec:
    """Create and register a ChainCodec from existing codec names.

    Idempotent: re-registering the same name returns the existing chain.
    Codec ID is derived deterministically from the ordered constituent IDs.
    """
    # Idempotent: return existing chain if same name already registered
    if name in _REGISTRY:
        existing = _REGISTRY[name]
        if isinstance(existing, ChainCodec):
            return existing

    codecs = [get_codec(n) for n in codec_names]
    chain_id = _derive_chain_id(name, *(c.codec_id for c in codecs))

    # Collision resolution: linear probe within 0x80..0xFF
    for _ in range(128):
        if chain_id not in _ID_MAP or _ID_MAP[chain_id].name == name:
            break
        chain_id = 0x80 + ((chain_id - 0x80 + 1) % 128)
    else:
        raise ValueError("No available chain codec IDs in 0x80..0xFF range")

    chain = ChainCodec(name, codecs, chain_id)
    register_codec(chain)
    return chain


# ---------------------------------------------------------------------------
# SGL wrappers for codec paths
# ---------------------------------------------------------------------------


class _CompressedSGL:
    """SET path: wraps pre-compressed bytes, duck-types the SGL interface."""

    def __init__(self, compressed_bytes: bytes):
        self._data = compressed_bytes
        self.size = len(compressed_bytes)
        self.reg_handle = 1  # forces non-zero-copy in RDMA client

    def to_bytes(self) -> bytes:
        return self._data


class _DecodeSGL:
    """GET path: receives compressed bytes, decompresses into host pointer."""

    def __init__(self, host_ptr: int, original_size: int):
        self._host_ptr = host_ptr
        self._original_size = original_size
        self.size = original_size  # upper bound for receive buffer
        self.reg_handle = 1       # forces non-zero-copy

    def from_bytes(self, data: bytes) -> None:
        raw = unwrap_value(data)
        if len(raw) != self._original_size:
            logger.warning(
                "Codec size mismatch: decoded %d bytes, expected %d — buffer may contain "
                "%d bytes of stale data. Check encoder/decoder dtype/shape agreement.",
                len(raw), self._original_size, abs(len(raw) - self._original_size),
            )
        ctypes.memmove(self._host_ptr, raw, min(len(raw), self._original_size))


# ---------------------------------------------------------------------------
# Auto-register built-ins
# ---------------------------------------------------------------------------

register_codec(NoneCodec())
register_codec(Int8Codec())
register_codec(ShuffleZstdCodec())
