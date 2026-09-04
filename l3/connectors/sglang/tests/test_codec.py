"""Unit tests for cama_module.codec — no live server required.

Tests are organized by component:

  TestCodecHeader              (4 tests)  — pack/unpack round-trip, magic, legacy passthrough
  TestNoneCodec                (2 tests)  — identity encode/decode
  TestInt8Codec                (5 tests)  — round-trip FP16, size ratio, all-zeros, 1MB tensor, lossy
  TestShuffleZstdCodec         (4 tests)  — round-trip, ratio > 1.0, custom level, lossless
  TestByteShuffle              (2 tests)  — roundtrip, groups bytes
  TestChainCodec               (4 tests)  — int8+zstd round-trip, codec_id, name, lossy prop
  TestChainIntermediateSizes   (3 tests)  — BUG 1: sub-header, expanding-intermediate round-trip
  TestGlobalMutation           (1 test)   — BUG 2: local ShuffleZstdCodec doesn't touch singleton
  TestDeterministicChainId     (3 tests)  — BUG 3: same chain→same ID, different→different, order
  TestDecodeSGLSizeMismatch    (1 test)   — BUG 4: size mismatch logs WARNING
  TestWrapUnwrapChain          (1 test)   — end-to-end wrap/unwrap through chain codec
  TestRegistry                 (4 tests)  — register, get by name/id, missing raises ValueError
  TestWrapUnwrap               (4 tests)  — round-trip, legacy passthrough, mixed data, none codec
  TestCompressedSGL            (2 tests)  — to_bytes, reg_handle
  TestDecodeSGL                (3 tests)  — from_bytes decompresses, legacy passthrough, size cap
"""

import ctypes
import logging
import struct
import unittest

import numpy as np

from cama_module.codec import (
    CODEC_MAGIC,
    HEADER_SIZE,
    HEADER_FMT,
    Codec,
    NoneCodec,
    Int8Codec,
    ShuffleZstdCodec,
    ChainCodec,
    _derive_chain_id,
    _REGISTRY,
    _ID_MAP,
    register_codec,
    register_chain,
    get_codec,
    get_codec_by_id,
    wrap_value,
    unwrap_value,
    _CompressedSGL,
    _DecodeSGL,
    _byte_shuffle,
    _byte_unshuffle,
)


class TestCodecHeader(unittest.TestCase):
    """Tests for the 8-byte header format."""

    def test_header_pack_unpack_roundtrip(self):
        """Pack then unpack yields original values."""
        codec_id = 0x01
        original_size = 131072  # 128 KB
        header = struct.pack(HEADER_FMT, CODEC_MAGIC, codec_id, 0, original_size)
        self.assertEqual(len(header), HEADER_SIZE)
        magic, cid, reserved, orig = struct.unpack(HEADER_FMT, header)
        self.assertEqual(magic, CODEC_MAGIC)
        self.assertEqual(cid, codec_id)
        self.assertEqual(reserved, 0)
        self.assertEqual(orig, original_size)

    def test_magic_bytes(self):
        """Magic is 0xCA 0xCA."""
        self.assertEqual(CODEC_MAGIC, b"\xCA\xCA")

    def test_header_size(self):
        """Header is exactly 8 bytes."""
        self.assertEqual(HEADER_SIZE, 8)

    def test_legacy_passthrough(self):
        """Data without magic prefix passes through unwrap_value unchanged."""
        raw = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        self.assertEqual(unwrap_value(raw), raw)


class TestNoneCodec(unittest.TestCase):
    """Tests for the identity codec."""

    def test_encode_identity(self):
        codec = NoneCodec()
        data = b"hello world"
        self.assertEqual(codec.encode(data), data)

    def test_decode_identity(self):
        codec = NoneCodec()
        data = b"hello world"
        self.assertEqual(codec.decode(data, len(data)), data)


class TestInt8Codec(unittest.TestCase):
    """Tests for INT8 symmetric quantization."""

    def test_roundtrip_fp16(self):
        """Encode then decode recovers FP16 values within tolerance."""
        codec = Int8Codec()
        rng = np.random.default_rng(42)
        fp16 = rng.standard_normal(1024).astype(np.float16)
        raw = fp16.tobytes()

        encoded = codec.encode(raw)
        decoded_bytes = codec.decode(encoded, len(raw))
        decoded = np.frombuffer(decoded_bytes, dtype=np.float16)

        self.assertTrue(np.allclose(fp16, decoded, atol=0.1))

    def test_compression_ratio(self):
        """Encoded size should be ~half + 4 bytes scale."""
        codec = Int8Codec()
        n = 2048
        fp16 = np.ones(n, dtype=np.float16)
        raw = fp16.tobytes()  # 4096 bytes
        encoded = codec.encode(raw)
        # int8 payload = n bytes + 4 bytes scale = 2052
        self.assertEqual(len(encoded), n + 4)
        self.assertLess(len(encoded), len(raw))

    def test_all_zeros(self):
        """All-zeros tensor encodes and decodes without NaN/Inf."""
        codec = Int8Codec()
        fp16 = np.zeros(512, dtype=np.float16)
        raw = fp16.tobytes()
        encoded = codec.encode(raw)
        decoded_bytes = codec.decode(encoded, len(raw))
        decoded = np.frombuffer(decoded_bytes, dtype=np.float16)
        self.assertTrue(np.all(decoded == 0))

    def test_1mb_tensor(self):
        """Round-trip on a 1 MB FP16 tensor."""
        codec = Int8Codec()
        n = 1024 * 1024 // 2  # 524288 float16 elements = 1 MB
        rng = np.random.default_rng(123)
        fp16 = rng.standard_normal(n).astype(np.float16)
        raw = fp16.tobytes()
        self.assertEqual(len(raw), 1024 * 1024)

        encoded = codec.encode(raw)
        decoded_bytes = codec.decode(encoded, len(raw))
        decoded = np.frombuffer(decoded_bytes, dtype=np.float16)
        self.assertTrue(np.allclose(fp16, decoded, atol=0.1))

    def test_is_lossy(self):
        codec = Int8Codec()
        self.assertTrue(codec.is_lossy)


class TestShuffleZstdCodec(unittest.TestCase):
    """Tests for byte-shuffle + zstd."""

    def test_roundtrip_exact(self):
        """Encode then decode is exact (lossless)."""
        codec = ShuffleZstdCodec(level=1)
        rng = np.random.default_rng(99)
        fp16 = rng.standard_normal(2048).astype(np.float16)
        raw = fp16.tobytes()
        encoded = codec.encode(raw)
        decoded = codec.decode(encoded, len(raw))
        self.assertEqual(raw, decoded)

    def test_compression_ratio(self):
        """Structured data should compress better than 1.0x."""
        codec = ShuffleZstdCodec(level=3)
        # Structured: repeating pattern compresses well
        fp16 = np.tile(np.arange(16, dtype=np.float16), 256)
        raw = fp16.tobytes()
        encoded = codec.encode(raw)
        self.assertLess(len(encoded), len(raw))

    def test_custom_level(self):
        """Higher zstd level should still produce valid output."""
        codec = ShuffleZstdCodec(level=9)
        fp16 = np.ones(512, dtype=np.float16)
        raw = fp16.tobytes()
        encoded = codec.encode(raw)
        decoded = codec.decode(encoded, len(raw))
        self.assertEqual(raw, decoded)

    def test_is_not_lossy(self):
        codec = ShuffleZstdCodec()
        self.assertFalse(codec.is_lossy)


class TestByteShuffle(unittest.TestCase):
    """Tests for the byte shuffle/unshuffle helpers."""

    def test_roundtrip(self):
        data = bytes(range(256)) * 4  # 1024 bytes
        shuffled = _byte_shuffle(data, element_size=2)
        unshuffled = _byte_unshuffle(shuffled, element_size=2)
        self.assertEqual(data, unshuffled)

    def test_shuffle_groups_bytes(self):
        """Shuffling [0x00,0x01, 0x02,0x03] with elem_size=2 gives [0x00,0x02,0x01,0x03]."""
        data = bytes([0x00, 0x01, 0x02, 0x03])
        shuffled = _byte_shuffle(data, element_size=2)
        self.assertEqual(shuffled, bytes([0x00, 0x02, 0x01, 0x03]))


class TestChainCodec(unittest.TestCase):
    """Tests for chaining multiple codecs."""

    def test_chain_roundtrip(self):
        """int8 + shuffle_zstd chain round-trips within INT8 tolerance.

        Also verifies decoded size == original size exactly, and that chain
        output matches manual step-by-step application.
        """
        chain = register_chain("test_int8+shuffle_zstd", "int8", "shuffle_zstd")
        rng = np.random.default_rng(7)
        fp16 = rng.standard_normal(2048).astype(np.float16)
        raw = fp16.tobytes()

        encoded = chain.encode(raw)
        decoded_bytes = chain.decode(encoded, len(raw))
        # Exact size match
        self.assertEqual(len(decoded_bytes), len(raw))
        decoded = np.frombuffer(decoded_bytes, dtype=np.float16)
        self.assertTrue(np.allclose(fp16, decoded, atol=0.1))

        # Verify chain output matches manual step-by-step
        int8_codec = get_codec("int8")
        zstd_codec = get_codec("shuffle_zstd")
        step1 = int8_codec.encode(raw)
        step2 = zstd_codec.encode(step1)
        # The chain payload (after sub-header) should equal step2
        n = encoded[0]
        sub_header_size = 1 + n * 4
        self.assertEqual(encoded[sub_header_size:], step2)

    def test_chain_codec_id_high_bit(self):
        """ChainCodec ids have bit 7 set (>= 0x80)."""
        chain = register_chain("test_chain_id", "none", "none")
        self.assertGreaterEqual(chain.codec_id, 0x80)

    def test_chain_name(self):
        chain = register_chain("test_mychain", "int8", "none")
        self.assertEqual(chain.name, "test_mychain")

    def test_chain_lossy_propagation(self):
        """Chain is lossy if any inner codec is lossy."""
        chain = register_chain("test_lossy_chain", "int8", "none")
        self.assertTrue(chain.is_lossy)
        chain2 = register_chain("test_lossless_chain", "none", "none")
        self.assertFalse(chain2.is_lossy)


class TestChainIntermediateSizes(unittest.TestCase):
    """BUG 1: ChainCodec sub-header records correct intermediate sizes."""

    def test_sub_header_sizes_match_intermediate(self):
        """Sub-header sizes match the actual input to each codec in the chain."""
        chain = register_chain("test_sizes_int8+zstd", "int8", "shuffle_zstd")
        rng = np.random.default_rng(42)
        fp16 = rng.standard_normal(2048).astype(np.float16)
        raw = fp16.tobytes()

        encoded = chain.encode(raw)

        # Parse sub-header
        n = encoded[0]
        self.assertEqual(n, 2)
        size0 = struct.unpack_from("<I", encoded, 1)[0]
        size1 = struct.unpack_from("<I", encoded, 5)[0]

        # size0 = input to int8 = len(raw)
        self.assertEqual(size0, len(raw))
        # size1 = input to shuffle_zstd = output of int8
        int8_codec = get_codec("int8")
        int8_output = int8_codec.encode(raw)
        self.assertEqual(size1, len(int8_output))

    def test_chain_roundtrip_with_expanding_intermediate(self):
        """Chain with an expanding intermediate codec round-trips correctly.

        This would FAIL with the old code: zstd.decompress would get
        max_output_size=original_size, which is smaller than the expanded
        intermediate data.
        """
        # Create a "doubler" codec that expands data (output > input)
        class DoublerCodec(Codec):
            @property
            def codec_id(self) -> int:
                return 0x10
            @property
            def name(self) -> str:
                return "_test_doubler"
            def encode(self, data: bytes) -> bytes:
                return data + data  # 2x expansion
            def decode(self, payload: bytes, original_size: int) -> bytes:
                return payload[:original_size]

        register_codec(DoublerCodec())
        chain = register_chain("test_doubler+zstd", "_test_doubler", "shuffle_zstd")

        rng = np.random.default_rng(99)
        fp16 = rng.standard_normal(1024).astype(np.float16)
        raw = fp16.tobytes()  # 2048 bytes

        encoded = chain.encode(raw)
        decoded = chain.decode(encoded, len(raw))
        self.assertEqual(decoded, raw)

    def test_sub_header_format(self):
        """Sub-header is count(1B) + sizes(4B each), followed by compressed data."""
        chain = register_chain("test_hdr_none+none", "none", "none")
        data = b"test1234"
        encoded = chain.encode(data)

        # count = 2
        self.assertEqual(encoded[0], 2)
        # size[0] = 8 (input to first none)
        self.assertEqual(struct.unpack_from("<I", encoded, 1)[0], 8)
        # size[1] = 8 (input to second none)
        self.assertEqual(struct.unpack_from("<I", encoded, 5)[0], 8)
        # payload after 9-byte sub-header = original data (none+none)
        self.assertEqual(encoded[9:], data)


class TestGlobalMutation(unittest.TestCase):
    """BUG 2: Creating local ShuffleZstdCodec doesn't touch global singleton."""

    def test_shuffle_zstd_level_isolation(self):
        """Local ShuffleZstdCodec(level=19) does not mutate _REGISTRY singleton."""
        global_zstd = _REGISTRY["shuffle_zstd"]
        original_level = global_zstd._level

        # Create a local instance (the BUG 2 fix pattern from cama_storage.py)
        local_zstd = ShuffleZstdCodec(level=19)

        # Global singleton is untouched
        self.assertEqual(global_zstd._level, original_level)
        self.assertEqual(local_zstd._level, 19)
        self.assertIsNot(local_zstd, global_zstd)


class TestDeterministicChainId(unittest.TestCase):
    """BUG 3: Chain codec_id is deterministic from constituent IDs."""

    def test_same_chain_same_id(self):
        """Calling _derive_chain_id with same args always returns same result."""
        id1 = _derive_chain_id("int8+shuffle_zstd", 0x01, 0x02)
        id2 = _derive_chain_id("int8+shuffle_zstd", 0x01, 0x02)
        self.assertEqual(id1, id2)

    def test_different_chains_different_ids(self):
        """Different names produce different chain IDs."""
        id_a = _derive_chain_id("chain_a", 0x01, 0x02)
        id_b = _derive_chain_id("chain_b", 0x01, 0x02)
        self.assertNotEqual(id_a, id_b)

    def test_order_matters(self):
        """Reversed codec order produces a different ID."""
        id_ab = _derive_chain_id("ab", 0x01, 0x02)
        id_ba = _derive_chain_id("ab", 0x02, 0x01)
        self.assertNotEqual(id_ab, id_ba)

    def test_id_in_range(self):
        """Derived IDs are always in 0x80..0xFF."""
        for a in range(8):
            for b in range(8):
                cid = _derive_chain_id(f"test_{a}_{b}", a, b)
                self.assertGreaterEqual(cid, 0x80)
                self.assertLessEqual(cid, 0xFF)

    def test_register_chain_idempotent(self):
        """Re-registering same chain name returns existing instance."""
        c1 = register_chain("test_idem_chain", "none", "int8")
        c2 = register_chain("test_idem_chain", "none", "int8")
        self.assertIs(c1, c2)


class TestDecodeSGLSizeMismatch(unittest.TestCase):
    """BUG 4: _DecodeSGL logs WARNING on size mismatch."""

    def test_warns_on_size_mismatch(self):
        """Decoded size != expected size triggers a WARNING log."""
        codec = get_codec("none")
        raw = b"hello world!!"  # 13 bytes
        wrapped = wrap_value(codec, raw)

        buf = (ctypes.c_char * 1024)()  # oversized buffer
        sgl = _DecodeSGL(ctypes.addressof(buf), 1024)  # expects 1024

        with self.assertLogs("cama_module.codec", level="WARNING") as cm:
            sgl.from_bytes(wrapped)

        # Check the warning message
        self.assertTrue(
            any("Codec size mismatch" in msg for msg in cm.output),
            f"Expected 'Codec size mismatch' in logs, got: {cm.output}",
        )

    def test_no_warning_on_exact_match(self):
        """No warning when decoded size matches expected size."""
        codec = get_codec("none")
        raw = b"exact match!"
        wrapped = wrap_value(codec, raw)

        buf = (ctypes.c_char * len(raw))()
        sgl = _DecodeSGL(ctypes.addressof(buf), len(raw))

        # Should not log any warnings
        logger = logging.getLogger("cama_module.codec")
        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError when no logs are emitted
            with self.assertLogs("cama_module.codec", level="WARNING"):
                sgl.from_bytes(wrapped)


class TestWrapUnwrapChain(unittest.TestCase):
    """End-to-end wrap/unwrap through chain codec with exact size verification."""

    def test_wrap_unwrap_chain_exact_size(self):
        """wrap_value + unwrap_value through int8+shuffle_zstd chain."""
        chain = register_chain("test_e2e_chain", "int8", "shuffle_zstd")
        rng = np.random.default_rng(123)
        fp16 = rng.standard_normal(2048).astype(np.float16)
        raw = fp16.tobytes()

        wrapped = wrap_value(chain, raw)
        # Verify header
        self.assertEqual(wrapped[:2], CODEC_MAGIC)
        self.assertEqual(wrapped[2], chain.codec_id)
        orig_size = struct.unpack_from("<I", wrapped, 4)[0]
        self.assertEqual(orig_size, len(raw))

        # Unwrap and verify exact size + approximate values
        unwrapped = unwrap_value(wrapped)
        self.assertEqual(len(unwrapped), len(raw))
        decoded = np.frombuffer(unwrapped, dtype=np.float16)
        self.assertTrue(np.allclose(fp16, decoded, atol=0.1))


class TestRegistry(unittest.TestCase):
    """Tests for codec registration and lookup."""

    def test_builtin_registered(self):
        """Built-in codecs are available by name."""
        self.assertIsInstance(get_codec("none"), NoneCodec)
        self.assertIsInstance(get_codec("int8"), Int8Codec)
        self.assertIsInstance(get_codec("shuffle_zstd"), ShuffleZstdCodec)

    def test_get_by_id(self):
        self.assertIsInstance(get_codec_by_id(0x00), NoneCodec)
        self.assertIsInstance(get_codec_by_id(0x01), Int8Codec)
        self.assertIsInstance(get_codec_by_id(0x02), ShuffleZstdCodec)

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError):
            get_codec("nonexistent_codec")

    def test_missing_id_raises(self):
        with self.assertRaises(ValueError):
            get_codec_by_id(0xFF)


class TestWrapUnwrap(unittest.TestCase):
    """Tests for wrap_value / unwrap_value with header."""

    def test_roundtrip_int8(self):
        """wrap + unwrap with int8 codec recovers data within tolerance."""
        codec = get_codec("int8")
        rng = np.random.default_rng(55)
        fp16 = rng.standard_normal(1024).astype(np.float16)
        raw = fp16.tobytes()

        wrapped = wrap_value(codec, raw)
        # Verify header present
        self.assertEqual(wrapped[:2], CODEC_MAGIC)
        self.assertEqual(wrapped[2], 0x01)  # int8 codec_id

        unwrapped = unwrap_value(wrapped)
        decoded = np.frombuffer(unwrapped, dtype=np.float16)
        self.assertTrue(np.allclose(fp16, decoded, atol=0.1))

    def test_legacy_passthrough(self):
        """Data without header passes through unwrap unchanged."""
        raw = b"\x00\x01\x02\x03" * 100
        self.assertEqual(unwrap_value(raw), raw)

    def test_short_data_passthrough(self):
        """Data shorter than header size passes through."""
        raw = b"\xCA\xCA\x01"  # 3 bytes, looks like partial header
        self.assertEqual(unwrap_value(raw), raw)

    def test_none_codec_roundtrip(self):
        """NoneCodec wraps/unwraps as identity."""
        codec = get_codec("none")
        raw = b"test data bytes"
        wrapped = wrap_value(codec, raw)
        self.assertEqual(unwrap_value(wrapped), raw)


class TestCompressedSGL(unittest.TestCase):
    """Tests for _CompressedSGL (SET path wrapper)."""

    def test_to_bytes(self):
        data = b"compressed payload here"
        sgl = _CompressedSGL(data)
        self.assertEqual(sgl.to_bytes(), data)
        self.assertEqual(sgl.size, len(data))

    def test_reg_handle(self):
        """reg_handle=1 forces non-zero-copy path."""
        sgl = _CompressedSGL(b"data")
        self.assertEqual(sgl.reg_handle, 1)


class TestDecodeSGL(unittest.TestCase):
    """Tests for _DecodeSGL (GET path wrapper)."""

    def test_from_bytes_decompresses(self):
        """from_bytes with int8-wrapped data decompresses into ctypes buffer."""
        codec = get_codec("int8")
        rng = np.random.default_rng(77)
        fp16 = rng.standard_normal(512).astype(np.float16)
        raw = fp16.tobytes()

        wrapped = wrap_value(codec, raw)

        # Allocate a ctypes buffer as the target
        buf = (ctypes.c_char * len(raw))()
        sgl = _DecodeSGL(ctypes.addressof(buf), len(raw))
        sgl.from_bytes(wrapped)

        result = np.frombuffer(bytes(buf), dtype=np.float16)
        self.assertTrue(np.allclose(fp16, result, atol=0.1))

    def test_legacy_passthrough(self):
        """from_bytes with uncompressed data copies directly."""
        raw = np.arange(64, dtype=np.float16).tobytes()
        buf = (ctypes.c_char * len(raw))()
        sgl = _DecodeSGL(ctypes.addressof(buf), len(raw))
        sgl.from_bytes(raw)
        self.assertEqual(bytes(buf), raw)

    def test_reg_handle(self):
        """reg_handle=1 forces non-zero-copy path."""
        sgl = _DecodeSGL(0, 1024)
        self.assertEqual(sgl.reg_handle, 1)


if __name__ == "__main__":
    unittest.main()
