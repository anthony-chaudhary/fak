package l3kv

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// syntheticAttentionKVTensors generates synthetic KV cache attention tensors
// across multiple layers, tokens, and heads. Attention key/value tensors
// exhibit typical structural properties: activation sparsity, clustered head
// subspaces, and repeated token features.
func syntheticAttentionKVTensors(tokens, layers, heads, headDim int) []byte {
	totalFloats := layers * tokens * heads * headDim * 2 // K and V
	buf := make([]byte, totalFloats*4)
	off := 0

	for l := 0; l < layers; l++ {
		for tok := 0; tok < tokens; tok++ {
			for h := 0; h < heads; h++ {
				for d := 0; d < headDim; d++ {
					// Key tensor: clustered activation values with repeated token features
					var kVal float32
					if d%4 == 0 {
						kVal = float32(tok%8) * 0.125
					} else if d%8 == 1 {
						kVal = float32(h%4) * 0.25
					} else {
						kVal = 0.0 // sparse activations
					}
					binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(kVal))
					off += 4

					// Value tensor: low-rank subspaces with repeated patterns
					var vVal float32
					if d%2 == 0 {
						vVal = float32(tok%4) * 0.5
					} else {
						vVal = 0.0
					}
					binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(vVal))
					off += 4
				}
			}
		}
	}
	return buf
}

// TestSpanCompressionRatioAndParity verifies that CompressSpan and DecompressSpan
// achieve >= 1.5x data reduction on compressible attention KV tensors while
// guaranteeing bit-identical round-trip restoration (#11040, parent #10964).
func TestSpanCompressionRatioAndParity(t *testing.T) {
	// Representative attention KV cache geometry:
	// 32 tokens, 4 layers, 8 heads, 64 head-dim -> 512 KiB raw tensor
	raw := syntheticAttentionKVTensors(32, 4, 8, 64)
	if len(raw) == 0 {
		t.Fatal("generated empty synthetic attention tensors")
	}

	compressed, err := CompressSpan(raw)
	if err != nil {
		t.Fatalf("CompressSpan failed: %v", err)
	}

	if !IsCompressedSpan(compressed) {
		t.Fatal("expected IsCompressedSpan to be true for compressed payload")
	}

	reduction := float64(len(raw)) / float64(len(compressed))
	t.Logf("raw=%d bytes, compressed=%d bytes, reduction=%.2fx", len(raw), len(compressed), reduction)

	if reduction < 1.5 {
		t.Fatalf("compression ratio = %.2fx, want >= 1.5x data reduction", reduction)
	}

	restored, err := DecompressSpan(compressed)
	if err != nil {
		t.Fatalf("DecompressSpan failed: %v", err)
	}

	if len(restored) != len(raw) {
		t.Fatalf("restored length mismatch: got %d, want %d", len(restored), len(raw))
	}

	if !bytes.Equal(restored, raw) {
		t.Fatal("restored bytes are not bit-identical to raw attention KV weights (max|delta| != 0)")
	}
}

// TestSpanCompressionCorruptionDetection validates fail-closed rejection when
// CRC32 checksums, magic headers, or LZ4 payloads are tampered.
func TestSpanCompressionCorruptionDetection(t *testing.T) {
	raw := syntheticAttentionKVTensors(16, 2, 4, 32)
	compressed, err := CompressSpan(raw)
	if err != nil {
		t.Fatalf("CompressSpan failed: %v", err)
	}
	if !IsCompressedSpan(compressed) {
		t.Fatal("expected IsCompressedSpan to be true")
	}

	t.Run("tampered CRC32 returns error", func(t *testing.T) {
		tampered := append([]byte(nil), compressed...)
		// Tamper one byte in the 4-byte CRC32 field (bytes 8..11)
		tampered[8] ^= 0xFF
		_, err := DecompressSpan(tampered)
		if err == nil {
			t.Fatal("expected CRC32 checksum mismatch error, got nil")
		}
		if !strings.Contains(err.Error(), "CRC32 checksum mismatch") {
			t.Fatalf("unexpected error message: %v", err)
		}
	})

	t.Run("tampered payload returns error", func(t *testing.T) {
		tampered := append([]byte(nil), compressed...)
		// Tamper a byte in the compressed payload area
		tampered[len(tampered)-1] ^= 0x55
		_, err := DecompressSpan(tampered)
		if err == nil {
			t.Fatal("expected error on tampered compressed payload, got nil")
		}
	})

	t.Run("truncated header returns error", func(t *testing.T) {
		// Contains magic but fewer than 12 header bytes
		truncated := compressed[:8]
		_, err := DecompressSpan(truncated)
		if err == nil {
			t.Fatal("expected error on truncated header, got nil")
		}
		if !strings.Contains(err.Error(), "truncated compressed span header") {
			t.Fatalf("unexpected error message: %v", err)
		}
	})
}

// TestSpanCompressionTransparentFailSafe verifies clean handling of uncompressed,
// incompressible, and empty payloads.
func TestSpanCompressionTransparentFailSafe(t *testing.T) {
	t.Run("uncompressed legacy payload returned cleanly", func(t *testing.T) {
		legacy := []byte("FAKL3SP2_uncompressed_span_payload_from_older_tier_version")
		restored, err := DecompressSpan(legacy)
		if err != nil {
			t.Fatalf("DecompressSpan on uncompressed payload failed: %v", err)
		}
		if !bytes.Equal(restored, legacy) {
			t.Fatal("uncompressed payload not preserved bit-identically")
		}
	})

	t.Run("incompressible payload returned uncompressed without inflation", func(t *testing.T) {
		// Small raw data where framing overhead would exceed raw length
		tiny := []byte("tiny_kv")
		out, err := CompressSpan(tiny)
		if err != nil {
			t.Fatalf("CompressSpan on tiny payload failed: %v", err)
		}
		if IsCompressedSpan(out) {
			t.Fatal("expected tiny payload to remain uncompressed to avoid inflation")
		}
		if !bytes.Equal(out, tiny) {
			t.Fatal("uncompressed output mismatch")
		}

		restored, err := DecompressSpan(out)
		if err != nil {
			t.Fatalf("DecompressSpan failed: %v", err)
		}
		if !bytes.Equal(restored, tiny) {
			t.Fatal("restored tiny payload mismatch")
		}
	})

	t.Run("empty payload handled cleanly", func(t *testing.T) {
		comp, err := CompressSpan(nil)
		if err != nil || len(comp) != 0 {
			t.Fatalf("CompressSpan(nil) err=%v len=%d", err, len(comp))
		}
		decomp, err := DecompressSpan(nil)
		if err != nil || len(decomp) != 0 {
			t.Fatalf("DecompressSpan(nil) err=%v len=%d", err, len(decomp))
		}
	})
}

// TestSpanCompressionTableDrivenSizes verifies bit-identical round trip across
// boundary payload sizes.
func TestSpanCompressionTableDrivenSizes(t *testing.T) {
	sizes := []int{1, 4, 12, 13, 32, 64, 256, 1024, 4096, 16384, 65536}
	for _, size := range sizes {
		// Repeating pattern to ensure compressibility on larger sizes
		raw := make([]byte, size)
		for i := range raw {
			raw[i] = byte(i % 13)
		}

		compressed, err := CompressSpan(raw)
		if err != nil {
			t.Fatalf("size %d: CompressSpan failed: %v", size, err)
		}

		restored, err := DecompressSpan(compressed)
		if err != nil {
			t.Fatalf("size %d: DecompressSpan failed: %v", size, err)
		}

		if !bytes.Equal(restored, raw) {
			t.Fatalf("size %d: restored not bit-identical", size)
		}
	}
}
