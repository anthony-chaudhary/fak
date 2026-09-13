package ggufload

import (
	"math"
	"testing"
)

// mxfp4_test.go is part of the issue #12951 witness set: it exercises the MXFP4
// E2M1 x UE8M0 per-32 path through the shared loader dequant under the
// independent re-read NMSE sampler. The dequant itself already lives in
// gguf_dequant.go (dequantMXFP4Scalar / kvaluesMXFP4 / e8m0ToF32Half) and is
// witnessed by dequant_mxfp4_test.go; this file adds the validator integration
// so a malformed or out-of-bounds MXFP4 payload fails closed before it can be
// admitted as a valid artifact.

// mxfp4EncodeBlock builds one packed MXFP4 block (1 E8M0 scale byte + 16 packed
// E2M1 codes) whose codebook values are the low/high nibble of j and j+16.
func mxfp4EncodeBlock(scale uint8, codes [16]byte) []byte {
	raw := make([]byte, blockMXFP4Bytes)
	raw[0] = scale
	copy(raw[1:], codes[:])
	return raw
}

// mxfp4ExpectedBlock reconstructs the f32 values dequantMXFP4Scalar must emit,
// independently of the production table (using the canonical E2M1 magnitudes).
func mxfp4ExpectedBlock(scale uint8, codes [16]byte) []float32 {
	e2m1 := [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	d := float32(math.Ldexp(1, int(scale)-127))
	out := make([]float32, qkMXFP4)
	for j := 0; j < 16; j++ {
		out[j] = e2m1[codes[j]&0x0f] * d
		out[j+16] = e2m1[codes[j]>>4] * d
	}
	return out
}

// TestMXFP4ValidatorNMSEAcceptsExact proves the sampler folds a bit-exact MXFP4
// reconstruction to NMSE 0 and the validator admits it.
func TestMXFP4ValidatorNMSEAcceptsExact(t *testing.T) {
	codes := [16]byte{}
	for j := range codes {
		codes[j] = byte(j << 4) // covers both nibbles across j and j+16
	}
	packed := mxfp4EncodeBlock(127, codes)
	original := mxfp4ExpectedBlock(127, codes)

	sample := &quantIntegrityNMSESample{}
	if err := mxfp4NMSESampler(sample, original, packed); err != nil {
		t.Fatalf("mxfp4NMSESampler: %v", err)
	}
	if v := sample.nmse(); v != 0 {
		t.Fatalf("exact MXFP4 reconstruction nmse = %g, want 0", v)
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, sample, nil); err != nil {
		t.Fatalf("in-bounds MXFP4 NMSE refused: %v", err)
	}
}

// TestMXFP4ValidatorNMSERefusesDrift proves a payload whose reconstructed
// values drift far from the original is rejected by the ceiling.
func TestMXFP4ValidatorNMSERefusesDrift(t *testing.T) {
	codes := [16]byte{}
	packed := mxfp4EncodeBlock(127, codes) // reconstructs all zeros
	original := make([]float32, qkMXFP4)
	for i := range original {
		original[i] = 100 // large original, reconstructed 0 => NMSE 1.0
	}
	sample := &quantIntegrityNMSESample{}
	if err := mxfp4NMSESampler(sample, original, packed); err != nil {
		t.Fatalf("mxfp4NMSESampler: %v", err)
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, sample, nil); err == nil {
		t.Fatal("drifted MXFP4 payload must fail the NMSE ceiling")
	}
}

// TestMXFP4ValidatorRefusesMalformed proves a payload that is not a whole number
// of blocks, or an original of the wrong element count, fails closed before any
// reconstruction is admitted.
func TestMXFP4ValidatorRefusesMalformed(t *testing.T) {
	sample := &quantIntegrityNMSESample{}
	original := make([]float32, qkMXFP4)
	if err := mxfp4NMSESampler(sample, original, make([]byte, blockMXFP4Bytes-1)); err == nil {
		t.Fatal("non-block-aligned payload must refuse")
	}
	if err := mxfp4NMSESampler(sample, original, nil); err == nil {
		t.Fatal("empty payload must refuse")
	}
	if err := mxfp4NMSESampler(sample, make([]float32, qkMXFP4-1), mxfp4EncodeBlock(127, [16]byte{})); err == nil {
		t.Fatal("wrong original element count must refuse")
	}
	if err := mxfp4NMSESampler(nil, original, mxfp4EncodeBlock(127, [16]byte{})); err == nil {
		t.Fatal("nil sample must refuse")
	}
}
