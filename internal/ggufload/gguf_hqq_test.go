package ggufload

import (
	"encoding/binary"
	"math"
	"testing"
)

// gguf_hqq_test.go — the HQQ 4-bit vision-tower golden block (#4876, Bonsai VLM).
//
// HQQ (Half-Quadratic Quantization) is NOT a ggml k-quant: it reconstructs in the
// quantized domain, y = scale*(q - zero), with an fp16 scale AND an fp16 zero-point
// learned per group (default group_size 64 for 4-bit). These tests pin that math to
// hand-derived reference values so `go test ./internal/ggufload -run HQQ` witnesses
// the dequant independently of any k-quant path. Invalidating assumptions (confirm
// against a live Bonsai mmproj header): the ggml type tag (TensorHQQ4=44 is
// fak-local), the group size (64), and the fp16 scale/zero block encoding.

// buildHQQ4Block packs one qkHQQ4=64-element HQQ group: little-endian f16 scale,
// little-endian f16 zero, then qkHQQ4/2 split-half interleaved 4-bit codes (low
// nibble of byte j = element j, high nibble = element j+qkHQQ4/2). It mirrors the
// on-disk layout dequantHQQ4Scalar reads, so the test packs exactly what the loader
// unpacks.
func buildHQQ4Block(scaleBits, zeroBits uint16, codes [qkHQQ4]byte) []byte {
	raw := make([]byte, blockHQQ4Bytes)
	binary.LittleEndian.PutUint16(raw[0:2], scaleBits)
	binary.LittleEndian.PutUint16(raw[2:4], zeroBits)
	for j := 0; j < qkHQQ4/2; j++ {
		lo := codes[j] & 0x0f
		hi := codes[j+qkHQQ4/2] & 0x0f
		raw[4+j] = lo | hi<<4
	}
	return raw
}

// TestDequantHQQ4GoldenBlock dequantizes a single hand-built HQQ group and checks
// every element against y = scale*(q - zero) computed in Go. scale=0.5 and zero=8.0
// are both exact in fp16, and q-zero is an integer, so every product is an exact
// multiple of 0.5 — the comparison is exact float32 equality, not a tolerance.
func TestDequantHQQ4GoldenBlock(t *testing.T) {
	if got := TensorHQQ4.String(); got != "HQQ4" {
		t.Fatalf("TensorHQQ4.String() = %q, want HQQ4", got)
	}

	const (
		scale     = float32(0.5)
		zero      = float32(8.0)
		scaleBits = uint16(0x3800) // fp16 0.5
		zeroBits  = uint16(0x4800) // fp16 8.0
	)
	var codes [qkHQQ4]byte
	for j := 0; j < qkHQQ4; j++ {
		codes[j] = byte(j & 0x0f) // 0..15, wrapping — a full sweep of the 4-bit range
	}
	raw := buildHQQ4Block(scaleBits, zeroBits, codes)

	tensor := TensorInfo{Name: "v.blk.0.attn_q.weight", Dims: []uint64{qkHQQ4}, Type: TensorHQQ4}
	payload, err := tensorPayloadBytes(tensor)
	if err != nil {
		t.Fatalf("tensorPayloadBytes: %v", err)
	}
	if payload != blockHQQ4Bytes {
		t.Fatalf("payload = %d, want %d", payload, blockHQQ4Bytes)
	}

	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("dequantF32: %v", err)
	}
	if len(got) != qkHQQ4 {
		t.Fatalf("len(got) = %d, want %d", len(got), qkHQQ4)
	}
	for j := 0; j < qkHQQ4; j++ {
		want := scale * (float32(int(codes[j]&0x0f)) - zero)
		if got[j] != want {
			t.Fatalf("value[%d] = %g, want %g", j, got[j], want)
		}
	}
}

// TestDequantHQQ4RoundTripTwoBlocks packs a deterministic code vector into two HQQ
// groups with DISTINCT fp16 scales and zeros and checks the dequant reproduces
// scale*(q-zero) exactly per group — proving block 1 reads its own scale/zero/codes
// and not block 0's (the per-block base arithmetic).
func TestDequantHQQ4RoundTripTwoBlocks(t *testing.T) {
	const elems = 2 * qkHQQ4
	scales := []float32{0.5, 1.5}         // exact in fp16
	zeros := []float32{8.0, 6.0}          // exact in fp16
	scaleBits := []uint16{0x3800, 0x3e00} // fp16 0.5, 1.5
	zeroBits := []uint16{0x4800, 0x4600}  // fp16 8.0, 6.0

	var codes [elems]byte
	for i := range codes {
		codes[i] = byte((i*7 + 3) & 0x0f)
	}

	raw := make([]byte, 2*blockHQQ4Bytes)
	for block := 0; block < 2; block++ {
		var bc [qkHQQ4]byte
		copy(bc[:], codes[block*qkHQQ4:(block+1)*qkHQQ4])
		copy(raw[block*blockHQQ4Bytes:], buildHQQ4Block(scaleBits[block], zeroBits[block], bc))
	}

	tensor := TensorInfo{Name: "v.blk.1.ffn_up.weight", Dims: []uint64{elems}, Type: TensorHQQ4}
	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("dequantF32: %v", err)
	}
	if len(got) != elems {
		t.Fatalf("len(got) = %d, want %d", len(got), elems)
	}
	for i, v := range got {
		g := i / qkHQQ4
		want := scales[g] * (float32(int(codes[i]&0x0f)) - zeros[g])
		if v != want {
			t.Fatalf("value[%d] = %g, want %g", i, v, want)
		}
	}
}

// TestDequantHQQ4FeedsMMProjPath confirms the vision-tower hook: an HQQ4 tensor named
// in the mmproj v.* namespace is recognized as a vision tensor AND dequantizes to f32
// through the exact call WeightSource.TensorF32 makes (dequantF32). That is the
// wiring the Bonsai vision forward (T8) consumes — the tower reaches f32 at its
// shipped ~4.5-bpw footprint.
func TestDequantHQQ4FeedsMMProjPath(t *testing.T) {
	name := "v.blk.0.attn_k.weight"
	if !isMMProjVisionTensor(name) {
		t.Fatalf("isMMProjVisionTensor(%q) = false, want true", name)
	}
	var codes [qkHQQ4]byte
	for j := range codes {
		codes[j] = byte((j * 3) & 0x0f)
	}
	raw := buildHQQ4Block(0x3800, 0x4800, codes) // scale 0.5, zero 8.0
	tensor := TensorInfo{Name: name, Dims: []uint64{qkHQQ4}, Type: TensorHQQ4}
	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("dequantF32: %v", err)
	}
	for j := 0; j < qkHQQ4; j++ {
		want := float32(0.5) * (float32(int(codes[j]&0x0f)) - 8.0)
		if math.Abs(float64(got[j]-want)) > 1e-6 {
			t.Fatalf("value[%d] = %g, want %g", j, got[j], want)
		}
	}
}

// TestDequantHQQ4RejectsPartialBlock refuses a tensor whose element count is not a
// whole number of HQQ groups — a truncated block must be a loud error, never a
// silently-misread tower.
func TestDequantHQQ4RejectsPartialBlock(t *testing.T) {
	tensor := TensorInfo{Name: "partial_hqq4", Dims: []uint64{qkHQQ4 - 1}, Type: TensorHQQ4}
	if _, err := tensorPayloadBytes(tensor); err == nil {
		t.Fatal("tensorPayloadBytes accepted a partial HQQ4 block")
	}
	if _, err := dequantF32(tensor, make([]byte, blockHQQ4Bytes)); err == nil {
		t.Fatal("dequantF32 accepted a partial HQQ4 block")
	}
}
