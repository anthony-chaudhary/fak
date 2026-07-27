package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// This block follows the Q1_0 (g128) layout of the Bonsai-27B 1-bit build —
// the binary sibling of the PrismML Q2_0 ternary type (#4871): one
// little-endian f16 scale followed by 128 contiguous 1-bit codes, eight
// low-to-high codes per byte. Code cardinality is 2: 0=-1, 1=+1.
func TestQ1_0GoldenReferenceBlock(t *testing.T) {
	if got := TensorQ1_0.String(); got != "Q1_0" {
		t.Fatalf("TensorQ1_0.String() = %q, want Q1_0", got)
	}

	raw := make([]byte, blockQ1_0Bytes)
	binary.LittleEndian.PutUint16(raw[:2], 0x3800) // IEEE f16 0.5
	for i := 0; i < 16; i++ {
		raw[2+i] = 0x5a // low-to-high bits 0,1,0,1,1,0,1,0
	}

	tensor := TensorInfo{Name: "golden_q1_0", Dims: []uint64{128}, Type: TensorQ1_0}
	payload, err := tensorPayloadBytes(tensor)
	if err != nil {
		t.Fatalf("tensorPayloadBytes: %v", err)
	}
	if payload != blockQ1_0Bytes {
		t.Fatalf("payload = %d, want %d", payload, blockQ1_0Bytes)
	}

	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("DequantF32: %v", err)
	}
	want := []float32{-0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5}
	for i, v := range got {
		if math.Abs(float64(v-want[i%len(want)])) > 1e-6 {
			t.Fatalf("value[%d] = %g, want %g", i, v, want[i%len(want)])
		}
	}
}

// TestQ1_0RoundTripTwoBlocks packs a deterministic reference sign vector into
// two Q1_0 blocks with distinct f16 scales and checks the dequant reproduces
// sign*scale exactly — a quantize→dequantize round trip that also exercises
// the per-block base arithmetic (block 1 must read its own scale and codes).
func TestQ1_0RoundTripTwoBlocks(t *testing.T) {
	const elems = 256
	scales := []float32{0.5, 1.5} // exact in f16
	scaleBits := []uint16{0x3800, 0x3e00}

	signs := make([]int, elems)
	for i := range signs {
		if (i*7+3)%5 < 2 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}

	raw := make([]byte, 2*blockQ1_0Bytes)
	for block := 0; block < 2; block++ {
		base := block * blockQ1_0Bytes
		binary.LittleEndian.PutUint16(raw[base:base+2], scaleBits[block])
		for j := 0; j < 128; j++ {
			if signs[block*128+j] > 0 {
				raw[base+2+j/8] |= 1 << uint(j%8)
			}
		}
	}

	tensor := TensorInfo{Name: "roundtrip_q1_0", Dims: []uint64{elems}, Type: TensorQ1_0}
	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("DequantF32: %v", err)
	}
	if len(got) != elems {
		t.Fatalf("len(got) = %d, want %d", len(got), elems)
	}
	for i, v := range got {
		want := float32(signs[i]) * scales[i/128]
		if v != want {
			t.Fatalf("value[%d] = %g, want %g", i, v, want)
		}
	}
}

func TestQ1_0RejectsPartialBlock(t *testing.T) {
	tensor := TensorInfo{Name: "partial_q1_0", Dims: []uint64{127}, Type: TensorQ1_0}
	if _, err := tensorPayloadBytes(tensor); err == nil {
		t.Fatal("tensorPayloadBytes accepted a partial Q1_0 block")
	}
}

// bonsaiQ1_0Fixture builds two real Q1_0 (g128) blocks and the f32 values they
// must dequantize to. Each block is a little-endian f16 scale followed by 16
// bytes of 1-bit codes, eight low-to-high codes per byte; code 1 = +1, code
// 0 = -1, so y = (2*code-1)*d. Both scales are exactly representable in f16, so
// the reference is compared without tolerance drift, and they differ between the
// blocks so a dequant that ignored the per-block scale stride would fail.
func bonsaiQ1_0Fixture() ([]byte, []float32) {
	scaleBits := []uint16{0x3800, 0x3e00} // f16 0.5, 1.5
	scales := []float32{0.5, 1.5}
	raw := make([]byte, 2*blockQ1_0Bytes)
	want := make([]float32, 2*128)
	for block := 0; block < 2; block++ {
		base := block * blockQ1_0Bytes
		binary.LittleEndian.PutUint16(raw[base:base+2], scaleBits[block])
		for j := 0; j < 128; j++ {
			sign := float32(-1)
			if (j+block)%3 != 0 { // deterministic mixed signs, not an all-ones block
				raw[base+2+j/8] |= 1 << uint(j%8)
				sign = 1
			}
			want[block*128+j] = sign * scales[block]
		}
	}
	return raw, want
}

// TestQ1_0BonsaiGGUFOpensAndDequantizes closes the FILE-level half of #4871's
// done condition — "a Bonsai 1-bit GGUF opens AND a golden Q1_0_g128 block
// dequantizes to f32 matching reference". The block-level tests above call
// dequantF32 on a hand-built TensorInfo, which never proves the loader admits
// ggml type 43: the header/tensor-directory parse, the tensorPayloadBytes sizing
// that fixes the tensor's on-disk extent, and the file bounds check all sit
// outside that in-memory path. This drives the real reader end to end —
// OpenWeights on a Bonsai-arch GGUF file, then TensorF32 via TensorBytes.
func TestQ1_0BonsaiGGUFOpensAndDequantizes(t *testing.T) {
	const (
		elems = 256 // two g128 blocks, so the per-block scale stride is exercised
		name  = "blk.0.ffn_down.weight"
	)
	raw, want := bonsaiQ1_0Fixture()

	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 2)
	writeKVString(&b, "general.architecture", "bonsai")
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, name, []uint64{elems}, TensorQ1_0, 0)
	padToAlignment(&b, 32)
	b.Write(raw)

	path := filepath.Join(t.TempDir(), "bonsai-27b-1bit.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights(Bonsai 1-bit GGUF): %v", err)
	}
	defer ws.Close()

	// The parsed directory must carry the tensor as Q1_0 — the loader admitting
	// type 43 rather than dropping it or refusing the file.
	info, ok := ws.Tensor(name)
	if !ok {
		t.Fatal("Q1_0 tensor missing from the opened file's tensor directory")
	}
	if info.Type != TensorQ1_0 {
		t.Fatalf("tensor type = %s (%d), want Q1_0 (%d)", info.Type, info.Type, TensorQ1_0)
	}

	// The on-disk extent is sized by the loader's own tensorPayloadBytes here,
	// not by the test's arithmetic: a wrong block size would overrun the file.
	payload, _, err := ws.TensorBytes(name)
	if err != nil {
		t.Fatalf("TensorBytes: %v", err)
	}
	if wantBytes := elems / 128 * blockQ1_0Bytes; len(payload) != wantBytes {
		t.Fatalf("payload = %d bytes, want %d (%d blocks x %d B)",
			len(payload), wantBytes, elems/128, blockQ1_0Bytes)
	}

	got, _, err := ws.TensorF32(name)
	if err != nil {
		t.Fatalf("TensorF32: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("value[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}
