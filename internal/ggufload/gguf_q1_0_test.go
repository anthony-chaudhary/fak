package ggufload

import (
	"encoding/binary"
	"math"
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
