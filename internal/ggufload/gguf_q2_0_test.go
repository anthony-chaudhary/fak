package ggufload

import (
	"encoding/binary"
	"math"
	"testing"
)

// This block is encoded according to PrismML's llama.cpp Q2_0 reference
// (fidan/q2_0-b9587, ggml type 42): one little-endian f16 scale followed by
// 128 contiguous 2-bit codes, four low-to-high codes per byte. Codes map as
// 00=-1, 01=0, 10=+1, 11=+2.
func TestQ2_0GoldenReferenceBlock(t *testing.T) {
	if got := TensorQ2_0.String(); got != "Q2_0" {
		t.Fatalf("TensorQ2_0.String() = %q, want Q2_0", got)
	}

	raw := make([]byte, blockQ2_0Bytes)
	binary.LittleEndian.PutUint16(raw[:2], 0x3800) // IEEE f16 0.5
	for i := 0; i < 32; i++ {
		raw[2+i] = 0xe4 // low-to-high codes 00,01,10,11
	}

	tensor := TensorInfo{Name: "golden_q2_0", Dims: []uint64{128}, Type: TensorQ2_0}
	payload, err := tensorPayloadBytes(tensor)
	if err != nil {
		t.Fatalf("tensorPayloadBytes: %v", err)
	}
	if payload != blockQ2_0Bytes {
		t.Fatalf("payload = %d, want %d", payload, blockQ2_0Bytes)
	}

	got, err := dequantF32(tensor, raw)
	if err != nil {
		t.Fatalf("DequantF32: %v", err)
	}
	want := []float32{-0.5, 0, 0.5, 1}
	for i, v := range got {
		if math.Abs(float64(v-want[i%len(want)])) > 1e-6 {
			t.Fatalf("value[%d] = %g, want %g", i, v, want[i%len(want)])
		}
	}
}

func TestQ2_0RejectsPartialBlock(t *testing.T) {
	tensor := TensorInfo{Name: "partial_q2_0", Dims: []uint64{127}, Type: TensorQ2_0}
	if _, err := tensorPayloadBytes(tensor); err == nil {
		t.Fatal("tensorPayloadBytes accepted a partial Q2_0 block")
	}
}
