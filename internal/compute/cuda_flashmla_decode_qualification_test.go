//go:build cuda && cgo && cudaarchspecific && fak_cuda_flashmla_decode

package compute

import (
	"math"
	"os"
	"testing"
)

// Real selected Go/cgo/archive/device witness. Never enabled in normal tests.
// Missing acceptance opt-in is a failure; record exact archive/device pins.
func TestFlashMLADecodeOwnedPhysical(t *testing.T) {
	if os.Getenv("FAK_FLASHMLA_DECODE_ACCEPT") != "1" {
		t.Fatal("selected qualification profile requires FAK_FLASHMLA_DECODE_ACCEPT=1")
	}
	if cudaDev == nil {
		t.Fatal("no actual CUDA backend")
	}
	for _, topk := range []int{640, 768} {
		for _, invariant := range []bool{false, true} {
			x := flashDecodeFixture()
			x.Indices = make([]int32, topk)
			x.Length = int32(topk)
			x.Sink = make([]float32, 64)
			x.BatchInvariant = invariant
			for d := 0; d < 512; d++ {
				x.KV[d] = 0x38
			}
			y, err := cudaDev.qualifyFlashMLADecode(x)
			if err != nil {
				t.Fatal(err)
			}
			want := float64(topk) / float64(topk+1)
			for _, v := range y.ValuesBF16 {
				if math.Abs(float64(math.Float32frombits(uint32(v)<<16))-want) > 0.02 {
					t.Fatal("sink-normalized values")
				}
			}
			for _, v := range y.LSE {
				if math.Abs(float64(v)-math.Log(float64(topk))) > 0.02 {
					t.Fatal("sink-excluded LSE")
				}
			}
			for i := range x.Indices {
				x.Indices[i] = -1
			}
			if _, err := cudaDev.qualifyFlashMLADecode(x); err != nil {
				t.Fatal("empty donor contract", err)
			}
		}
	}
}
