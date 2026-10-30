//go:build cuda && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// Hardware pickup: packed residency, block alignment, output/token tails and
// real resident decode/prefill vs the existing independent CPU Q3_K decoder.
// fak-test:runtime integration est=3s lane=optin
func TestCUDAQ3KResidentMatMul(t *testing.T) {
	cb := cudaOrSkip(t)
	for _, p := range []int{1, 2, 4, 5, 129} {
		for _, in := range []int{256, 5120, 17408} {
			t.Run(fmt.Sprintf("p%d_in%d", p, in), func(t *testing.T) {
				const out = 9
				raw := make([]byte, out*(in/256)*110)
				seed := uint32(13991)
				for i := range raw {
					seed = 1664525*seed + 1013904223
					raw[i] = byte(seed >> 24)
				}
				for off := 0; off < len(raw); off += 110 {
					binary.LittleEndian.PutUint16(raw[off+108:], 0x2400)
				}
				host := NewQ3K(Default(), []int{out, in}, raw)
				w := cb.UploadClass(host, Q3_K, MemoryOffload, "q3k-acceptance")
				defer cb.Free(w)
				b, ok := w.buf.(*cudaBuf)
				if !ok || b.ptr == nil || b.n != len(raw) || w.Dtype != Q3_K || w.Quant.Bits != 3 {
					t.Fatalf("invalid packed resident descriptor: dtype=%v quant=%+v", w.Dtype, w.Quant)
				}
				x := make([]float32, p*in)
				for i := range x {
					x[i] = float32((i*17)%31-15) / 16
				}
				shape := []int{p, in}
				if p == 1 {
					shape = []int{in}
				}
				dx := cb.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "q3k-acceptance-input")
				defer cb.Free(dx)
				var y Tensor
				if p == 1 {
					y = cb.MatMul(w, dx)
				} else {
					y = cb.BatchedMatMul(w, dx, p)
				}
				defer cb.Free(y)
				got := cb.Read(y)
				weights := make([]float32, out*in)
				DequantQ3K(weights, raw)
				if len(got) != p*out {
					t.Fatalf("output length=%d", len(got))
				}
				for token := 0; token < p; token++ {
					for row := 0; row < out; row++ {
						var want, absolute float64
						for k := 0; k < in; k++ {
							v := float64(weights[row*in+k]) * float64(x[token*in+k])
							want += v
							absolute += math.Abs(v)
						}
						actual := float64(got[token*out+row])
						if math.IsNaN(actual) || math.IsInf(actual, 0) || math.Abs(actual-want) > 2e-5*math.Max(1, absolute) {
							t.Fatalf("token=%d row=%d got=%g want=%g absdot=%g", token, row, actual, want, absolute)
						}
					}
				}
			})
		}
	}
}
