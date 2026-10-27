//go:build cuda && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestCUDAKQuantPanelMatchesDecode holds each token to the unchanged tile-one
// CUDA arithmetic, not a new approximation threshold. The existing raw k-quant
// reference test separately qualifies decode against cpuref. This witness covers
// both partial eight-row output blocks and the four-token dispatch/tail boundary.
// Require mode must not turn an absent device into a green acceptance skip.
//
// fak-test:runtime integration est=5s lane=optin
func TestCUDAKQuantPanelMatchesDecode(t *testing.T) {
	cb, ok := Pick("cuda").(*cudaBackend)
	if !ok {
		if os.Getenv("FAK_REQUIRE_CUDA_KQUANT_PANEL") == "1" {
			t.Fatal("native CUDA backend unavailable")
		}
		t.Skip("native CUDA backend unavailable")
	}
	ref := Default()
	const out, in, rows = 9, 5120, 129
	x := make([]float32, rows*in)
	for token := 0; token < rows; token++ {
		for k := 0; k < in; k++ {
			x[token*in+k] = float32(((token+3)*(k+7))%79-39) / 79
		}
	}
	for _, tc := range []struct {
		dt        Dtype
		blockSize int
		newTensor func(Backend, []int, []byte) Tensor
	}{
		{Q5_K, 176, NewQ5K},
		{Q6_K, 210, NewQ6K},
	} {
		t.Run(tc.dt.String(), func(t *testing.T) {
			raw := make([]byte, out*(in/256)*tc.blockSize)
			for i := range raw {
				raw[i] = byte((i*37 + 11) % 251)
			}
			for b := 0; b < len(raw); b += tc.blockSize {
				if tc.dt == Q5_K {
					binary.LittleEndian.PutUint16(raw[b:], 0x3000)
					binary.LittleEndian.PutUint16(raw[b+2:], 0x2c00)
				} else {
					binary.LittleEndian.PutUint16(raw[b+208:], 0x3000)
				}
			}
			dw := cb.Upload(tc.newTensor(ref, []int{out, in}, raw), tc.dt)
			defer cb.Free(dw)
			want := make([]float32, rows*out)
			for token := 0; token < rows; token++ {
				dx := cb.Upload(NewF32(ref, []int{in}, x[token*in:(token+1)*in]), F32)
				dy := cb.MatMul(dw, dx)
				row := cb.Read(dy)
				cb.Free(dy)
				cb.Free(dx)
				if len(row) != out {
					t.Fatalf("decode row length=%d want=%d", len(row), out)
				}
				copy(want[token*out:], row)
			}
			for _, p := range []int{1, 3, 4, 5, 128, rows} {
				t.Run(fmt.Sprintf("P%d", p), func(t *testing.T) {
					dx := cb.Upload(NewF32(ref, []int{p, in}, x[:p*in]), F32)
					defer cb.Free(dx)
					dy := cb.BatchedMatMul(dw, dx, p)
					defer cb.Free(dy)
					got := cb.Read(dy) // Synchronize before any equality claim.
					if len(got) != p*out {
						t.Fatalf("panel length=%d want=%d", len(got), p*out)
					}
					for i, value := range got {
						if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value) != math.Float32bits(want[i]) {
							t.Fatalf("token=%d output=%d panel=%g (%08x) decode=%g (%08x)", i/out, i%out, value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
						}
					}
				})
			}
		})
	}
}
