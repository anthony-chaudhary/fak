package model

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41MHCProjectorBorrowsFlatInput(t *testing.T) {
	const H = 33
	const width = 4 * H
	input := make([]float32, width)
	for i := range input {
		input[i] = float32(i%17-8) / 9
	}
	inputBefore := append([]float32(nil), input...)
	logical := make([]float32, v41MHCMixWidth*width)
	for i := range logical {
		logical[i] = float32(i%23-11) / 13
	}
	var ss float32
	for _, v := range input {
		ss += v * v
	}
	rsqrt := float32(1 / math.Sqrt(float64(ss/float32(width)+1e-6)))
	want := make([]float32, v41MHCMixWidth)
	for i := range want {
		// The optimization must retain fdot's exact reduction, not replace it
		// with a serial accumulator or move the RMS scale into the input.
		want[i] = fdot(logical[i*width:(i+1)*width], input) * rsqrt
	}
	for _, leaf := range []string{"mhc.mixes.weight", "mhc.ffn_mixes.weight"} {
		for _, transposed := range []bool{false, true} {
			name := leaf + "/logical"
			if transposed {
				name = leaf + "/transposed"
			}
			t.Run(name, func(t *testing.T) {
				weights := append([]float32(nil), logical...)
				shape := []int{v41MHCMixWidth, width}
				if transposed {
					shape = []int{width, v41MHCMixWidth}
					for out := 0; out < v41MHCMixWidth; out++ {
						for in := 0; in < width; in++ {
							weights[in*v41MHCMixWidth+out] = logical[out*width+in]
						}
					}
				}
				raw := make([]byte, 4*len(weights))
				for i, v := range weights {
					binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
				}
				m := &Model{manifest: map[string]tensorMeta{layerName(0, leaf): {Dtype: "f32", Shape: shape, Nbytes: len(raw)}}, raw: raw}
				project := m.v41MHCProjectorNamed(0, leaf, H, 1e-6, true, transposed, &v41ProjScratch{})
				var got []float32
				var err error
				allocs := testing.AllocsPerRun(5, func() { got, err = project(input) })
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("projection = %v, %v; want %v", got, err, want)
				}
				limit := float64(1) // owned coefficients only; no copied 4H input
				if transposed {
					limit++ // existing transposed-weight column scratch
				}
				if allocs > limit {
					t.Fatalf("steady-state allocations = %g, want <= %g without re-flattening", allocs, limit)
				}
				got[0]++
				if !reflect.DeepEqual(input, inputBefore) || !reflect.DeepEqual(m.tensor(layerName(0, leaf)), weights) {
					t.Fatal("projection mutated borrowed input or model weights")
				}
			})
		}
	}
}
