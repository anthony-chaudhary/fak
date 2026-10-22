package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4CPUOracle(t *testing.T) {
	t.Parallel()
	// One full codebook in each block. A second row proves scale indexing is
	// row-major, not a reused first-row scale. Expected values are independent
	// constants, not values obtained from the implementation's decoder.
	m := NVFP4Matrix{Rows: 2, Cols: 32, Packed: make([]byte, 32), BlockScales: []byte{0x38, 0x40, 0x01, 0x7e}, GlobalScale: .5}
	for i := range m.Packed {
		m.Packed[i] = byte((2*i)%16 | ((2*i+1)%16)<<4)
	}
	codebook := []float32{0, .5, 1, 1.5, 2, 3, 4, 6, math.Float32frombits(0x80000000), -.5, -1, -1.5, -2, -3, -4, -6}
	blockMultipliers := []float32{.5, 1, 1.0 / 1024, 224}
	got, err := m.Dequantize()
	if err != nil || len(got) != 64 {
		t.Fatalf("decode: len=%d err=%v", len(got), err)
	}
	for i, v := range got {
		want := codebook[i%16] * blockMultipliers[i/16]
		if math.Float32bits(v) != math.Float32bits(want) {
			t.Fatalf("element %d: bits=%08x want=%08x", i, math.Float32bits(v), math.Float32bits(want))
		}
	}
	x := make([]float32, 32)
	for i := range x {
		if i%16 < 8 {
			x[i] = 1
		}
	}
	y, err := m.GEMV(x)
	// Positive magnitudes sum to 18. Distinct block/global scales must apply.
	if err != nil || len(y) != 2 || y[0] != 27 || y[1] != 4032.017578125 {
		t.Fatalf("GEMV=%v err=%v", y, err)
	}
	for _, global := range []float32{0, math.SmallestNonzeroFloat32} {
		z := m
		z.GlobalScale = global
		v, err := z.Dequantize()
		if err != nil || math.Float32bits(v[8]) != 0x80000000 || math.Float32bits(v[0]) != 0 {
			t.Fatalf("signed zero global=%g: %v err=%v", global, v, err)
		}
		// Replacing the global .5 with 2^-149 leaves block zero's scale at 1:
		// magnitude 1 gives 2^-149, magnitude 2 gives 2^-148. Index 33 instead gives
		// .5 * (1/512) * 2^-149 = 2^-159, which rounds to positive zero.
		if global != 0 && (math.Float32bits(v[2]) != 1 || math.Float32bits(v[4]) != 2 || math.Float32bits(v[33]) != 0) {
			t.Fatalf("subnormal/underflow: %v", v)
		}
	}
	bad := []struct {
		name string
		edit func(*NVFP4Matrix)
	}{
		{"zero rows", func(m *NVFP4Matrix) { m.Rows = 0 }},
		{"negative columns", func(m *NVFP4Matrix) { m.Cols = -16 }},
		{"partial block", func(m *NVFP4Matrix) { m.Cols = 17 }},
		{"shape overflow", func(m *NVFP4Matrix) { m.Rows = int(^uint(0) >> 1) }},
		{"short payload", func(m *NVFP4Matrix) { m.Packed = m.Packed[:31] }},
		{"extra scale", func(m *NVFP4Matrix) { m.BlockScales = append(m.BlockScales, 0) }},
		{"scale NaN", func(m *NVFP4Matrix) { m.BlockScales[3] = 0x7f }},
		{"scale negative", func(m *NVFP4Matrix) { m.BlockScales[3] = 0xb8 }},
		{"scale negative zero", func(m *NVFP4Matrix) { m.BlockScales[3] = 0x80 }},
		{"global NaN", func(m *NVFP4Matrix) { m.GlobalScale = float32(math.NaN()) }},
		{"global infinity", func(m *NVFP4Matrix) { m.GlobalScale = float32(math.Inf(1)) }},
		{"global negative", func(m *NVFP4Matrix) { m.GlobalScale = -1 }},
		{"global negative zero", func(m *NVFP4Matrix) { m.GlobalScale = math.Float32frombits(0x80000000) }},
		{"decoded overflow", func(m *NVFP4Matrix) { m.GlobalScale = math.MaxFloat32 }},
	}
	for _, tt := range bad {
		t.Run(tt.name, func(t *testing.T) {
			b := m
			b.BlockScales = append([]byte(nil), m.BlockScales...)
			tt.edit(&b)
			if v, err := b.Dequantize(); err == nil || v != nil {
				t.Fatalf("decode accepted invalid matrix: %v err=%v", v, err)
			}
			if v, err := b.GEMV(x); err == nil || v != nil {
				t.Fatalf("GEMV accepted invalid matrix: %v err=%v", v, err)
			}
		})
	}
	productOverflow := append([]float32(nil), x...)
	productOverflow[6] = math.MaxFloat32
	sumOverflow := append([]float32(nil), x...)
	sumOverflow[2], sumOverflow[3] = math.MaxFloat32, math.MaxFloat32
	for _, vector := range [][]float32{x[:31], append([]float32{float32(math.NaN())}, x[1:]...), productOverflow, sumOverflow} {
		if v, err := m.GEMV(vector); err == nil || v != nil {
			t.Fatalf("GEMV accepted invalid/overflowing vector: %v err=%v", v, err)
		}
	}
	round := NVFP4Matrix{Rows: 1, Cols: 16, Packed: []byte{0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22}, BlockScales: []byte{0x38}, GlobalScale: 1}
	rx := make([]float32, 16)
	rx[0], rx[1], rx[2] = 16777216, 1, -16777216
	if v, err := round.GEMV(rx); err != nil || len(v) != 1 || v[0] != 0 {
		t.Fatalf("GEMV must round each FP32 addition: %v err=%v", v, err)
	}
}
