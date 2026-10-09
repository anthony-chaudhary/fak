package model

import (
	"math"
	"testing"
)

// TestV41IndexerScoreRoundsEveryFloat32Operation pins the scalar reduction
// contract independently of whether this host compiler normally emits FMA.
// Go permits fusion across assignments; explicit conversions forbid skipping
// their rounding: https://go.dev/ref/spec#Floating_point_operators.
// These are hand-derived IEEE binary32 controls, not checkpoint measurements.
//
// fak-test:runtime fast est=1ms lane=default
func TestV41IndexerScoreRoundsEveryFloat32Operation(t *testing.T) {
	t.Parallel()

	// 2^24 + 1 rounds to 2^24 in binary32, so subtracting 2^24 gives zero.
	// A binary64 reduction retains the unit and would change the winning row.
	wide := float64(0x1p24)
	widened := float32(float64(wide+1) - wide)
	if bits := math.Float32bits(widened); bits != 0x3f800000 {
		t.Fatalf("binary64 control bits = %#08x, want 0x3f800000", bits)
	}

	// a*b = 1 + 2^-24 - 2^-47 rounds to 1 in binary32. Subtracting 1
	// after that rounding gives +0; fusion leaves 2^-24 - 2^-47. The exact
	// residual fits binary32, so this binary64 FMA also gives the F32 FMA
	// control without double rounding. Its positive sign survives ReLU.
	a := math.Float32frombits(0x3f800001) // 1 + 2^-23
	b := math.Float32frombits(0x3f7fffff) // 1 - 2^-24
	fused := float32(math.FMA(float64(a), float64(b), -1))
	if bits := math.Float32bits(fused); bits != 0x337ffffe {
		t.Fatalf("FMA control bits = %#08x, want 0x337ffffe", bits)
	}

	for _, tc := range []struct {
		name           string
		q, keys        []float32
		weights        []float32
		headDim        int
		wantBits       []uint32
		unroundedFirst float32
	}{
		{
			name:    "dot_sum_rounding",
			q:       []float32{0x1p24, 1, -0x1p24},
			keys:    []float32{1, 1, 1, 0, 0.5, 0},
			weights: []float32{1}, headDim: 3,
			wantBits: []uint32{0, 0x3f000000}, unroundedFirst: widened,
		},
		{
			name: "head_sum_rounding",
			q:    []float32{0x1p24, 1, 0x1p24}, keys: []float32{1},
			weights: []float32{1, 1, -1}, headDim: 1,
			wantBits: []uint32{0}, unroundedFirst: widened,
		},
		{
			name: "dot_product_rounding",
			q:    []float32{-1, a}, keys: []float32{1, b, -0x1p-25, 0},
			weights: []float32{1}, headDim: 2,
			wantBits: []uint32{0, 0x33000000}, unroundedFirst: fused,
		},
		{
			name: "head_product_rounding",
			q:    []float32{1, a}, keys: []float32{1},
			weights: []float32{-1, b}, headDim: 1,
			wantBits: []uint32{0}, unroundedFirst: fused,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if math.Float32bits(tc.unroundedFirst) == tc.wantBits[0] {
				t.Fatal("control does not distinguish omitted rounding")
			}
			scores, err := V41IndexerScore(tc.q, tc.keys, tc.weights, len(tc.weights), tc.headDim, len(tc.wantBits))
			if err != nil {
				t.Fatal(err)
			}
			if len(scores) != len(tc.wantBits) {
				t.Fatalf("got %d scores, want %d", len(scores), len(tc.wantBits))
			}
			for i, want := range tc.wantBits {
				if bits := math.Float32bits(scores[i]); bits != want {
					t.Fatalf("score[%d] bits = %#08x, want %#08x", i, bits, want)
				}
			}
			if len(scores) == 2 {
				rows, err := V41SelectIndexRows(scores, 2, 1, 0, nil)
				if err != nil || len(rows) != 1 || rows[0] != 1 {
					t.Fatalf("rounded scores select %v, err %v; want row 1", rows, err)
				}
				if tc.unroundedFirst <= scores[1] {
					t.Fatal("omitted-rounding control would not change the winning row")
				}
			}
		})
	}
}
