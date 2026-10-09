package model

import (
	"errors"
	"math"
	"testing"
)

// The literals distinguish the BF16 projection, F32 wrapped-scalar multiply,
// and BF16 result boundaries in the explicit PyTorch v2.9.0 CUDA profile.
// They are not runtime parity or FP8/FP4 qualification. Estimate is unmeasured.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerWeightsBF16(t *testing.T) {
	t.Parallel()
	checkBits := func(t *testing.T, got []float32, want []uint32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("width %d, want %d", len(got), len(want))
		}
		for i, value := range got {
			if bits := math.Float32bits(value); bits != want[i] {
				t.Fatalf("element %d bits=%08x, want %08x", i, bits, want[i])
			}
		}
	}
	ones, scaled := make([]uint32, 32), make([]uint32, 32)
	for i := range ones {
		ones[i], scaled[i] = 0x3f800000, 0x3c800000
	}
	for _, tc := range []struct {
		name      string
		dim       int
		projected []uint32
		want      []uint32
	}{
		{"128-by-32-scale", 128, ones, scaled},
		// 265/256 rounds to even 33/32 at projection. Its product with
		// F32(1/sqrt(3)) rounds to 0.59375. Skipping the first cast or
		// casting the scalar to BF16 first would instead give 0.59765625.
		{"projection-and-opmath-order", 3, []uint32{0x3f848000}, []uint32{0x3f180000}},
		{"projection-ties-and-signed-zero", 1,
			[]uint32{0x3f808000, 0x3f818000, 0xbf808000, 0x80000000},
			[]uint32{0x3f000000, 0x3f020000, 0xbf000000, 0x80000000}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float32, len(tc.projected))
			for i, bits := range tc.projected {
				input[i] = math.Float32frombits(bits)
			}
			got, err := v41IndexWeightsBF16(3, input, tc.dim, len(input))
			if err != nil {
				t.Fatal(err)
			}
			checkBits(t, got, tc.want)
			checkBits(t, input, tc.projected)
			got[0] = 0
			checkBits(t, input, tc.projected)
		})
	}
	t.Run("failure-is-atomic-before-scaling", func(t *testing.T) {
		// Finite F32 max would survive the small scalar if projection
		// copyback were skipped. The NaN payload would wrap to zero if
		// bit rounding happened before validation.
		for _, bad := range []uint32{0x7fffffff, 0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff} {
			input := []float32{math.Float32frombits(0x3f808000), math.Float32frombits(bad)}
			got, err := v41IndexWeightsBF16(3, input, 128, 2)
			var operation *V41ProjectionOperationError
			if got != nil || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &operation) || operation.Layer != 3 || operation.Leaf != "indexer.weights_proj.weight" || operation.Stage != string(v41StageIndexer) {
				t.Fatalf("bits=%08x lost atomic indexer error: got=%v err=%v", bad, got, err)
			}
			checkBits(t, input, []uint32{0x3f808000, bad})
		}
	})
	t.Run("geometry-is-checked", func(t *testing.T) {
		input := []float32{1}
		got, err := v41IndexWeightsBF16(3, input, 128, 2)
		var operation *V41ProjectionOperationError
		if got != nil || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &operation) || operation.Leaf != "indexer.weights_proj.weight" {
			t.Fatalf("invalid weight geometry was accepted: got=%v err=%v", got, err)
		}
		checkBits(t, input, []uint32{0x3f800000})
	})
}
