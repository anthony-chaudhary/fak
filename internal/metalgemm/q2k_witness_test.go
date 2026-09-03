// q2k_witness_test.go — pure-Go witness tests for Q2_K wire format and dequant arithmetic.
// Runs in every build without requiring a Metal GPU.

package metalgemm

import (
	"testing"
)

func TestQ2KFormatConstantsAndSizing(t *testing.T) {
	if Q2KBlockWeights != 256 {
		t.Fatalf("Q2KBlockWeights = %d, want 256", Q2KBlockWeights)
	}
	if Q2KBlockBytes != 84 {
		t.Fatalf("Q2KBlockBytes = %d, want 84", Q2KBlockBytes)
	}
	for _, tc := range []struct {
		out, in, want int
	}{
		{0, 256, 0},
		{64, 0, 0},
		{64, 100, 0}, // not a multiple of 256
		{1, 256, 84},
		{4, 512, 4 * 2 * 84},
		{128, 1024, 128 * 4 * 84},
	} {
		if got := Q2KPayloadBytes(tc.out, tc.in); got != tc.want {
			t.Fatalf("Q2KPayloadBytes(%d, %d) = %d, want %d", tc.out, tc.in, got, tc.want)
		}
	}
}

func TestQ2KDequantBlockProducesFiniteNonZeroWeights(t *testing.T) {
	raw := q2kTestRaw(1, 256, 0x1234)
	weights := make([]float32, Q2KBlockWeights)
	q2kDequantBlock(weights, raw)

	nonZero := false
	for i, w := range weights {
		if w != 0 {
			nonZero = true
		}
		if w > 100.0 || w < -100.0 {
			t.Fatalf("weight[%d] = %v out of expected reasonable range", i, w)
		}
	}
	if !nonZero {
		t.Fatal("all dequantized weights were zero")
	}
}

func TestQ2KReferenceGEMVDeterministic(t *testing.T) {
	const out, in = 4, 512
	raw := q2kTestRaw(out, in, 0x5678)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i%19)-9) * 0.1
	}

	y1 := q2kReference(raw, out, in, x)
	y2 := q2kReference(raw, out, in, x)

	if len(y1) != out {
		t.Fatalf("len(y1) = %d, want %d", len(y1), out)
	}
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Fatalf("y1[%d] = %v != y2[%d] = %v", i, y1[i], i, y2[i])
		}
	}
}
