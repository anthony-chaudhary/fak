package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestV41CompressorRingGeometryRefusals(t *testing.T) {
	row := NewF32(Default(), []int{4}, make([]float32, 4))
	ring := NewF32(Default(), []int{2, 4}, make([]float32, 8))
	other := NewF32(Default(), []int{2, 4}, make([]float32, 8))
	for name, err := range map[string]error{
		"ratio one":     ValidateV41CompressorRingAlloc(1, 4),
		"ratio cap":     ValidateV41CompressorRingAlloc(V41CompressorPoolMaxRatio+1, 4),
		"zero width":    ValidateV41CompressorRingAlloc(2, 0),
		"index range":   ValidateV41CompressorRingAlloc(V41CompressorPoolMaxRatio, math.MaxInt32),
		"slot high":     ValidateV41CompressorRingStore(ring, other, row, row, 2, 2, 4),
		"slot negative": ValidateV41CompressorRingStore(ring, other, row, row, -1, 2, 4),
		"short ring":    ValidateV41CompressorRingStore(row, other, row, row, 0, 2, 4),
		"short row":     ValidateV41CompressorRingStore(ring, other, NewF32(Default(), []int{3}, make([]float32, 3)), row, 0, 2, 4),
		"aliased rings": ValidateV41CompressorRingStore(ring, ring, row, row, 0, 2, 4),
	} {
		if !errors.Is(err, ErrV41CompressorRingGeometry) {
			t.Errorf("%s: got %v, want ErrV41CompressorRingGeometry", name, err)
		}
	}
	if err := ValidateV41CompressorRingStore(ring, other, row, row, 1, 2, 4); err != nil {
		t.Fatalf("valid store refused: %v", err)
	}
}

// fak-test:runtime fast est=5ms
func TestDecodeV41CompressorRingStatus(t *testing.T) {
	pooled := []float32{1, -2, 0.5}
	values := append(append([]float32(nil), pooled...), 0, 0, 0)
	got, err := DecodeV41CompressorRing(values, 3)
	if err != nil || len(got) != 3 || got[0] != 1 || got[1] != -2 || got[2] != 0.5 {
		t.Fatalf("clean status = %v, %v", got, err)
	}
	got[0] = 9
	if values[0] != 1 {
		t.Fatal("decoded row aliases the readback")
	}
	for status, dim := range map[float32]int{V41CompressorRingKVNonFinite: 1, V41CompressorRingScoreNonFinite: 2, V41CompressorRingKVNonFinite | V41CompressorRingScoreNonFinite: 0} {
		bad := append([]float32(nil), values...)
		bad[3+dim] = status
		_, err := DecodeV41CompressorRing(bad, 3)
		var typed *V41CompressorRingNonFiniteError
		if !errors.Is(err, ErrV41CompressorRingNonFinite) || !errors.As(err, &typed) || typed.Dim != dim || typed.Status != int(status) {
			t.Errorf("status %v at %d: got %v", status, dim, err)
		}
	}
	for name, bad := range map[string][]float32{
		"unknown bit": {1, 2, 3, 0, 4, 0},
		"fraction":    {1, 2, 3, 0.5, 0, 0},
		"nan word":    {1, 2, 3, float32(math.NaN()), 0, 0},
		"short":       {1, 2, 3, 0, 0},
	} {
		if _, err := DecodeV41CompressorRing(bad, 3); !errors.Is(err, ErrV41CompressorRingStatus) {
			t.Errorf("%s: got %v, want ErrV41CompressorRingStatus", name, err)
		}
	}
}

// v41CompressorPoolFloat64Oracle is the binary64 gated pool every device
// class is pinned to: per dimension, a max-shifted softmax over the ratio rows
// of scores gates kv.
func v41CompressorPoolFloat64Oracle(kv, scores []float32, ratio, width int) []float64 {
	out := make([]float64, width)
	for d := 0; d < width; d++ {
		maxS := math.Inf(-1)
		for r := 0; r < ratio; r++ {
			maxS = math.Max(maxS, float64(scores[r*width+d]))
		}
		var den, acc float64
		for r := 0; r < ratio; r++ {
			w := math.Exp(float64(scores[r*width+d]) - maxS)
			den += w
			acc += float64(kv[r*width+d]) * w
		}
		out[d] = acc / den
	}
	return out
}

// fak-test:runtime fast est=5ms
func TestV41CompressorPoolFloat64OracleClosedForms(t *testing.T) {
	const ratio, width = 4, 3
	kv := []float32{1, -2, 8, 3, 4, 8, 5, -6, 8, 7, 8, 8}
	equal := make([]float32, ratio*width)
	got := v41CompressorPoolFloat64Oracle(kv, equal, ratio, width)
	for d, want := range []float64{4, 1, 8} {
		if math.Abs(got[d]-want) > 1e-12 {
			t.Fatalf("equal scores dim %d = %g, want mean %g", d, got[d], want)
		}
	}
	dominant := make([]float32, ratio*width)
	for d := 0; d < width; d++ {
		dominant[2*width+d] = 1000
	}
	got = v41CompressorPoolFloat64Oracle(kv, dominant, ratio, width)
	for d := 0; d < width; d++ {
		if got[d] != float64(kv[2*width+d]) {
			t.Fatalf("dominant row dim %d = %g, want %g", d, got[d], kv[2*width+d])
		}
	}
}

// fak-test:runtime fast est=5ms
func TestV41CompressorRingStatusRowRule(t *testing.T) {
	if err := v41CompressorRingStatusErr([]float32{0, 0, 0}); err != nil {
		t.Fatalf("clean status row refused: %v", err)
	}
	var typed *V41CompressorRingNonFiniteError
	if err := v41CompressorRingStatusErr([]float32{0, 3, 1}); !errors.As(err, &typed) || typed.Dim != 1 || typed.Status != 3 {
		t.Fatalf("first non-finite dimension: got %v", err)
	}
	if err := v41CompressorRingStatusErr([]float32{0, 4}); !errors.Is(err, ErrV41CompressorRingStatus) {
		t.Fatalf("unknown status bit: got %v", err)
	}
}
