package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestV41CompressorNormSignedFiniteDomain(t *testing.T) {
	t.Parallel()
	for _, bits := range []uint32{0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff} {
		if !v41CompressorFinite32(math.Float32frombits(bits)) {
			t.Fatalf("finite value %08x rejected", bits)
		}
	}
	for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000} {
		bad := math.Float32frombits(bits)
		if v41CompressorFinite32(bad) {
			t.Fatalf("nonfinite value %08x accepted", bits)
		}
		for _, gainBad := range []bool{false, true} {
			pooled, gain := []float32{1, 1}, []float32{1, 1}
			if gainBad {
				gain[1] = bad
			} else {
				pooled[1] = bad
			}
			if _, err := V41CompressorNorm(pooled, gain, 1); !errors.Is(err, ErrV41CompressorNormDomain) {
				t.Fatalf("nonfinite %08x gain=%v: error=%v", bits, gainBad, err)
			}
		}
	}
	// Both squared inputs equal one, so eps=1 gives 1/sqrt(2). BF16
	// rounds its magnitude to 0x3f35; the negative input/gain set each sign.
	got, err := V41CompressorNorm([]float32{-1, 1}, []float32{1, -1}, 1)
	if err != nil || len(got) != 2 {
		t.Fatalf("signed row: %v, %v", got, err)
	}
	for i, value := range got {
		if bits := math.Float32bits(value); bits != 0xbf350000 {
			t.Fatalf("signed row[%d]=%08x, want bf350000", i, bits)
		}
	}
	got, err = V41CompressorNorm([]float32{math.Float32frombits(0x80000000), 0}, []float32{1, 1}, 1)
	if err != nil || len(got) != 2 || math.Float32bits(got[0]) != 0x80000000 || math.Float32bits(got[1]) != 0 {
		t.Fatalf("signed zero row: %v, %v", got, err)
	}
}
