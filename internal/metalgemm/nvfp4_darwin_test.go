//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"slices"
	"strings"
	"testing"
)

// TestNVFP4M5GEMVCandidateMatchesIndependentOracle is the source-grounded
// first witness. It calls the kernel directly so non-M5 Apple builders can
// validate its arithmetic, while the public selector remains M5-only.
func TestNVFP4M5GEMVCandidateMatchesIndependentOracle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetNVFP4()

	const out, in = 5, 64
	packed := make([]byte, out*in/2)
	for i := range packed {
		lo := byte((i*7 + 1) & 15)
		hi := byte((i*11 + 5) & 15)
		packed[i] = lo | hi<<4
	}
	// Include subnormal, unity, negative (abs-required), and maximum finite scales.
	scaleCodes := []byte{0x01, 0x38, 0xbc, 0x7e}
	scales := make([]byte, out*in/NVFP4BlockWeights)
	for i := range scales {
		scales[i] = scaleCodes[i%len(scaleCodes)]
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*29)%61-30) / 31
	}

	w := uploadNVFP4(packed, scales, out, in)
	if w == nil {
		t.Fatal("candidate upload failed")
	}
	got := make([]float32, out)
	if executed := w.GEMV(x, got); executed != NVFP4ExecutedM5GEMV {
		t.Fatalf("executed=%v, want M5 NVFP4 GEMV", executed)
	}

	// Deliberately independent from nvfp4Reference and its decode tables.
	decodeWeight := func(code byte) float32 {
		magnitude := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6}
		v := magnitude[code&7]
		if code&8 != 0 {
			return -v
		}
		return v
	}
	decodeScale := func(raw byte) float32 {
		mag := raw & 0x7f
		exponent, mantissa := mag>>3, mag&7
		var v float64
		if exponent == 0 {
			v = float64(mantissa) / 512
		} else {
			v = math.Ldexp(float64(8+mantissa), int(exponent)-10)
		}
		return float32(v) // sign intentionally omitted: the kernel requires abs(scale).
	}
	want := make([]float32, out)
	blocks := in / NVFP4BlockWeights
	for row := 0; row < out; row++ {
		var sum float32
		for k := 0; k < in; k++ {
			pair := packed[(row*in+k)/2]
			code := pair & 15
			if k&1 != 0 {
				code = pair >> 4
			}
			sum += decodeWeight(code) * decodeScale(scales[row*blocks+k/NVFP4BlockWeights]) * x[k]
		}
		want[row] = sum
	}
	for i := range want {
		delta := math.Abs(float64(got[i] - want[i]))
		limit := 2e-5 * math.Max(1, math.Abs(float64(want[i])))
		if delta > limit {
			t.Fatalf("row %d = %g, independent oracle %g (delta %g > %g)", i, got[i], want[i], delta, limit)
		}
	}

	if !strings.Contains(DeviceName(), "M5") && UploadNVFP4(packed, scales, out, in) != nil {
		t.Fatalf("public selector widened candidate beyond M5 device %q", DeviceName())
	}
	untouched := []float32{9936, 9936, 9936, 9936, 9936}
	before := slices.Clone(untouched)
	if executed := w.GEMV(x[:in-1], untouched); executed != NVFP4NotExecuted || !slices.Equal(untouched, before) {
		t.Fatalf("invalid M!=1/shape call did not fail closed: executed=%v output=%v", executed, untouched)
	}
}
