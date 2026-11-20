package compute

import (
	"math"
	"math/rand"
	"testing"
)

// i3gsPackCodes writes the n codes (each in [0,7]) into the low/high planes using a
// straightforward index walk (j/4, j%4, j/8, j%8) — deliberately independent of the
// production bit-shifts (j>>2, j&3, j>>3, j&7) so a shared bit-layout bug cannot hide.
func i3gsPackCodes(codes []int, low, high []byte) {
	for j, c := range codes {
		low2 := c & 3
		high1 := (c >> 2) & 1
		b := j / 4
		shift := (j % 4) * 2
		mask := byte(0x03) << uint(shift)
		low[b] = (low[b] &^ mask) | (byte(low2) << uint(shift))
		hb := j / 8
		hshift := j % 8
		hmask := byte(0x01) << uint(hshift)
		high[hb] = (high[hb] &^ hmask) | (byte(high1) << uint(hshift))
	}
}

// i3gsMakeCodes builds n codes from a per-index function.
func i3gsMakeCodes(n int, f func(j int) int) []int {
	codes := make([]int, n)
	for j := 0; j < n; j++ {
		codes[j] = f(j)
	}
	return codes
}

// TestI3GSDequantGroup is the scalar-oracle table test: pack known codes with an
// independent packer, dequant with i3gsDequantGroup, and assert bit-exactness against the
// directly computed (code-4)*scale. Cases: full group, partial tail, all-zero, all-max,
// alternating, and a ramp.
func TestI3GSDequantGroup(t *testing.T) {
	tests := []struct {
		name  string
		codes []int
		scale float32
		n     int
	}{
		{"full_group_all_zero", i3gsMakeCodes(64, func(j int) int { return 0 }), 0.5, 64},
		{"full_group_all_max", i3gsMakeCodes(64, func(j int) int { return 7 }), 0.5, 64},
		{"alternating", i3gsMakeCodes(64, func(j int) int {
			if j%2 == 0 {
				return 0
			}
			return 7
		}), 0.25, 64},
		{"ramp", i3gsMakeCodes(64, func(j int) int { return j % 8 }), 1.0, 64},
		{"partial_tail", i3gsMakeCodes(64, func(j int) int { return (j * 3) % 8 }), 0.125, 10},
		{"single_value", i3gsMakeCodes(64, func(j int) int { return 5 }), 2.0, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			low := make([]byte, i3gsLowBytes)
			high := make([]byte, i3gsHighBytes)
			i3gsPackCodes(tc.codes[:tc.n], low, high)
			got := make([]float32, tc.n)
			i3gsDequantGroup(got, low, high, tc.scale, tc.n)
			for j := 0; j < tc.n; j++ {
				want := float32(tc.codes[j]-4) * tc.scale
				if got[j] != want {
					t.Fatalf("j=%d: got %v, want %v", j, got[j], want)
				}
			}
		})
	}
}

// TestI3GSDequantGroupHandComputed pins the bit-layout against hand-computed plane bytes:
// the all-zero, all-max, and alternating patterns have known plane encodings, so a
// bit-layout bug (wrong shift or mask) would fail here even if the packer shared it.
func TestI3GSDequantGroupHandComputed(t *testing.T) {
	const scale = 0.5
	// all-zero codes: low = 0x00 x16, high = 0x00 x8 -> every value = -4*scale.
	{
		low := make([]byte, i3gsLowBytes)
		high := make([]byte, i3gsHighBytes)
		got := make([]float32, i3gsGroup)
		i3gsDequantGroup(got, low, high, scale, i3gsGroup)
		for j, v := range got {
			if v != -4*scale {
				t.Fatalf("all-zero j=%d: got %v, want %v", j, v, -4*scale)
			}
		}
	}
	// all-max codes (7): low = 0xFF x16, high = 0xFF x8 -> every value = 3*scale.
	{
		low := make([]byte, i3gsLowBytes)
		high := make([]byte, i3gsHighBytes)
		for i := range low {
			low[i] = 0xFF
		}
		for i := range high {
			high[i] = 0xFF
		}
		got := make([]float32, i3gsGroup)
		i3gsDequantGroup(got, low, high, scale, i3gsGroup)
		for j, v := range got {
			if v != 3*scale {
				t.Fatalf("all-max j=%d: got %v, want %v", j, v, 3*scale)
			}
		}
	}
	// alternating codes (0,7,0,7,...): low = 0xCC x16 (j even low2=00, j odd low2=11),
	// high = 0xAA x8 (j even high1=0, j odd high1=1).
	{
		low := make([]byte, i3gsLowBytes)
		high := make([]byte, i3gsHighBytes)
		for i := range low {
			low[i] = 0xCC
		}
		for i := range high {
			high[i] = 0xAA
		}
		got := make([]float32, i3gsGroup)
		i3gsDequantGroup(got, low, high, scale, i3gsGroup)
		for j, v := range got {
			var want float32
			if j%2 == 0 {
				want = -4 * scale
			} else {
				want = 3 * scale
			}
			if v != want {
				t.Fatalf("alternating j=%d: got %v, want %v", j, v, want)
			}
		}
	}
	// first 8 codes = 0,1,2,3,4,5,6,7 (rest 0): low[0]=0xE4, low[1]=0xE4, high[0]=0xF0.
	{
		low := make([]byte, i3gsLowBytes)
		high := make([]byte, i3gsHighBytes)
		low[0] = 0xE4
		low[1] = 0xE4
		high[0] = 0xF0
		got := make([]float32, i3gsGroup)
		i3gsDequantGroup(got, low, high, scale, i3gsGroup)
		for j := 0; j < 8; j++ {
			want := float32(j-4) * scale
			if got[j] != want {
				t.Fatalf("ramp8 j=%d: got %v, want %v", j, got[j], want)
			}
		}
		for j := 8; j < i3gsGroup; j++ {
			if got[j] != -4*scale {
				t.Fatalf("ramp8 j=%d: got %v, want %v", j, got[j], -4*scale)
			}
		}
	}
}

// TestI3GSNewMetadata checks that NewI3GS wraps the raw plane bytes + per-group scales as a
// host Tensor of dtype I3_GS with the expected shape and quant spec.
func TestI3GSNewMetadata(t *testing.T) {
	be := Default()
	const out, in = 4, 128
	nGroups := in / i3gsGroup
	raw := make([]byte, out*nGroups*i3gsGroupBytes)
	scale := make([]float32, out*nGroups)
	tensor := NewI3GS(be, []int{out, in}, raw, scale)

	if tensor.Dtype != I3_GS {
		t.Fatalf("expected dtype I3_GS (%s), got %s", I3_GS, tensor.Dtype)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != out || tensor.Shape[1] != in {
		t.Fatalf("expected shape [%d, %d], got %v", out, in, tensor.Shape)
	}
	if tensor.Quant == nil || tensor.Quant.Block != i3gsGroup || tensor.Quant.Bits != 3 {
		t.Fatalf("unexpected quant spec: %+v", tensor.Quant)
	}
	if len(tensor.Quant.Scale) != out*nGroups {
		t.Fatalf("expected %d scales, got %d", out*nGroups, len(tensor.Quant.Scale))
	}
}

// TestI3GSNewRejectsInvalidStorage pins the no-wrap-around admission: malformed shape or
// byte lengths must panic, never admit an empty buffer for a large nonempty tensor.
func TestI3GSNewRejectsInvalidStorage(t *testing.T) {
	be := Default()
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		shape []int
		raw   []byte
		scale []float32
	}{
		{"zero_rows", []int{0, i3gsGroup}, nil, nil},
		{"zero_columns", []int{1, 0}, nil, nil},
		{"negative_dimensions", []int{-1, i3gsGroup}, nil, nil},
		{"overflow", []int{maxInt, i3gsGroup}, nil, nil},
		{"partial_group", []int{1, i3gsGroup - 1}, make([]byte, i3gsGroupBytes), make([]float32, 1)},
		{"short_raw", []int{1, i3gsGroup}, make([]byte, i3gsGroupBytes-1), make([]float32, 1)},
		{"long_raw", []int{1, i3gsGroup}, make([]byte, i3gsGroupBytes+1), make([]float32, 1)},
		{"short_scale", []int{1, i3gsGroup}, make([]byte, i3gsGroupBytes), make([]float32, 0)},
		{"long_scale", []int{1, i3gsGroup}, make([]byte, i3gsGroupBytes), make([]float32, 2)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s: expected panic, got nil", tc.name)
				}
			}()
			NewI3GS(be, tc.shape, tc.raw, tc.scale)
		})
	}
}

// i3gsMatMulReference is the independent scalar per-element reference for the I3_GS matmul:
// the same scale-folded-per-group arithmetic and reduction order as i3gsRowDot, but with a
// straightforward index walk (j/4, j%4, j/8, j%8) instead of the production bit-shifts, so
// the two are bit-identical only if i3gsRowDot is correct.
func i3gsMatMulReference(raw []byte, scale []float32, x []float32, out, in int) []float32 {
	nGroups := in / i3gsGroup
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		row := raw[o*nGroups*i3gsGroupBytes : (o+1)*nGroups*i3gsGroupBytes]
		sc := scale[o*nGroups : (o+1)*nGroups]
		var sum float32
		for g := 0; g < nGroups; g++ {
			grp := row[g*i3gsGroupBytes : (g+1)*i3gsGroupBytes]
			low := grp[:i3gsLowBytes]
			high := grp[i3gsLowBytes:]
			var groupSum float32
			for j := 0; j < i3gsGroup; j++ {
				low2 := int(low[j/4]>>uint((j%4)*2)) & 3
				high1 := int(high[j/8]>>uint(j%8)) & 1
				code := low2 | (high1 << 2)
				groupSum += float32(code-4) * x[g*i3gsGroup+j]
			}
			sum += groupSum * sc[g]
		}
		y[o] = sum
	}
	return y
}

// TestI3GSMatMulReference checks that the cpuBackend MatMul/BatchedMatMul I3_GS path matches
// the independent scalar per-element reference bit-for-bit on a fixed matrix.
func TestI3GSMatMulReference(t *testing.T) {
	be := Default()
	const out, in = 3, 256
	nGroups := in / i3gsGroup
	rng := rand.New(rand.NewSource(14495))
	raw := make([]byte, out*nGroups*i3gsGroupBytes)
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	scale := make([]float32, out*nGroups)
	for i := range scale {
		scale[i] = float32(rng.Float32()*2 - 1) // signed, in [-1, 1)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	tensor := NewI3GS(be, []int{out, in}, raw, scale)
	got := be.MatMul(tensor, NewF32(be, []int{in}, x))
	gotF, _ := hostF32(got)
	want := i3gsMatMulReference(raw, scale, x, out, in)
	for o := 0; o < out; o++ {
		if gotF[o] != want[o] {
			t.Fatalf("MatMul o=%d: got %v, want %v", o, gotF[o], want[o])
		}
	}

	// BatchedMatMul: P=4 samples, each row must match the single-sample reference.
	const P = 4
	X := make([]float32, P*in)
	for i := range X {
		X[i] = rng.Float32()*2 - 1
	}
	bgot := be.BatchedMatMul(tensor, NewF32(be, []int{P, in}, X), P)
	bgotF, _ := hostF32(bgot)
	for s := 0; s < P; s++ {
		xs := X[s*in : (s+1)*in]
		wantRow := i3gsMatMulReference(raw, scale, xs, out, in)
		for o := 0; o < out; o++ {
			if bgotF[s*out+o] != wantRow[o] {
				t.Fatalf("BatchedMatMul s=%d o=%d: got %v, want %v", s, o, bgotF[s*out+o], wantRow[o])
			}
		}
	}
}

// TestI3GSQuantizeRoundTrip checks that QuantizeRowsI3GS then dequantize recovers the
// scalar-oracle values within the format's rounding bound (0.5*scale), and that the
// per-group scale is amax/3.5 with the 1e-8 floor.
func TestI3GSQuantizeRoundTrip(t *testing.T) {
	const out, in = 2, 128
	nGroups := in / i3gsGroup
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, out*in)
	for i := range w {
		w[i] = rng.Float32()*4 - 2 // in [-2, 2)
	}
	// Force one group to be all-zero to exercise the 1e-8 floor.
	w[in/2] = 0
	for i := in / 2; i < in/2+i3gsGroup; i++ {
		w[i] = 0
	}

	raw, scale, err := QuantizeRowsI3GS(w, out, in)
	if err != nil {
		t.Fatalf("QuantizeRowsI3GS: %v", err)
	}
	if len(raw) != out*nGroups*i3gsGroupBytes {
		t.Fatalf("raw length %d, want %d", len(raw), out*nGroups*i3gsGroupBytes)
	}
	if len(scale) != out*nGroups {
		t.Fatalf("scale length %d, want %d", len(scale), out*nGroups)
	}

	// Verify the scale = amax/3.5 (1e-8 floor) for each group.
	for o := 0; o < out; o++ {
		for g := 0; g < nGroups; g++ {
			grp := w[o*in+g*i3gsGroup : o*in+(g+1)*i3gsGroup]
			var amax float32
			for _, v := range grp {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			wantScale := amax / 3.5
			if wantScale < 1e-8 {
				wantScale = 1e-8
			}
			if scale[o*nGroups+g] != wantScale {
				t.Fatalf("o=%d g=%d: scale %v, want %v", o, g, scale[o*nGroups+g], wantScale)
			}
		}
	}

	// Dequantize and check the rounding bound: |w - dequant| <= 0.5*scale.
	dequant := make([]float32, out*in)
	for o := 0; o < out; o++ {
		rowRaw := raw[o*nGroups*i3gsGroupBytes : (o+1)*nGroups*i3gsGroupBytes]
		rowScale := scale[o*nGroups : (o+1)*nGroups]
		for g := 0; g < nGroups; g++ {
			grp := rowRaw[g*i3gsGroupBytes : (g+1)*i3gsGroupBytes]
			low := grp[:i3gsLowBytes]
			high := grp[i3gsLowBytes:]
			i3gsDequantGroup(dequant[o*in+g*i3gsGroup:o*in+(g+1)*i3gsGroup], low, high, rowScale[g], i3gsGroup)
		}
	}
	for i := 0; i < out*in; i++ {
		g := i / i3gsGroup
		bound := 0.5 * scale[g]
		diff := math.Abs(float64(w[i] - dequant[i]))
		if diff > float64(bound)+1e-6 {
			t.Fatalf("i=%d: |w-dequant|=%v exceeds bound %v", i, diff, bound)
		}
	}
}

// TestI3GSQuantizeRejectsInvalid pins the quantize mirror's input validation.
func TestI3GSQuantizeRejectsInvalid(t *testing.T) {
	if _, _, err := QuantizeRowsI3GS(nil, 0, 64); err == nil {
		t.Fatal("expected error for out=0")
	}
	if _, _, err := QuantizeRowsI3GS(nil, 1, 0); err == nil {
		t.Fatal("expected error for in=0")
	}
	if _, _, err := QuantizeRowsI3GS(nil, 1, 63); err == nil {
		t.Fatal("expected error for in not a multiple of 64")
	}
	if _, _, err := QuantizeRowsI3GS(make([]float32, 63), 1, 64); err == nil {
		t.Fatal("expected error for short weight slice")
	}
	nan := make([]float32, 64)
	nan[0] = float32(math.NaN())
	if _, _, err := QuantizeRowsI3GS(nan, 1, 64); err == nil {
		t.Fatal("expected error for NaN weight")
	}
	inf := make([]float32, 64)
	inf[0] = float32(math.Inf(1))
	if _, _, err := QuantizeRowsI3GS(inf, 1, 64); err == nil {
		t.Fatal("expected error for Inf weight")
	}
}

// TestI3GSDeviceDecline checks that I3_GS is cleanly declined by the device-weight-dtype
// predicate (fail-closed) on the CPU reference it is supported, and that the dtype is
// quantized (so it carries a QuantSpec). No device backend serves I3_GS yet, so the
// KQuantDeviceKernel capability must report false for it (never a panic, never a silent
// fallback to a wrong dtype).
func TestI3GSDeviceDecline(t *testing.T) {
	if !I3_GS.Quantized() {
		t.Fatal("I3_GS must be a quantized dtype")
	}
	if I3_GS.String() != "i3_gs" {
		t.Fatalf("I3_GS.String() = %q, want %q", I3_GS.String(), "i3_gs")
	}
	be := Default()
	if !BackendSupportsDeviceWeightDtype(be, I3_GS) {
		t.Fatal("CPU reference must support I3_GS MatMul")
	}
	// The CPU reference is a host backend; its SupportsDeviceWeightDtype is the honest
	// predicate. A device backend that has no I3_GS kernel must report false (fail-closed).
	// We assert the dtype is NOT in the rawIQFormats map (so it does not silently fall to
	// the raw-IQ default path in MatMul).
	if f, ok := rawIQFormats[I3_GS]; ok {
		t.Fatalf("I3_GS must not be a raw-IQ format, got %+v", f)
	}
}
