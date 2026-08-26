package gdn

import (
	"bytes"
	"math"
	"runtime"
	"testing"
)

// TestShapesAreTheRealLayer pins the derived widths against the Qwen3.6-27B linear_attn
// dims. Every buffer in Layer is cut from these, so a slipped constant would reshape the
// whole experiment silently rather than failing to compile.
func TestShapesAreTheRealLayer(t *testing.T) {
	if KeyDim != 2048 || ValDim != 6144 || ConvDim != 10240 {
		t.Fatalf("derived widths drifted: KeyDim=%d ValDim=%d ConvDim=%d want 2048/6144/10240",
			KeyDim, ValDim, ConvDim)
	}
	if ConvDim != 2*KeyDim+ValDim {
		t.Fatalf("ConvDim must hold q,k (KeyDim each) and v (ValDim): %d != 2*%d+%d", ConvDim, KeyDim, ValDim)
	}
	if NV%NK != 0 {
		t.Fatalf("value heads must be a whole multiple of key heads (GQA-style repeat): nV=%d nK=%d", NV, NK)
	}
}

// TestActivations pins the defining values and both saturation limits, so a flipped sign in
// an exponent cannot pass.
func TestActivations(t *testing.T) {
	if got := Silu(0); got != 0 {
		t.Errorf("Silu(0) = %v, want 0", got)
	}
	if got := Sigmoidf(0); got != 0.5 {
		t.Errorf("Sigmoidf(0) = %v, want 0.5", got)
	}
	if got := Softplus(0); math.Abs(float64(got)-math.Ln2) > 1e-6 {
		t.Errorf("Softplus(0) = %v, want ln2 (%v)", got, math.Ln2)
	}
	if got := Silu(20); math.Abs(float64(got)-20) > 1e-3 { // gate saturates to 1
		t.Errorf("Silu(20) = %v, want ~20", got)
	}
	if got := Silu(-20); math.Abs(float64(got)) > 1e-6 {
		t.Errorf("Silu(-20) = %v, want ~0", got)
	}
	if got := Sigmoidf(20); got < 0.999 {
		t.Errorf("Sigmoidf(20) = %v, want ~1", got)
	}
	if got := Sigmoidf(-20); got > 0.001 {
		t.Errorf("Sigmoidf(-20) = %v, want ~0", got)
	}
	if got := Softplus(20); math.Abs(float64(got)-20) > 1e-3 { // log1p(e^x) -> x
		t.Errorf("Softplus(20) = %v, want ~20", got)
	}
}

// TestL2NormIntoUsesSumNotMean is the guard on the divergence flagged in
// gdn-recurrence-bench: qwen35.go:l2normInto divides by the SUM of squares, and a
// mean-of-squares variant differs by sqrt(len) — a factor of ~11.3 on a 128-wide head.
// With 128 ones the sum form gives 1/sqrt(128); the mean form would give ~1.
func TestL2NormIntoUsesSumNotMean(t *testing.T) {
	src := make([]float32, KHd)
	for i := range src {
		src[i] = 1
	}
	dst := make([]float32, KHd)
	L2NormInto(dst, src, 1e-6)

	want := 1 / math.Sqrt(float64(KHd))
	if math.Abs(float64(dst[0])-want) > 1e-6 {
		t.Fatalf("L2NormInto(ones) = %v, want %v (sum-of-squares); a mean-of-squares form would give ~1", dst[0], want)
	}
	// The result must be a unit vector.
	var ss float64
	for _, v := range dst {
		ss += float64(v) * float64(v)
	}
	if math.Abs(ss-1) > 1e-5 {
		t.Fatalf("normalized vector has squared norm %v, want 1", ss)
	}
}

// TestRMSNormGain1p pins BOTH properties the GDN input norm depends on: the mean (not sum)
// of squares, and the (1+w) gain. src=2 everywhere gives rms 2, so inv=1/2 and the
// normalized value is 1; the gain then scales it by (1+w).
func TestRMSNormGain1p(t *testing.T) {
	src := []float32{2, 2, 2, 2}
	dst := make([]float32, 4)

	RMSNormGain1p(dst, src, []float32{0, 0, 0, 0}, 1e-6)
	if math.Abs(float64(dst[0])-1) > 1e-5 {
		t.Fatalf("RMSNormGain1p(src=2, w=0) = %v, want 1 (mean-of-squares; a sum form gives 0.5)", dst[0])
	}
	RMSNormGain1p(dst, src, []float32{0.5, 0.5, 0.5, 0.5}, 1e-6)
	if math.Abs(float64(dst[0])-1.5) > 1e-5 {
		t.Fatalf("RMSNormGain1p(src=2, w=0.5) = %v, want 1.5 ((1+w) gain; a plain-w form gives 0.5)", dst[0])
	}
}

// TestRMSNormGatedInPlace pins the GDN readout norm's two departures from the input norm:
// the weight is applied PLAIN (not 1+w) and the result is gated by Silu(gate).
func TestRMSNormGatedInPlace(t *testing.T) {
	x := []float32{2, 2, 2, 2}
	w := []float32{3, 3, 3, 3}
	gate := []float32{20, 20, 20, 20} // Silu(20) ~ 20
	RMSNormGatedInPlace(x, w, gate, 1e-6)
	// normalized 2 -> 1; times plain w=3 -> 3; times Silu(20)~20 -> ~60.
	if math.Abs(float64(x[0])-60) > 1e-2 {
		t.Fatalf("RMSNormGatedInPlace = %v, want ~60 (plain w; a (1+w) form gives ~80)", x[0])
	}
	// A zero gate must zero the output (Silu(0)=0) — the gate is multiplicative, not additive.
	y := []float32{2, 2, 2, 2}
	RMSNormGatedInPlace(y, w, []float32{0, 0, 0, 0}, 1e-6)
	if y[0] != 0 {
		t.Fatalf("RMSNormGatedInPlace with a zero gate = %v, want 0", y[0])
	}
}

// TestQuantF16Regimes pins the three declared regimes: f16-exact values round-trip
// unchanged, overflow saturates with the sign kept, and subnormals flush to zero.
func TestQuantF16Regimes(t *testing.T) {
	for _, x := range []float32{0, 1, -1, 0.5, 2, 1.5, -0.25} {
		if got := QuantF16(x); got != x {
			t.Errorf("QuantF16(%v) = %v, want unchanged (f16-exact)", x, got)
		}
	}
	if got := QuantF16(1e30); got != 65504 {
		t.Errorf("QuantF16(1e30) = %v, want 65504 (saturate)", got)
	}
	if got := QuantF16(-1e30); got != -65504 {
		t.Errorf("QuantF16(-1e30) = %v, want -65504 (saturate)", got)
	}
	if got := QuantF16(1e-30); got != 0 {
		t.Errorf("QuantF16(1e-30) = %v, want 0 (flush)", got)
	}
}

// TestQuantF16RoundsHalfToEven pins the tie rule the doc comment claims. 1+0.5ulp sits
// exactly between 1 (even mantissa) and 1+1ulp (odd), so it must land on 1; 1+1.5ulp sits
// between 1+1ulp (odd) and 1+2ulp (even), so it must land on 1+2ulp. A round-half-UP
// implementation gets the first case wrong.
func TestQuantF16RoundsHalfToEven(t *testing.T) {
	ulp := float32(math.Ldexp(1, -10)) // one f16 mantissa step at exponent 0
	if got := QuantF16(1 + ulp/2); got != 1 {
		t.Errorf("QuantF16(1+0.5ulp) = %v, want 1 (tie -> even); half-up would give %v", got, 1+ulp)
	}
	if got, want := QuantF16(1+ulp*3/2), 1+2*ulp; got != want {
		t.Errorf("QuantF16(1+1.5ulp) = %v, want %v (tie -> even)", got, want)
	}
	// Non-ties round to the nearest neighbour in both directions.
	if got := QuantF16(1 + ulp*0.4); got != 1 {
		t.Errorf("QuantF16(1+0.4ulp) = %v, want 1", got)
	}
	if got, want := QuantF16(1+ulp*0.6), 1+ulp; got != want {
		t.Errorf("QuantF16(1+0.6ulp) = %v, want %v", got, want)
	}
}

// TestQuantF16IsIdempotentOnEveryF16 is the exhaustive witness: a value that is already
// exactly representable in binary16 must survive the round-trip untouched, or the modelled
// f16 store injects error a real one would not. Sweeps all 61440 normal f16 patterns.
func TestQuantF16IsIdempotentOnEveryF16(t *testing.T) {
	moved := 0
	var first float32
	for h := 0; h < 1<<16; h++ {
		e := int32(h>>10) & 0x1F
		if e == 0 || e == 0x1F { // subnormal / inf / nan: outside the declared normal range
			continue
		}
		sign := uint32(h>>15) & 1
		m := uint32(h) & 0x3FF
		x := math.Float32frombits((sign << 31) | (uint32(e-15+127) << 23) | (m << 13))
		if got := QuantF16(x); got != x {
			if moved == 0 {
				first = x
			}
			moved++
		}
	}
	if moved != 0 {
		t.Fatalf("QuantF16 moved %d exactly-representable f16 values (first %v); a correct round-trip moves none", moved, first)
	}
}

// TestDepthwiseCausalSiluIsCausal drives a single-channel impulse through the K=4 tap
// window. The taps are ordered so tap K-1 is the CURRENT position: an impulse at t=0 must
// appear at outputs 0..K-1 with descending tap weights and then vanish, and an impulse at
// the last position must reach only that position (no future leakage).
func TestDepthwiseCausalSiluIsCausal(t *testing.T) {
	const steps, channels, kernel = 5, 1, 4
	w := []float32{1, 2, 3, 4}

	src := make([]float32, steps*channels)
	src[0] = 1 // impulse at position 0
	dst := make([]float32, steps*channels)
	DepthwiseCausalSilu(dst, src, w, steps, channels, kernel)
	for i, want := range []float32{Silu(4), Silu(3), Silu(2), Silu(1), Silu(0)} {
		if dst[i] != want {
			t.Fatalf("impulse at 0: dst[%d] = %v, want %v (taps walk backwards then the window clears)", i, dst[i], want)
		}
	}

	src = make([]float32, steps*channels)
	src[steps-1] = 1 // impulse at the last position
	dst = make([]float32, steps*channels)
	DepthwiseCausalSilu(dst, src, w, steps, channels, kernel)
	for i := 0; i < steps-1; i++ {
		if dst[i] != 0 {
			t.Fatalf("future leakage: dst[%d] = %v, want 0 for an impulse at position %d", i, dst[i], steps-1)
		}
	}
	if dst[steps-1] != Silu(4) {
		t.Fatalf("current position: dst[%d] = %v, want Silu(4) = %v", steps-1, dst[steps-1], Silu(4))
	}
}

// TestParMatmulMatchesSerialBitExactly checks the parallel GEMM against a plain serial
// reference and, separately, that its output does not depend on GOMAXPROCS. The second
// property is what makes every rho in these experiments reproducible: workers own disjoint
// output rows and each dot product keeps a fixed ascending-j order.
func TestParMatmulMatchesSerialBitExactly(t *testing.T) {
	const P, outDim, inDim = 3, 7, 5
	X := make([]float32, P*inDim)
	for i := range X {
		X[i] = float32(math.Sin(float64(i)*1.7)) * 2
	}
	W := make([]float32, outDim*inDim)
	for i := range W {
		W[i] = float32(math.Cos(float64(i)*0.9)) * 3
	}

	want := make([]float32, P*outDim)
	for tk := 0; tk < P; tk++ {
		for i := 0; i < outDim; i++ {
			var acc float32
			for j := 0; j < inDim; j++ {
				acc += W[i*inDim+j] * X[tk*inDim+j]
			}
			want[tk*outDim+i] = acc
		}
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	var prev []float32
	for _, procs := range []int{1, 2, 3, 8, 16} {
		runtime.GOMAXPROCS(procs)
		got := make([]float32, P*outDim)
		ParMatmul(got, X, W, P, outDim, inDim)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("GOMAXPROCS=%d: element %d = %v, want %v (serial reference)", procs, i, got[i], want[i])
			}
		}
		if prev != nil {
			for i := range got {
				if got[i] != prev[i] {
					t.Fatalf("GOMAXPROCS=%d changed element %d: %v vs %v", procs, i, got[i], prev[i])
				}
			}
		}
		prev = got
	}
}

// TestRelDiv pins the identity, a known perturbation, and the zero-reference guard.
func TestRelDiv(t *testing.T) {
	if got := RelDiv([]float32{3, 4}, []float32{3, 4}); got != 0 {
		t.Errorf("RelDiv(equal) = %v, want 0", got)
	}
	if got := RelDiv([]float32{3, 4}, []float32{0, 0}); math.Abs(got-1) > 1e-12 {
		t.Errorf("RelDiv([3,4],[0,0]) = %v, want 1", got)
	}
	if got := RelDiv([]float32{0, 0}, []float32{1, 1}); got != 0 {
		t.Errorf("RelDiv(zero ref) = %v, want 0 (den==0 guard)", got)
	}
	// ||a-b||=5 against ||a||=5 -> 1.
	if got := RelDiv([]float32{3, 4}, []float32{3, 9}); math.Abs(got-1) > 1e-6 {
		t.Errorf("RelDiv = %v, want 1", got)
	}
}

// TestNewLayerWeightsShapes checks every buffer is cut to the tensor it names — the
// projections against the hidden size passed in, the control tensors against the fixed
// per-layer widths.
func TestNewLayerWeightsShapes(t *testing.T) {
	const h = 64
	lw := NewLayerWeights(h)
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"Hidden", lw.Hidden, h},
		{"WIn", len(lw.WIn), h},
		{"Wqkv", len(lw.Wqkv), ConvDim * h},
		{"Wz", len(lw.Wz), ValDim * h},
		{"Wb", len(lw.Wb), NV * h},
		{"Wa", len(lw.Wa), NV * h},
		{"Conv", len(lw.Conv), ConvDim * K},
		{"ALog", len(lw.ALog), NV},
		{"DtB", len(lw.DtB), NV},
		{"NormW", len(lw.NormW), VHd},
		{"WOut", len(lw.WOut), h * ValDim},
	} {
		if c.got != c.want {
			t.Errorf("%s: %d, want %d", c.name, c.got, c.want)
		}
	}
}

// TestFillIsLayerSeeded checks the fixture is a pure function of the layer index — two
// weight sets filled with the same layer are bit-identical, different layers are not —
// which is what lets the reference and test runs share weights exactly.
func TestFillIsLayerSeeded(t *testing.T) {
	const h = 64
	a, b, c := NewLayerWeights(h), NewLayerWeights(h), NewLayerWeights(h)
	a.Fill(3, -2.0)
	b.Fill(3, -2.0)
	c.Fill(4, -2.0)

	for i := range a.Wqkv {
		if a.Wqkv[i] != b.Wqkv[i] {
			t.Fatalf("same layer index produced different weights at %d: %v vs %v", i, a.Wqkv[i], b.Wqkv[i])
		}
	}
	same := true
	for i := range a.Wqkv {
		if a.Wqkv[i] != c.Wqkv[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("layers 3 and 4 produced identical weights; the per-layer seed is not being used")
	}
}

// TestFillALogMeanShiftsOnlyALog is the guard on the parameter that replaced two different
// baked-in constants: aLogMean must shift A_log by exactly its delta and leave every other
// tensor bit-identical. Both draws consume the PRNG in the same order, so the difference is
// exact rather than statistical.
func TestFillALogMeanShiftsOnlyALog(t *testing.T) {
	const h = 64
	a, b := NewLayerWeights(h), NewLayerWeights(h)
	a.Fill(0, -2.0)
	b.Fill(0, -5.0)

	for i := range a.ALog {
		if d := float64(a.ALog[i] - b.ALog[i]); math.Abs(d-3.0) > 1e-5 {
			t.Fatalf("ALog[%d] delta = %v, want 3.0 (the -2.0 vs -5.0 mean shift)", i, d)
		}
	}
	for name, pair := range map[string][2][]float32{
		"WIn": {a.WIn, b.WIn}, "Wqkv": {a.Wqkv, b.Wqkv}, "Wz": {a.Wz, b.Wz},
		"Wb": {a.Wb, b.Wb}, "Wa": {a.Wa, b.Wa}, "Conv": {a.Conv, b.Conv},
		"DtB": {a.DtB, b.DtB}, "NormW": {a.NormW, b.NormW}, "WOut": {a.WOut, b.WOut},
	} {
		for i := range pair[0] {
			if pair[0][i] != pair[1][i] {
				t.Fatalf("%s[%d] moved with aLogMean (%v vs %v); the knob must touch A_log only",
					name, i, pair[0][i], pair[1][i])
			}
		}
	}
}

// TestLayerIsDeterministic runs the same layer twice and demands bit-identical output. Every
// divergence these experiments report is a difference between two Layer calls, so a Layer
// that is not reproducible would make every reported rho noise.
func TestLayerIsDeterministic(t *testing.T) {
	lw, X, P := smallLayer(t)
	a := Layer(lw, X, P, ModeForward, 1e-6)
	b := Layer(lw, X, P, ModeForward, 1e-6)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("Layer is not deterministic: element %d = %v vs %v", i, a[i], b[i])
		}
	}
	if RelDiv(a, b) != 0 {
		t.Fatalf("RelDiv of two identical forward runs = %v, want exactly 0", RelDiv(a, b))
	}
	// A layer that returned zeros would pass the equality check vacuously.
	nonzero := false
	for _, v := range a {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("Layer produced an all-zero output; the equality checks above would be vacuous")
	}
}

// TestScanModesPerturbButStaySmall is the property the divergence experiment measures in
// miniature: reordering the scan's reduction (ModeReverse) must change the result — it is a
// different rounding — but only slightly, while storing the state in f16 (ModeF16State) must
// perturb it strictly MORE, since it drops 13 mantissa bits every step.
func TestScanModesPerturbButStaySmall(t *testing.T) {
	lw, X, P := smallLayer(t)
	ref := Layer(lw, X, P, ModeForward, 1e-6)
	rev := Layer(lw, X, P, ModeReverse, 1e-6)
	f16 := Layer(lw, X, P, ModeF16State, 1e-6)

	rhoRev := RelDiv(ref, rev)
	rhoF16 := RelDiv(ref, f16)
	if rhoRev <= 0 {
		t.Fatalf("reverse-order reduction produced rho=%v; reordering f32 sums must change the rounding", rhoRev)
	}
	if rhoRev > 1e-3 {
		t.Fatalf("reverse-order rho=%v is too large to be pure f32 reduction-order rounding", rhoRev)
	}
	if rhoF16 <= rhoRev {
		t.Fatalf("f16 state rho=%v must exceed reduction-order rho=%v (it drops 13 mantissa bits per step)", rhoF16, rhoRev)
	}
	if rhoF16 > 0.5 {
		t.Fatalf("f16 state rho=%v is implausibly large for a half-precision accumulator", rhoF16)
	}
}

// TestKeyOrderVisitsEveryIndexOnce pins the reduction order each mode implies: forward walks
// 0..KHd-1 ascending, reverse walks it descending, and both cover every index exactly once. An
// order that skipped or repeated an index would drop or double-count a key slice while still
// producing plausible numbers.
func TestKeyOrderVisitsEveryIndexOnce(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mode      ScanMode
		wantFirst int
		wantLast  int
	}{
		{"forward", ModeForward, 0, KHd - 1},
		{"f16state", ModeF16State, 0, KHd - 1}, // f16state changes precision, not order
		{"reverse", ModeReverse, KHd - 1, 0},
	} {
		i, stride := keyOrder(tc.mode)
		seen := make([]bool, KHd)
		first, last := i, i
		for n := 0; n < KHd; n++ {
			if i < 0 || i >= KHd {
				t.Fatalf("%s: step %d ran out of range at index %d", tc.name, n, i)
			}
			if seen[i] {
				t.Fatalf("%s: index %d visited twice", tc.name, i)
			}
			seen[i] = true
			last = i
			i += stride
		}
		if first != tc.wantFirst || last != tc.wantLast {
			t.Fatalf("%s: walked %d..%d, want %d..%d", tc.name, first, last, tc.wantFirst, tc.wantLast)
		}
	}
}

// TestReductionsMatchTheirExplicitOrder is the guard on collapsing the two hand-written
// direction branches into one stride-driven loop: each mode's result must be BIT-identical to
// a reference that spells the loop out in that direction. Float addition is not associative,
// so "same set of terms" is not enough — the order has to match exactly.
func TestReductionsMatchTheirExplicitOrder(t *testing.T) {
	st := make([]float32, KHd*VHd)
	for i := range st {
		st[i] = float32(math.Sin(float64(i) * 0.013))
	}
	kn := make([]float32, KHd)
	qn := make([]float32, KHd)
	delta := make([]float32, VHd)
	for i := range kn {
		kn[i] = float32(math.Cos(float64(i) * 0.071))
		qn[i] = float32(math.Sin(float64(i) * 0.029))
	}
	for d := range delta {
		delta[d] = float32(math.Cos(float64(d) * 0.037))
	}

	// accumulate, both directions, against explicit references.
	for _, tc := range []struct {
		name    string
		mode    ScanMode
		descend bool
	}{{"forward", ModeForward, false}, {"reverse", ModeReverse, true}} {
		want := make([]float32, VHd)
		for n := 0; n < KHd; n++ {
			i := n
			if tc.descend {
				i = KHd - 1 - n
			}
			ki := kn[i]
			base := i * VHd
			for d := 0; d < VHd; d++ {
				want[d] += st[base+d] * ki
			}
		}
		got := make([]float32, VHd)
		accumulate(got, st, kn, tc.mode)
		for d := range want {
			if got[d] != want[d] {
				t.Fatalf("accumulate/%s: element %d = %v, want %v (bit-exact)", tc.name, d, got[d], want[d])
			}
		}
	}

	// readout mutates st, so each direction gets its own copy.
	for _, tc := range []struct {
		name    string
		mode    ScanMode
		descend bool
	}{{"forward", ModeForward, false}, {"reverse", ModeReverse, true}} {
		wantSt := append([]float32(nil), st...)
		wantOd := make([]float32, VHd)
		for n := 0; n < KHd; n++ {
			i := n
			if tc.descend {
				i = KHd - 1 - n
			}
			ki, qi := kn[i], qn[i]
			base := i * VHd
			for d := 0; d < VHd; d++ {
				wantSt[base+d] += ki * delta[d]
				wantOd[d] += wantSt[base+d] * qi
			}
		}
		gotSt := append([]float32(nil), st...)
		gotOd := make([]float32, VHd)
		readout(gotOd, gotSt, kn, qn, delta, tc.mode)
		for d := range wantOd {
			if gotOd[d] != wantOd[d] {
				t.Fatalf("readout/%s: output %d = %v, want %v (bit-exact)", tc.name, d, gotOd[d], wantOd[d])
			}
		}
		for i := range wantSt {
			if gotSt[i] != wantSt[i] {
				t.Fatalf("readout/%s: state %d = %v, want %v (bit-exact)", tc.name, i, gotSt[i], wantSt[i])
			}
		}
	}

	// The two orders must actually differ, or the test above would be vacuous.
	fwd, rev := make([]float32, VHd), make([]float32, VHd)
	accumulate(fwd, st, kn, ModeForward)
	accumulate(rev, st, kn, ModeReverse)
	if RelDiv(fwd, rev) == 0 {
		t.Fatal("forward and reverse reductions agreed bit-for-bit; the mode is not reaching the loop")
	}
}

// smallLayer builds a layer-weight set and input at a small hidden size — the head/state
// dims (NV, KHd, VHd, K) stay REAL, only the projection fan-in shrinks, so the recurrence
// under test is the true one.
func smallLayer(t *testing.T) (*LayerWeights, []float32, int) {
	t.Helper()
	const h, P = 64, 3
	lw := NewLayerWeights(h)
	lw.Fill(0, -2.0)
	X := make([]float32, P*h)
	for i := range X {
		X[i] = float32(math.Sin(float64(i) * 0.37))
	}
	return lw, X, P
}

// TestEmitJSON pins the wire shape every experiment's -json mode emits: 2-space indent and a
// trailing newline.
func TestEmitJSON(t *testing.T) {
	var buf bytes.Buffer
	if err := EmitJSON(&buf, struct {
		A int    `json:"a"`
		B string `json:"b"`
	}{1, "x"}); err != nil {
		t.Fatalf("EmitJSON: %v", err)
	}
	want := "{\n  \"a\": 1,\n  \"b\": \"x\"\n}\n"
	if got := buf.String(); got != want {
		t.Fatalf("EmitJSON = %q, want %q", got, want)
	}
}

func TestVerdict(t *testing.T) {
	if got, want := Verdict("first clause.", "Second clause."), "first clause. Second clause."; got != want {
		t.Fatalf("Verdict = %q, want %q", got, want)
	}
}
