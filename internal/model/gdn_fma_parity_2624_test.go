//go:build amd64

package model

// fak-private#2624 — cross-package and prefill-vs-decode witnesses for the GDN FMA-parity
// invariant. Companion to internal/compute/gdn_fma_parity_2624_test.go, which pins the
// kernel-level bit-identity on the canonical 128x128 geometry.
//
// The spec's required invariant: the Go GDN recurrence and the AVX-512 assembly kernel must
// produce BIT-IDENTICAL results (exactly equal float32 bit patterns, not "within
// tolerance") for the recurrent state `st`, the readout `od`, `kvmem` and `delta`, because a
// separate multiply-add rounds the product BEFORE the add while VFMADD231PS does not. A
// radix prefix-cache HIT replays state produced by the decode kernel while a cache MISS /
// cold prefill recomputes the SAME span with the Go scan — so identical input must yield
// identical bytes, or a greedy argmax can flip onto an already-emitted token.
//
// What this file adds over the compute package file:
//   - clause 5: internal/model.ScalarHeadStep (the model's scalar decode reference) must be
//     bit-identical to internal/compute.Wave32GatedDeltaNetStep on the canonical geometry.
//     These are "the Go path" from two different packages, and a divergence between them is
//     the same class of bug.
//   - clause 6: the level at which the defect actually manifested — a multi-token PREFILL
//     scan versus a sequence of single-token DECODE steps over the same token span, compared
//     on the resulting recurrent state, on the canonical 128x128 geometry so the decode arm
//     really does run the AVX-512 kernel.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const gdnDim2624 = 128

// gdnModelGates2624 is the (beta, decay) grid the spec mandates. It intentionally includes
// the degenerate corners — beta=0 (no selection), decay=0 (total forget) and decay=1 (no
// forget) — because those are where a rounding-order change is easiest to hide.
var gdnModelGates2624 = []struct {
	beta, decay float32
}{
	{0, float32(math.Exp(-1e-6))},
	{0, 0},
	{1e-8, 0.25},
	{0.001, float32(math.Exp(-1e-4))},
	{0.001, 0.01},
	{0.25, float32(math.Exp(-0.05))},
	{0.25, 0.125},
	{0.5, float32(math.Exp(-1e-6))},
	{0.5, 0.5},
	{0.6, 0.25},
	{0.75, 0.125},
	{0.9, 0.5},
	{0.999, 0.01},
	{1, 1},
	{1, 0},
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type gdnOps2624 struct {
	st, qn, kn, vh, od, kvmem, delta []float32
}

func gdnNewOps2624() *gdnOps2624 {
	d := gdnDim2624
	return &gdnOps2624{
		st: make([]float32, d*d), qn: make([]float32, d), kn: make([]float32, d),
		vh: make([]float32, d), od: make([]float32, d), kvmem: make([]float32, d), delta: make([]float32, d),
	}
}

func (o *gdnOps2624) clone() *gdnOps2624 {
	c := gdnNewOps2624()
	copy(c.st, o.st)
	copy(c.qn, o.qn)
	copy(c.kn, o.kn)
	copy(c.vh, o.vh)
	copy(c.od, o.od)
	copy(c.kvmem, o.kvmem)
	copy(c.delta, o.delta)
	return c
}

var gdnBufNames2624 = []string{"st", "od", "kvmem", "delta"}

func (o *gdnOps2624) buf(n string) []float32 {
	switch n {
	case "st":
		return o.st
	case "od":
		return o.od
	case "kvmem":
		return o.kvmem
	case "delta":
		return o.delta
	}
	panic("unknown buffer " + n)
}

// assertGDNBits2624 asserts EXACT float32 bit-pattern equality and reports the first few
// offenders with full hex bit patterns.
func assertGDNBits2624(tb testing.TB, ctx string, want, got []float32) {
	tb.Helper()
	if len(want) != len(got) {
		tb.Fatalf("%s: length mismatch got=%d want=%d", ctx, len(got), len(want))
	}
	reported := 0
	for i := range want {
		wb, gb := math.Float32bits(want[i]), math.Float32bits(got[i])
		if wb == gb {
			continue
		}
		if reported == 0 {
			tb.Errorf("%s: NOT bit-identical", ctx)
		}
		if reported < 8 {
			tb.Errorf("%s: [%d] want bits=0x%08x (%g) got bits=0x%08x (%g) ulpDelta=%d",
				ctx, i, wb, want[i], gb, got[i], int32(gb)-int32(wb))
		}
		reported++
	}
	if reported > 8 {
		tb.Errorf("%s: ... %d differing elements total", ctx, reported)
	}
}

func gdnMaxMag2624(v []float32) float64 {
	var m float64
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			continue
		}
		if a := math.Abs(float64(x)); a > m {
			m = a
		}
	}
	return m
}

func gdnNonZero2624(bufs ...[]float32) bool {
	for _, b := range bufs {
		for _, x := range b {
			if x != 0 {
				return true
			}
		}
	}
	return false
}

func gdnSeedOps2624(rng *rand.Rand, o *gdnOps2624) {
	for i := range o.st {
		o.st[i] = rng.Float32()*0.2 - 0.1
	}
	for i := 0; i < gdnDim2624; i++ {
		o.qn[i] = rng.Float32()*0.2 - 0.1
		o.kn[i] = rng.Float32()*0.2 - 0.1
		o.vh[i] = rng.Float32()*0.2 - 0.1
		o.od[i] = rng.Float32()*0.02 - 0.01
	}
}

// ---------------------------------------------------------------------------
// 5. cross-package consistency
// ---------------------------------------------------------------------------

// TestGDNFMA2624ScalarHeadStepMatchesComputeBitwise pins spec clause 5: model.ScalarHeadStep
// (the decode scalar reference in internal/model) must be BIT-IDENTICAL to
// compute.Wave32GatedDeltaNetStep on the canonical 128x128 geometry, for all four buffers,
// across the full gate grid. Both are "the Go path" from different packages, so a divergence
// between them is the same class of bug as the FMA split.
//
// A regression: a ULP difference on any buffer for any gate pair, reported with both hex bit
// patterns. That is exactly the "two Go implementations disagree" failure mode.
func TestGDNFMA2624ScalarHeadStepMatchesComputeBitwise(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	live := 0
	for gi, gate := range gdnModelGates2624 {
		rng := rand.New(rand.NewSource(int64(2625000 + gi)))
		base := gdnNewOps2624()
		gdnSeedOps2624(rng, base)
		ctx := fmt.Sprintf("scalar-vs-compute gate[%02d] beta=%g decay=%g", gi, gate.beta, gate.decay)

		modelOps := base.clone()
		ScalarHeadStep(modelOps.st, modelOps.qn, modelOps.kn, modelOps.vh, gate.beta, gate.decay,
			modelOps.od, modelOps.kvmem, modelOps.delta)

		computeOps := base.clone()
		compute.Wave32GatedDeltaNetStep(computeOps.st, computeOps.qn, computeOps.kn, computeOps.vh,
			gate.beta, gate.decay, computeOps.od, computeOps.kvmem, computeOps.delta)

		for _, name := range gdnBufNames2624 {
			assertGDNBits2624(t, ctx+" buf="+name, modelOps.buf(name), computeOps.buf(name))
		}
		if gdnNonZero2624(modelOps.st, modelOps.od, modelOps.kvmem, modelOps.delta) {
			live++
		}
	}
	if live < 12 {
		t.Fatalf("non-vacuity too low: only %d/%d gate pairs produced a non-zero result", live, len(gdnModelGates2624))
	}
	t.Logf("ScalarHeadStep vs compute.Wave32GatedDeltaNetStep: %d gate pairs, %d non-vacuous, all four buffers bit-identical",
		len(gdnModelGates2624), live)
}

// TestGDNFMA2624ScalarHeadStepMultiStepBitwise pins clause 5 over a RECURRENCE, not a single
// step: drive model.ScalarHeadStep and compute.Wave32GatedDeltaNetStep over 64 steps with the
// same inputs, carrying `st` forward, and assert bitwise equality of the recurrent state
// after every step. A single-step comparison can pass while the recurrent state drifts.
//
// A regression: per-step fused-vs-split drift > 0 in the recurrent state, i.e. the two Go
// implementations no longer agree on the bytes that a prefix cache would persist.
func TestGDNFMA2624ScalarHeadStepMultiStepBitwise(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	const steps = 64
	pairs := []struct{ beta, decay float32 }{
		{0.5, float32(math.Exp(-1e-6))},
		{0.9, 0.5},
		{0.25, 0.125},
		{1, 0.25},
		{0.001, 0.5},
	}
	for pi, p := range pairs {
		rng := rand.New(rand.NewSource(int64(2625100 + pi)))
		base := gdnNewOps2624()
		gdnSeedOps2624(rng, base)
		modelOps, computeOps := base.clone(), base.clone()

		for step := 0; step < steps; step++ {
			ScalarHeadStep(modelOps.st, modelOps.qn, modelOps.kn, modelOps.vh, p.beta, p.decay,
				modelOps.od, modelOps.kvmem, modelOps.delta)
			compute.Wave32GatedDeltaNetStep(computeOps.st, computeOps.qn, computeOps.kn, computeOps.vh,
				p.beta, p.decay, computeOps.od, computeOps.kvmem, computeOps.delta)
			ctx := fmt.Sprintf("multi-step pair[%d] beta=%g decay=%g step=%d", pi, p.beta, p.decay, step)
			for _, name := range gdnBufNames2624 {
				assertGDNBits2624(t, ctx+" buf="+name, modelOps.buf(name), computeOps.buf(name))
			}
		}
		stMag, odMag := gdnMaxMag2624(modelOps.st), gdnMaxMag2624(modelOps.od)
		if stMag == 0 || odMag == 0 {
			t.Fatalf("pair[%d] beta=%g decay=%g: %d-step recurrence collapsed to zero (max|st|=%g, max|od|=%g) — vacuous",
				pi, p.beta, p.decay, steps, stMag, odMag)
		}
		t.Logf("pair[%d] beta=%g decay=%g: %d steps bit-identical, final max|st|=%g max|od|=%g",
			pi, p.beta, p.decay, steps, stMag, odMag)
	}
}

// TestGDNFMA2624VectorizedHeadStepMatchesScalarBitwise pins clause 5 at the model's own
// DISPATCH seam: on the canonical 128x128 geometry model.VectorizedHeadStep routes to
// compute.Wave32GatedDeltaNetStep and must be bit-identical to model.ScalarHeadStep; on a
// non-canonical geometry it must route to the scalar reference, and the counter must prove
// the arm that actually ran (presence != invokability: an existing kernel is not evidence a
// production entrypoint invokes it).
//
// A regression: the vectorized arm disagreeing with the scalar arm (the FMA split), or the
// routing taking the wrong arm for a geometry.
func TestGDNFMA2624VectorizedHeadStepMatchesScalarBitwise(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	if !HasVectorizedDeltaNetFor(gdnDim2624, gdnDim2624) {
		t.Skip("canonical 128x128 vectorized GDN kernel unavailable on this host (AVX-512F + OS ZMM state)")
	}

	rng := rand.New(rand.NewSource(int64(2625200)))
	base := gdnNewOps2624()
	gdnSeedOps2624(rng, base)
	beta, decay := float32(0.6), float32(0.25)

	// The counters are process-global, so reset immediately before each measured arm.
	scalarOps := base.clone()
	ResetGDNStepCounters()
	ScalarHeadStep(scalarOps.st, scalarOps.qn, scalarOps.kn, scalarOps.vh, beta, decay,
		scalarOps.od, scalarOps.kvmem, scalarOps.delta)
	if n := ScalarGDNStepCalls(); n != 1 {
		t.Errorf("ScalarHeadStep called %d time(s), want 1", n)
	}

	vecOps := base.clone()
	ResetGDNStepCounters()
	VectorizedHeadStep(vecOps.st, vecOps.qn, vecOps.kn, vecOps.vh, beta, decay,
		vecOps.od, vecOps.kvmem, vecOps.delta)
	vecCalls := VectorizedGDNStepCalls()
	scalarCalls := ScalarGDNStepCalls()

	if vecCalls == 0 {
		t.Fatal("VectorizedHeadStep on the canonical geometry never invoked compute.Wave32GatedDeltaNetStep")
	}
	if scalarCalls != 0 {
		t.Errorf("VectorizedHeadStep fell through to ScalarHeadStep (%d calls) on the canonical geometry", scalarCalls)
	}
	for _, name := range gdnBufNames2624 {
		assertGDNBits2624(t, "vectorized-vs-scalar buf="+name, scalarOps.buf(name), vecOps.buf(name))
	}
	t.Logf("VectorizedHeadStep invoked the AVX-512 arm %d time(s) and is bit-identical to ScalarHeadStep", vecCalls)

	// Non-canonical geometry: the model must NOT advertise or take the vectorized arm.
	if HasVectorizedDeltaNetFor(64, 64) {
		t.Error("model advertised the AVX-512 GDN kernel for the non-canonical 64x64 geometry")
	}
	kHd, vHd := 64, 64
	small := &gdnOps2624{
		st: make([]float32, kHd*vHd), qn: make([]float32, kHd), kn: make([]float32, kHd),
		vh: make([]float32, vHd), od: make([]float32, vHd), kvmem: make([]float32, vHd), delta: make([]float32, vHd),
	}
	for i := range small.st {
		small.st[i] = rng.Float32()*0.2 - 0.1
	}
	for i := 0; i < kHd; i++ {
		small.qn[i], small.kn[i] = rng.Float32()*0.2-0.1, rng.Float32()*0.2-0.1
	}
	for i := 0; i < vHd; i++ {
		small.vh[i], small.od[i] = rng.Float32()*0.2-0.1, rng.Float32()*0.02-0.01
	}
	ref := &gdnOps2624{
		st: append([]float32(nil), small.st...), qn: append([]float32(nil), small.qn...),
		kn: append([]float32(nil), small.kn...), vh: append([]float32(nil), small.vh...),
		od: append([]float32(nil), small.od...), kvmem: make([]float32, vHd), delta: make([]float32, vHd),
	}
	ResetGDNStepCounters()
	VectorizedHeadStep(ref.st, ref.qn, ref.kn, ref.vh, 0.5, 0.5, ref.od, ref.kvmem, ref.delta)
	if got := VectorizedGDNStepCalls(); got != 0 {
		t.Errorf("non-canonical 64x64 geometry invoked the AVX-512 arm %d time(s)", got)
	}
	if got := ScalarGDNStepCalls(); got != 1 {
		t.Errorf("non-canonical 64x64 geometry: ScalarHeadStep called %d times, want 1", got)
	}
	if !gdnNonZero2624(ref.st, ref.od, ref.kvmem, ref.delta) {
		t.Error("non-canonical 64x64 routing produced an all-zero result — the routing witness is vacuous")
	}
}

// ---------------------------------------------------------------------------
// 6. prefill-vs-decode parity at the level the defect manifested
// ---------------------------------------------------------------------------

// gdnCanonicalCfg2624 is a synthetic Qwen hybrid config whose linear-attention head geometry
// is the CANONICAL 128x128, so both the prefill scan and the decode step exercise the real
// head shape and the decode step really does reach the AVX-512 kernel.
func gdnCanonicalCfg2624() Config {
	cfg := qwen35HybridTestCfg()
	cfg.HiddenSize = gdnDim2624
	cfg.LinearKeyHeadDim = gdnDim2624
	cfg.LinearValueHeadDim = gdnDim2624
	return cfg
}

// TestGDNFMA2624PrefillScanVsDecodeStepsStateBitwise pins spec clause 6 at the level the
// defect actually manifested. It drives the SAME token span twice at the canonical 128x128
// geometry:
//
//   - PREFILL arm: model.linearAttnSeqBatchedState over the whole span from a zero recurrent
//     state (the cold prefill / cache-MISS recomputation), which uses the pure-Go GDN scan.
//   - DECODE arm: Session.linearAttnStep once per token, carrying the session's persistent
//     recurrent state forward (what a radix prefix-cache HIT replays), which routes through
//     model.VectorizedHeadStep -> compute.Wave32GatedDeltaNetStep -> the AVX-512 kernel.
//
// It then asserts the resulting RECURRENT STATE is bit-identical per head, and that the
// per-position outputs are bit-identical. This is the assertion that would have caught
// fak-private#2624: a last-bit split between the prefill scan and the decode kernel makes the
// two arms compute different bytes for the SAME span, so identical input yields different
// logits and greedy argmax can flip onto an already-emitted token.
//
// A regression: any per-head recurrent-state ULP difference between the prefill arm and the
// decode arm, reported with both hex bit patterns.
func TestGDNFMA2624PrefillScanVsDecodeStepsStateBitwise(t *testing.T) {
	cfg := gdnCanonicalCfg2624()
	m := NewSynthetic(cfg)
	_, nV, kHd, vHd, _, _, _ := cfg.linearAttnDims()
	layer := 0
	for !cfg.isLinearAttnLayer(layer) {
		layer++
	}
	if kHd != gdnDim2624 || vHd != gdnDim2624 {
		t.Fatalf("test config is not the canonical geometry: kHd=%d vHd=%d", kHd, vHd)
	}
	vectorized := HasVectorizedDeltaNetFor(kHd, vHd)
	t.Logf("canonical geometry kHd=%d vHd=%d; AVX-512 decode arm reachable=%v", kHd, vHd, vectorized)

	const span = 12
	panel := make([][]float32, span)
	for r := range panel {
		panel[r] = make([]float32, cfg.HiddenSize)
		for i := range panel[r] {
			panel[r][i] = float32(math.Sin(float64((r+1)*(i+3)*2624) * 0.017))
		}
	}

	// DECODE arm: one Session.linearAttnStep per token, carrying the persistent state.
	ds := m.NewSession()
	defer ds.Close()
	mat := residentKernel{m}
	if ds.Cache.linear == nil {
		ds.Cache.linear = newLinearAttnCache(cfg)
	}
	decOut := make([][]float32, span)
	for r := range panel {
		decOut[r] = ds.linearAttnStep(layer, panel[r], mat)
	}
	decState := ds.Cache.linear.layer(cfg, layer)

	// PREFILL arm: the multi-token scan over the identical span from a zero state.
	preState := newLinearAttnLayerState(cfg)
	preOut, err := m.linearAttnSeqBatchedState(layer, panel, &preState)
	if err != nil {
		t.Fatalf("prefill scan refused the canonical panel: %v", err)
	}
	if len(preOut) != span {
		t.Fatalf("prefill scan returned %d rows, want %d", len(preOut), span)
	}

	// The per-head recurrent state is the thing a prefix cache persists and replays.
	if len(preState.recurrent) != nV || len(decState.recurrent) != nV {
		t.Fatalf("recurrent head count mismatch: prefill=%d decode=%d want=%d",
			len(preState.recurrent), len(decState.recurrent), nV)
	}
	for h := 0; h < nV; h++ {
		assertGDNBits2624(t, fmt.Sprintf("prefill-vs-decode recurrent head %d (span %d)", h, span),
			decState.recurrent[h], preState.recurrent[h])
	}
	// The convolution window is part of the same replayed prefix state.
	if len(preState.conv) != len(decState.conv) {
		t.Fatalf("conv window length mismatch: prefill=%d decode=%d", len(preState.conv), len(decState.conv))
	}
	for i := range decState.conv {
		assertGDNBits2624(t, fmt.Sprintf("prefill-vs-decode conv row %d", i), decState.conv[i], preState.conv[i])
	}
	// Per-position outputs, which is where the split would surface as differing logits.
	for r := range preOut {
		assertGDNBits2624(t, fmt.Sprintf("prefill-vs-decode out_proj row %d", r), decOut[r], preOut[r])
	}

	// Non-vacuity: a zero state would make the comparison trivially true.
	liveHeads := 0
	for h := 0; h < nV; h++ {
		if gdnMaxMag2624(preState.recurrent[h]) > 1e-6 {
			liveHeads++
		}
	}
	if liveHeads == 0 {
		t.Fatal("every prefill recurrent head is zero — the prefill-vs-decode comparison is vacuous")
	}
	outMag := gdnMaxMag2624(preOut[len(preOut)-1])
	if outMag == 0 {
		t.Fatal("prefill out_proj row is all zero — the prefill-vs-decode comparison is vacuous")
	}
	t.Logf("span=%d: %d/%d recurrent heads live (max|st| head0=%g), final out_proj max|x|=%g, prefill and decode agree bit-for-bit",
		span, liveHeads, nV, gdnMaxMag2624(preState.recurrent[0]), outMag)
}

// TestGDNFMA2624PrefillArmIsGoScan pins the premise of the prefill-vs-decode witness: the
// multi-token prefill scan really is an INDEPENDENT pure-Go GDN recurrence — it invokes
// neither the AVX-512 kernel nor model.ScalarHeadStep, it runs its own inline gdnFMA32 scan —
// while the decode step really does reach the AVX-512 kernel on the canonical geometry.
//
// That premise is what gives the clause-6 comparison its teeth. If the prefill scan routed
// through the same entry point as the decode step, the two arms would share one
// implementation and the comparison would prove nothing about the FMA split; asserting the
// independence here means a future refactor that collapses the two arms into one is itself a
// finding.
//
// A regression: the prefill arm starting to invoke the AVX-512 kernel (arms share an
// implementation, the cross-implementation invariant goes untested), or the decode arm
// ceasing to invoke it.
func TestGDNFMA2624PrefillArmIsGoScan(t *testing.T) {
	cfg := gdnCanonicalCfg2624()
	m := NewSynthetic(cfg)
	_, _, kHd, vHd, _, _, _ := cfg.linearAttnDims()
	layer := 0
	for !cfg.isLinearAttnLayer(layer) {
		layer++
	}
	if !HasVectorizedDeltaNetFor(kHd, vHd) {
		t.Skip("canonical 128x128 vectorized GDN kernel unavailable; the arm-premise contrast is not observable")
	}
	const span = 6
	panel := make([][]float32, span)
	for r := range panel {
		panel[r] = make([]float32, cfg.HiddenSize)
		for i := range panel[r] {
			panel[r][i] = float32(math.Sin(float64((r+2)*(i+5)*2624) * 0.013))
		}
	}

	preState := newLinearAttnLayerState(cfg)
	ResetGDNStepCounters()
	if _, err := m.linearAttnSeqBatchedState(layer, panel, &preState); err != nil {
		t.Fatalf("prefill scan refused the canonical panel: %v", err)
	}
	preVec := VectorizedGDNStepCalls()
	preScalar := ScalarGDNStepCalls()
	if preVec != 0 {
		t.Errorf("the multi-token prefill scan invoked the AVX-512 arm %d time(s); it must stay the pure-Go scan "+
			"so that the clause-6 prefill-vs-decode comparison still crosses implementations", preVec)
	}
	// The prefill scan is a third, independent Go implementation (an inline gdnFMA32 loop in
	// qwen35_chunked.go) — it routes through NEITHER model.ScalarHeadStep NOR the AVX-512
	// kernel. That is exactly the geometry of the original defect: a cache MISS recomputes
	// the span with a Go loop while a cache HIT replays kernel-produced state.
	t.Logf("arm premise: the prefill scan invoked 0 vectorized and %d scalar head step(s) — it is an "+
		"independent inline Go scan, so the clause-6 comparison genuinely crosses implementations", preScalar)

	ds := m.NewSession()
	defer ds.Close()
	if ds.Cache.linear == nil {
		ds.Cache.linear = newLinearAttnCache(cfg)
	}
	ResetGDNStepCounters()
	for r := range panel {
		ds.linearAttnStep(layer, panel[r], residentKernel{m})
	}
	decVec := VectorizedGDNStepCalls()
	if decVec == 0 {
		t.Error("the decode step never invoked the AVX-512 arm on the canonical geometry; " +
			"the clause-6 prefill-vs-decode comparison would then be Go-vs-Go and vacuous")
	}
	t.Logf("arm premise: decode used %d vectorized GDN step(s) over %d tokens on the canonical 128x128 geometry", decVec, span)
}

// TestGDNFMA2624EnvSuppressesModelDispatch pins spec clause 4 from the model side: with
// FAK_VECTORIZED_DELTANET set to a disable spelling, the model must not advertise or take the
// vectorized arm on the canonical geometry, and the resulting bytes must equal the pure scalar
// reference. t.Setenv keeps the environment from leaking.
//
// A regression: model.HasVectorizedDeltaNetFor(128,128) still true under suppression, or the
// suppressed dispatch differing from ScalarHeadStep.
func TestGDNFMA2624EnvSuppressesModelDispatch(t *testing.T) {
	cfg := gdnCanonicalCfg2624()
	_, _, kHd, vHd, _, _, _ := cfg.linearAttnDims()
	rng := rand.New(rand.NewSource(int64(2625300)))
	base := gdnNewOps2624()
	gdnSeedOps2624(rng, base)
	beta, decay := float32(0.75), float32(0.125)

	ref := base.clone()
	ScalarHeadStep(ref.st, ref.qn, ref.kn, ref.vh, beta, decay, ref.od, ref.kvmem, ref.delta)

	for _, v := range []string{"0", "false", "no", "off", "OFF"} {
		t.Setenv("FAK_VECTORIZED_DELTANET", v)
		if HasVectorizedDeltaNet() {
			t.Errorf("model.HasVectorizedDeltaNet() is true under FAK_VECTORIZED_DELTANET=%q", v)
		}
		if HasVectorizedDeltaNetFor(kHd, vHd) {
			t.Errorf("model advertised the vectorized GDN arm for the canonical geometry under %q", v)
		}
		got := base.clone()
		ResetGDNStepCounters()
		VectorizedHeadStep(got.st, got.qn, got.kn, got.vh, beta, decay, got.od, got.kvmem, got.delta)
		if n := VectorizedGDNStepCalls(); n != 0 {
			t.Errorf("%q: the vectorized arm ran %d time(s) under suppression", v, n)
		}
		if n := ScalarGDNStepCalls(); n != 1 {
			t.Errorf("%q: ScalarHeadStep ran %d time(s), want 1", v, n)
		}
		for _, name := range gdnBufNames2624 {
			assertGDNBits2624(t, "env="+v+" buf="+name, ref.buf(name), got.buf(name))
		}
	}
	if !gdnNonZero2624(ref.st, ref.od, ref.kvmem, ref.delta) {
		t.Fatal("the scalar reference produced an all-zero result — the suppression identity is vacuous")
	}

	// Enable spellings must make the model capability equal the compute-package detection.
	for _, v := range []string{"1", "true", "on"} {
		t.Setenv("FAK_VECTORIZED_DELTANET", v)
		if got, want := HasVectorizedDeltaNet(), compute.HasVectorizedDeltaNet(); got != want {
			t.Errorf("%q: model capability=%v compute capability=%v", v, got, want)
		}
		if got, want := HasVectorizedDeltaNetFor(kHd, vHd), compute.HasVectorizedDeltaNetFor(kHd, vHd); got != want {
			t.Errorf("%q: model geometry capability=%v compute geometry capability=%v", v, got, want)
		}
	}
}
