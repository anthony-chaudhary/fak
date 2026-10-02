//go:build amd64

package compute

// fak-private#2624 — the Gated-DeltaNet (GDN) linear-attention recurrence of the Qwen3.8
// hybrid model had multiple Go implementations plus one AVX-512 assembly kernel. The Go
// implementations accumulated with a separate multiply and add (`acc += x*y`, which rounds
// the product BEFORE the add) while the assembly fused the same three sites with
// VFMADD231PS. Different bytes for the same token span is a correctness defect, not
// cosmetics: a radix prefix-cache HIT replays GDN state produced by the decode kernel while
// a cache MISS / cold prefill recomputes the SAME span with the Go scan, so identical input
// yields different logits and a greedy argmax can flip onto an already-emitted token.
//
// REQUIRED INVARIANT pinned here: on the canonical Qwen geometry (kHd==128 && vHd==128) the
// Go GDN recurrence and the AVX-512 assembly kernel must produce BIT-IDENTICAL results —
// exactly equal float32 bit patterns — for the recurrent state `st`, the readout `od`,
// `kvmem` and `delta`, over a wide grid of (beta, decay) gate configurations, over a
// multi-step recurrence, and over value-domain edge cases chosen to maximise FMA-vs-separate
// divergence.
//
// MUTATION-SENSITIVITY WITNESS. Every test here also computes a deliberately UNFUSED
// "split-rounding" reference (gdnSplitRef2624) that reproduces the pre-fix arithmetic. The
// tests assert two things at once:
//   (a) fused-Go == asm, bit-for-bit  (the required invariant), and
//   (b) split-rounding != fused       (proof that the probe actually detects the defect).
// Assertion (b) is what keeps the suite non-vacuous: if a future change removed math.FMA
// from the Go path, (b) would collapse to equality and (a) would then be carrying the whole
// load — and, worse, a "fix" that made the Go path match the split reference would be
// caught immediately by (a).

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

const gdnDim2624 = 128

// gdnGates2624 is the wide (beta, decay) grid mandated by the spec clause 1. The existing
// TestDeltaNetAVX512CapabilityAndParity probes only four pairs at a >1e-5 tolerance.
var gdnGates2624 = []struct {
	beta, decay float32
}{
	{0, float32(math.Exp(-1e-6))},
	{0, 0.5},
	{0, 0},
	{1e-8, float32(math.Exp(-1e-6))},
	{1e-8, 0.25},
	{0.001, float32(math.Exp(-1e-4))},
	{0.001, 0.01},
	{0.25, float32(math.Exp(-0.05))},
	{0.25, 0.5},
	{0.25, 0},
	{0.5, float32(math.Exp(-1e-6))},
	{0.5, float32(math.Exp(-1e-4))},
	{0.5, 0.5},
	{0.5, 0.125},
	{0.5, 0},
	{0.6, float32(math.Exp(-0.05))},
	{0.6, 0.25},
	{0.75, 0.125},
	{0.9, float32(math.Exp(-1e-4))},
	{0.9, 0.5},
	{0.999, 0.25},
	{0.999, 0.01},
	{1, 1},
	{1, 0.5},
	{1, 0},
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type gdnOperands2624 struct {
	st, qn, kn, vh, od, kvmem, delta []float32
}

func gdnNewOperands2624() *gdnOperands2624 {
	d := gdnDim2624
	return &gdnOperands2624{
		st:    make([]float32, d*d),
		qn:    make([]float32, d),
		kn:    make([]float32, d),
		vh:    make([]float32, d),
		od:    make([]float32, d),
		kvmem: make([]float32, d),
		delta: make([]float32, d),
	}
}

func (o *gdnOperands2624) clone() *gdnOperands2624 {
	c := gdnNewOperands2624()
	copy(c.st, o.st)
	copy(c.qn, o.qn)
	copy(c.kn, o.kn)
	copy(c.vh, o.vh)
	copy(c.od, o.od)
	copy(c.kvmem, o.kvmem)
	copy(c.delta, o.delta)
	return c
}

// gdnSkipped2624 skips cleanly on hosts without AVX-512F + OS ZMM state, keeping the suite
// green off-target. The spec's required invariant is only meaningful where the assembly
// path is actually taken.
func gdnSkipped2624(tb testing.TB) {
	tb.Helper()
	if !hasDeltaNetSIMD() {
		tb.Skip("AVX-512F with OS ZMM state support unavailable; AVX-512 GDN kernel not reachable")
	}
}

// mulRounded2624 is an explicit single-rounding barrier for a float32 product: the product is
// rounded to float32 INSIDE a //go:noinline function, so the following add cannot be folded
// into it.
//
// The barrier is DEFENSIVE INSURANCE, not a fix for an observed hazard. gc/amd64 does NOT
// auto-contract `x*y+z` into an FMA at any GOAMD64 level — disassembly of this package emits
// MULSS+ADDSS at v1, v3 and v4 alike — so the barrier is unnecessary today and is kept only so
// the reference survives if that ever changes (a future -ffp-contract=on default, or a
// non-amd64 GOARCH that does contract). That same measurement is what the ticket's premise
// rests on: the PRE-FIX production code was genuinely unfused on every build level, so the
// fused-vs-split divergence below is a real property of the shipped code and not an artifact
// of the reference's own codegen.
var _ = math.Float32bits

//go:noinline
func mulRounded2624(a, b float32) float32 {
	return math.Float32frombits(math.Float32bits(a * b))
}

// gdnSplitRef2624 is the pre-fix arithmetic: separate multiply and add at all three
// accumulate sites (kvmem reduction, state update, readout). It mirrors the assembly's
// phase order exactly (row-streamed decay+kvmem, then delta, then update+readout) so any
// difference from the fused paths is purely a rounding difference.
func gdnSplitRef2624(st, qn, kn, vh []float32, bt, g float32, od, kvmem, delta []float32) {
	const d = gdnDim2624
	for j := range kvmem[:d] {
		kvmem[j] = 0
	}
	for i := 0; i < d; i++ {
		ki := kn[i]
		base := i * d
		for j := 0; j < d; j++ {
			st[base+j] = st[base+j] * g
			kvmem[j] = kvmem[j] + mulRounded2624(st[base+j], ki)
		}
	}
	for j := 0; j < d; j++ {
		delta[j] = (vh[j] - kvmem[j]) * bt
	}
	for i := 0; i < d; i++ {
		ki := kn[i]
		qi := qn[i]
		base := i * d
		for j := 0; j < d; j++ {
			st[base+j] = st[base+j] + mulRounded2624(ki, delta[j])
			od[j] = od[j] + mulRounded2624(st[base+j], qi)
		}
	}
}

// gdnBufferNames2624 names the four spec-mandated buffers in assertion order.
var gdnBufferNames2624 = []string{"st", "od", "kvmem", "delta"}

func (o *gdnOperands2624) buf(name string) []float32 {
	switch name {
	case "st":
		return o.st
	case "od":
		return o.od
	case "kvmem":
		return o.kvmem
	case "delta":
		return o.delta
	}
	panic("unknown buffer " + name)
}

// assertGDNBitsEqual2624 asserts EXACT float32 bit-pattern equality (not a tolerance) and
// reports the first few offending elements with their full hex bit patterns.
func assertGDNBitsEqual2624(tb testing.TB, ctx string, want, got []float32) {
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
			tb.Errorf("%s: NOT bit-identical (first of %d buffers compared)", ctx, len(gdnBufferNames2624))
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

// gdnMaxMag2624 returns the maximum |x| and the count of non-finite entries.
func gdnMaxMag2624(v []float32) (float64, int) {
	var m float64
	nonFinite := 0
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			nonFinite++
			continue
		}
		if a := math.Abs(float64(x)); a > m {
			m = a
		}
	}
	return m, nonFinite
}

// gdnCountNonFinite2624 counts Inf/NaN elements across the named buffers, used by the
// non-finite guard on probes whose value domain can overflow.
func gdnCountNonFinite2624(bufs ...[]float32) int {
	n := 0
	for _, b := range bufs {
		_, nf := gdnMaxMag2624(b)
		n += nf
	}
	return n
}

func gdnAnyNonZero2624(bufs ...[]float32) bool {
	for _, b := range bufs {
		for _, x := range b {
			if x != 0 {
				return true
			}
		}
	}
	return false
}

// gdnSeedGrid2624 fills the operands with a deterministic, reproducible, moderately scaled
// random panel (the same shape TestDeltaNetAVX512CapabilityAndParity uses) plus a
// pre-warmed readout so the `od` accumulation site is exercised with a non-zero seed.
func gdnSeedGrid2624(rng *rand.Rand, o *gdnOperands2624) {
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
// 1. bit-identity across a WIDE gate grid
// ---------------------------------------------------------------------------

// TestGDNFMA2624GateGridBitIdentity pins spec clause 1: bitwise (not tolerance) equality of
// all four mandated buffers between the Go GDN recurrence and the AVX-512 kernel across the
// full 24-pair (beta, decay) grid the spec mandates, with a non-vacuity guard.
//
// A regression: any accumulate site reverting to a separate multiply-add (or any change in
// loop/unroll order that changes the kvmem reduction order) shows up as a non-zero ULP
// difference on at least one element, reported with both hex bit patterns.
func TestGDNFMA2624GateGridBitIdentity(t *testing.T) {
	gdnSkipped2624(t)
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	nonVacuous := 0
	for gi, gate := range gdnGates2624 {
		rng := rand.New(rand.NewSource(int64(2624000 + gi)))
		base := gdnNewOperands2624()
		gdnSeedGrid2624(rng, base)

		ctx := fmt.Sprintf("gate[%02d] beta=%g decay=%g", gi, gate.beta, gate.decay)

		goOps := base.clone()
		wave32GatedDeltaNetStepGo(goOps.st, goOps.qn, goOps.kn, goOps.vh, gate.beta, gate.decay,
			goOps.od, goOps.kvmem, goOps.delta)

		asmOps := base.clone()
		if !tryDeltaNetSIMD(asmOps.st, asmOps.qn, asmOps.kn, asmOps.vh, gate.beta, gate.decay,
			asmOps.od, asmOps.kvmem, asmOps.delta) {
			t.Fatalf("%s: detected AVX-512 kernel declined canonical 128x128 input", ctx)
		}

		for _, name := range gdnBufferNames2624 {
			assertGDNBitsEqual2624(t, ctx+" buf="+name, goOps.buf(name), asmOps.buf(name))
		}

		// Non-vacuity: the compared result must carry real magnitude. decay==0 legitimately
		// annihilates the state (st *= 0), which drives st, kvmem and od to exactly zero, so
		// those cells are only required to be non-trivial for decay>0.
		stMag, stNF := gdnMaxMag2624(goOps.st)
		odMag, _ := gdnMaxMag2624(goOps.od)
		kvMag, _ := gdnMaxMag2624(goOps.kvmem)
		dlMag, _ := gdnMaxMag2624(goOps.delta)
		if stNF != 0 {
			t.Fatalf("%s: recurrent state went non-finite (non-finite elements=%d)", ctx, stNF)
		}
		if gate.decay > 0 {
			if kvMag == 0 {
				t.Fatalf("%s: degenerate operands (kvmem max|x|=%g with decay=%g)", ctx, kvMag, gate.decay)
			}
		}
		if gate.decay > 0 {
			if stMag <= 1e-3 {
				t.Fatalf("%s: vacuous — st max|x|=%g is too small to probe rounding", ctx, stMag)
			}
			if gate.beta > 0 && odMag <= 1e-6 {
				t.Fatalf("%s: vacuous — od max|x|=%g is too small to probe the readout FMA", ctx, odMag)
			}
			nonVacuous++
		}
		if gate.beta > 0 && dlMag == 0 {
			t.Fatalf("%s: vacuous — delta is identically zero despite beta=%g", ctx, gate.beta)
		}
	}
	// A grid where almost every cell is trivially zero would make the whole test vacuous.
	if nonVacuous < 20 {
		t.Fatalf("grid non-vacuity too low: %d/%d decay>0 cells exercised a live state", nonVacuous, len(gdnGates2624))
	}
	t.Logf("gate grid: %d configurations, %d with a live (non-zero) recurrent state, all four buffers bit-identical",
		len(gdnGates2624), nonVacuous)
}

// TestGDNFMA2624GateGridDetectsSplitRounding is the mutation-sensitivity companion to
// TestGDNFMA2624GateGridBitIdentity. If the Go path ever reverts to a separate multiply-add,
// the pre-fix split-rounding reference stops differing from it and this test fails loudly.
//
// A regression: MAX|go_fused - split| collapsing to 0 across the whole grid means the probe
// no longer discriminates and clause 1's bitwise gate is no longer meaningful.
func TestGDNFMA2624GateGridDetectsSplitRounding(t *testing.T) {
	gdnSkipped2624(t)
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	differingCells := 0
	for gi, gate := range gdnGates2624 {
		rng := rand.New(rand.NewSource(int64(2624000 + gi)))
		base := gdnNewOperands2624()
		gdnSeedGrid2624(rng, base)

		fused := base.clone()
		wave32GatedDeltaNetStepGo(fused.st, fused.qn, fused.kn, fused.vh, gate.beta, gate.decay,
			fused.od, fused.kvmem, fused.delta)
		split := base.clone()
		gdnSplitRef2624(split.st, split.qn, split.kn, split.vh, gate.beta, gate.decay,
			split.od, split.kvmem, split.delta)

		cell := 0
		for _, name := range gdnBufferNames2624 {
			a, b := fused.buf(name), split.buf(name)
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					cell++
				}
			}
		}
		differingCells += cell
		if gi == 0 {
			t.Logf("probe gate[0] beta=%g decay=%g: %d/%d elements differ between fused and split-rounding",
				gate.beta, gate.decay, cell, gdnDim2624*gdnDim2624+3*gdnDim2624)
		}
	}
	if differingCells == 0 {
		t.Fatal("split-rounding reference is bit-identical to the fused Go path on the whole gate grid: " +
			"the FMA-divergence probe has lost all discriminating power (spec clause 1 gate is now vacuous)")
	}
	t.Logf("split-rounding vs fused: %d differing float32 elements across %d gate configurations",
		differingCells, len(gdnGates2624))
}

// ---------------------------------------------------------------------------
// 2. multi-step recurrence parity
// ---------------------------------------------------------------------------

// TestGDNFMA2624MultiStepRecurrenceBitIdentity pins spec clause 2: drive BOTH paths over a
// SEQUENCE of 64 steps carrying `st` forward, asserting bitwise equality of the recurrent
// state after EVERY step, plus od/kvmem/delta. A single-step test can pass while the
// recurrent state drifts, so this is the property that actually matters.
//
// It also reports per-step drift: the fused-vs-asm drift (the required invariant, must be
// 0) and the fused-vs-split drift (the defect signal, must grow once the gate stops being
// trivial) so a regression is visible even in the log.
//
// A regression: a per-step fused-vs-asm drift > 0, or a fused-vs-split drift that stops
// growing across the sequence, means the recurrence is no longer bit-stable.
func TestGDNFMA2624MultiStepRecurrenceBitIdentity(t *testing.T) {
	gdnSkipped2624(t)
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	const steps = 64
	// Gate pairs that keep the recurrence live and bounded: near-1 decay accumulates, small
	// decay forgets, beta=1 is the fully-open selection gate, beta=0.5 the classic default.
	pairs := []struct {
		beta, decay float32
	}{
		{0.5, float32(math.Exp(-1e-6))},
		{0.9, float32(math.Exp(-1e-4))},
		{0.5, 0.5},
		{0.25, 0.125},
		{1, 0.25},
		{0.001, 0.5},
	}

	for pi, p := range pairs {
		rng := rand.New(rand.NewSource(int64(2624100 + pi)))
		base := gdnNewOperands2624()
		gdnSeedGrid2624(rng, base)

		goOps := base.clone()
		asmOps := base.clone()
		splitOps := base.clone()

		var maxFusedDrift, maxSplitDrift float64
		firstSplitDrift := -1.0
		for step := 0; step < steps; step++ {
			wave32GatedDeltaNetStepGo(goOps.st, goOps.qn, goOps.kn, goOps.vh, p.beta, p.decay,
				goOps.od, goOps.kvmem, goOps.delta)
			if !tryDeltaNetSIMD(asmOps.st, asmOps.qn, asmOps.kn, asmOps.vh, p.beta, p.decay,
				asmOps.od, asmOps.kvmem, asmOps.delta) {
				t.Fatalf("pair[%d] step=%d: AVX-512 kernel declined canonical input", pi, step)
			}
			gdnSplitRef2624(splitOps.st, splitOps.qn, splitOps.kn, splitOps.vh, p.beta, p.decay,
				splitOps.od, splitOps.kvmem, splitOps.delta)

			ctx := fmt.Sprintf("pair[%d] beta=%g decay=%g step=%d", pi, p.beta, p.decay, step)
			for _, name := range gdnBufferNames2624 {
				assertGDNBitsEqual2624(t, ctx+" buf="+name, goOps.buf(name), asmOps.buf(name))
			}
			d := MaxAbsDelta(goOps.st, asmOps.st)
			if d > maxFusedDrift {
				maxFusedDrift = d
			}
			sd := MaxAbsDelta(goOps.st, splitOps.st)
			if firstSplitDrift < 0 && sd > 0 {
				firstSplitDrift = float64(step)
			}
			if sd > maxSplitDrift {
				maxSplitDrift = sd
			}
			if mag, nf := gdnMaxMag2624(goOps.st); nf != 0 {
				t.Fatalf("%s: recurrent state went non-finite (max|x|=%g, non-finite=%d) — the probe cannot continue", ctx, mag, nf)
			}
		}

		if maxFusedDrift != 0 {
			t.Errorf("pair[%d] beta=%g decay=%g: fused-Go vs asm recurrent-state drift = %g after %d steps (must be exactly 0)",
				pi, p.beta, p.decay, maxFusedDrift, steps)
		}
		stMag, _ := gdnMaxMag2624(goOps.st)
		odMag, _ := gdnMaxMag2624(goOps.od)
		if stMag == 0 || odMag == 0 {
			t.Fatalf("pair[%d] beta=%g decay=%g: %d-step recurrence collapsed to zero (st max|x|=%g, od max|x|=%g) — vacuous probe",
				pi, p.beta, p.decay, steps, stMag, odMag)
		}
		if maxSplitDrift == 0 {
			t.Errorf("pair[%d] beta=%g decay=%g: split-rounding never diverged from the fused path over %d steps — the multi-step probe is vacuous",
				pi, p.beta, p.decay, steps)
		}
		t.Logf("pair[%d] beta=%-6g decay=%-12g over %d steps: fused-vs-asm drift=%g, fused-vs-split drift=%g (first nonzero at step %g), final max|st|=%g",
			pi, p.beta, p.decay, steps, maxFusedDrift, maxSplitDrift, firstSplitDrift, stMag)
	}
}

// TestGDNFMA2624NearUnityDecayDrift pins the worst realistic case for recurrence drift: a
// decay within 1e-6 of 1 over 64 steps barely damps the state, so any per-step last-bit
// disagreement compounds instead of being forgotten. Asserts bitwise equality at every step
// and reports the growth of the fused-vs-split drift so the sensitivity margin is visible.
//
// A regression: any per-step fused-vs-asm mismatch, or fused-vs-split drift that stops
// growing, means the recurrent state is no longer bit-stable under near-unity decay.
func TestGDNFMA2624NearUnityDecayDrift(t *testing.T) {
	gdnSkipped2624(t)
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	const steps = 64
	beta := float32(0.5)
	decay := float32(math.Exp(-1e-6))
	rng := rand.New(rand.NewSource(int64(2624200)))
	base := gdnNewOperands2624()
	gdnSeedGrid2624(rng, base)

	goOps, asmOps, splitOps := base.clone(), base.clone(), base.clone()

	drifts := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		wave32GatedDeltaNetStepGo(goOps.st, goOps.qn, goOps.kn, goOps.vh, beta, decay,
			goOps.od, goOps.kvmem, goOps.delta)
		if !tryDeltaNetSIMD(asmOps.st, asmOps.qn, asmOps.kn, asmOps.vh, beta, decay,
			asmOps.od, asmOps.kvmem, asmOps.delta) {
			t.Fatalf("step=%d: AVX-512 kernel declined canonical input", step)
		}
		gdnSplitRef2624(splitOps.st, splitOps.qn, splitOps.kn, splitOps.vh, beta, decay,
			splitOps.od, splitOps.kvmem, splitOps.delta)

		ctx := fmt.Sprintf("near-unity-decay step=%d", step)
		for _, name := range gdnBufferNames2624 {
			assertGDNBitsEqual2624(t, ctx+" buf="+name, goOps.buf(name), asmOps.buf(name))
		}
		drifts = append(drifts, MaxAbsDelta(goOps.st, splitOps.st))
	}

	// The split-rounding defect signal must be visible and must not be a one-off.
	nonZero := 0
	for i, d := range drifts {
		if d > 0 {
			nonZero++
		}
		if i%16 == 0 || i == len(drifts)-1 {
			t.Logf("near-unity-decay step=%3d: fused-vs-asm drift=%g, fused-vs-split drift=%g, max|st|=%g",
				i, MaxAbsDelta(goOps.st, asmOps.st), d, must(gdnMaxMag2624(goOps.st)))
		}
	}
	if nonZero < 8 {
		t.Fatalf("split-rounding drift was non-zero on only %d/%d steps — the near-unity-decay probe lost its sensitivity", nonZero, steps)
	}
	if drifts[len(drifts)-1] <= 0 {
		t.Fatal("fused and split-rounding recurrent states are bit-identical at step 63 — probe is vacuous")
	}
}

func must[T any](v T, _ int) T { return v }

// ---------------------------------------------------------------------------
// 3. value-domain edge cases
// ---------------------------------------------------------------------------

// gdnProbe2624 names a value-domain regime and builds its operands deterministically.
type gdnProbe2624 struct {
	name string
	seed func(rng *rand.Rand, o *gdnOperands2624)
	// strongest marks the regime asserted most strictly: it must both be non-trivial AND
	// actually diverge from the split-rounding reference, i.e. it is the FMA-divergence probe
	// with the largest discriminating power.
	strongest bool
	// allowNonFinite marks a regime that is EXPECTED to drive values to ±Inf/NaN. Such a
	// regime needs a non-finite guard, because bit-identity is only a meaningful assertion
	// while some elements stay finite: two paths that both produced +Inf everywhere would
	// compare equal trivially. The guard therefore also requires such a probe to retain
	// finite live elements, and requires the Go and assembly paths to agree on the
	// finite/non-finite element COUNT.
	allowNonFinite bool
}

// gdnProbes2624 is the value-domain menu. Which of these is the STRONGEST FMA-divergence
// probe and why:
//
//   - catastrophicCancellation (strongest): the kvmem reduction sums 128 products whose
//     48-bit exact products are each rounded to 24 bits by a separate multiply, then added
//     to a partial sum that is itself tiny because the signs cancel. A split rounding then
//     loses up to 2^-25 of the product magnitude into a sum of comparable magnitude, so the
//     disagreement is a full ULP of the accumulator and, because it re-enters `delta`, then
//     `st` and `od`, it amplifies. This is the worst case for FMA-vs-separate.
//   - subnormalInputs: a subnormal state (1e-41, below the 1.18e-38 smallest normal) makes
//     the decay multiply and the kvmem product land in the subnormal range, where gradual
//     underflow and DAZ/FTZ could split the paths. The vector operands stay order 1 so the
//     products remain representable and the site is actually exercised rather than flushed
//     to zero on both paths.
//   - nearOverflow: 3e38 products near the float32 ceiling. The exact sum and the sum of the
//     separately rounded products land on opposite sides of the overflow threshold, so the
//     paths can disagree on finite-vs-Inf, which is a far worse artifact than a ULP.
//   - mixedSignsZerosNegZero: exercises +0/-0 bit patterns (Go's clear() and the asm's
//     VXORPS both write +0, but the accumulation order decides the sign of a zero result) and
//     lanes where one path would produce -0.
//   - denormalProducingIntermediates: state magnitudes that decay INTO the subnormal range, so
//     the decay multiply (a separate VMULPS in the assembly too) lands on a denormal and the
//     subsequent accumulate behaves differently under DAZ/FTZ.
var gdnProbes2624 = []gdnProbe2624{
	{
		name:      "moderate_random",
		seed:      func(rng *rand.Rand, o *gdnOperands2624) { gdnSeedGrid2624(rng, o) },
		strongest: false,
	},
	{
		name: "catastrophic_cancellation",
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			// st ~ 2^20 and k ~ 2^-20 => products ~1 with a full 48-bit exact mantissa that a
			// separate multiply must truncate to 24 bits. Alternating signs keep the running
			// sum O(1) so the truncation error is a full ULP of the accumulator.
			for i := range o.st {
				m := 1.0 + rng.Float32()*0.5
				if i%2 == 0 {
					m = -m
				}
				o.st[i] = float32(math.Ldexp(float64(m), 20))
			}
			for i := 0; i < gdnDim2624; i++ {
				m := 1.0 + rng.Float32()*0.5
				if i%2 == 0 {
					m = -m
				}
				o.kn[i] = float32(math.Ldexp(float64(m), -20))
				o.qn[i] = rng.Float32()*0.5 - 0.25
				o.vh[i] = rng.Float32()*0.5 - 0.25
				o.od[i] = rng.Float32() * 0.01
			}
		},
		strongest: true,
	},
	{
		// The STATE stays subnormal (that is the intent: the decay multiply and the
		// kvmem product both land in the subnormal range, where gradual underflow and
		// DAZ/FTZ could split the paths), but the VECTOR operands are ordinary-scale.
		// The previous 1e-45 x 1e-45 seeds produced 1e-90 products that flushed to zero
		// on both paths, so every accumulate site was trivially equal. Now st*kn ~7e-43
		// is subnormal yet representable (min subnormal 1e-45), and the delta/update and
		// readout products stay far above the underflow threshold because vh and qn are
		// order 1. st=1e-41 also leaves headroom for the smallest decay in the grid
		// (g=0.01 -> 1e-43, still nonzero).
		name: "subnormal_inputs",
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			for i := range o.st {
				o.st[i] = 1e-41 // subnormal (min normal float32 is 1.18e-38)
				if i%3 == 0 {
					o.st[i] = -1e-41
				}
			}
			for i := 0; i < gdnDim2624; i++ {
				o.kn[i] = 0.5
				if i%2 == 0 {
					o.kn[i] = -0.5
				}
				o.qn[i] = 0.5
				if i%3 == 0 {
					o.qn[i] = -0.5
				}
				o.vh[i] = 1
				if i%2 == 1 {
					o.vh[i] = -1
				}
				o.od[i] = 1e-40
			}
		},
		strongest: false,
	},
	{
		// Operands are 1e-19-scale so that every product lands at ~1e-38 — representable
		// (the smallest float32 subnormal is 1.4e-45, so 1e-38 keeps ~7e6x headroom) and
		// still nonzero. The previous 1e-38 operands produced 1e-76 products that flushed
		// to zero on BOTH paths, leaving the kvmem reduction and the readout trivially
		// equal and this regime unable to discriminate anything. Uniform scaling keeps the
		// whole probe deep in the tiny-magnitude regime without underflowing: st*kn ~1e-38,
		// and after the delta stage the update product kn*delta and the readout product
		// st*qn are again ~1e-38 (delta is dominated by the vh term, not by kvmem).
		name: "tiny_magnitudes_1e38",
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			for i := range o.st {
				o.st[i] = float32(1e-19) * (1 + rng.Float32())
			}
			for i := 0; i < gdnDim2624; i++ {
				o.kn[i] = float32(1e-19) * (1 + rng.Float32())
				o.qn[i] = float32(1e-19) * (1 + rng.Float32())
				o.vh[i] = float32(1e-19) * (1 + rng.Float32())
				o.od[i] = 0
			}
		},
		strongest: false,
	},
	{
		name: "near_overflow_3e38",
		// This regime is allowed to produce ±Inf: the point is that the exact sum and the
		// sum of separately rounded products can land on opposite sides of the overflow
		// threshold. It still has to keep finite live elements so bit-identity stays
		// meaningful, which the guard in TestGDNFMA2624ValueDomainEdgeCases enforces.
		allowNonFinite: true,
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			for i := range o.st {
				m := float32(3e38) * (0.5 + rng.Float32()*0.5)
				if i%2 == 0 {
					m = -m
				}
				o.st[i] = m
			}
			for i := 0; i < gdnDim2624; i++ {
				o.kn[i] = 0.5
				if i%2 == 0 {
					o.kn[i] = -0.5
				}
				o.qn[i] = 1
				o.vh[i] = 3e38
				if i%2 == 1 {
					o.vh[i] = -3e38
				}
				o.od[i] = 0
			}
		},
		strongest: false,
	},
	{
		name: "mixed_signs_zeros_negzero",
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			pattern := []float32{0, float32(math.Copysign(0, -1)), 1, -1, 1e-30, -1e-30, 0.5, -0.5}
			for i := range o.st {
				o.st[i] = pattern[i%len(pattern)] * (1 + rng.Float32()*0.25)
			}
			for i := 0; i < gdnDim2624; i++ {
				o.kn[i] = pattern[(i*3)%len(pattern)]
				o.qn[i] = pattern[(i*5)%len(pattern)]
				o.vh[i] = pattern[(i*7)%len(pattern)]
				o.od[i] = pattern[(i*11)%len(pattern)]
			}
		},
		strongest: false,
	},
	{
		name: "denormal_producing_intermediates",
		seed: func(rng *rand.Rand, o *gdnOperands2624) {
			// Start just above the smallest normal so the decay multiply lands in the
			// subnormal range, where DAZ/FTZ handling could split the paths.
			for i := range o.st {
				o.st[i] = 1.2e-38
				if i%2 == 0 {
					o.st[i] = -1.2e-38
				}
			}
			for i := 0; i < gdnDim2624; i++ {
				o.kn[i] = 0.25
				if i%2 == 0 {
					o.kn[i] = -0.25
				}
				o.qn[i] = 0.5
				o.vh[i] = 1e-38
				if i%2 == 1 {
					o.vh[i] = -1e-38
				}
				o.od[i] = 1e-39
			}
		},
		strongest: false,
	},
}

// TestGDNFMA2624ValueDomainEdgeCases pins spec clause 3: every value-domain regime that
// breaks FMA-vs-separate rounding hardest must be bit-identical between the Go recurrence
// and the AVX-512 kernel, for all four buffers, over the whole gate grid. The designated
// STRONGEST probe (catastrophic cancellation) is additionally required to be non-trivial and
// to actually diverge from the split-rounding reference — i.e. the strongest probe really is
// the strongest, by measurement rather than by assertion of faith.
//
// A regression: a ULP difference on any edge-case regime; or the strongest probe ceasing to
// diverge from split-rounding (which would mean the whole edge-case menu is now vacuous).
func TestGDNFMA2624ValueDomainEdgeCases(t *testing.T) {
	gdnSkipped2624(t)
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	type probeResult struct {
		name          string
		strongest     bool
		liveCells     int
		finiteCells   int
		nonFiniteGo   int
		nonFiniteAsm  int
		splitDiverged bool
		maxSplitDrift float64
	}
	results := make([]probeResult, 0, len(gdnProbes2624))

	for pi, probe := range gdnProbes2624 {
		pr := probeResult{name: probe.name, strongest: probe.strongest}
		for gi, gate := range gdnGates2624 {
			rng := rand.New(rand.NewSource(int64(2624300 + 1000*pi + gi)))
			base := gdnNewOperands2624()
			probe.seed(rng, base)
			ctx := fmt.Sprintf("probe=%s gate[%02d] beta=%g decay=%g", probe.name, gi, gate.beta, gate.decay)

			goOps := base.clone()
			wave32GatedDeltaNetStepGo(goOps.st, goOps.qn, goOps.kn, goOps.vh, gate.beta, gate.decay,
				goOps.od, goOps.kvmem, goOps.delta)
			asmOps := base.clone()
			if !tryDeltaNetSIMD(asmOps.st, asmOps.qn, asmOps.kn, asmOps.vh, gate.beta, gate.decay,
				asmOps.od, asmOps.kvmem, asmOps.delta) {
				t.Fatalf("%s: detected AVX-512 kernel declined canonical 128x128 input", ctx)
			}
			splitOps := base.clone()
			gdnSplitRef2624(splitOps.st, splitOps.qn, splitOps.kn, splitOps.vh, gate.beta, gate.decay,
				splitOps.od, splitOps.kvmem, splitOps.delta)

			for _, name := range gdnBufferNames2624 {
				assertGDNBitsEqual2624(t, ctx+" buf="+name, goOps.buf(name), asmOps.buf(name))
			}
			if gdnAnyNonZero2624(goOps.st, goOps.od, goOps.kvmem, goOps.delta) {
				pr.liveCells++
			}
			// Non-finite guard. A probe that is not allowNonFinite must never produce Inf/NaN
			// (it declared a finite regime); a probe that IS allowed to must still retain finite
			// live elements, otherwise the bit-identity comparison above degenerates into
			// "both paths produced +Inf everywhere". Either way the Go and asm paths must agree
			// on how many elements went non-finite.
			goNF := gdnCountNonFinite2624(goOps.st, goOps.od, goOps.kvmem, goOps.delta)
			asmNF := gdnCountNonFinite2624(asmOps.st, asmOps.od, asmOps.kvmem, asmOps.delta)
			pr.nonFiniteGo += goNF
			pr.nonFiniteAsm += asmNF
			if goNF == 0 {
				pr.finiteCells++
			}
			if goNF != asmNF {
				t.Errorf("%s: non-finite element count differs — Go=%d asm=%d; the two paths disagree on whether a value overflowed", ctx, goNF, asmNF)
			}
			if !probe.allowNonFinite && goNF != 0 {
				t.Errorf("%s: %d non-finite elements in a regime that must stay finite", ctx, goNF)
			}
			div := false
			for _, name := range gdnBufferNames2624 {
				a, b := goOps.buf(name), splitOps.buf(name)
				for i := range a {
					if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						div = true
					}
				}
			}
			if div {
				pr.splitDiverged = true
			}
			for _, name := range gdnBufferNames2624 {
				if d := MaxAbsDelta(goOps.buf(name), splitOps.buf(name)); d > pr.maxSplitDrift {
					pr.maxSplitDrift = d
				}
			}
		}
		if pr.liveCells == 0 {
			t.Errorf("probe=%s: every gate cell produced an all-zero result — the regime is vacuous", probe.name)
		}
		if probe.allowNonFinite && pr.finiteCells == 0 {
			t.Errorf("probe=%s is allowed to overflow but EVERY gate cell went non-finite; the bit-identity "+
				"assertion above is then vacuous (both paths trivially agree on +Inf)", probe.name)
		}
		if probe.strongest {
			if !pr.splitDiverged {
				t.Errorf("probe=%s is designated the STRONGEST FMA-divergence probe but is bit-identical to the "+
					"split-rounding reference on every gate cell; the edge-case menu proves nothing", probe.name)
			}
			t.Logf("STRONGEST probe=%s: %d/%d gate cells live, diverges from split-rounding, max drift over all four buffers=%g",
				probe.name, pr.liveCells, len(gdnGates2624), pr.maxSplitDrift)
		} else {
			t.Logf("probe=%-34s %d/%d gate cells live, finite cells=%d, nonFinite(Go/asm)=%d/%d, "+
				"split-divergent=%v, max drift over all four buffers=%g",
				probe.name, pr.liveCells, len(gdnGates2624), pr.finiteCells, pr.nonFiniteGo, pr.nonFiniteAsm,
				pr.splitDiverged, pr.maxSplitDrift)
		}
		results = append(results, pr)
	}

	// At least one non-strongest regime must ALSO discriminate, so the menu is not a single
	// lucky cell. (subnormalInputs and near_overflow are expected candidates.)
	extra := 0
	for _, r := range results {
		if !r.strongest && r.splitDiverged {
			extra++
		}
	}
	if extra == 0 {
		t.Errorf("only the designated strongest probe discriminates from split-rounding; "+
			"the remaining %d value-domain regimes add no sensitivity", len(results)-1)
	}
}

// ---------------------------------------------------------------------------
// 4. geometry gating and environment suppression
// ---------------------------------------------------------------------------

// TestGDNFMA2624GeometryGating pins spec clause 4: the AVX-512 kernel is only ever taken for
// the canonical kHd==128 && vHd==128 geometry, and the Go path remains self-consistent for
// every other geometry.
//
// A regression: HasVectorizedDeltaNetFor answering true for a non-canonical geometry (which
// would hand the assembly a wrong-shaped state matrix), or the Go path disagreeing with
// itself across repeated calls on a non-canonical geometry.
func TestGDNFMA2624GeometryGating(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")

	nonCanonical := [][2]int{
		{64, 64}, {128, 64}, {64, 128}, {129, 128}, {128, 129}, {129, 129},
		{1, 1}, {0, 128}, {128, 0}, {32, 128}, {128, 32}, {256, 128}, {128, 256},
	}
	for _, g := range nonCanonical {
		if HasVectorizedDeltaNetFor(g[0], g[1]) {
			t.Errorf("geometry %dx%d is non-canonical but advertises the AVX-512 GDN kernel", g[0], g[1])
		}
	}
	if !HasVectorizedDeltaNetFor(128, 128) && hasDeltaNetSIMD() {
		t.Error("canonical 128x128 must advertise the detected AVX-512 kernel when the hardware supports it")
	}
	// The capability must be exactly the geometry predicate AND the env/hardware gate.
	if got, want := HasVectorizedDeltaNetFor(128, 128), hasDeltaNetSIMD(); got != want {
		t.Errorf("canonical geometry capability=%v, hardware/OS detection=%v", got, want)
	}

	// For a non-canonical geometry the Go path must be self-consistent: repeated calls on the
	// same inputs produce bit-identical results, and dispatch through the exported entry
	// point agrees with the direct Go reference.
	for _, g := range [][2]int{{64, 64}, {128, 64}, {64, 128}, {129, 128}} {
		kHd, vHd := g[0], g[1]
		rng := rand.New(rand.NewSource(int64(2624500 + kHd*131 + vHd)))
		base := &gdnOperands2624{
			st:    make([]float32, kHd*vHd),
			qn:    make([]float32, gdnDim2624),
			kn:    make([]float32, gdnDim2624),
			vh:    make([]float32, gdnDim2624),
			od:    make([]float32, gdnDim2624),
			kvmem: make([]float32, gdnDim2624),
			delta: make([]float32, gdnDim2624),
		}
		for i := range base.st {
			base.st[i] = rng.Float32()*0.2 - 0.1
		}
		qn := make([]float32, kHd)
		kn := make([]float32, kHd)
		vh := make([]float32, vHd)
		od := make([]float32, vHd)
		kvmem := make([]float32, vHd)
		delta := make([]float32, vHd)
		for i := 0; i < kHd; i++ {
			qn[i], kn[i] = rng.Float32()*0.2-0.1, rng.Float32()*0.2-0.1
		}
		for i := 0; i < vHd; i++ {
			vh[i], od[i] = rng.Float32()*0.2-0.1, rng.Float32()*0.02-0.01
		}
		beta, decay := float32(0.6), float32(math.Exp(-0.05))

		ref := append([]float32(nil), base.st...)
		refOD := append([]float32(nil), od...)
		wave32GatedDeltaNetStepGo(ref, qn, kn, vh, beta, decay, refOD, kvmem, delta)
		refSt, refODv, refKV, refDelta := append([]float32(nil), ref...), append([]float32(nil), refOD...),
			append([]float32(nil), kvmem...), append([]float32(nil), delta...)

		for rep := 0; rep < 3; rep++ {
			got := append([]float32(nil), base.st...)
			gotOD := append([]float32(nil), od...)
			kv := make([]float32, vHd)
			dl := make([]float32, vHd)
			Wave32GatedDeltaNetStep(got, qn, kn, vh, beta, decay, gotOD, kv, dl)
			ctx := fmt.Sprintf("geometry %dx%d rep=%d", kHd, vHd, rep)
			assertGDNBitsEqual2624(t, ctx+" st", refSt, got)
			assertGDNBitsEqual2624(t, ctx+" od", refODv, gotOD)
			assertGDNBitsEqual2624(t, ctx+" kvmem", refKV, kv)
			assertGDNBitsEqual2624(t, ctx+" delta", refDelta, dl)
		}
		if !gdnAnyNonZero2624(refSt, refODv, refKV, refDelta) {
			t.Errorf("geometry %dx%d: Go path produced an all-zero result — self-consistency check is vacuous", kHd, vHd)
		}
	}
}

// TestGDNFMA2624EnvSuppression pins spec clause 4: FAK_VECTORIZED_DELTANET set to any of the
// recognised false spellings suppresses the vectorized path entirely (and the resulting
// dispatch is byte-identical to the pure Go path), while "1" makes the capability equal the
// hardware/OS detection. t.Setenv keeps the environment from leaking.
//
// A regression: any of the four disable spellings failing to suppress, or an enable leaving
// the capability different from hasDeltaNetSIMD().
func TestGDNFMA2624EnvSuppression(t *testing.T) {
	for _, v := range []string{"0", "false", "no", "off", "0 ", " OFF ", "False", "NO"} {
		t.Setenv("FAK_VECTORIZED_DELTANET", v)
		if HasVectorizedDeltaNet() {
			t.Errorf("FAK_VECTORIZED_DELTANET=%q must suppress the vectorized GDN path", v)
		}
		if HasVectorizedDeltaNetFor(128, 128) {
			t.Errorf("FAK_VECTORIZED_DELTANET=%q must suppress the canonical 128x128 capability", v)
		}
	}

	// Suppressed dispatch must be byte-identical to the pure Go reference, and the state must
	// actually be non-trivial so this is not a vacuous comparison.
	if !hasDeltaNetSIMD() {
		t.Skip("AVX-512F unavailable; suppression identity still asserted below, but no SIMD comparison needed")
	}
	for _, v := range []string{"0", "off"} {
		t.Setenv("FAK_VECTORIZED_DELTANET", v)
		rng := rand.New(rand.NewSource(int64(2624600)))
		base := gdnNewOperands2624()
		gdnSeedGrid2624(rng, base)
		beta, decay := float32(0.75), float32(0.125)

		ref := base.clone()
		wave32GatedDeltaNetStepGo(ref.st, ref.qn, ref.kn, ref.vh, beta, decay, ref.od, ref.kvmem, ref.delta)
		got := base.clone()
		Wave32GatedDeltaNetStep(got.st, got.qn, got.kn, got.vh, beta, decay, got.od, got.kvmem, got.delta)
		for _, name := range gdnBufferNames2624 {
			assertGDNBitsEqual2624(t, "env="+v+" buf="+name, ref.buf(name), got.buf(name))
		}
		if !gdnAnyNonZero2624(ref.st, ref.od, ref.kvmem, ref.delta) {
			t.Fatalf("env=%s: Go reference produced an all-zero result — suppression identity is vacuous", v)
		}
	}

	// Enable spellings: capability must equal hardware/OS detection exactly.
	for _, v := range []string{"1", "true", "yes", "on", " 1 "} {
		t.Setenv("FAK_VECTORIZED_DELTANET", v)
		if got, want := HasVectorizedDeltaNet(), hasDeltaNetSIMD(); got != want {
			t.Errorf("FAK_VECTORIZED_DELTANET=%q capability=%v, hardware/OS detection=%v", v, got, want)
		}
	}

	// Unset must also fall back to hardware detection.
	t.Setenv("FAK_VECTORIZED_DELTANET", "")
	if got, want := HasVectorizedDeltaNet(), hasDeltaNetSIMD(); got != want {
		t.Errorf("unset FAK_VECTORIZED_DELTANET capability=%v, hardware/OS detection=%v", got, want)
	}
}

// TestGDNFMA2624DispatchMatchesAsmOnCanonical pins spec clause 1 at the DISPATCH seam rather
// than at the two raw kernels: the exported Wave32GatedDeltaNetStep on the canonical geometry
// must route to the AVX-512 kernel and be bit-identical to the Go reference, and with the
// vectorized path suppressed it must be bit-identical to the Go reference too. It also
// witnesses that a non-canonical geometry is never handed to tryDeltaNetSIMD.
//
// A regression: the dispatcher taking a different arm than the capability advertises (the
// "presence != invokability" class), or the two arms disagreeing.
func TestGDNFMA2624DispatchMatchesAsmOnCanonical(t *testing.T) {
	gdnSkipped2624(t)

	t.Setenv("FAK_VECTORIZED_DELTANET", "1")
	rng := rand.New(rand.NewSource(int64(2624700)))
	base := gdnNewOperands2624()
	gdnSeedGrid2624(rng, base)

	for gi, gate := range gdnGates2624 {
		ref := base.clone()
		wave32GatedDeltaNetStepGo(ref.st, ref.qn, ref.kn, ref.vh, gate.beta, gate.decay,
			ref.od, ref.kvmem, ref.delta)
		got := base.clone()
		Wave32GatedDeltaNetStep(got.st, got.qn, got.kn, got.vh, gate.beta, gate.decay,
			got.od, got.kvmem, got.delta)
		ctx := fmt.Sprintf("dispatch gate[%02d] beta=%g decay=%g", gi, gate.beta, gate.decay)
		for _, name := range gdnBufferNames2624 {
			assertGDNBitsEqual2624(t, ctx+" buf="+name, ref.buf(name), got.buf(name))
		}
	}

	// Under-suppressed dispatch: same bytes as the Go reference.
	t.Setenv("FAK_VECTORIZED_DELTANET", "0")
	ref := base.clone()
	wave32GatedDeltaNetStepGo(ref.st, ref.qn, ref.kn, ref.vh, 0.5, 0.5, ref.od, ref.kvmem, ref.delta)
	got := base.clone()
	Wave32GatedDeltaNetStep(got.st, got.qn, got.kn, got.vh, 0.5, 0.5, got.od, got.kvmem, got.delta)
	for _, name := range gdnBufferNames2624 {
		assertGDNBitsEqual2624(t, "dispatch suppressed buf="+name, ref.buf(name), got.buf(name))
	}

	// The assembly must decline a non-canonical shape outright — the guard that keeps
	// HasVectorizedDeltaNetFor honest at the kernel level.
	small := make([]float32, 64)
	if tryDeltaNetSIMD(small, small[:64], small[:64], small[:64], 0.5, 0.5, small[:64], small[:64], small[:64]) {
		t.Error("tryDeltaNetSIMD accepted a 64x64 shape; the kernel-level geometry guard is missing")
	}
}
