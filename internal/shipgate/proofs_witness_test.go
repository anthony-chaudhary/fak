package shipgate

// proofs_witness_test.go closes OPEN math-proof obligations for internal/shipgate.
//
// OPEN (1) [measurement-deterministic]: Evaluate is deterministic — for any fixed
// Witness w, repeated evaluations Evaluate(w) yield the identical (Decision, keep-bit),
// with no dependence on RNG, wall-clock, goroutine scheduling, map-iteration order, or
// mutable global state.
//   mechanism: shipgate.go:54 (Witness.improved) and shipgate.go:64 (Evaluate).
//
// Discipline: fak/docs/proofs/00-METHOD.md. Stdlib only; fixed RNG seed; no wall-clock,
// no shared mutable state. The assertion is a real invariant comparison (every repeat /
// every goroutine yields a result bit-identical to a single reference evaluation), not a
// smoke test.

import (
	"math"
	"math/rand"
	"sync"
	"testing"
	"testing/quick"
)

// genWitness builds a Witness from a fixed-seed RNG. We deliberately exercise the full
// input surface Evaluate reads: Metric (string, irrelevant to the decision but part of
// the value), Before/After (including NaN/Inf and equal-value boundaries), LowerBetter,
// SuiteGreen, TruthClean. improvedBit is left zero — it is set only by Evaluate, so a
// caller cannot pre-seed it; a fixed input value of the *exported* surface is what
// determinism is claimed over.
func genWitness(r *rand.Rand) Witness {
	pick := func() float64 {
		switch r.Intn(7) {
		case 0:
			return math.NaN()
		case 1:
			return math.Inf(1)
		case 2:
			return math.Inf(-1)
		case 3:
			return 0
		case 4:
			return 1e300 * (r.Float64()*2 - 1)
		case 5:
			return r.NormFloat64()
		default:
			return r.Float64()*2000 - 1000
		}
	}
	metrics := []string{"p50_ns", "vdso_hit_rate", "", "throughput", "rss_bytes"}
	before := pick()
	after := pick()
	// Occasionally force Before==After to hit the strict-boundary (no-gain) case.
	if r.Intn(4) == 0 {
		after = before
	}
	return Witness{
		Metric:      metrics[r.Intn(len(metrics))],
		Before:      before,
		After:       after,
		LowerBetter: r.Intn(2) == 0,
		SuiteGreen:  r.Intn(2) == 0,
		TruthClean:  r.Intn(2) == 0,
	}
}

// result is the full observable output of Evaluate: the typed decision plus the
// non-forgeable keep-bit read back via Kept(). Both must be reproduced exactly.
type result struct {
	d    Decision
	kept bool
}

func evalOnce(w Witness) result {
	d, out := Evaluate(w)
	return result{d: d, kept: out.Kept()}
}

// consistent: a KEEP must carry the keep-bit and a REVERT must not — Evaluate never
// emits an inconsistent (Decision, keep-bit) pair. This pins what "identical output"
// means so the determinism check below cannot pass on a degenerate always-false bit.
func (r result) consistent() bool {
	switch r.d {
	case KEEP:
		return r.kept
	case REVERT:
		return !r.kept
	default:
		// Evaluate only ever returns KEEP or REVERT (never ESCALATE); any other
		// decision is itself a violation.
		return false
	}
}

// TestEvaluateDeterministicRepeat asserts OPEN(1): over a fixed-seed sweep of the full
// Witness input surface, evaluating the SAME witness many times in a row always yields
// the bit-identical (Decision, keep-bit). A reference result is taken once and every
// subsequent repeat must equal it exactly. Non-vacuous: it also requires the pair to be
// internally consistent (KEEP<=>kept), and asserts the sweep actually exercised BOTH
// outcomes so the equality is not trivially over a constant.
func TestEvaluateDeterministicRepeat(t *testing.T) {
	r := rand.New(rand.NewSource(0x5eed_1234))
	const cases = 4000
	const repeats = 64
	sawKeep, sawRevert := false, false
	for i := 0; i < cases; i++ {
		w := genWitness(r)
		ref := evalOnce(w)
		if !ref.consistent() {
			t.Fatalf("case %d: inconsistent (Decision,keep-bit) pair: %+v for %+v", i, ref, w)
		}
		switch ref.d {
		case KEEP:
			sawKeep = true
		case REVERT:
			sawRevert = true
		}
		for j := 0; j < repeats; j++ {
			got := evalOnce(w)
			if got != ref {
				t.Fatalf("case %d repeat %d: Evaluate not deterministic: got %+v want %+v for witness %+v",
					i, j, got, ref, w)
			}
		}
	}
	if !sawKeep || !sawRevert {
		t.Fatalf("sweep was vacuous: sawKeep=%v sawRevert=%v (need both outcomes present)", sawKeep, sawRevert)
	}
}

// TestEvaluateDeterministicConcurrent asserts the scheduling/shared-state half of
// OPEN(1): the same witness evaluated from many goroutines at once yields exactly one
// distinct (Decision, keep-bit) result — concurrent evaluation cannot diverge, proving
// there is no mutable global state or scheduling dependence. Run this package with -race
// to also rule out a data race on any hidden shared cell.
func TestEvaluateDeterministicConcurrent(t *testing.T) {
	r := rand.New(rand.NewSource(0xC0ffee_99))
	const cases = 256
	const goroutines = 32
	for i := 0; i < cases; i++ {
		w := genWitness(r)
		ref := evalOnce(w)

		var wg sync.WaitGroup
		results := make([]result, goroutines)
		start := make(chan struct{})
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				<-start // line them up to maximize interleaving
				results[idx] = evalOnce(w)
			}(g)
		}
		close(start)
		wg.Wait()

		for g := 0; g < goroutines; g++ {
			if results[g] != ref {
				t.Fatalf("case %d goroutine %d: concurrent Evaluate diverged: got %+v want %+v for %+v",
					i, g, results[g], ref, w)
			}
		}
	}
}

// TestEvaluateDeterministicQuick is a property-based restatement of OPEN(1) via
// testing/quick with a FIXED seed: for every generated witness, two independent
// evaluations agree and the pair is consistent. Independent generator from the loops
// above, so it widens input coverage without RNG sharing.
func TestEvaluateDeterministicQuick(t *testing.T) {
	prop := func(metric string, before, after float64, lower, green, clean bool) bool {
		w := Witness{
			Metric:      metric,
			Before:      before,
			After:       after,
			LowerBetter: lower,
			SuiteGreen:  green,
			TruthClean:  clean,
		}
		a := evalOnce(w)
		b := evalOnce(w)
		return a == b && a.consistent()
	}
	cfg := &quick.Config{
		MaxCount: 5000,
		Rand:     rand.New(rand.NewSource(0xABCD_4242)),
	}
	if err := quick.Check(prop, cfg); err != nil {
		t.Fatalf("Evaluate not deterministic/consistent under quick.Check: %v", err)
	}
}
