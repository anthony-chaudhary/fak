package turntaxmeter

import "testing"

// TestSamplerShouldSampleUnderCostCap is the observer-effect cost fence (issue #1156,
// ticket T4 of epic #1147) the docs/standards/observer-effect.md standard requires: a
// meter fak puts on a hot path must cost less than a declared cap, and that cost must
// itself be MEASURED — you cannot honestly report an overhead you never bounded.
//
// Sampler is the rate-bound a per-turn meter calls once per hot-path event to decide
// whether to record it. Its ShouldSample is one atomic increment plus a modulo on the
// common (non-admit) path — no I/O, no slice growth, no closure — so its declared cap
// is the tightest a hot-path gate can have: ZERO heap allocations per call.
// testing.AllocsPerRun is deterministic (same input → the same count on every run and
// every host), so this is a WITNESSED bound, not a noisy wall-clock one that would
// read OBSERVED. This is the companion to internal/spec's AcceptanceMeter cost cap:
// that pins the meter's PER-SAMPLE cost at 0 allocs; this pins the rate gate that
// decides how OFTEN the meter samples at 0 allocs too — together they bound both
// halves of the observer effect (how much each sample costs, and how many fire).
func TestSamplerShouldSampleUnderCostCap(t *testing.T) {
	const maxAllocsPerCall = 0.0 // declared cap: the rate gate allocates nothing per event
	// A non-trivial period so the run exercises BOTH the admit branch (one extra
	// atomic add) and the far-more-common reject branch; the cap must hold on each.
	s := NewSampler(8)
	got := testing.AllocsPerRun(10000, func() {
		s.ShouldSample()
	})
	if got > maxAllocsPerCall {
		t.Fatalf("Sampler.ShouldSample = %g allocs/op, want <= %g — the observer-effect fence: the meter's own per-event cost must stay under its declared cap", got, maxAllocsPerCall)
	}
	// Sanity: the gate actually admitted events over the run, so the zero-cost path is
	// the real sampling path, not a no-op masquerading as cheap (a cap on a gate that
	// never admits would be vacuous, exactly the failure the doc's clamp-to-1 guards).
	if s.Admitted() == 0 {
		t.Fatal("Sampler admitted 0 events after 10000 calls at rate 8 — the cap would be vacuous on a gate that does nothing")
	}
}
