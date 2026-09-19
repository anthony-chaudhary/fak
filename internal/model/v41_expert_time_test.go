package model

// v41_expert_time_test.go — the #13299 witness: the V4.1 routed-expert phase
// ledger records SEPARATE fault, dequantization and contraction DURATIONS (not
// just byte counts), attributes them to the phase the session declared, and
// names the contraction backend that actually ran.
//
// The load-bearing property is the SEPARATION at the three real call boundaries
// (v41_forward.go: expertCheckpoint.fault -> expertWeightF32Into -> v41SwiGLU):
// counts alone cannot say whether the 492 s first token (fak#13294) was disk
// wait, f32 materialization or the scalar contraction. A phase-less ledger that
// accumulates all three into one number would reproduce exactly the blind spot
// this leaf removes, so the witness drives a CONTROLLED clock: the forward can
// only read time through the ledger's clock seam, so the test can make each
// boundary take a distinct, precisely known amount and then assert the three
// totals are independent -- never double-counted -- and that the phases stay
// separated.
//
// The contraction-backend identity is asserted as an OBSERVED value: the host
// scalar arm must report "host", and a declared Vulkan engine selection must
// report "vulkan" WITHOUT the forward executing a GPU contraction (the identity
// names the selected backend, it does not fabricate a device receipt).

import (
	"math"
	"testing"
)

// v41TestClock is a deterministic monotonic clock the ledger reads through its
// injectable seam. Every read advances the clock by perTick, so a boundary
// bracketed by two reads measures exactly perTick regardless of the real
// elapsed time -- the durations become a function of how many boundaries ran,
// which is the separation property under test.
type v41TestClock struct {
	ticks   int64
	perTick int64
}

func (c *v41TestClock) now() int64 {
	c.ticks++
	return c.ticks * c.perTick
}

// TestV41ExpertTime separates the three timed boundaries under a controlled
// clock and asserts the phase ledgers carry independent fault/dequant/
// contraction totals, with per-token rates consistent with their denominators.
func TestV41ExpertTime(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("tier-only admission error = %v, want nil", err)
	}

	// Install the fake clock and a known tick so the durations are exact
	// integers. The forward reads time ONLY through this seam.
	clk := &v41TestClock{perTick: 100}
	m.v41SetExpertTimeClock(clk.now)

	// Inert-by-default: before any phase is set, the snapshot stays zero even
	// though the clock is installed.
	if got := m.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("snapshot before any phase = %+v, want zero", got)
	}

	// Prefill: two tokens, with each timed boundary advancing the clock by
	// exactly one tick. The tier-only fixture faults 3 projections per pick and
	// contracts one SwiGLU per pick, so the counts are known from the routing
	// geometry.
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	act, err := m.forwardV41([]int{1, 3}, nil)
	m.v41NoteExpertFaultToken(2)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if err != nil {
		t.Fatalf("tier-only prefill error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("prefill produced %d logit rows, want 2", len(act.Logits))
	}

	p := m.V41ExpertFaultAttribution().Prefill
	if p.Faults <= 0 {
		t.Fatalf("prefill Faults = %d, want > 0", p.Faults)
	}
	// Every faulted projection and every contraction advanced the clock once.
	if p.FaultDoorNanos != int64(p.Faults)*clk.perTick {
		t.Fatalf("prefill FaultNanos = %d for %d faults, want %d",
			p.FaultDoorNanos, p.Faults, int64(p.Faults)*clk.perTick)
	}
	if p.DequantNanos != int64(p.Faults)*clk.perTick {
		t.Fatalf("prefill DequantNanos = %d for %d dequants, want %d",
			p.DequantNanos, p.Faults, int64(p.Faults)*clk.perTick)
	}
	// Each routed token contracts V41RouterTopK experts exactly once.
	if want := 2 * V41RouterTopK; p.Contractions != want {
		t.Fatalf("prefill Contractions = %d, want %d (2 tokens x top-%d)",
			p.Contractions, want, V41RouterTopK)
	}
	if p.ContractionNanos != int64(p.Contractions)*clk.perTick {
		t.Fatalf("prefill ContractionNanos = %d for %d contractions, want %d",
			p.ContractionNanos, p.Contractions, int64(p.Contractions)*clk.perTick)
	}
	if p.ContractionBackend != "host" {
		t.Fatalf("prefill ContractionBackend = %q, want host (the fixture runs the scalar host arm)", p.ContractionBackend)
	}
	// The three totals must be independent, not one number copied three times.
	if p.FaultDoorNanos == p.ContractionNanos {
		t.Fatalf("FaultNanos == ContractionNanos (%d): the boundaries are not separated", p.FaultDoorNanos)
	}
	if want := float64(p.FaultDoorNanos) / float64(p.Tokens); math.Abs(p.FaultNanosPerToken-want) > 0.5 {
		t.Fatalf("prefill FaultNanosPerToken = %v, want %v", p.FaultNanosPerToken, want)
	}
	if want := float64(p.DequantNanos) / float64(p.Tokens); math.Abs(p.DequantNanosPerToken-want) > 0.5 {
		t.Fatalf("prefill DequantNanosPerToken = %v, want %v", p.DequantNanosPerToken, want)
	}
	if want := float64(p.ContractionNanos) / float64(p.Tokens); math.Abs(p.ContractionNanosPerToken-want) > 0.5 {
		t.Fatalf("prefill ContractionNanosPerToken = %v, want %v", p.ContractionNanosPerToken, want)
	}

	// Decode phase: one token, its own ledgers, and the prefill ledger must be
	// unchanged by the decode step.
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	act2, err := m.forwardV41([]int{5}, nil)
	m.v41NoteExpertFaultToken(1)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if err != nil {
		t.Fatalf("tier-only decode error = %v, want nil", err)
	}
	if len(act2.Logits) != 1 {
		t.Fatalf("decode produced %d logit rows, want 1", len(act2.Logits))
	}
	got := m.V41ExpertFaultAttribution()
	d := got.Decode
	if d.Tokens != 1 || d.Faults <= 0 || d.Contractions != V41RouterTopK {
		t.Fatalf("decode ledger Tokens=%d Faults=%d Contractions=%d, want 1 with Faults>0 and Contractions=top-%d",
			d.Tokens, d.Faults, d.Contractions, V41RouterTopK)
	}
	if d.FaultDoorNanos <= 0 || d.DequantNanos <= 0 || d.ContractionNanos <= 0 {
		t.Fatalf("decode durations must all be > 0: %+v", d)
	}
	if got.Prefill != p {
		t.Fatalf("prefill ledger changed by the decode step: %+v, want %+v", got.Prefill, p)
	}
}

// TestV41ExpertTimeBackendIdentity pins the contraction-backend identity: the
// host scalar arm reports "host", an explicit Vulkan engine selection reports
// "vulkan", and an unset selection falls back to "host" -- the identity the
// planner needs to know WHICH engine's contraction the durations describe.
func TestV41ExpertTimeBackendIdentity(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	clk := &v41TestClock{perTick: 1}
	m.v41SetExpertTimeClock(clk.now)

	// Default (unset) selection: host.
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	m.v41NoteExpertContraction()
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if got := m.V41ExpertFaultAttribution().Decode.ContractionBackend; got != "host" {
		t.Fatalf("default ContractionBackend = %q, want host", got)
	}

	// Declared Vulkan engine selection: the identity follows the selection, and
	// the note is honest that no device receipt was taken (it is a selection).
	m.v41SetContractionBackend("vulkan")
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	m.v41NoteExpertContraction()
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if got := m.V41ExpertFaultAttribution().Decode.ContractionBackend; got != "vulkan" {
		t.Fatalf("selected ContractionBackend = %q, want vulkan", got)
	}
}

// TestV41ExpertTimeResetSemantics pins reset: a fresh ledger reports zero
// durations, and a phase-less ledger drops every timed note so a stray caller
// cannot fabricate a denominator.
func TestV41ExpertTimeResetSemantics(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	// No phase ever set: timed notes must be dropped.
	m.v41NoteExpertFaultDoorNanos(11)
	m.v41NoteExpertDequantNanos(13)
	m.v41NoteExpertContraction()
	if got := m.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("unscoped timed notes = %+v, want zero", got)
	}
	// A live but empty phase reports zero durations and zero rates, never NaN.
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	got := m.V41ExpertFaultAttribution().Prefill
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if got.FaultDoorNanos != 0 || got.DequantNanos != 0 || got.ContractionNanos != 0 {
		t.Fatalf("empty live phase durations = %+v, want zero", got)
	}
	if math.IsNaN(got.FaultNanosPerToken) || math.IsNaN(got.ContractionNanosPerToken) {
		t.Fatalf("empty live phase reports NaN rates: %+v, want zero", got)
	}
}
