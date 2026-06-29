package callavoid

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// classobserve_test.go — regression floor for #819 acceptance (3): the per-tool-class
// admission verdict is read off (m, v, c) DERIVED from observation, not hand-asserted. The
// three cases the issue names: a stable read-heavy class is ADMITTED (a measured net win), a
// write-churned class is DECLINED (a measured net loss), and a class with no observation
// ABSTAINS (unmeasured -> declined, never admitted on a guess).

// TestFoldObservationAdmitsStableReadHeavyClass: a class reused many times whose entries are
// almost never invalidated by a write yields a LOW observed mutation rate (m ≈ 0), so its
// DERIVED economics prove and it is admitted — a witnessed (measured) admit, the net-win
// case. 50 reuse attempts, only 1 invalidated by an intervening write -> m = 0.02.
func TestFoldObservationAdmitsStableReadHeavyClass(t *testing.T) {
	g := FoldClassObservation(ClassObservation{
		Class:         "Read",
		ReuseAttempts: 50,
		Invalidations: 1, // m = 1/50 = 0.02 — a calm, read-heavy class.
	})
	if !g.Measured {
		t.Fatalf("a class with observed reuse attempts must be MEASURED, got abstained (%s)", g.Note)
	}
	if !g.Decision.Admit || g.Decision.Proof.Status != ProofProven {
		t.Fatalf("a stable read-heavy class must ADMIT/PROVE, got admit=%v %s (%s)",
			g.Decision.Admit, g.Decision.Proof.Status, g.Note)
	}
	// The admit is grounded in the DERIVED mutation rate, not a supplied one.
	if !approx(g.Calibration.MutationRate, 0.02) {
		t.Errorf("derived mutation rate = %v, want 0.02 (1 invalidation / 50 reuses)", g.Calibration.MutationRate)
	}
	if g.Calibration.Accesses != 50 {
		t.Errorf("representative k = %d, want the observed reuse count 50", g.Calibration.Accesses)
	}
	if g.Decision.Class != "Read" {
		t.Errorf("class = %q, want Read", g.Decision.Class)
	}
}

// TestFoldObservationDeclinesWriteChurnedClass is the heart of #819: a class whose entries
// are churned OUT by writes faster than they hit — most reuse attempts find the entry
// invalidated — yields a HIGH observed mutation rate, so its DERIVED economics REFUTE and it
// is declined. This is the write-heavy-session net-loss the issue describes: the global
// world-version strands the class's reads, so caching it is a measured loss. 50 reuse
// attempts, 49 invalidated -> m = 0.98, D = 1 - 0.98 - v - 0.98c < 0 -> never break even.
func TestFoldObservationDeclinesWriteChurnedClass(t *testing.T) {
	g := FoldClassObservation(ClassObservation{
		Class:         "Grep",
		ReuseAttempts: 50,
		Invalidations: 49, // m = 49/50 = 0.98 — writes strand it faster than it hits.
	})
	if !g.Measured {
		t.Fatalf("a class with observed reuse attempts must be MEASURED, got abstained")
	}
	if g.Decision.Admit || g.Decision.Proof.Status != ProofRefuted {
		t.Fatalf("a write-churned class must DECLINE/REFUTE, got admit=%v %s (%s)",
			g.Decision.Admit, g.Decision.Proof.Status, g.Note)
	}
	if !approx(g.Calibration.MutationRate, 0.98) {
		t.Errorf("derived mutation rate = %v, want 0.98 (49 invalidations / 50 reuses)", g.Calibration.MutationRate)
	}
	// The per-reuse net gain is negative — the cache never pays for this class at any reuse.
	if g.Decision.Proof.PerReuseNetGain > 0 {
		t.Errorf("per-reuse net gain = %v, want <= 0 (a write-churned class is a measured net loss)", g.Decision.Proof.PerReuseNetGain)
	}
	if g.Decision.Proof.BreakEvenAccesses != neverBreakEven {
		t.Errorf("break-even = %d, want never (no reuse count repays a 0.98-churn class)", g.Decision.Proof.BreakEvenAccesses)
	}
}

// TestFoldObservationAbstainsWithNoObservation: a class with ZERO reuse attempts has nothing
// to measure its mutation rate from, so the fold ABSTAINS — Measured=false, declined, no
// fabricated input. The calibrate-don't-assume law: an un-observed class is never admitted
// on a guess. Critically, this is DISTINCT from a measured refute (the net-loss case): the
// proof is empty because there was no input to prove over.
func TestFoldObservationAbstainsWithNoObservation(t *testing.T) {
	g := FoldClassObservation(ClassObservation{Class: "WebFetch"}) // no reuse attempts observed.
	if g.Measured {
		t.Fatalf("a class with no observed reuse attempts must ABSTAIN (Measured=false), got measured")
	}
	if g.Decision.Admit {
		t.Fatalf("an abstained class must be DECLINED, got admit=true (%s)", g.Note)
	}
	// The abstain is honestly attributed, not dressed up as a refuting proof.
	if g.Decision.Proof.Status == ProofProven || g.Decision.Proof.Status == ProofRefuted {
		t.Errorf("an abstain must carry NO proof verdict, got %q (it is unmeasured, not refuted)", g.Decision.Proof.Status)
	}
	if g.Decision.Class != "WebFetch" {
		t.Errorf("class = %q, want WebFetch even on abstain", g.Decision.Class)
	}
	if !strings.Contains(g.Note, "abstain") || !strings.Contains(g.Note, "unmeasured") {
		t.Errorf("abstain note must say it is unmeasured: %q", g.Note)
	}
}

// TestFoldObservationDerivesCostsFromSamples: the validate/capture costs are the MEANS of
// their observed samples, not asserted — and a class with no cost samples falls back to the
// cheap non-zero default (never free, the #817 stale-miss-is-a-loss floor).
func TestFoldObservationDerivesCostsFromSamples(t *testing.T) {
	// Validate cost sampled: sum 0.36 over 6 samples -> mean 0.06. Capture unsampled -> default.
	g := FoldClassObservation(ClassObservation{
		Class:               "Glob",
		ReuseAttempts:       20,
		Invalidations:       1,
		ValidateCostSamples: 0.36,
		ValidateCostCount:   6,
	})
	if !approx(g.Calibration.ValidateCost, 0.06) {
		t.Errorf("derived validate cost = %v, want 0.06 (0.36 / 6 samples)", g.Calibration.ValidateCost)
	}
	if g.Calibration.CaptureCost != defaultObservedCaptureCost {
		t.Errorf("unsampled capture cost = %v, want the cheap default %v (never free)",
			g.Calibration.CaptureCost, defaultObservedCaptureCost)
	}
	// A class with no validate samples at all also falls back, never to zero.
	g2 := FoldClassObservation(ClassObservation{Class: "Glob", ReuseAttempts: 20, Invalidations: 1})
	if g2.Calibration.ValidateCost != defaultObservedValidateCost {
		t.Errorf("unsampled validate cost = %v, want the cheap default %v", g2.Calibration.ValidateCost, defaultObservedValidateCost)
	}
}

// TestFoldObservationClampsMalformedInvalidations: a malformed invalidation count (negative,
// or exceeding the reuse attempts) is clamped into [0, ReuseAttempts] so the derived mutation
// rate can never leave [0,1] — a bad count cannot push the gate to a nonsense verdict.
func TestFoldObservationClampsMalformedInvalidations(t *testing.T) {
	over := FoldClassObservation(ClassObservation{Class: "X", ReuseAttempts: 10, Invalidations: 99})
	if over.Calibration.MutationRate != 1.0 {
		t.Errorf("over-count mutation rate = %v, want clamped to 1.0", over.Calibration.MutationRate)
	}
	neg := FoldClassObservation(ClassObservation{Class: "X", ReuseAttempts: 10, Invalidations: -5})
	if neg.Calibration.MutationRate != 0.0 {
		t.Errorf("negative-count mutation rate = %v, want clamped to 0.0", neg.Calibration.MutationRate)
	}
}

// TestFoldObservationsBatchAndAdmittedProjection: the batch fold decides per class in order,
// and AdmittedFromObservations returns exactly the MEASURED-and-proving classes — the
// tier-2 allow-set. A measured-refute and an unmeasured-abstain are both excluded, so the
// allow-set is built only from observed evidence that a class pays.
func TestFoldObservationsBatchAndAdmittedProjection(t *testing.T) {
	obs := []ClassObservation{
		{Class: "Read", ReuseAttempts: 50, Invalidations: 1},  // measured admit (m=0.02).
		{Class: "Grep", ReuseAttempts: 50, Invalidations: 49}, // measured decline (m=0.98).
		{Class: "WebFetch"}, // abstain (unmeasured).
		{Class: "Glob", ReuseAttempts: 30, Invalidations: 0}, // measured admit (m=0).
	}
	gates := FoldClassObservations(obs)
	if len(gates) != 4 {
		t.Fatalf("batch length = %d, want 4", len(gates))
	}
	if gates[0].Decision.Class != "Read" || gates[2].Decision.Class != "WebFetch" {
		t.Errorf("batch order not preserved: %q, %q", gates[0].Decision.Class, gates[2].Decision.Class)
	}
	if gates[2].Measured {
		t.Errorf("WebFetch must abstain (unmeasured), got measured")
	}
	admitted := AdmittedFromObservations(obs)
	if !reflect.DeepEqual(admitted, []string{"Read", "Glob"}) {
		t.Errorf("admitted = %v, want [Read Glob] (Grep declined, WebFetch abstained)", admitted)
	}
}

// TestFoldObservationDeterministic: pure — identical observation yields identical verdict.
func TestFoldObservationDeterministic(t *testing.T) {
	obs := ClassObservation{
		Class: "Read", ReuseAttempts: 17, Invalidations: 3,
		ValidateCostSamples: 0.4, ValidateCostCount: 8,
		CaptureCostSamples: 0.2, CaptureCostCount: 5,
	}
	if !reflect.DeepEqual(FoldClassObservation(obs), FoldClassObservation(obs)) {
		t.Error("FoldClassObservation is not deterministic")
	}
}

// TestFoldObservationJSONRoundTrips: the observed-gate artifact marshals to stable JSON a
// caller (the deferred kernel.Reap seam's logger) consumes, carrying the measured flag and
// the derived calibration so the admission decision is auditable back to its observation.
func TestFoldObservationJSONRoundTrips(t *testing.T) {
	// 1/4 = 0.25 is exactly representable, so the round-tripped rate compares cleanly.
	g := FoldClassObservation(ClassObservation{Class: "Read", ReuseAttempts: 4, Invalidations: 1})
	b, err := json.Marshal(g)
	if err != nil {
		t.Fatalf("marshal ObservedClassGate: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["measured"] != true {
		t.Errorf("measured = %v, want true", m["measured"])
	}
	cal, ok := m["calibration"].(map[string]any)
	if !ok || cal["mutation_rate"] != 0.25 {
		t.Errorf("calibration.mutation_rate = %v, want 0.25 (derived, surfaced for audit)", m["calibration"])
	}
}
