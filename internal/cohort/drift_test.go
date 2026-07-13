package cohort

import (
	"strings"
	"testing"
)

// goodProv is a fully-attributed provenance: every field #4581 requires is set.
func goodProv() DriftProvenance {
	return DriftProvenance{
		Model:     "qwen3-8b",
		Tokenizer: "qwen3-bpe",
		Engine:    "native",
		Oracle:    "exact-match",
		Revision:  "rev-1",
		Baseline:  "nightly-baseline@rev-0",
	}
}

// baseObs is one cohort observed exactly at its baseline on every scope signal —
// the "clean" run every drift test perturbs from.
func baseObs(cohort string, tier DriftTier) CohortObservation {
	return CohortObservation{
		Cohort:      cohort,
		Tier:        tier,
		CostSeconds: 12.5,
		Provenance:  goodProv(),
		Baseline: map[string]float64{
			SignalMix:          0.50,
			SignalLength:       128,
			SignalDegeneration: 0.02,
			SignalRubric:       0.90,
		},
		Observed: map[string]float64{
			SignalMix:          0.50,
			SignalLength:       128,
			SignalDegeneration: 0.02,
			SignalRubric:       0.90,
		},
		Tolerance: map[string]float64{
			SignalMix:          0.05,
			SignalLength:       16,
			SignalDegeneration: 0.01,
			SignalRubric:       0.03,
		},
	}
}

// TestDriftLocalizedWithoutGlobalMisattribution is the witness: three cohorts run,
// a representative defect is planted in exactly one (its rubric score collapses),
// and the monitor must flag THAT cohort while reporting the other two stable —
// then, after the fix restores the score within tolerance, the whole run is clean.
func TestDriftLocalizedWithoutGlobalMisattribution(t *testing.T) {
	alpha := baseObs("alpha", DriftTierNightly)
	bravo := baseObs("bravo", DriftTierNightly)
	gamma := baseObs("gamma", DriftTierNightly)

	// Plant the defect: gamma's rubric score drops 0.20, far past its 0.03 tolerance.
	gamma.Observed[SignalRubric] = 0.70

	got := MonitorDrift("rev-1", []CohortObservation{alpha, bravo, gamma})

	if got.Clean {
		t.Fatal("planted rubric defect in gamma but report is Clean")
	}
	if len(got.Drifts) != 1 {
		t.Fatalf("drift misattributed: %d cohorts flagged, want exactly 1 (gamma)", len(got.Drifts))
	}
	d := got.Drifts[0]
	if d.Cohort != "gamma" || d.State != DriftDrifted {
		t.Fatalf("wrong cohort/state flagged: %+v, want gamma drifted", d)
	}
	if d.FirstDivergence == nil || d.FirstDivergence.Signal != SignalRubric {
		t.Fatalf("first divergence=%+v, want signal %q", d.FirstDivergence, SignalRubric)
	}
	// The two innocent cohorts must be reported stable, not smeared by gamma's drift.
	if want := []string{"alpha", "bravo"}; !equalStrings(got.Stable, want) {
		t.Fatalf("stable cohorts=%v, want %v (no global misattribution)", got.Stable, want)
	}

	// Scrubbed replay artifact: numbers and labels only, no raw model text.
	if d.Replay == nil {
		t.Fatal("drifted cohort emitted no replay artifact")
	}
	if d.Replay.Cohort != "gamma" || d.Replay.Signal != SignalRubric || d.Replay.Revision != "rev-1" {
		t.Fatalf("replay=%+v, want gamma/rubric/rev-1", d.Replay)
	}
	if d.Replay.Observed != 0.70 || d.Replay.Baseline != 0.90 {
		t.Fatalf("replay values=%+v, want observed 0.70 baseline 0.90", d.Replay)
	}

	// After the fix: gamma's score is restored within tolerance; the run goes clean.
	gamma.Observed[SignalRubric] = 0.89
	fixed := MonitorDrift("rev-1", []CohortObservation{alpha, bravo, gamma})
	if !fixed.Clean {
		t.Fatalf("after fix report not clean: %s", ExplainDrift(fixed))
	}
	if len(fixed.Drifts) != 0 {
		t.Fatalf("after fix still %d drift(s)", len(fixed.Drifts))
	}
	if want := []string{"alpha", "bravo", "gamma"}; !equalStrings(fixed.Stable, want) {
		t.Fatalf("after fix stable=%v, want %v", fixed.Stable, want)
	}
}

// TestDriftFailsClosedOnMissingOrInconclusiveEvidence proves "no evidence" and
// "unclear evidence" are never a pass: incomplete provenance, an unassigned tier,
// an absent baseline, an undeclared tolerance, and a missing observation each
// block instead of reporting stable.
func TestDriftFailsClosedOnMissingOrInconclusiveEvidence(t *testing.T) {
	cases := []struct {
		name  string
		mut   func(o *CohortObservation)
		state DriftState
	}{
		{"incomplete provenance", func(o *CohortObservation) { o.Provenance.Tokenizer = "" }, DriftInconclusive},
		{"neither seed nor oracle", func(o *CohortObservation) { o.Provenance.Oracle = "" }, DriftInconclusive},
		{"unassigned tier", func(o *CohortObservation) { o.Tier = "" }, DriftInconclusive},
		{"no baseline", func(o *CohortObservation) { o.Baseline = nil }, DriftMissing},
		{"undeclared tolerance", func(o *CohortObservation) { delete(o.Tolerance, SignalRubric) }, DriftInconclusive},
		{"missing observation", func(o *CohortObservation) { delete(o.Observed, SignalRubric) }, DriftInconclusive},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o := baseObs("solo", DriftTierPR)
			tc.mut(&o)
			got := MonitorDrift("rev-1", []CohortObservation{o})
			if got.Clean {
				t.Fatalf("%s reported Clean, want fail-closed", tc.name)
			}
			if len(got.Drifts) != 1 || got.Drifts[0].State != tc.state {
				t.Fatalf("%s state=%v, want %v", tc.name, got.Drifts, tc.state)
			}
			if len(got.Stable) != 0 {
				t.Fatalf("%s marked a cohort stable: %v", tc.name, got.Stable)
			}
		})
	}
}

// TestDriftFirstDivergenceIsDeterministic proves that when several signals drift
// the monitor localizes the first in sorted signal order, so the reported "first
// actionable divergence" is stable across Go's randomized map iteration.
func TestDriftFirstDivergenceIsDeterministic(t *testing.T) {
	o := baseObs("multi", DriftTierRelease)
	// Both degeneration and rubric drift; "degeneration" sorts before "rubric".
	o.Observed[SignalDegeneration] = 0.20
	o.Observed[SignalRubric] = 0.10

	for i := 0; i < 8; i++ {
		got := MonitorDrift("rev-1", []CohortObservation{o})
		if len(got.Drifts) != 1 {
			t.Fatalf("iter %d: %d drifts, want 1", i, len(got.Drifts))
		}
		fd := got.Drifts[0].FirstDivergence
		if fd == nil || fd.Signal != SignalDegeneration {
			t.Fatalf("iter %d: first divergence=%+v, want deterministic %q", i, fd, SignalDegeneration)
		}
	}
}

// TestExplainDriftReadout checks the operator readout marks the first actionable
// cohort and never leaks anything but labels and numbers.
func TestExplainDriftReadout(t *testing.T) {
	bad := baseObs("gamma", DriftTierNightly)
	bad.Observed[SignalRubric] = 0.40
	out := ExplainDrift(MonitorDrift("rev-1", []CohortObservation{baseObs("alpha", DriftTierNightly), bad}))
	if !strings.HasPrefix(out, "DRIFT") {
		t.Fatalf("readout=%q, want DRIFT prefix", out)
	}
	if !strings.Contains(out, "-> ") || !strings.Contains(out, "gamma") {
		t.Fatalf("readout missing first-actionable marker or cohort: %q", out)
	}
	clean := ExplainDrift(MonitorDrift("rev-1", []CohortObservation{baseObs("alpha", DriftTierNightly)}))
	if !strings.HasPrefix(clean, "CLEAN") {
		t.Fatalf("clean readout=%q, want CLEAN prefix", clean)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
