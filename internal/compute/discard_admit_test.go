package compute

import (
	"reflect"
	"testing"
)

// TestDecideDiscardAdmissionFailsOpen pins the load-bearing fence: a candidate with no
// known-discard signal ALWAYS proceeds, in every phase. This is what makes wiring the
// decision into a scheduler safe — it can never refuse a sequence the historical path
// would have run (issue #808; #807 advisory-never-trust posture).
func TestDecideDiscardAdmissionFailsOpen(t *testing.T) {
	for _, phase := range []AdmitPhase{PhaseQueued, PhaseRunning} {
		got := DecideDiscardAdmission(DiscardCandidate{Phase: phase, WillDiscard: false})
		if got != AdmitProceed {
			t.Fatalf("no-discard candidate in phase %v: got %v, want proceed", phase, got)
		}
	}
	// The zero value (the default-constructed decision) must proceed.
	if got := DecideDiscardAdmission(DiscardCandidate{}); got != AdmitProceed {
		t.Fatalf("zero candidate: got %v, want proceed", got)
	}
}

// TestDecideDiscardAdmissionVerdicts pins the two positive verdicts: a known-discard
// sequence drops while queued (never dispatched) and is preempted while running (slot
// freed).
func TestDecideDiscardAdmissionVerdicts(t *testing.T) {
	cases := []struct {
		name  string
		cand  DiscardCandidate
		want  AdmitVerdict
		print string
	}{
		{"queued-discard-drops", DiscardCandidate{Phase: PhaseQueued, WillDiscard: true}, AdmitDrop, "drop"},
		{"running-discard-preempts", DiscardCandidate{Phase: PhaseRunning, WillDiscard: true}, AdmitPreempt, "preempt"},
		{"queued-clean-proceeds", DiscardCandidate{Phase: PhaseQueued}, AdmitProceed, "proceed"},
		{"running-clean-proceeds", DiscardCandidate{Phase: PhaseRunning}, AdmitProceed, "proceed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecideDiscardAdmission(tc.cand)
			if got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			if got.String() != tc.print {
				t.Fatalf("verdict string: got %q, want %q", got.String(), tc.print)
			}
		})
	}
}

// TestPlanDiscardAdmissionStats checks the aggregate throughput/cost-win numbers over a
// mixed batch: dropped queued candidates count their prefill rows as never-dispatched, and
// preempted running candidates count their KV slots as reclaimed.
func TestPlanDiscardAdmissionStats(t *testing.T) {
	cands := []DiscardCandidate{
		{Phase: PhaseQueued, WillDiscard: true, PrefillRows: 100, Reason: "tool-cancelled"}, // drop, save 100 rows
		{Phase: PhaseRunning, WillDiscard: true, KVSlots: 4, Reason: "turn-superseded"},     // preempt, free 4 slots
		{Phase: PhaseQueued, WillDiscard: false, PrefillRows: 50},                           // proceed (no opinion)
		{Phase: PhaseRunning, WillDiscard: false, KVSlots: 2},                               // proceed (no opinion)
		{Phase: PhaseQueued, WillDiscard: true, PrefillRows: 30, Reason: "branch-refuted"},  // drop, save 30 rows
	}
	decisions, stats := PlanDiscardAdmission(cands)

	if len(decisions) != len(cands) {
		t.Fatalf("decisions length: got %d, want %d", len(decisions), len(cands))
	}
	for i, d := range decisions {
		if d.Index != i {
			t.Fatalf("decision %d carries index %d", i, d.Index)
		}
	}
	want := DiscardAdmissionStats{
		Candidates:       5,
		Dropped:          2,
		Preempted:        1,
		Proceeded:        2,
		SlotsFreed:       4,
		PrefillRowsSaved: 130,
	}
	if stats != want {
		t.Fatalf("stats: got %+v, want %+v", stats, want)
	}
}

// TestPlanDiscardAdmissionDeterministic pins byte-determinism: the same input yields an
// identical plan every time (the house requirement for a scheduling policy that must not
// drift with hardware), and re-deciding a dropped candidate is stable — the idempotent /
// fail-safe contract.
func TestPlanDiscardAdmissionDeterministic(t *testing.T) {
	cands := []DiscardCandidate{
		{Phase: PhaseRunning, WillDiscard: true, KVSlots: 3},
		{Phase: PhaseQueued, WillDiscard: true, PrefillRows: 64},
		{Phase: PhaseQueued, WillDiscard: false, PrefillRows: 64},
	}
	d1, s1 := PlanDiscardAdmission(cands)
	d2, s2 := PlanDiscardAdmission(cands)
	if !reflect.DeepEqual(d1, d2) || s1 != s2 {
		t.Fatalf("non-deterministic plan:\n d1=%+v s1=%+v\n d2=%+v s2=%+v", d1, s1, d2, s2)
	}

	// Re-deciding an already-dropped candidate is stable (idempotent): the drop verdict
	// does not depend on prior decisions.
	dropped := cands[1]
	if v := DecideDiscardAdmission(dropped); v != AdmitDrop {
		t.Fatalf("first decide: got %v, want drop", v)
	}
	if v := DecideDiscardAdmission(dropped); v != AdmitDrop {
		t.Fatalf("re-decide (idempotent): got %v, want drop", v)
	}
}

// TestPlanDiscardAdmissionEmpty pins the empty/nil contract: no candidates -> nil
// decisions and zero stats, never a panic.
func TestPlanDiscardAdmissionEmpty(t *testing.T) {
	d, s := PlanDiscardAdmission(nil)
	if d != nil {
		t.Fatalf("nil input: got decisions %v, want nil", d)
	}
	if (s != DiscardAdmissionStats{}) {
		t.Fatalf("nil input: got stats %+v, want zero", s)
	}
}
