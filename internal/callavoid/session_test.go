package callavoid

import (
	"encoding/json"
	"reflect"
	"testing"
)

// session_test.go — the regression floor for the parent #815 CONSOLIDATION: the single
// AccountFromObservations entrypoint composes the three children (#818 Counters fold, #820
// witnessed fan-out, #819 observed class admit-set) into ONE honest headline that
//   - is FAITHFUL to hand-stitching the children (it adds no behaviour, only wiring),
//   - never DOUBLE-COUNTS a deny (hard vs productive),
//   - never credits an UNWITNESSED path into the grade (speculative + empty witnesses),
//   - never credits an UNMEASURED class into the admitted set,
//   - accounts the pure no-tool-call (zero Execute) window correctly.

// TestAccountFromObservationsFaithfulToHandStitch pins that the composed entrypoint is
// EXACTLY the hand-wired pipeline the children expose — same Turns report, field for field —
// so the consolidation is a wiring convenience that can never silently diverge from the
// pieces it composes. If a child changes, this test breaks unless the composition tracks it.
func TestAccountFromObservationsFaithfulToHandStitch(t *testing.T) {
	c := Counters{EngineCalls: 6, VDSOHits: 30, Transforms: 4, Denies: 5}
	wits := []WitnessedRedirect{
		{Rule: "policy.read-scope", Variants: []string{"a", "b", "c"}},
		{Rule: "cap.fs.write", Variants: []string{"x", "y"}},
	}
	spec := []int{50, 50}

	// The hand-stitched pipeline a caller would otherwise write by hand.
	handTally := TallyFromCountersWitnessed(c, wits)
	handTally.Redirects = spec
	handReport := Account(handTally)

	got := AccountFromObservations(SessionInput{
		Counters:             c,
		WitnessedDenies:      wits,
		SpeculativeRedirects: spec,
	})

	if !reflect.DeepEqual(got.Turns, handReport) {
		t.Fatalf("composed Turns diverged from the hand-stitched pipeline:\n composed=%+v\n hand=%+v", got.Turns, handReport)
	}
	if got.Schema != "fak.callavoid.session.v1" {
		t.Errorf("schema = %q, want fak.callavoid.session.v1", got.Schema)
	}
}

// TestAccountFromObservationsNoDoubleCount proves the composed headline counts each deny
// exactly once. Of 5 denies, 2 are witnessed productive (netted out of HardDeny) and 3 stay
// hard. RawTurns must reconcile with the counters (every disposition once), and the witnessed
// fan-out must be credited ONLY from the enumerated variants — never added on top of a deny
// that was also counted as hard.
func TestAccountFromObservationsNoDoubleCount(t *testing.T) {
	c := Counters{EngineCalls: 6, VDSOHits: 30, Transforms: 4, Denies: 5}
	wits := []WitnessedRedirect{
		{Variants: []string{"a", "b", "c"}}, // 3 distinct
		{Variants: []string{"x", "y"}},      // 2 distinct
	}
	r := AccountFromObservations(SessionInput{Counters: c, WitnessedDenies: wits}).Turns

	// 5 denies = 3 hard + 2 witnessed: the witnessed ones were MOVED out of HardDeny, not added.
	if r.HardDenies != 3 {
		t.Errorf("hard denies = %d, want 3 (5 denies - 2 witnessed, netted not added)", r.HardDenies)
	}
	if r.WitnessedDenies != 2 {
		t.Errorf("witnessed denies = %d, want 2", r.WitnessedDenies)
	}
	// RawTurns counts each disposition exactly once: 6 exec + 30 memo + 4 repair + 3 hard + 2 witnessed = 45.
	if r.RawTurns != 45 {
		t.Errorf("raw turns = %d, want 45 (each deny counted once, no double-count)", r.RawTurns)
	}
	// The witnessed credit is the enumerated variant count (3+2=5), nothing more.
	if r.WitnessedPruned != 5 {
		t.Errorf("witnessed pruned = %d, want 5 (sum of distinct enumerated variants)", r.WitnessedPruned)
	}
	// Effective = realized work + avoidance + witnessed fan-out, with hard denies symmetric:
	// 6 exec + 30 memo + 4 repair + 5 witnessed = 45; hard denies add 0.
	if !approx(r.EffectiveTurns, 45) {
		t.Errorf("effective = %v, want 45", r.EffectiveTurns)
	}
}

// TestAccountFromObservationsNoUnwitnessedCredit proves the grade is built from realized,
// witnessed dispositions ONLY: an empty-variant witness credits nothing, and a flood of
// purely-speculative (non-enumerated) redirects cannot move the graded amplification — they
// ride the excluded speculative axis exactly as for a hand-built Tally (#816).
func TestAccountFromObservationsNoUnwitnessedCredit(t *testing.T) {
	base := AccountFromObservations(SessionInput{
		Counters: Counters{EngineCalls: 1, VDSOHits: 9},
	}).Turns

	flood := make([]int, 5000)
	for i := range flood {
		flood[i] = 1 << 20 // each far over the per-deny cap
	}
	withSpec := AccountFromObservations(SessionInput{
		Counters:             Counters{EngineCalls: 1, VDSOHits: 9, Denies: 1},
		WitnessedDenies:      []WitnessedRedirect{{Variants: []string{"", ""}}}, // names NOTHING -> empty witness
		SpeculativeRedirects: flood,
	}).Turns

	// The empty witness credited zero realized fan-out and is surfaced, never inflated.
	if withSpec.WitnessedPruned != 0 || withSpec.WitnessedEmpty != 1 {
		t.Errorf("empty witness credited fan-out: pruned=%d empty=%d, want 0/1", withSpec.WitnessedPruned, withSpec.WitnessedEmpty)
	}
	// The graded amplification/grade/status are IDENTICAL with vs without the speculative flood
	// and the empty witness — neither unwitnessed path can touch the grade.
	if !approx(base.Amplification, withSpec.Amplification) || base.Grade != withSpec.Grade || base.Status != withSpec.Status {
		t.Fatalf("an unwitnessed path moved the grade: base amp/grade/status=%v/%s/%s, with=%v/%s/%s",
			base.Amplification, base.Grade, base.Status, withSpec.Amplification, withSpec.Grade, withSpec.Status)
	}
	// The speculative upper bound is bounded in aggregate, not ~5e9.
	if withSpec.SpeculativePrunedTurns != DefaultMaxSpeculativePrunedTurns || !withSpec.SpeculativeAggregateCapped {
		t.Errorf("speculative pruned = %d capped=%v, want saturated at %d and flagged",
			withSpec.SpeculativePrunedTurns, withSpec.SpeculativeAggregateCapped, DefaultMaxSpeculativePrunedTurns)
	}
}

// TestAccountFromObservationsNoUnmeasuredCredit proves the admitted-class advisory is the
// projection of MEASURED-and-proving classes only (#819's calibrate-don't-assume): a stable
// read-heavy class is admitted, a write-churned class is declined as a measured net-loss, and
// an unmeasured class abstains. None of the three touches the amplification grade.
func TestAccountFromObservationsNoUnmeasuredCredit(t *testing.T) {
	obs := []ClassObservation{
		{Class: "Read", ReuseAttempts: 100, Invalidations: 1},   // stable -> admit
		{Class: "Write", ReuseAttempts: 100, Invalidations: 95}, // write-churned -> measured decline
		{Class: "Glob"}, // no reuse attempts -> abstain (unmeasured)
	}
	rep := AccountFromObservations(SessionInput{
		Counters:          Counters{EngineCalls: 2, VDSOHits: 8},
		ClassObservations: obs,
	})

	if !reflect.DeepEqual(rep.AdmittedClasses, []string{"Read"}) {
		t.Errorf("admitted = %v, want [Read] (only the measured-and-proving class)", rep.AdmittedClasses)
	}
	// All three verdicts are surfaced so a caller sees WHY a class was declined.
	if len(rep.ObservedClasses) != 3 {
		t.Fatalf("observed classes = %d, want 3", len(rep.ObservedClasses))
	}
	// The abstained class is honestly marked unmeasured, distinct from the measured decline.
	var glob, write ObservedClassGate
	for _, g := range rep.ObservedClasses {
		switch g.Decision.Class {
		case "Glob":
			glob = g
		case "Write":
			write = g
		}
	}
	if glob.Measured || glob.Decision.Admit {
		t.Errorf("Glob should abstain (unmeasured, not admitted), got measured=%v admit=%v", glob.Measured, glob.Decision.Admit)
	}
	if !write.Measured || write.Decision.Admit {
		t.Errorf("Write should be a MEASURED decline, got measured=%v admit=%v", write.Measured, write.Decision.Admit)
	}
	// The admitted-class advisory NEVER feeds the grade: dropping the observations entirely
	// leaves the amplification identical.
	noObs := AccountFromObservations(SessionInput{Counters: Counters{EngineCalls: 2, VDSOHits: 8}}).Turns
	if !approx(noObs.Amplification, rep.Turns.Amplification) || noObs.Grade != rep.Turns.Grade {
		t.Errorf("class observations moved the grade: with=%v/%s, without=%v/%s",
			rep.Turns.Amplification, rep.Turns.Grade, noObs.Amplification, noObs.Grade)
	}
}

// TestAccountFromObservationsPureNoToolCall pins the no-tool-call (pure-avoid) accounting the
// parent issue names: a window with ZERO real engine dispatches — only memo hits, repairs, and
// a witnessed productive deny — must still account honestly. The executed work is just the
// near-free memo validations; the effective turns tower over it; and a window of ONLY repairs
// (which cost nothing and validate nothing) is the all-avoided +Inf-amplification regime that
// must grade A without any unwitnessed credit.
func TestAccountFromObservationsPureNoToolCall(t *testing.T) {
	// Only avoidance: 0 execute, 12 memo hits, 3 repairs, 1 witnessed deny (2 variants).
	r := AccountFromObservations(SessionInput{
		Counters:        Counters{VDSOHits: 12, Transforms: 3, Denies: 1},
		WitnessedDenies: []WitnessedRedirect{{Variants: []string{"p", "q"}}},
	}).Turns

	// Executed = only the 12 memo validations at the floor (0.01 each) = 0.12. No dispatch paid.
	wantExec := 12 * ValidateFloor
	if !approx(r.ExecutedTurns, wantExec) {
		t.Fatalf("executed = %v, want %v (no dispatch — only memo validations)", r.ExecutedTurns, wantExec)
	}
	// Effective = 12 avoided reads + 3 avoided retries + 2 witnessed pruned = 17.
	if !approx(r.EffectiveTurns, 17) {
		t.Fatalf("effective = %v, want 17 (12 memo + 3 repair + 2 witnessed)", r.EffectiveTurns)
	}
	if r.Status != "amplifying" || r.Amplification <= 1 {
		t.Fatalf("pure-avoid window status/amp = %s/%v, want amplifying/>1", r.Status, r.Amplification)
	}

	// The fully-avoided extreme: a window of only repairs pays nothing and validates nothing, so
	// executed work is 0 and amplification is +Inf — graded A, the all-avoided regime — but it is
	// realized (repairs are Counter-backed), NOT an unwitnessed counterfactual.
	allAvoided := AccountFromObservations(SessionInput{Counters: Counters{Transforms: 4}}).Turns
	if allAvoided.ExecutedTurns != 0 || allAvoided.Grade != "A" {
		t.Errorf("all-repair window: executed=%v grade=%s, want 0/A (all avoided, realized)", allAvoided.ExecutedTurns, allAvoided.Grade)
	}
}

// TestAccountFromObservationsEmptyIsBreakEven: an all-zero SessionInput is the honest "nothing
// happened" headline — break-even, grade-neutral, no fabricated credit and no admitted classes.
func TestAccountFromObservationsEmptyIsBreakEven(t *testing.T) {
	rep := AccountFromObservations(SessionInput{})
	if rep.Turns.Status != "break_even" || rep.Turns.RawTurns != 0 {
		t.Errorf("empty session = %s/%d turns, want break_even/0", rep.Turns.Status, rep.Turns.RawTurns)
	}
	if len(rep.AdmittedClasses) != 0 || len(rep.ObservedClasses) != 0 {
		t.Errorf("empty session credited classes: admitted=%v observed=%v", rep.AdmittedClasses, rep.ObservedClasses)
	}
}

// TestSessionReportJSONRoundTrips: the composed report marshals to the stable shape a tier-4
// caller renders, with both the realized turns and the advisory admitted-class set present.
func TestSessionReportJSONRoundTrips(t *testing.T) {
	rep := AccountFromObservations(SessionInput{
		Counters:          Counters{EngineCalls: 2, VDSOHits: 8, Denies: 1},
		WitnessedDenies:   []WitnessedRedirect{{Rule: "r", Variants: []string{"a"}}},
		ClassObservations: []ClassObservation{{Class: "Read", ReuseAttempts: 50, Invalidations: 1}},
	})
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["schema"] != "fak.callavoid.session.v1" {
		t.Errorf("schema = %v, want fak.callavoid.session.v1", m["schema"])
	}
	if _, ok := m["turns"]; !ok {
		t.Errorf("report missing turns headline: %v", m)
	}
	ac, ok := m["admitted_classes"].([]any)
	if !ok || len(ac) != 1 || ac[0] != "Read" {
		t.Errorf("admitted_classes = %v, want [Read]", m["admitted_classes"])
	}
}
