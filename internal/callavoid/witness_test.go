package callavoid

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// witness_test.go — regression floor for #820 (witnessed productive-deny credit),
// #819 (per-tool-class economics gate), and #818 (Counters -> Tally fold).

// ---------------------------------------------------------------------------
// #820 — a WITNESSED productive deny carries enumerated variants and graduates to
// realized credit, while a bare Redirects count stays speculative.
// ---------------------------------------------------------------------------

// TestWitnessedDenyCreditsRealized: a deny whose pruned variants are ENUMERATED folds its
// (deduplicated) fan-out into the graded EffectiveTurns/Amplification — unlike a bare
// Redirects count, which stays on the excluded speculative axis. One real execute plus a
// witnessed deny that names 3 distinct futile variants reaches the same state a naive
// agent spends 1+3 = 4 round-trips on, for 1 unit of paid work.
func TestWitnessedDenyCreditsRealized(t *testing.T) {
	r := Account(Tally{
		Execute: 1,
		WitnessedRedirects: []WitnessedRedirect{
			{Rule: "policy.read-scope", Variants: []string{"a", "b", "c"}},
		},
	})
	if r.WitnessedDenies != 1 || r.WitnessedPruned != 3 {
		t.Fatalf("witnessed denies/pruned = %d/%d, want 1/3", r.WitnessedDenies, r.WitnessedPruned)
	}
	// EffectiveTurns = execute(1) + witnessed fan-out(3) = 4; executed = 1 -> amp 4.
	if !approx(r.EffectiveTurns, 4) || !approx(r.ExecutedTurns, 1) {
		t.Fatalf("effective/executed = %v/%v, want 4/1 (witnessed credit IS realized)", r.EffectiveTurns, r.ExecutedTurns)
	}
	if !approx(r.Amplification, 4) || r.Status != "amplifying" {
		t.Errorf("amp/status = %v/%s, want 4/amplifying", r.Amplification, r.Status)
	}
	// The realized credit must be surfaced as an action (never silent).
	if !hasAction(r.Actions, "witnessed productive deny") {
		t.Errorf("witnessed credit must surface an action:\n%v", r.Actions)
	}
}

// TestWitnessedVsSpeculativeBoundary is the heart of #820: the SAME pruned size credits
// the grade when WITNESSED (enumerated variants) but is excluded when SPECULATIVE (a bare
// count). With one execute, a 3-variant witnessed deny amplifies to 4x; a bare Redirects
// count of 3 stays break-even.
func TestWitnessedVsSpeculativeBoundary(t *testing.T) {
	witnessed := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{{Variants: []string{"a", "b", "c"}}}})
	speculative := Account(Tally{Execute: 1, Redirects: []int{3}})

	if !approx(witnessed.Amplification, 4) {
		t.Errorf("witnessed amplification = %v, want 4 (enumerated variants ARE credited)", witnessed.Amplification)
	}
	if !approx(speculative.Amplification, 1) || speculative.Status != "break_even" {
		t.Errorf("speculative amplification/status = %v/%s, want 1/break_even (a bare count is NOT credited)",
			speculative.Amplification, speculative.Status)
	}
	// Same nominal pruning size, opposite grade treatment — that is the whole point.
	if witnessed.Grade == speculative.Grade {
		t.Errorf("witnessed and speculative graded the same (%s); the witness must change the grade", witnessed.Grade)
	}
}

// TestWitnessedDedupAndBlankCreditNothing: a witness is non-gameable on enumeration —
// listing the same variant repeatedly, or padding with blanks, credits each DISTINCT
// non-empty variant exactly once. "a","a","","b" credits 2, not 4.
func TestWitnessedDedupAndBlankCreditNothing(t *testing.T) {
	r := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{
		{Variants: []string{"a", "a", "", "b"}},
	}})
	if r.WitnessedPruned != 2 {
		t.Errorf("witnessed pruned = %d, want 2 (dedup + drop blanks)", r.WitnessedPruned)
	}
	if !approx(r.EffectiveTurns, 3) { // 1 execute + 2 distinct variants.
		t.Errorf("effective = %v, want 3", r.EffectiveTurns)
	}
}

// TestEmptyWitnessIsAHardDeny: a witnessed redirect that names NO variants (or only
// blanks) credits nothing — it is an effective hard deny — and is surfaced as such, so an
// un-enumerated "productive" deny can never inflate the realized headline.
func TestEmptyWitnessIsAHardDeny(t *testing.T) {
	r := Account(Tally{Execute: 2, WitnessedRedirects: []WitnessedRedirect{
		{Rule: "policy.x", Variants: nil},
		{Rule: "policy.y", Variants: []string{"", ""}},
	}})
	if r.WitnessedDenies != 0 || r.WitnessedPruned != 0 {
		t.Errorf("witnessed denies/pruned = %d/%d, want 0/0 (no variants named)", r.WitnessedDenies, r.WitnessedPruned)
	}
	if r.WitnessedEmpty != 2 {
		t.Errorf("witnessed empty = %d, want 2", r.WitnessedEmpty)
	}
	// No realized credit: 2 executes, nothing avoided -> break-even.
	if !approx(r.Amplification, 1) || r.Status != "break_even" {
		t.Errorf("amp/status = %v/%s, want 1/break_even (empty witnesses credit nothing)", r.Amplification, r.Status)
	}
	if !hasRisk(r.Risks, "named NO variants") {
		t.Errorf("an empty witness must surface a risk:\n%v", r.Risks)
	}
}

// TestWitnessedPerDenyCapBounds: a single witnessed deny cannot credit an unbounded
// fan-out — its distinct-variant count is clamped to the per-deny cap, the clamp is
// surfaced, and the credit is a lower bound, never inflated.
func TestWitnessedPerDenyCapBounds(t *testing.T) {
	variants := make([]string, DefaultMaxRedirectFanout+50)
	for i := range variants {
		variants[i] = "v" + strconv.Itoa(i) // each distinct, so the dedup keeps them all up to the cap.
	}
	r := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{{Variants: variants}}})
	if r.WitnessedPruned != DefaultMaxRedirectFanout {
		t.Errorf("witnessed pruned = %d, want clamped to %d", r.WitnessedPruned, DefaultMaxRedirectFanout)
	}
	if r.WitnessedCapped != 1 {
		t.Errorf("witnessed capped = %d, want 1 (the clamp surfaced)", r.WitnessedCapped)
	}
	if !hasRisk(r.Risks, "clamped to the per-deny cap") {
		t.Errorf("a clamped witness must surface a risk:\n%v", r.Risks)
	}
}

// TestWitnessedAggregateCapBounds: an unbounded NUMBER of witnessed denies cannot push the
// realized credit past the aggregate ceiling — the witnessed total saturates at
// DefaultMaxSpeculativePrunedTurns and the saturation is surfaced.
func TestWitnessedAggregateCapBounds(t *testing.T) {
	// Each witness credits DefaultMaxRedirectFanout (1024); enough of them to blow past
	// the aggregate cap (1<<20). 2000*1024 > 1<<20.
	wits := make([]WitnessedRedirect, 2000)
	for i := range wits {
		vs := make([]string, DefaultMaxRedirectFanout)
		for j := range vs {
			vs[j] = "w" + strconv.Itoa(i) + "_" + strconv.Itoa(j)
		}
		wits[i] = WitnessedRedirect{Variants: vs}
	}
	r := Account(Tally{Execute: 1, WitnessedRedirects: wits})
	if r.WitnessedPruned != DefaultMaxSpeculativePrunedTurns || !r.WitnessedAggregateCapped {
		t.Fatalf("witnessed pruned = %d capped=%v, want saturated at %d and flagged",
			r.WitnessedPruned, r.WitnessedAggregateCapped, DefaultMaxSpeculativePrunedTurns)
	}
	if !hasRisk(r.Risks, "saturated the aggregate cap") {
		t.Errorf("an aggregate-capped witnessed total must surface a risk:\n%v", r.Risks)
	}
}

// ---------------------------------------------------------------------------
// #819 — the per-tool-class economics gate.
// ---------------------------------------------------------------------------

// TestGateClassAdmitsStableClass: a calm, cheaply-validated class (low m, tiny v/c)
// PROVES and is admitted.
func TestGateClassAdmitsStableClass(t *testing.T) {
	d := GateClass(ClassMemoInput{Class: "Read", Accesses: 10, ValidateCost: 0.02, MutationRate: 0.02, CaptureCost: 0.02})
	if !d.Admit || d.Proof.Status != ProofProven {
		t.Fatalf("stable class must ADMIT/PROVE, got admit=%v %s (%s)", d.Admit, d.Proof.Status, d.Proof.Reason)
	}
	if d.Class != "Read" {
		t.Errorf("class = %q, want Read", d.Class)
	}
}

// TestGateClassDeclinesVolatileClass: a class the world almost always mutates under (high
// m) REFUTES at every reuse and is declined — the #819 acceptance bullet that a volatile /
// non-fingerprintable class is not admitted to the tier-2 cache.
func TestGateClassDeclinesVolatileClass(t *testing.T) {
	d := GateClass(ClassMemoInput{Class: "Bash", Accesses: 100, ValidateCost: 0.02, MutationRate: 0.99, CaptureCost: 0.02})
	if d.Admit || d.Proof.Status != ProofRefuted {
		t.Fatalf("volatile class must DECLINE/REFUTE, got admit=%v %s (%s)", d.Admit, d.Proof.Status, d.Proof.Reason)
	}
	if d.Proof.BreakEvenAccesses != neverBreakEven {
		t.Errorf("volatile class break-even = %d, want never", d.Proof.BreakEvenAccesses)
	}
}

// TestGateClassDeclinesNonFingerprintable: a class whose validate cost rivals execution
// (validate ~= 1) never pays at any reuse and is declined.
func TestGateClassDeclinesNonFingerprintable(t *testing.T) {
	d := GateClass(ClassMemoInput{Class: "WebFetch", Accesses: 50, ValidateCost: 1.0, MutationRate: 0.05, CaptureCost: 0.02})
	if d.Admit {
		t.Fatalf("a class as costly to validate as to run must DECLINE, got admit=true (%s)", d.Note)
	}
}

// TestGateClassDefaultsToBreakEvenProbe: an unset Accesses is judged on the per-reuse
// economics via the break-even probe of 2, not on a degenerate single-access window — so a
// stable class still admits even when the caller gives no representative k.
func TestGateClassDefaultsToBreakEvenProbe(t *testing.T) {
	d := GateClass(ClassMemoInput{Class: "Glob", ValidateCost: 0.02, MutationRate: 0.02, CaptureCost: 0.02})
	if d.Proof.Accesses != 2 {
		t.Errorf("default accesses = %d, want the break-even probe of 2", d.Proof.Accesses)
	}
	if !d.Admit {
		t.Errorf("a stable class must admit on the probe, got decline (%s)", d.Note)
	}
}

// TestGateClassesAndAdmittedProjection: the batch gate decides per class in order, and the
// admitted projection returns exactly the proving classes — the allow-set the seam builds.
func TestGateClassesAndAdmittedProjection(t *testing.T) {
	in := []ClassMemoInput{
		{Class: "Read", Accesses: 10, ValidateCost: 0.02, MutationRate: 0.02, CaptureCost: 0.02}, // admit
		{Class: "Bash", Accesses: 10, ValidateCost: 0.02, MutationRate: 0.99, CaptureCost: 0.02}, // decline (volatile)
		{Class: "Glob", Accesses: 10, ValidateCost: 0.02, MutationRate: 0.01, CaptureCost: 0.02}, // admit
	}
	decisions := GateClasses(in)
	if len(decisions) != 3 || decisions[0].Class != "Read" || decisions[1].Class != "Bash" {
		t.Fatalf("batch order/shape wrong: %+v", decisions)
	}
	admitted := AdmittedClasses(in)
	if !reflect.DeepEqual(admitted, []string{"Read", "Glob"}) {
		t.Errorf("admitted = %v, want [Read Glob] (Bash declined)", admitted)
	}
}

// TestGateClassDeterministic: pure — identical class calibration yields identical decision.
func TestGateClassDeterministic(t *testing.T) {
	in := ClassMemoInput{Class: "Read", Accesses: 7, ValidateCost: 0.03, MutationRate: 0.15, CaptureCost: 0.2}
	if !reflect.DeepEqual(GateClass(in), GateClass(in)) {
		t.Error("GateClass is not deterministic")
	}
}

// ---------------------------------------------------------------------------
// #818 — the Counters -> Tally fold.
// ---------------------------------------------------------------------------

// TestTallyFromCounters maps a session's kernel counters onto a Tally per the documented
// mapping, and Account over it reconciles with the same counters (no double-count).
func TestTallyFromCounters(t *testing.T) {
	c := Counters{EngineCalls: 4, VDSOHits: 6, Transforms: 2, Denies: 3}
	tally := TallyFromCounters(c)
	if tally.Execute != 4 || tally.MemoHit != 6 || tally.Repair != 2 || tally.HardDeny != 3 {
		t.Fatalf("mapping = %+v, want Execute4 MemoHit6 Repair2 HardDeny3", tally)
	}
	r := Account(tally)
	// RawTurns reconciles with the counters: 4+6+2+3 = 15 (denies are hard, symmetric).
	if r.RawTurns != 15 {
		t.Errorf("raw turns = %d, want 15 (reconciles with the counters)", r.RawTurns)
	}
	// Hard denies are symmetric, so the amplification is execute+memo only.
	if !approx(r.EffectiveTurns, 4+6+2) {
		t.Errorf("effective = %v, want 12 (denies add nothing — they are hard)", r.EffectiveTurns)
	}
}

// TestTallyFromCountersWitnessedNetsDenies: folding witnessed productive denies MOVES them
// out of HardDeny (never adds on top), so a deny is counted once — either hard or
// productive, never both.
func TestTallyFromCountersWitnessedNetsDenies(t *testing.T) {
	c := Counters{EngineCalls: 1, Denies: 3}
	wits := []WitnessedRedirect{{Variants: []string{"a", "b"}}} // 1 of the 3 denies was witnessed.
	tally := TallyFromCountersWitnessed(c, wits)
	if tally.HardDeny != 2 {
		t.Errorf("hard deny = %d, want 2 (3 denies - 1 witnessed)", tally.HardDeny)
	}
	if len(tally.WitnessedRedirects) != 1 {
		t.Errorf("witnessed redirects = %d, want 1", len(tally.WitnessedRedirects))
	}
	r := Account(tally)
	// RawTurns counts each deny once: 1 execute + 2 hard + 1 witnessed = 4.
	if r.RawTurns != 4 {
		t.Errorf("raw turns = %d, want 4 (no double-count of the witnessed deny)", r.RawTurns)
	}
	// The witnessed deny credits its 2 variants into the realized headline.
	if r.WitnessedPruned != 2 {
		t.Errorf("witnessed pruned = %d, want 2", r.WitnessedPruned)
	}
}

// TestTallyFromCountersWitnessedSurplusFloors: more witnessed denies than recorded floors
// HardDeny at zero rather than going negative, and the witnessed credit still comes from
// the enumerated variants (the witness, not the count it was netted against).
func TestTallyFromCountersWitnessedSurplusFloors(t *testing.T) {
	c := Counters{Denies: 1}
	wits := []WitnessedRedirect{
		{Variants: []string{"a"}},
		{Variants: []string{"b"}},
	}
	tally := TallyFromCountersWitnessed(c, wits)
	if tally.HardDeny != 0 {
		t.Errorf("hard deny = %d, want 0 (floored, never negative)", tally.HardDeny)
	}
	r := Account(tally)
	if r.WitnessedPruned != 2 {
		t.Errorf("witnessed pruned = %d, want 2 (credit comes from the witness)", r.WitnessedPruned)
	}
}

// TestTallyFromCountersNegativeGuards: a negative counter (defense-in-depth) floors to zero
// — it can never inflate the mapped tally.
func TestTallyFromCountersNegativeGuards(t *testing.T) {
	tally := TallyFromCounters(Counters{EngineCalls: -5, VDSOHits: -1, Transforms: -2, Denies: -3})
	if tally.Execute != 0 || tally.MemoHit != 0 || tally.Repair != 0 || tally.HardDeny != 0 {
		t.Errorf("negative counters not floored: %+v", tally)
	}
}

// TestFoldJSONRoundTrips: the new artifact shapes marshal to stable JSON a caller consumes.
func TestFoldJSONRoundTrips(t *testing.T) {
	cb, err := json.Marshal(Counters{EngineCalls: 1, VDSOHits: 2})
	if err != nil {
		t.Fatalf("marshal Counters: %v", err)
	}
	var cm map[string]any
	if err := json.Unmarshal(cb, &cm); err != nil {
		t.Fatalf("unmarshal Counters: %v", err)
	}
	if cm["engine_calls"] != float64(1) {
		t.Errorf("engine_calls = %v, want 1", cm["engine_calls"])
	}
	db, err := json.Marshal(GateClass(ClassMemoInput{Class: "Read", Accesses: 5, ValidateCost: 0.02, MutationRate: 0.02, CaptureCost: 0.02}))
	if err != nil {
		t.Fatalf("marshal ClassGateDecision: %v", err)
	}
	var dm map[string]any
	if err := json.Unmarshal(db, &dm); err != nil {
		t.Fatalf("unmarshal ClassGateDecision: %v", err)
	}
	if dm["admit"] != true || dm["class"] != "Read" {
		t.Errorf("decision JSON = %v", dm)
	}
}

// ---------------------------------------------------------------------------
// helpers local to this file.
// ---------------------------------------------------------------------------

func hasAction(actions []string, needle string) bool {
	for _, a := range actions {
		if strings.Contains(a, needle) {
			return true
		}
	}
	return false
}
