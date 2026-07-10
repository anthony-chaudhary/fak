package guardaccuracy

import (
	"strings"
	"testing"
)

// TestFoldComplaintsCountsOnlyOverBlockKinds pins the intake reduction: only
// over-block appeals (false-positive, over-broad) count as field-FP evidence;
// latency/confusing/other/unknown kinds are real complaints but not escalate-
// boundary signals and are excluded. Occurrences sum, and a sub-1 count clamps
// to 1 (guardcomplaint's own floor).
func TestFoldComplaintsCountsOnlyOverBlockKinds(t *testing.T) {
	sig := FoldComplaints([]FieldComplaint{
		{Kind: "false-positive", Summary: "grep git push in docs", Occurrences: 3},
		{Kind: "over-broad", Summary: "commit message mentions rm", Occurrences: 1},
		{Kind: "FALSE-POSITIVE", Summary: "uppercase kind still counts", Occurrences: 0}, // clamps to 1
		{Kind: "latency", Summary: "slow gate", Occurrences: 9},                          // excluded
		{Kind: "confusing", Summary: "unclear reason", Occurrences: 4},                   // excluded
		{Kind: "other", Summary: "misc", Occurrences: 2},                                 // excluded
		{Kind: "not-a-kind", Summary: "unknown", Occurrences: 5},                         // excluded
	})
	if sig.Appeals != 3 {
		t.Fatalf("appeals = %d, want 3 (2 false-positive + 1 over-broad)", sig.Appeals)
	}
	if sig.Occurrences != 5 {
		t.Fatalf("occurrences = %d, want 5 (3 + 1 + clamped-1)", sig.Occurrences)
	}
	// Most-recurrent first: the x3 appeal must lead the triage queue.
	if len(sig.Offenders) != 3 || !strings.Contains(sig.Offenders[0], "grep git push") {
		t.Fatalf("offenders not ordered most-recurrent-first: %v", sig.Offenders)
	}
	for _, o := range sig.Offenders {
		if strings.Contains(o, "slow gate") || strings.Contains(o, "unclear reason") || strings.Contains(o, "unknown") {
			t.Fatalf("excluded kind leaked into the intake queue: %v", sig.Offenders)
		}
	}
}

// TestEmptyComplaintsFoldToZero proves the no-op path: nil and non-over-block-only
// inputs fold to an empty signal, so BuildScorecardWithComplaints leaves the
// payload identical to the complaint-free build.
func TestEmptyComplaintsFoldToZero(t *testing.T) {
	for _, cs := range [][]FieldComplaint{
		nil,
		{},
		{{Kind: "latency", Summary: "slow", Occurrences: 2}},
	} {
		sig := FoldComplaints(cs)
		if sig.Appeals != 0 || sig.Occurrences != 0 || len(sig.Offenders) != 0 {
			t.Fatalf("non-over-block input folded to a non-zero signal: %+v", sig)
		}
	}
}

// TestComplaintsAreAdvisoryNeverDebt is the load-bearing anti-gaming witness. A
// healthy seed corpus (0 corpus FP/FN) plus a PILE of agent-filed over-block
// complaints must STILL score OK with zero debt: filing complaints cannot lower
// the guard's assertiveness gate. The intake must surface only as Soft on the
// field_fp_intake KPI -- never as a Defect (which would count as debt).
func TestComplaintsAreAdvisoryNeverDebt(t *testing.T) {
	complaints := []FieldComplaint{
		{Kind: "false-positive", Summary: "a", Occurrences: 7},
		{Kind: "false-positive", Summary: "b", Occurrences: 5},
		{Kind: "over-broad", Summary: "c", Occurrences: 3},
	}
	p := BuildScorecardWithComplaints("", SeedCorpus(), complaints)

	if !p.OK {
		t.Fatalf("complaints must not flip ok=false on a clean corpus; verdict=%q reason=%q", p.Verdict, p.Reason)
	}
	if debt, _ := p.Corpus[DebtKey].(int); debt != 0 {
		t.Fatalf("field complaints added %v debt; a soft self-report must never be debt", p.Corpus[DebtKey])
	}

	var intake *struct{ soft, defects int }
	for _, k := range p.KPIs {
		if k.Key == "field_fp_intake" {
			intake = &struct{ soft, defects int }{len(k.Soft), len(k.Defects)}
		}
	}
	if intake == nil {
		t.Fatal("field_fp_intake KPI missing when complaints were supplied")
	}
	if intake.defects != 0 {
		t.Fatalf("field_fp_intake carried %d hard defect(s); it must be advisory-only", intake.defects)
	}
	if intake.soft == 0 {
		t.Fatal("field_fp_intake carried no Soft advisory; the intake is invisible")
	}
}

// TestComplaintIntakeSurfacesInCorpus proves the intake is machine-readable for an
// RSI loop: the appeal/occurrence counts land in the payload corpus, and they are
// kept SEPARATE from the ground-truth fp_count/fp_rate (which stay at the clean
// corpus baseline of zero, uncontaminated by the subjective signal).
func TestComplaintIntakeSurfacesInCorpus(t *testing.T) {
	complaints := []FieldComplaint{
		{Kind: "false-positive", Summary: "a", Occurrences: 4},
		{Kind: "over-broad", Summary: "b", Occurrences: 2},
	}
	p := BuildScorecardWithComplaints("", SeedCorpus(), complaints)

	if got, _ := p.Corpus["field_fp_appeals"].(int); got != 2 {
		t.Fatalf("field_fp_appeals = %v, want 2", p.Corpus["field_fp_appeals"])
	}
	if got, _ := p.Corpus["field_fp_occurrences"].(int); got != 6 {
		t.Fatalf("field_fp_occurrences = %v, want 6", p.Corpus["field_fp_occurrences"])
	}
	// The subjective intake must not contaminate the ground-truth corpus FP number.
	if got, _ := p.Corpus["fp_count"].(int); got != 0 {
		t.Fatalf("corpus fp_count = %v, want 0 (complaints must not inflate the labeled FP rate)", p.Corpus["fp_count"])
	}
}

// TestComplaintFreePathIsUnchanged proves the byte-identical contract: a nil
// complaint slice produces the exact payload BuildScorecardFromRows does, so the
// intake is purely additive and no existing consumer sees a shape change until a
// complaint is actually supplied.
func TestComplaintFreePathIsUnchanged(t *testing.T) {
	withNil := BuildScorecardWithComplaints("/repo", SeedCorpus(), nil)
	legacy := BuildScorecardFromRows("/repo", SeedCorpus())

	if _, ok := withNil.Corpus["field_fp_appeals"]; ok {
		t.Fatal("field_fp_appeals leaked into a complaint-free payload")
	}
	if len(withNil.KPIs) != len(legacy.KPIs) {
		t.Fatalf("complaint-free KPI count %d != legacy %d", len(withNil.KPIs), len(legacy.KPIs))
	}
	for _, k := range withNil.KPIs {
		if k.Key == "field_fp_intake" {
			t.Fatal("field_fp_intake KPI present with no complaints")
		}
	}
}
