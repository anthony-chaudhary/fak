package skillvalue

import (
	"strings"
	"testing"
)

// row is a terse SessionRow builder for the table tests.
func row(sess, class string, pass bool, cost, lat float64, skills ...string) SessionRow {
	return SessionRow{
		Schema:    LedgerSchema,
		SessionID: sess,
		TaskClass: class,
		Skills:    skills,
		Pass:      pass,
		CostUSD:   cost,
		LatencyMS: lat,
	}
}

func find(r Rollup, id string) (SkillValue, bool) {
	for _, s := range r.Skills {
		if s.SkillID == id {
			return s, true
		}
	}
	return SkillValue{}, false
}

func TestComputePositiveLiftGrounded(t *testing.T) {
	// "helper" loaded -> passes; matched same-class sessions without it fail.
	sessions := []SessionRow{
		row("s1", "refactor", true, 1, 100, "helper"),
		row("s2", "refactor", true, 1, 100, "helper"),
		row("s3", "refactor", false, 2, 300), // matched baseline, no helper
		row("s4", "refactor", false, 2, 300),
	}
	r := Compute(sessions, map[string]string{"helper": "ablation:matched-pass-delta"})
	sv, ok := find(r, "helper")
	if !ok {
		t.Fatal("helper missing from rollup")
	}
	if sv.TaskClasses != 1 || sv.ComparableN != 2 || sv.BaselineN != 2 {
		t.Fatalf("arm counts wrong: %+v", sv)
	}
	if sv.LoadedPass != 1 || sv.BaselinePass != 0 {
		t.Fatalf("pass rates wrong: loaded=%v base=%v", sv.LoadedPass, sv.BaselinePass)
	}
	if sv.PassLift != 1 {
		t.Fatalf("want pass lift 1, got %v", sv.PassLift)
	}
	if sv.CostDelta != -1 { // 1 loaded - 2 baseline
		t.Fatalf("want cost delta -1, got %v", sv.CostDelta)
	}
	if sv.LatencyDelta != -200 {
		t.Fatalf("want latency delta -200, got %v", sv.LatencyDelta)
	}
	if sv.HasFlag(FlagNetNegative) {
		t.Fatal("positive-lift skill must not be flagged net-negative")
	}
	if sv.HasFlag(FlagNoValuationBasis) {
		t.Fatal("grounded skill must not be flagged no-valuation-basis")
	}
	if got := r.AutoRevert(); len(got) != 0 {
		t.Fatalf("no skill should auto-revert, got %v", got)
	}
	if g := r.Gate(); !g.OK || len(g.Ungrounded) != 0 {
		t.Fatalf("grounded skill must pass the gate, got %+v", g)
	}
}

func TestComputeNetNegativeAutoReverts(t *testing.T) {
	// "harmful" loaded -> fails; matched sessions without it pass. lift < 0.
	sessions := []SessionRow{
		row("s1", "debug", false, 3, 500, "harmful"),
		row("s2", "debug", false, 3, 500, "harmful"),
		row("s3", "debug", true, 1, 100),
		row("s4", "debug", true, 1, 100),
	}
	r := Compute(sessions, map[string]string{"harmful": "ablation:matched-pass-delta"})
	sv, _ := find(r, "harmful")
	if sv.PassLift != -1 {
		t.Fatalf("want pass lift -1, got %v", sv.PassLift)
	}
	if !sv.HasFlag(FlagNetNegative) {
		t.Fatalf("net-negative skill must be flagged, flags=%v", sv.Flags)
	}
	got := r.AutoRevert()
	if len(got) != 1 || got[0] != "harmful" {
		t.Fatalf("harmful must auto-revert, got %v", got)
	}
}

func TestZeroLiftIsNetNegative(t *testing.T) {
	// lift exactly 0 (<=0) must revert — a skill that changes nothing earns no keep.
	sessions := []SessionRow{
		row("s1", "test", true, 1, 100, "noop"),
		row("s2", "test", false, 1, 100, "noop"),
		row("s3", "test", true, 1, 100),
		row("s4", "test", false, 1, 100),
	}
	r := Compute(sessions, map[string]string{"noop": "ablation:matched-pass-delta"})
	sv, _ := find(r, "noop")
	if sv.PassLift != 0 {
		t.Fatalf("want pass lift 0, got %v", sv.PassLift)
	}
	if !sv.HasFlag(FlagNetNegative) {
		t.Fatalf("zero-lift skill must revert, flags=%v", sv.Flags)
	}
}

func TestInsufficientEvidenceNotReverted(t *testing.T) {
	// "lonely" is the only skill and every session in its class loaded it — no
	// matched baseline exists, so its lift is unmeasurable. It must be reported
	// not-yet, never auto-reverted on absence of evidence.
	sessions := []SessionRow{
		row("s1", "solo", false, 1, 100, "lonely"),
		row("s2", "solo", false, 1, 100, "lonely"),
	}
	r := Compute(sessions, map[string]string{"lonely": "ablation:matched-pass-delta"})
	sv, _ := find(r, "lonely")
	if !sv.HasFlag(FlagInsufficientEvidence) {
		t.Fatalf("no-baseline skill must be insufficient-evidence, flags=%v", sv.Flags)
	}
	if sv.HasFlag(FlagNetNegative) {
		t.Fatal("insufficient-evidence skill must not be flagged net-negative")
	}
	if got := r.AutoRevert(); len(got) != 0 {
		t.Fatalf("insufficient-evidence must NOT auto-revert, got %v", got)
	}
}

func TestValuationBasisGateFlagsUngrounded(t *testing.T) {
	// "grounded" carries a basis; "ungrounded" does not. Only the latter is a gate
	// finding — the #2796 mirror keys off the basis, not the load count.
	sessions := []SessionRow{
		row("s1", "a", true, 1, 100, "grounded", "ungrounded"),
		row("s2", "a", true, 1, 100, "grounded"),
		row("s3", "a", false, 1, 100, "ungrounded"),
		row("s4", "a", false, 1, 100),
	}
	r := Compute(sessions, map[string]string{"grounded": "ablation:matched-pass-delta"})
	g := r.Gate()
	if g.OK {
		t.Fatal("gate must fail when a skill is ungrounded")
	}
	if len(g.Ungrounded) != 1 || g.Ungrounded[0] != "ungrounded" {
		t.Fatalf("gate must flag exactly the ungrounded skill, got %v", g.Ungrounded)
	}
	gr, _ := find(r, "grounded")
	if gr.HasFlag(FlagNoValuationBasis) {
		t.Fatal("grounded skill must not carry the no-basis flag")
	}
}

func TestTaskClassMatchingIsApplesToApples(t *testing.T) {
	// "x" helps in class A (all loaded pass, baseline fails) but class B has only
	// loaded sessions (no baseline). The lift must be computed from class A alone —
	// class B contributes to TotalLoaded but not to the matched comparison.
	sessions := []SessionRow{
		row("a1", "A", true, 1, 10, "x"),
		row("a2", "A", false, 1, 10), // baseline in A
		row("b1", "B", false, 1, 10, "x"),
		row("b2", "B", false, 1, 10, "x"),
	}
	r := Compute(sessions, map[string]string{"x": "basis"})
	sv, _ := find(r, "x")
	if sv.TotalLoaded != 3 {
		t.Fatalf("want total loaded 3, got %d", sv.TotalLoaded)
	}
	if sv.TaskClasses != 1 || sv.ComparableN != 1 || sv.BaselineN != 1 {
		t.Fatalf("only class A is comparable: %+v", sv)
	}
	if sv.LoadedPass != 1 || sv.BaselinePass != 0 || sv.PassLift != 1 {
		t.Fatalf("lift must come from class A alone: %+v", sv)
	}
}

func TestParseLedgerFiltersSchema(t *testing.T) {
	content := strings.Join([]string{
		`{"schema":"fak-skill-value-ledger/1","session_id":"s1","task_class":"a","skills":["k"],"pass":true}`,
		``,
		`{"schema":"some-other-ledger/1","session_id":"x","skills":["k"]}`,
		`not json`,
		`{"schema":"fak-skill-value-ledger/1","session_id":"s2","task_class":"a","skills":[],"pass":false}`,
	}, "\n")
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("want 2 kept rows, got %d: %+v", len(rows), rows)
	}
	if rows[0].SessionID != "s1" || rows[1].SessionID != "s2" {
		t.Fatalf("wrong rows kept: %+v", rows)
	}
}

func TestComputeIsDeterministicAndWorstFirst(t *testing.T) {
	sessions := []SessionRow{
		row("s1", "c", true, 1, 10, "good"),
		row("s2", "c", false, 1, 10, "bad"),
		row("s3", "c", false, 1, 10),
		row("s4", "c", true, 1, 10),
	}
	basis := map[string]string{"good": "b", "bad": "b"}
	r1 := Compute(sessions, basis)
	r2 := Compute(sessions, basis)
	if len(r1.Skills) != 2 {
		t.Fatalf("want 2 skills, got %d", len(r1.Skills))
	}
	// good: loaded pass rate 1 vs baseline 0.5 -> +0.5; bad: 0 vs 0.5 -> -0.5.
	if r1.Skills[0].SkillID != "bad" {
		t.Fatalf("worst-lift skill must lead, got %s", r1.Skills[0].SkillID)
	}
	for i := range r1.Skills {
		if r1.Skills[i].SkillID != r2.Skills[i].SkillID || r1.Skills[i].PassLift != r2.Skills[i].PassLift {
			t.Fatalf("non-deterministic at %d: %+v vs %+v", i, r1.Skills[i], r2.Skills[i])
		}
	}
}
