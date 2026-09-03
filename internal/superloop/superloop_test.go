package superloop

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// TestRegistryScorecardRefsReal is the no-drift witness: every scorecard member in
// the registry must reference a REAL control-pane card key, so a super loop can never
// send an operator at a scorecard that does not exist. It re-derives the valid keys
// from scorecardpane.Cards (the same source the control pane folds) and fails if a
// member drifts away from it.
func TestRegistryScorecardRefsReal(t *testing.T) {
	valid := map[string]bool{}
	for _, c := range scorecardpane.Cards {
		valid[c.Key] = true
	}
	for _, ref := range ScorecardRefs() {
		if !valid[ref] {
			t.Errorf("scorecard member %q is not a real control-pane card key (drifted from scorecardpane.Cards)", ref)
		}
	}
}

// TestRegistryWellFormed checks the structural invariants the shell relies on: every
// super loop has a name, a title, at least one member, and every member has a kind +
// ref.
func TestRegistryWellFormed(t *testing.T) {
	for _, s := range Registry() {
		if s.Name == "" || s.Title == "" {
			t.Errorf("super loop %+v missing name/title", s)
		}
		if len(s.Members) == 0 {
			t.Errorf("super loop %q has no members", s.Name)
		}
		for _, m := range s.Members {
			if m.Kind == "" || m.Ref == "" {
				t.Errorf("super loop %q has a malformed member %+v", s.Name, m)
			}
		}
		if got, ok := Lookup(s.Name); !ok || got.Name != s.Name {
			t.Errorf("Lookup(%q) did not round-trip", s.Name)
		}
	}
}

func TestManageBenchmarksBridgesToBenchLoop(t *testing.T) {
	s, ok := Lookup("manage-benchmarks")
	if !ok {
		t.Fatal("manage-benchmarks not registered")
	}
	refs := map[MemberKind]map[string]bool{}
	for _, m := range s.Members {
		if refs[m.Kind] == nil {
			refs[m.Kind] = map[string]bool{}
		}
		refs[m.Kind][m.Ref] = true
	}
	if !refs[KindScorecard]["bench_dx"] {
		t.Fatal("manage-benchmarks must include the benchmark-DX scorecard")
	}
	if !refs[KindLoop]["nightrun"] {
		t.Fatal("manage-benchmarks must include the nightrun collection loop")
	}
	if !refs[KindSurface]["fak bench-loop status"] {
		t.Fatal("manage-benchmarks must descend to the concrete bench-loop status surface")
	}

	rep := Walk(s, []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "bench_dx"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoop, Ref: "nightrun"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindSurface, Ref: "fak bench-loop status"}, Container: true, Detail: "domain fold"},
	})
	if len(rep.Worklist) != 1 {
		t.Fatalf("surface descend pointer should remain in the worklist, got %d items", len(rep.Worklist))
	}
	if got := rep.Worklist[0].Action; got != "enter `fak bench-loop status`" {
		t.Fatalf("surface action = %q", got)
	}
}

func TestDrainIssuesTracksGoalScopedDispatchLoops(t *testing.T) {
	s, ok := Lookup("drain-issues")
	if !ok {
		t.Fatal("drain-issues not registered")
	}
	children := map[string]Member{}
	aggregate := false
	for _, m := range s.Members {
		switch {
		case m.Kind == KindLoop && m.Ref == "dispatch":
			aggregate = true
		case m.Kind == KindSuperloop && (m.Ref == "drain-throughput" || m.Ref == "drain-high-priority"):
			children[m.Ref] = m
		default:
			t.Errorf("drain-issues member %q must be aggregate dispatch or a goal superloop, got %q", m.Ref, m.Kind)
		}
	}
	if !aggregate {
		t.Fatal("drain-issues must keep the legacy aggregate dispatch member visible")
	}
	for _, ref := range []string{"drain-throughput", "drain-high-priority"} {
		if _, ok := children[ref]; !ok {
			t.Fatalf("drain-issues missing goal superloop %q", ref)
		}
	}

	goalLoops := map[string]struct {
		ref  string
		goal string
	}{
		"drain-throughput":    {ref: "loopmgr:issue-resolve-dispatch/claude/throughput", goal: "throughput"},
		"drain-high-priority": {ref: "loopmgr:issue-resolve-dispatch/claude/high-priority", goal: "high-priority"},
	}
	for name, want := range goalLoops {
		goalLoop, ok := Lookup(name)
		if !ok {
			t.Fatalf("%s not registered", name)
		}
		if v := Classify(FactsFor(goalLoop)); !v.IsSuper {
			t.Fatalf("%s must classify as a first-class superloop: %s", name, v.Reason)
		}
		if len(goalLoop.Members) != 1 {
			t.Fatalf("%s must walk exactly its goal ledger, got %d members", name, len(goalLoop.Members))
		}
		m := goalLoop.Members[0]
		if m.Kind != KindLoop || m.Ref != want.ref {
			t.Fatalf("%s member = %+v, want loop %q", name, m, want.ref)
		}
		if !strings.Contains(m.Enter, "--goal "+want.goal) {
			t.Fatalf("%s enter hint = %q, want --goal %s", name, m.Enter, want.goal)
		}
		rep := Walk(goalLoop, []MemberStatus{{Member: m, Measured: true, Debt: 1}})
		if len(rep.Worklist) != 1 || !strings.Contains(rep.Worklist[0].Action, m.Enter) {
			t.Fatalf("%s must produce a directly runnable action for its goal loop, got %+v", name, rep.Worklist)
		}
	}

	loops, ok := Lookup("improve-loops")
	if !ok {
		t.Fatal("improve-loops not registered")
	}
	found := false
	for _, m := range loops.Members {
		if m.Kind == KindSuperloop && m.Ref == "drain-issues" {
			found = true
		}
	}
	if !found {
		t.Fatal("improve-loops must descend into drain-issues")
	}
}

// TestClassifySuperVsNormal is the differentiation witness: a registered super loop
// satisfies all five properties; a normal leaf loop satisfies none of the structural
// ones and is classified NOT super, with the reason naming the first failing rung.
func TestClassifySuperVsNormal(t *testing.T) {
	s, ok := Lookup("improve-quality")
	if !ok {
		t.Fatal("improve-quality not registered")
	}
	v := Classify(FactsFor(s))
	if !v.IsSuper {
		t.Fatalf("registered super loop classified as not-super: %s", v.Reason)
	}
	if len(v.Properties) != 5 {
		t.Fatalf("want 5 properties, got %d", len(v.Properties))
	}
	for _, p := range v.Properties {
		if !p.Holds {
			t.Errorf("super loop property %q does not hold: got=%v want=%v", p.Name, p.Got, p.Want)
		}
	}

	leaf := Classify(LeafFacts("dispatch-tick"))
	if leaf.IsSuper {
		t.Error("a normal leaf loop must not classify as a super loop")
	}
	// The first failing rung for a bare leaf is has_members.
	if want := "has_members"; !containsProp(leaf, want, false) {
		t.Errorf("leaf should fail %q", want)
	}
}

// TestClassifyPartialNotSuper proves the AND-gate: a loop that walks members and reads
// them first but does NOT select worst-first is still not a super loop — every rung
// must hold.
func TestClassifyPartialNotSuper(t *testing.T) {
	partial := LoopFacts{
		Name:              "half-super",
		MemberCount:       3,
		WalksFirst:        true,
		SelectsWorstFirst: false, // the missing rung
		ExitsOnAggregate:  true,
		ActsAtOwnAltitude: false,
	}
	v := Classify(partial)
	if v.IsSuper {
		t.Fatal("a loop missing worst-first selection must not classify as super")
	}
	if !containsProp(v, "selects_worst_first", false) {
		t.Errorf("reason should name selects_worst_first; got %q", v.Reason)
	}
}

// TestWalkWorstFirst checks the SELECT step: dark/unmeasured leaves rank first, then
// debt descending, containers in the descend band, and a clean measured leaf is
// dropped from the worklist. Aggregate debt sums only measured leaves (not the
// container).
func TestWalkWorstFirst(t *testing.T) {
	s := Super{
		Name:  "t",
		Title: "test",
		Floor: 0,
		Members: []Member{
			{Kind: KindScorecard, Ref: "a"},
			{Kind: KindScorecard, Ref: "b"},
			{Kind: KindScorecard, Ref: "clean"},
			{Kind: KindLoop, Ref: "darkloop"},
			{Kind: KindGarden, Ref: "garden"},
		},
	}
	statuses := []MemberStatus{
		{Member: s.Members[0], Debt: 10, Measured: true},         // debt 10
		{Member: s.Members[1], Debt: 600, Measured: true},        // debt 600 (heaviest)
		{Member: s.Members[2], Debt: 0, Measured: true},          // clean -> dropped
		{Member: s.Members[3], Dark: true, Measured: true},       // dark -> most urgent
		{Member: s.Members[4], Container: true, Measured: false}, // descend pointer
	}
	rep := Walk(s, statuses)

	if rep.TotalDebt != 610 {
		t.Errorf("total debt: want 610 (10+600, container excluded), got %d", rep.TotalDebt)
	}
	if rep.Dark != 1 {
		t.Errorf("dark count: want 1, got %d", rep.Dark)
	}
	if rep.Unmeasured != 0 {
		t.Errorf("unmeasured: want 0 (container is not counted), got %d", rep.Unmeasured)
	}
	if rep.Satisfied {
		t.Error("must not be satisfied with debt and a dark loop")
	}
	// Worklist excludes the clean member: 5 members - 1 clean = 4.
	if len(rep.Worklist) != 4 {
		t.Fatalf("worklist len: want 4, got %d", len(rep.Worklist))
	}
	// Order: dark leaf, then debt 600, then debt 10, then container.
	wantOrder := []string{"darkloop", "b", "a", "garden"}
	for i, ref := range wantOrder {
		if rep.Worklist[i].Member.Ref != ref {
			t.Errorf("worklist[%d]: want %q, got %q", i, ref, rep.Worklist[i].Member.Ref)
		}
		if rep.Worklist[i].Rank != i+1 {
			t.Errorf("worklist[%d] rank: want %d, got %d", i, i+1, rep.Worklist[i].Rank)
		}
	}
	if rep.Finding != "superloop_dark" {
		t.Errorf("finding: want superloop_dark, got %q", rep.Finding)
	}
}

// TestClassifyWork pins the gardening/throughput/neutral buckets (#3126): scorecard/
// garden/surface tend quality (gardening), issue-drain loops move the backlog
// (throughput), and non-drain loops + capacity signals are neutral.
func TestClassifyWork(t *testing.T) {
	cases := []struct {
		m    Member
		want WorkClass
	}{
		{Member{Kind: KindScorecard, Ref: "slop"}, WorkGardening},
		{Member{Kind: KindGarden, Ref: "garden"}, WorkGardening},
		{Member{Kind: KindSurface, Ref: "fak bench-loop status"}, WorkGardening},
		{Member{Kind: KindLoop, Ref: "dispatch"}, WorkThroughput},
		{Member{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput"}, WorkThroughput},
		{Member{Kind: KindSuperloop, Ref: "drain-issues"}, WorkThroughput},
		{Member{Kind: KindLoop, Ref: "cadence"}, WorkNeutral}, // report loop, not drain
		{Member{Kind: KindLoop, Ref: "loopmgr:recent-feature-dogfood"}, WorkNeutral},
		{Member{Kind: KindSuperloop, Ref: "improve-quality"}, WorkNeutral}, // a container, not itself drain
		{Member{Kind: KindUtilization, Ref: "account-limits"}, WorkNeutral},
	}
	for _, tc := range cases {
		if got := classifyWork(tc.m); got != tc.want {
			t.Errorf("classifyWork(%s:%s) = %q, want %q", tc.m.Kind, tc.m.Ref, got, tc.want)
		}
	}
}

// TestFavoredClass pins the soft-target decision: a tie-break leans toward the
// under-represented class, stays silent when the mix is on target, and never leans at
// all unless BOTH classes are present (with only one class there is no tie to trade).
func TestFavoredClass(t *testing.T) {
	cases := []struct {
		name                          string
		gardening, throughput, target int
		want                          WorkClass
	}{
		{"throughput under target leans throughput", 3, 1, 50, WorkThroughput},
		{"throughput over target leans gardening", 1, 3, 50, WorkGardening},
		{"exactly on target does not lean", 1, 1, 50, ""},
		{"no throughput present cannot lean", 4, 0, 50, ""},
		{"no gardening present cannot lean", 0, 4, 50, ""},
		{"a throughput-hungry target leans throughput on a balanced mix", 1, 1, 100, WorkThroughput},
		{"a gardening-first target leans gardening on a balanced mix", 1, 1, 0, WorkGardening},
	}
	for _, tc := range cases {
		if got := favoredClass(tc.gardening, tc.throughput, tc.target); got != tc.want {
			t.Errorf("%s: favoredClass(%d,%d,%d) = %q, want %q", tc.name, tc.gardening, tc.throughput, tc.target, got, tc.want)
		}
	}
}

// TestWalkMixRollup: a walk with both classes on the worklist reports the split, echoes
// the resolved (default) target, and — because the default 50% target is unmet by a
// 2:1 gardening-heavy mix — leans throughput.
func TestWalkMixRollup(t *testing.T) {
	s := Super{Name: "mix", Title: "mix", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "garden-a"},
		{Kind: KindLoop, Ref: "dispatch"}, // throughput
		{Kind: KindScorecard, Ref: "garden-b"},
		{Kind: KindUtilization, Ref: "account-limits"}, // neutral, still counted
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Debt: 5, Measured: true},
		{Member: s.Members[1], Debt: 5, Measured: true},
		{Member: s.Members[2], Debt: 5, Measured: true},
		{Member: s.Members[3], Debt: 5, Measured: true},
	})
	if rep.Mix.Gardening != 2 || rep.Mix.Throughput != 1 || rep.Mix.Neutral != 1 {
		t.Errorf("mix counts: got g=%d t=%d n=%d, want 2/1/1", rep.Mix.Gardening, rep.Mix.Throughput, rep.Mix.Neutral)
	}
	if rep.Mix.TargetThroughputPct != DefaultThroughputTargetPct {
		t.Errorf("target: got %d, want default %d", rep.Mix.TargetThroughputPct, DefaultThroughputTargetPct)
	}
	if rep.Mix.Favor != WorkThroughput {
		t.Errorf("a gardening-heavy mix under the 50%% target must lean throughput, got %q", rep.Mix.Favor)
	}
}

// TestWalkMixTieBreak is the load-bearing one: among members of EQUAL urgency and EQUAL
// debt, the soft tie-break moves the favored class ahead of declared order — but only
// there. It also proves the two guarantees: a higher-debt member of the non-favored
// class still wins (the nudge never overrides debt), and with only one class present no
// reordering happens at all.
func TestWalkMixTieBreak(t *testing.T) {
	// A hard throughput lean (target 100) with garden declared FIRST: the tied drain
	// member must jump ahead of both gardening members it would otherwise trail.
	s := Super{Name: "mix", Title: "mix", Floor: 0, ThroughputTargetPct: 100, Members: []Member{
		{Kind: KindScorecard, Ref: "garden-a"},
		{Kind: KindLoop, Ref: "dispatch"}, // throughput
		{Kind: KindScorecard, Ref: "garden-b"},
	}}
	tied := []MemberStatus{
		{Member: s.Members[0], Debt: 5, Measured: true},
		{Member: s.Members[1], Debt: 5, Measured: true},
		{Member: s.Members[2], Debt: 5, Measured: true},
	}
	rep := Walk(s, tied)
	if got := worklistRefs(rep); !equalStrings(got, []string{"dispatch", "garden-a", "garden-b"}) {
		t.Errorf("tie-break should float the favored throughput member first: got %v, want [dispatch garden-a garden-b]", got)
	}

	// The nudge must NOT override debt: give garden-a the heaviest debt and it stays
	// first despite the throughput lean; the tie-break only reorders the remaining tie.
	heavier := []MemberStatus{
		{Member: s.Members[0], Debt: 9, Measured: true}, // heaviest -> first regardless of class
		{Member: s.Members[1], Debt: 5, Measured: true},
		{Member: s.Members[2], Debt: 5, Measured: true},
	}
	rep = Walk(s, heavier)
	if got := worklistRefs(rep); !equalStrings(got, []string{"garden-a", "dispatch", "garden-b"}) {
		t.Errorf("tie-break must never move a member past a heavier one: got %v, want [garden-a dispatch garden-b]", got)
	}

	// With only gardening present, favor is "" and declared order is preserved untouched.
	solo := Super{Name: "solo", Title: "solo", Floor: 0, ThroughputTargetPct: 100, Members: []Member{
		{Kind: KindScorecard, Ref: "garden-a"},
		{Kind: KindScorecard, Ref: "garden-b"},
	}}
	rep = Walk(solo, []MemberStatus{
		{Member: solo.Members[0], Debt: 5, Measured: true},
		{Member: solo.Members[1], Debt: 5, Measured: true},
	})
	if rep.Mix.Favor != "" {
		t.Errorf("a single-class worklist must not lean, got %q", rep.Mix.Favor)
	}
	if got := worklistRefs(rep); !equalStrings(got, []string{"garden-a", "garden-b"}) {
		t.Errorf("single-class walk must keep declared order: got %v", got)
	}
}

// worklistRefs pulls the worklist member refs in rank order for ordering assertions.
func worklistRefs(rep WalkReport) []string {
	out := make([]string, len(rep.Worklist))
	for i, w := range rep.Worklist {
		out[i] = w.Member.Ref
	}
	return out
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

// TestWalkSatisfied: all leaves measured-clean and live, no container in the way ->
// satisfied, verdict OK.
func TestWalkSatisfied(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"}, {Kind: KindScorecard, Ref: "b"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Debt: 0, Measured: true},
		{Member: s.Members[1], Debt: 0, Measured: true},
	})
	if !rep.Satisfied {
		t.Errorf("want satisfied; reason=%q", rep.Reason)
	}
	if rep.Verdict != "OK" || rep.Finding != "superloop_satisfied" {
		t.Errorf("want OK/superloop_satisfied, got %s/%s", rep.Verdict, rep.Finding)
	}
	if len(rep.Worklist) != 0 {
		t.Errorf("clean walk should have an empty worklist, got %d", len(rep.Worklist))
	}
}

// TestWalkUnmeasuredBlocks: an unreadable leaf can never read as clean — it blocks
// Satisfied and raises the unmeasured finding even at zero measured debt.
func TestWalkUnmeasuredBlocks(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"},
	}}
	rep := Walk(s, []MemberStatus{{Member: s.Members[0], Measured: false}})
	if rep.Satisfied {
		t.Error("an unmeasured member must block satisfied")
	}
	if rep.Finding != "superloop_unmeasured" {
		t.Errorf("want superloop_unmeasured, got %q", rep.Finding)
	}
	if rep.Unmeasured != 1 {
		t.Errorf("want 1 unmeasured, got %d", rep.Unmeasured)
	}
}

// TestWalkSatisfiedWithContainer pins the load-bearing container rule: a clean,
// all-measured walk that ALSO carries a container (a descend pointer) is still
// SATISFIED — the container is excluded from the unmeasured tally, so it cannot flip
// a clean intent to permanently-unsatisfied, yet it is still surfaced for descent. A
// regression that counted the container as unmeasured would red every container-
// bearing intent (improve-quality, improve-loops, manage-benchmarks all carry one)
// while the rest of the suite stayed green.
func TestWalkSatisfiedWithContainer(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "a"},
		{Kind: KindGarden, Ref: "garden"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Debt: 0, Measured: true},
		{Member: s.Members[1], Container: true, Measured: false},
	})
	if !rep.Satisfied {
		t.Errorf("a clean walk carrying a container must be satisfied; reason=%q unmeasured=%d", rep.Reason, rep.Unmeasured)
	}
	if rep.Unmeasured != 0 {
		t.Errorf("a container must not count as unmeasured, got %d", rep.Unmeasured)
	}
	if rep.Verdict != "OK" {
		t.Errorf("want OK, got %s", rep.Verdict)
	}
	// The container is still surfaced as a descend pointer even on a satisfied walk.
	if len(rep.Worklist) != 1 || rep.Worklist[0].Member.Ref != "garden" {
		t.Errorf("container should remain a descend pointer in the worklist, got %+v", rep.Worklist)
	}
}

// TestWalkIssueTargetGate pins the issue-target fold (WithIssueProgress): a declared
// headline is a GATE only when the shell hands in a live progress count, and only then
// does an unmet target keep an otherwise-clean walk unsatisfied. The whole point is that
// the ~200-issue overnight headline stops being decorative prose and becomes a witnessed
// number the walk reds on until it is met — while an UNMEASURED walk stays surface-only,
// so an unread issue layer can never fabricate "target met".
func TestWalkIssueTargetGate(t *testing.T) {
	// A clean, all-measured, live intent that also declares a 200-issue headline.
	clean := Super{Name: "run-the-night", Title: "t", Floor: 0, IssueTarget: 200, Members: []Member{
		{Kind: KindScorecard, Ref: "a"},
	}}
	cleanStatuses := func() []MemberStatus {
		return []MemberStatus{{Member: clean.Members[0], Debt: 0, Measured: true}}
	}

	t.Run("shortfall gates an otherwise-clean walk", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses(), WithIssueProgress(120))
		if !rep.IssueProgressMeasured {
			t.Fatal("progress was handed in; IssueProgressMeasured must be true")
		}
		if rep.IssueProgressed != 120 {
			t.Errorf("IssueProgressed: want 120, got %d", rep.IssueProgressed)
		}
		if rep.IssueShortfall != 80 {
			t.Errorf("IssueShortfall: want 80 (200-120), got %d", rep.IssueShortfall)
		}
		if rep.Satisfied {
			t.Error("an unmet headline must keep a debt-clean walk unsatisfied")
		}
		if rep.Verdict != "ACTION" || rep.Finding != "superloop_issue_shortfall" {
			t.Errorf("want ACTION/superloop_issue_shortfall, got %s/%s", rep.Verdict, rep.Finding)
		}
	})

	t.Run("target met clears the gate", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses(), WithIssueProgress(200))
		if rep.IssueShortfall != 0 {
			t.Errorf("IssueShortfall: want 0 at target, got %d", rep.IssueShortfall)
		}
		if !rep.Satisfied {
			t.Errorf("a debt-clean walk that met its headline must be satisfied; reason=%q", rep.Reason)
		}
		if rep.Verdict != "OK" || rep.Finding != "superloop_satisfied" {
			t.Errorf("want OK/superloop_satisfied, got %s/%s", rep.Verdict, rep.Finding)
		}
	})

	t.Run("overshoot is not a shortfall", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses(), WithIssueProgress(250))
		if rep.IssueShortfall != 0 || !rep.Satisfied {
			t.Errorf("progress beyond target must clear the gate, got shortfall=%d satisfied=%v", rep.IssueShortfall, rep.Satisfied)
		}
	})

	t.Run("measured zero is a shortfall, not unmeasured", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses(), WithIssueProgress(0))
		if !rep.IssueProgressMeasured {
			t.Error("a measured zero must still count as MEASURED (distinct from surface-only)")
		}
		if rep.IssueShortfall != 200 {
			t.Errorf("IssueShortfall: want 200 (a real zero owes the whole headline), got %d", rep.IssueShortfall)
		}
		if rep.Satisfied || rep.Finding != "superloop_issue_shortfall" {
			t.Errorf("measured-zero progress must gate, got satisfied=%v finding=%q", rep.Satisfied, rep.Finding)
		}
	})

	t.Run("unmeasured walk stays surface-only (no gate)", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses()) // no WithIssueProgress
		if rep.IssueProgressMeasured {
			t.Error("no progress handed in; IssueProgressMeasured must be false")
		}
		if rep.IssueShortfall != 0 {
			t.Errorf("a surface-only walk must not compute a shortfall, got %d", rep.IssueShortfall)
		}
		if !rep.Satisfied {
			t.Errorf("without a measured count the headline is surface-only and must not gate; reason=%q", rep.Reason)
		}
		if rep.IssueTarget != 200 {
			t.Errorf("the declared target is still surfaced, want 200 got %d", rep.IssueTarget)
		}
	})

	t.Run("negative progress clamps to zero", func(t *testing.T) {
		rep := Walk(clean, cleanStatuses(), WithIssueProgress(-5))
		if rep.IssueProgressed != 0 {
			t.Errorf("negative progress must clamp to 0, got %d", rep.IssueProgressed)
		}
		if rep.IssueShortfall != 200 {
			t.Errorf("IssueShortfall after clamp: want 200, got %d", rep.IssueShortfall)
		}
	})

	t.Run("no declared target: progress never gates", func(t *testing.T) {
		noTarget := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{{Kind: KindScorecard, Ref: "a"}}}
		rep := Walk(noTarget, []MemberStatus{{Member: noTarget.Members[0], Debt: 0, Measured: true}}, WithIssueProgress(3))
		if rep.IssueShortfall != 0 || !rep.Satisfied {
			t.Errorf("with no declared target, progress must not gate; shortfall=%d satisfied=%v", rep.IssueShortfall, rep.Satisfied)
		}
	})

	t.Run("member debt still gates before the headline", func(t *testing.T) {
		// A member carrying debt reds via superloop_debt even if the headline is met —
		// the issue-target gate is folded AFTER member-level debt, not instead of it.
		rep := Walk(clean, []MemberStatus{{Member: clean.Members[0], Debt: 5, Measured: true}}, WithIssueProgress(200))
		if rep.Satisfied {
			t.Error("member debt must keep the walk unsatisfied")
		}
		if rep.Finding != "superloop_debt" {
			t.Errorf("member debt should surface before the headline gate, got %q", rep.Finding)
		}
	})
}

// TestWalkUnmeasuredBeatsDark pins walkVerdict's precedence: when a walk has BOTH an
// unmeasured leaf and a dark leaf, the unmeasured finding wins (a status we could not
// even read is more conservative than a known-dark loop). If the dark branch were
// reordered above unmeasured, this would silently flip and the unmeasured-only /
// dark-only tests would both still pass.
func TestWalkUnmeasuredBeatsDark(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "unread"},
		{Kind: KindLoop, Ref: "darkloop"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: false},
		{Member: s.Members[1], Dark: true, Measured: true},
	})
	if rep.Finding != "superloop_unmeasured" {
		t.Errorf("unmeasured must take precedence over dark; got finding %q", rep.Finding)
	}
	if rep.Verdict != "ACTION" {
		t.Errorf("want ACTION, got %s", rep.Verdict)
	}
	if rep.Unmeasured != 1 || rep.Dark != 1 {
		t.Errorf("want unmeasured=1 dark=1, got unmeasured=%d dark=%d", rep.Unmeasured, rep.Dark)
	}
}

// TestSubwalkStatusHonest pins the DESCEND fold's conservative mapping: a satisfied
// sub-walk arrives as a clean measured leaf; an UNSATISFIED sub-walk with zero
// measured debt (unmeasured/dark members inside) still carries one unit of debt at
// the parent's altitude — it can never read clean; measured debt passes through and
// a dark member below propagates the Dark bit.
func TestSubwalkStatusHonest(t *testing.T) {
	m := Member{Kind: KindSuperloop, Ref: "sub"}

	sat := SubwalkStatus(m, WalkReport{Satisfied: true, TotalDebt: 0, Verdict: "OK", Finding: "superloop_satisfied"})
	if !sat.Measured || sat.Container || sat.Debt != 0 || sat.Dark {
		t.Errorf("satisfied sub-walk should fold to a clean measured leaf, got %+v", sat)
	}

	unm := SubwalkStatus(m, WalkReport{Satisfied: false, TotalDebt: 0, Unmeasured: 2, Verdict: "ACTION", Finding: "superloop_unmeasured"})
	if unm.Debt < 1 {
		t.Errorf("an unsatisfied sub-walk must carry at least one unit of debt (got %d) — it can never read clean", unm.Debt)
	}
	if !unm.Measured {
		t.Error("a descended sub-walk was actually read; it must be Measured")
	}

	deep := SubwalkStatus(m, WalkReport{Satisfied: false, TotalDebt: 42, Dark: 1, Verdict: "ACTION", Finding: "superloop_dark"})
	if deep.Debt != 42 {
		t.Errorf("measured sub-debt must pass through, want 42 got %d", deep.Debt)
	}
	if !deep.Dark {
		t.Error("a dark member below must propagate the Dark bit to the parent")
	}
}

// TestSubwalkStatus_IssueShortfallOutranksTrivialDebt pins the shortfall fold (#3151):
// a descended sub-walk unsatisfied ONLY by an unmet ~200-issue headline (zero member
// debt) must out-rank a sibling carrying trivial member debt at the root fold. Before
// the fold it contributed a single floored unit and sank below any debt-2 sibling; now
// it carries the shortfall magnitude, so the night's biggest gap sorts worst-first.
func TestSubwalkStatus_IssueShortfallOutranksTrivialDebt(t *testing.T) {
	root, ok := Lookup("tend")
	if !ok {
		t.Fatal("tend not registered")
	}
	// The descended run-the-night sub-walk: unsatisfied by a 200-issue headline miss with
	// zero member debt. Its folded debt must carry the shortfall, not a floored 1.
	sub := SubwalkStatus(
		Member{Kind: KindSuperloop, Ref: "run-the-night"},
		WalkReport{Satisfied: false, IssueShortfall: 200, TotalDebt: 0, Members: 3,
			Verdict: "ACTION", Finding: "superloop_issue_shortfall"},
	)
	if sub.Debt < 200 {
		t.Fatalf("a descended shortfall of 200 must fold into the member debt, got %d", sub.Debt)
	}
	// A sibling carrying only trivial member debt (kind is immaterial to a same-tier
	// debt ranking; a measured leaf with debt 2 stands in for any trivial-debt member).
	sibling := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "slop"}, Measured: true, Debt: 2}

	// Root-fold both — the sibling is passed FIRST so declared/insertion order cannot be
	// what puts the sub-walk ahead; only its folded debt can.
	rep := Walk(root, []MemberStatus{sibling, sub})
	if len(rep.Worklist) < 2 {
		t.Fatalf("both members carry work; worklist should hold 2, got %d", len(rep.Worklist))
	}
	if rep.Worklist[0].Member.Ref != "run-the-night" {
		t.Errorf("the headline miss (shortfall 200) must sort ahead of trivial debt 2; worklist head = %q (debt %d)",
			rep.Worklist[0].Member.Ref, rep.Worklist[0].Debt)
	}
}

// TestSuperloopMembersResolveAcyclic is the recursion no-drift witness: every
// KindSuperloop member ref must resolve in the registry (the shell reds an unknown
// ref as UNMEASURED, so drift would permanently red its parent), and the
// KindSuperloop edge graph must be acyclic so the shell's descent terminates without
// tripping its cycle guard.
func TestSuperloopMembersResolveAcyclic(t *testing.T) {
	for _, s := range Registry() {
		for _, m := range s.Members {
			if m.Kind != KindSuperloop {
				continue
			}
			if m.Ref == s.Name {
				t.Errorf("super loop %q lists itself as a member (self-cycle)", s.Name)
			}
			if _, ok := Lookup(m.Ref); !ok {
				t.Errorf("super loop %q member %q is not a registered super loop (registry drift)", s.Name, m.Ref)
			}
		}
	}

	// DFS over KindSuperloop edges: a back edge is a cycle.
	const (
		visiting = 1
		done     = 2
	)
	state := map[string]int{}
	var visit func(name string, path []string)
	visit = func(name string, path []string) {
		switch state[name] {
		case visiting:
			t.Fatalf("super-loop registry cycle: %v -> %s", path, name)
		case done:
			return
		}
		state[name] = visiting
		if s, ok := Lookup(name); ok {
			for _, m := range s.Members {
				if m.Kind == KindSuperloop {
					visit(m.Ref, append(path, name))
				}
			}
		}
		state[name] = done
	}
	for _, name := range Names() {
		visit(name, nil)
	}
}

// TestTendWalksEveryOtherSuperloop pins the ROOT intent: every other registered
// intent must be REACHABLE from "tend" over KindSuperloop edges — directly a member,
// or nested below another intent (sweep-surfaces under improve-quality) — so a new
// intent cannot silently escape the root walk, while a nested intent stays counted
// once instead of being forced into a debt-double-counting direct membership.
func TestTendWalksEveryOtherSuperloop(t *testing.T) {
	tend, ok := Lookup("tend")
	if !ok {
		t.Fatal("root intent \"tend\" not registered")
	}
	for _, m := range tend.Members {
		if m.Kind != KindSuperloop {
			t.Errorf("tend member %q must be a sub-super-loop, got kind %q", m.Ref, m.Kind)
		}
	}
	reachable := map[string]bool{}
	var visit func(name string)
	visit = func(name string) {
		if reachable[name] {
			return
		}
		reachable[name] = true
		if s, ok := Lookup(name); ok {
			for _, m := range s.Members {
				if m.Kind == KindSuperloop {
					visit(m.Ref)
				}
			}
		}
	}
	visit("tend")
	for _, s := range Registry() {
		if !reachable[s.Name] {
			t.Errorf("tend must reach registered intent %q over KindSuperloop edges (add it as a member of tend or nest it under one)", s.Name)
		}
	}

	// A descended-status walk folds like leaves: two unsatisfied subs red the root.
	rep := Walk(tend, []MemberStatus{
		SubwalkStatus(tend.Members[0], WalkReport{Satisfied: false, TotalDebt: 5}),
		SubwalkStatus(tend.Members[1], WalkReport{Satisfied: false, TotalDebt: 0, Unmeasured: 1}),
		SubwalkStatus(tend.Members[2], WalkReport{Satisfied: true, TotalDebt: 0}),
	})
	if rep.Satisfied {
		t.Error("root walk with unsatisfied sub-intents must not be satisfied")
	}
	if rep.TotalDebt != 6 {
		t.Errorf("root debt should fold sub-debts 5+1+0=6, got %d", rep.TotalDebt)
	}
	if rep.Unmeasured != 0 {
		t.Errorf("descended subs are measured; want 0 unmeasured, got %d", rep.Unmeasured)
	}
	if len(rep.Worklist) != 2 || rep.Worklist[0].Member.Ref != tend.Members[0].Ref {
		t.Errorf("worst-first: want the debt-5 sub ranked first and the clean sub dropped, got %+v", rep.Worklist)
	}
}

// TestSweepSurfacesSevenSurfaces pins the seven-surface sweep intent: exactly the
// seven named quality surfaces, every one a scorecard member carrying a concrete
// Enter hint (the worklist action must be directly runnable), and the intent is
// NESTED — improve-quality descends it, and improve-quality holds no direct
// scorecard member duplicating a swept surface (that would double-count its debt).
func TestSweepSurfacesSevenSurfaces(t *testing.T) {
	sweep, ok := Lookup("sweep-surfaces")
	if !ok {
		t.Fatal("sweep-surfaces not registered")
	}
	want := []string{"code", "appeal", "agent", "slop", "disambiguation", "learning", "tooling_quality"}
	if len(sweep.Members) != len(want) {
		t.Fatalf("sweep-surfaces must walk exactly %d surfaces, got %d", len(want), len(sweep.Members))
	}
	got := map[string]Member{}
	for _, m := range sweep.Members {
		if m.Kind != KindScorecard {
			t.Errorf("sweep-surfaces member %q must be a scorecard, got %q", m.Ref, m.Kind)
		}
		if strings.TrimSpace(m.Enter) == "" {
			t.Errorf("surface %q has no Enter hint — the sweep worklist must be directly runnable", m.Ref)
		}
		got[m.Ref] = m
	}
	for _, ref := range want {
		if _, ok := got[ref]; !ok {
			t.Errorf("sweep-surfaces is missing surface %q", ref)
		}
	}

	iq, ok := Lookup("improve-quality")
	if !ok {
		t.Fatal("improve-quality not registered")
	}
	descends := false
	for _, m := range iq.Members {
		if m.Kind == KindSuperloop && m.Ref == "sweep-surfaces" {
			descends = true
		}
		if m.Kind == KindScorecard {
			if _, dup := got[m.Ref]; dup {
				t.Errorf("improve-quality holds scorecard %q directly AND via sweep-surfaces — its debt would fold twice", m.Ref)
			}
		}
	}
	if !descends {
		t.Error("improve-quality must descend sweep-surfaces as a KindSuperloop member")
	}
}

// TestRootFoldCountsEachScorecardOnce is the once-only fold witness: across every
// intent reachable from the root "tend", no scorecard key may be walked by two
// different intents — a duplicated key would fold its debt twice into the root
// aggregate and distort the worst-first ranking.
func TestRootFoldCountsEachScorecardOnce(t *testing.T) {
	seenIn := map[string]string{}
	visited := map[string]bool{}
	var visit func(name string)
	visit = func(name string) {
		if visited[name] {
			return
		}
		visited[name] = true
		s, ok := Lookup(name)
		if !ok {
			return
		}
		for _, m := range s.Members {
			switch m.Kind {
			case KindSuperloop:
				visit(m.Ref)
			case KindScorecard:
				if prior, dup := seenIn[m.Ref]; dup {
					t.Errorf("scorecard %q is walked by both %q and %q — the root fold would count its debt twice", m.Ref, prior, name)
					continue
				}
				seenIn[m.Ref] = name
			}
		}
	}
	visit("tend")
}

// TestWalkActionUsesEnterHint pins the useful-action rung: a measured, debt-bearing
// scorecard member with an Enter hint gets that concrete command in its worklist
// action; an unmeasured one still gets the measure-it action (you cannot retire what
// you have not read).
func TestWalkActionUsesEnterHint(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindScorecard, Ref: "slop", Enter: "/slop-score"},
		{Kind: KindScorecard, Ref: "appeal", Enter: "/appeal-score"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Debt: 5},
		{Member: s.Members[1], Measured: false},
	})
	var slopAction, appealAction string
	for _, it := range rep.Worklist {
		switch it.Member.Ref {
		case "slop":
			slopAction = it.Action
		case "appeal":
			appealAction = it.Action
		}
	}
	if !strings.Contains(slopAction, "/slop-score") {
		t.Errorf("measured member's action must carry its Enter hint, got %q", slopAction)
	}
	if strings.Contains(appealAction, "/appeal-score") || !strings.Contains(appealAction, "measure") {
		t.Errorf("unmeasured member must keep the measure action, got %q", appealAction)
	}
}

// TestRunTheNightWalksThreeDimensions pins the overnight meta-loop's shape: exactly the
// three night-productivity dimensions, in worst-first-able form — the issue-drain intent
// descended (a KindSuperloop, reachable and folded once per parent), and two live
// capacity pools as KindUtilization members carrying a concrete Enter hint so their
// worklist action is runnable. It also pins that tend descends it, so the root walk sees
// the night's productivity alongside quality/loop/benchmark debt.
func TestRunTheNightWalksThreeDimensions(t *testing.T) {
	night, ok := Lookup("run-the-night")
	if !ok {
		t.Fatal("run-the-night not registered")
	}
	if len(night.Members) != 3 {
		t.Fatalf("run-the-night must walk exactly 3 dimensions, got %d", len(night.Members))
	}
	// The operator's headline overnight number is a DECLARED field, not buried prose —
	// so a walk can surface it and a test can pin it. (Measuring live progress against
	// it is the named follow-on; here we only pin that the target is declared and
	// echoed, never that the walk fabricates a progress count.)
	if night.IssueTarget != 200 {
		t.Errorf("run-the-night must declare the ~200-issue overnight target, got IssueTarget=%d", night.IssueTarget)
	}
	if rep := Walk(night, nil); rep.IssueTarget != 200 {
		t.Errorf("Walk must echo the declared IssueTarget into the report, got %d", rep.IssueTarget)
	}
	byRef := map[string]Member{}
	for _, m := range night.Members {
		byRef[m.Ref] = m
	}
	if m, ok := byRef["drain-issues"]; !ok || m.Kind != KindSuperloop {
		t.Errorf("run-the-night must descend drain-issues as a KindSuperloop member, got %+v", m)
	}
	for _, ref := range []string{"account-limits", "node-resources"} {
		m, ok := byRef[ref]
		if !ok {
			t.Errorf("run-the-night is missing the %q utilization member", ref)
			continue
		}
		if m.Kind != KindUtilization {
			t.Errorf("night member %q must be KindUtilization, got %q", ref, m.Kind)
		}
		if strings.TrimSpace(m.Enter) == "" {
			t.Errorf("utilization member %q has no Enter hint — its idle-capacity action must be runnable", ref)
		}
	}

	tend, ok := Lookup("tend")
	if !ok {
		t.Fatal("tend not registered")
	}
	descends := false
	for _, m := range tend.Members {
		if m.Kind == KindSuperloop && m.Ref == "run-the-night" {
			descends = true
		}
	}
	if !descends {
		t.Error("tend must descend run-the-night so the root walk sees the night's productivity")
	}
}

// TestTendScoreboardsWalksReportingSurfaces pins the reporting-family intent's shape: the
// outward-facing report scorecards fak posts to Slack (product, release, steerability,
// milestone, osp-residual) as MEASURABLE members, the operator-steerability overlay's
// maintenance loop (#5039) as the one liveness member, plus the Slack-beat feed-delivery
// surface as a descend pointer. It also pins the once-only guarantee (none of the cards is
// walked by another intent, so the root fold counts each once) and that tend descends it.
func TestTendScoreboardsWalksReportingSurfaces(t *testing.T) {
	sb, ok := Lookup("tend-scoreboards")
	if !ok {
		t.Fatal("tend-scoreboards not registered")
	}
	wantCards := map[string]bool{"product": true, "release": true, "steer": true, "milestone": true, "osp_residual": true}
	gotCards := map[string]Member{}
	var surfaces, loops int
	for _, m := range sb.Members {
		switch m.Kind {
		case KindScorecard:
			if strings.TrimSpace(m.Enter) == "" {
				t.Errorf("report scorecard %q has no Enter hint — its worklist action must be runnable", m.Ref)
			}
			gotCards[m.Ref] = m
		case KindLoop:
			loops++
			if m.Ref != steerprOverlayLoopRef {
				t.Errorf("tend-scoreboards loop member = %q, want the steerability-overlay maintenance loop %q", m.Ref, steerprOverlayLoopRef)
			}
		case KindSurface:
			surfaces++
			if m.Ref != "fak slack beat" {
				t.Errorf("tend-scoreboards surface member = %q, want the slack-beat delivery-liveness surface", m.Ref)
			}
		default:
			t.Errorf("tend-scoreboards member %q has unexpected kind %q (want scorecard, loop or surface)", m.Ref, m.Kind)
		}
	}
	for ref := range wantCards {
		if _, ok := gotCards[ref]; !ok {
			t.Errorf("tend-scoreboards is missing the %q report scorecard", ref)
		}
	}
	if surfaces != 1 {
		t.Errorf("tend-scoreboards must carry exactly one feed-delivery surface pointer, got %d", surfaces)
	}
	if loops != 1 {
		t.Errorf("tend-scoreboards must carry exactly one liveness loop member (the overlay), got %d", loops)
	}

	// The four report scorecards must not be walked by any OTHER intent — else the root
	// fold would double-count their debt (the once-only invariant, pinned live here).
	for _, s := range Registry() {
		if s.Name == "tend-scoreboards" {
			continue
		}
		for _, m := range s.Members {
			if m.Kind == KindScorecard && wantCards[m.Ref] {
				t.Errorf("report scorecard %q is also walked by %q — the root fold would count its debt twice", m.Ref, s.Name)
			}
		}
	}

	// A measured, debt-bearing report scorecard produces its concrete Enter action; the
	// surface pointer rides along as a descend item but never blocks a clean fold.
	rep := Walk(sb, []MemberStatus{
		{Member: gotCards["milestone"], Measured: true, Debt: 7},
		{Member: gotCards["product"], Measured: true, Debt: 0},
		{Member: gotCards["release"], Measured: true, Debt: 0},
		{Member: gotCards["steer"], Measured: true, Debt: 0},
		{Member: gotCards["osp_residual"], Measured: true, Debt: 0},
		{Member: steerprOverlayMember(t, sb), Measured: true, Debt: 0},
		{Member: Member{Kind: KindSurface, Ref: "fak slack beat"}, Container: true},
	})
	if rep.TotalDebt != 7 {
		t.Errorf("total debt should fold only the measured scorecards, want 7 got %d", rep.TotalDebt)
	}
	if len(rep.Worklist) == 0 || rep.Worklist[0].Member.Ref != "milestone" {
		t.Fatalf("worst-first should enter the debt-bearing milestone scorecard, got %+v", rep.Worklist)
	}
	if !strings.Contains(rep.Worklist[0].Action, "/milestone-score") {
		t.Errorf("milestone action must carry its Enter hint, got %q", rep.Worklist[0].Action)
	}

	tend, ok := Lookup("tend")
	if !ok {
		t.Fatal("tend not registered")
	}
	descends := false
	for _, m := range tend.Members {
		if m.Kind == KindSuperloop && m.Ref == "tend-scoreboards" {
			descends = true
		}
	}
	if !descends {
		t.Error("tend must descend tend-scoreboards so the root walk sees the reporting surfaces")
	}
}

// TestUtilizationActionUsesEnterHint pins the KindUtilization worklist action: a measured,
// debt-bearing utilization member (idle capacity) gets its Enter hint as the concrete
// spend-the-capacity command; an unmeasured one gets the read-utilization action (you
// cannot spend headroom you have not read).
func TestUtilizationActionUsesEnterHint(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindUtilization, Ref: "node-resources", Enter: "fak lab status --all"},
		{Kind: KindUtilization, Ref: "account-limits", Enter: "fak accounts next"},
	}}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Debt: 3},
		{Member: s.Members[1], Measured: false},
	})
	var nodeAction, acctAction string
	for _, it := range rep.Worklist {
		switch it.Member.Ref {
		case "node-resources":
			nodeAction = it.Action
		case "account-limits":
			acctAction = it.Action
		}
	}
	if !strings.Contains(nodeAction, "fak lab status --all") {
		t.Errorf("measured utilization action must carry its Enter hint, got %q", nodeAction)
	}
	if strings.Contains(acctAction, "fak accounts next") || !strings.Contains(acctAction, "utilization") {
		t.Errorf("unmeasured utilization member must keep the read-utilization action, got %q", acctAction)
	}
}

func TestWalkLoopActionUsesEnterHint(t *testing.T) {
	s := Super{Name: "t", Title: "t", Floor: 0, Members: []Member{
		{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/high-priority", Enter: "fak dispatch auto --goal high-priority"},
	}}
	rep := Walk(s, []MemberStatus{{
		Member:   s.Members[0],
		Measured: true,
		Debt:     1,
	}})
	if len(rep.Worklist) != 1 {
		t.Fatalf("worklist len = %d, want 1", len(rep.Worklist))
	}
	if action := rep.Worklist[0].Action; !strings.Contains(action, "fak dispatch auto --goal high-priority") {
		t.Fatalf("loop action should carry Enter hint, got %q", action)
	}

	rep = Walk(s, []MemberStatus{{
		Member:   s.Members[0],
		Measured: true,
		Dark:     true,
	}})
	if len(rep.Worklist) != 1 {
		t.Fatalf("dark worklist len = %d, want 1", len(rep.Worklist))
	}
	if action := rep.Worklist[0].Action; !strings.Contains(action, "revive via") || !strings.Contains(action, "fak dispatch auto --goal high-priority") {
		t.Fatalf("dark loop action should be a concrete revive hint, got %q", action)
	}
}

func containsProp(v Verdict, name string, holds bool) bool {
	for _, p := range v.Properties {
		if p.Name == name {
			return p.Holds == holds
		}
	}
	return false
}

// TestWalkBudgetRowsAndShares is the divide-down witness: a walk emits exactly one
// budget row per contract dimension (Time/Tokens/Workers/Review, in that order),
// each carrying the declared cap and the per-worklist-member share (declared cap /
// worklist length, floored), and every worklist member is annotated with the same
// share. It also pins the load-bearing distinction between a budgeted-but-floored-to-
// zero share (workers 2 / 3 members = 0, still budgeted, NOT held) and an unbudgeted
// dimension (a hold).
func TestWalkBudgetRowsAndShares(t *testing.T) {
	s := Super{
		Name:   "t",
		Title:  "t",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 30, TokenCeiling: 200000, MaxWorkers: 2, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindScorecard, Ref: "a"},
			{Kind: KindScorecard, Ref: "b"},
			{Kind: KindScorecard, Ref: "c"},
		},
	}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Debt: 3},
		{Member: s.Members[1], Measured: true, Debt: 2},
		{Member: s.Members[2], Measured: true, Debt: 1},
	})
	if len(rep.Worklist) != 3 {
		t.Fatalf("worklist len = %d, want 3", len(rep.Worklist))
	}
	// Exactly four rows, in the contract's dimension order.
	wantDims := []string{BudgetTime, BudgetTokens, BudgetWorkers, BudgetReview}
	if len(rep.Budget) != len(wantDims) {
		t.Fatalf("budget rows = %d, want %d", len(rep.Budget), len(wantDims))
	}
	byDim := map[string]BudgetRow{}
	for i, row := range rep.Budget {
		if row.Dimension != wantDims[i] {
			t.Errorf("budget row[%d] dimension = %q, want %q", i, row.Dimension, wantDims[i])
		}
		if row.Stream != "gen/now" {
			t.Errorf("budget row %q stream = %q, want gen/now", row.Dimension, row.Stream)
		}
		if row.Members != 3 {
			t.Errorf("budget row %q members = %d, want 3", row.Dimension, row.Members)
		}
		byDim[row.Dimension] = row
	}
	// per_member = declared / worklist length, floored.
	cases := []struct {
		dim      string
		total    int
		perMem   int
		budgeted bool
	}{
		{BudgetTime, 30, 10, true},
		{BudgetTokens, 200000, 66666, true},
		{BudgetWorkers, 2, 0, true}, // 2/3 floors to 0 but the dimension is still BUDGETED
		{BudgetReview, 1, 0, true},  // 1/3 floors to 0, still budgeted
	}
	for _, c := range cases {
		row := byDim[c.dim]
		if row.Total != c.total || row.PerMember != c.perMem || row.Budgeted != c.budgeted {
			t.Errorf("row %q = {total %d per_member %d budgeted %v}, want {total %d per_member %d budgeted %v}",
				c.dim, row.Total, row.PerMember, row.Budgeted, c.total, c.perMem, c.budgeted)
		}
		if row.Hold != "" {
			t.Errorf("budgeted row %q must not carry a hold reason, got %q", c.dim, row.Hold)
		}
	}
	// Every worklist member carries the same divided share, and nothing is held.
	for _, it := range rep.Worklist {
		a := it.Allocation
		if a.MaxMinutes != 10 || a.TokenCeiling != 66666 || a.MaxWorkers != 0 || a.ReviewSlots != 0 {
			t.Errorf("member %q allocation = %+v, want minutes 10 tokens 66666 workers 0 review 0", it.Member.Ref, a)
		}
		if len(a.Held) != 0 {
			t.Errorf("member %q must hold no dimensions (all four are budgeted), got %v", it.Member.Ref, a.Held)
		}
	}
}

// TestWalkBudgetHoldForUnbudgetedDimension pins the contract's "no row = hold"
// case: a dimension with a zero cap is UNBUDGETED — its row carries Budgeted=false
// and a hold reason, and every worklist member lists it under Held — never silently
// treated as an unlimited grant.
func TestWalkBudgetHoldForUnbudgetedDimension(t *testing.T) {
	s := Super{
		Name:   "t",
		Title:  "t",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 20, TokenCeiling: 100000, MaxWorkers: 2}, // ReviewSlots 0 = held
		Members: []Member{
			{Kind: KindScorecard, Ref: "a"},
			{Kind: KindScorecard, Ref: "b"},
		},
	}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Debt: 4},
		{Member: s.Members[1], Measured: true, Debt: 1},
	})
	var review BudgetRow
	for _, row := range rep.Budget {
		if row.Dimension == BudgetReview {
			review = row
		}
	}
	if review.Budgeted {
		t.Error("a zero review cap must render as UNBUDGETED")
	}
	if review.Total != 0 || review.PerMember != 0 {
		t.Errorf("held review row must have total 0 / per_member 0, got total %d per_member %d", review.Total, review.PerMember)
	}
	if strings.TrimSpace(review.Hold) == "" {
		t.Error("an unbudgeted dimension must carry a hold reason for the operator")
	}
	for _, it := range rep.Worklist {
		held := map[string]bool{}
		for _, h := range it.Allocation.Held {
			held[h] = true
		}
		if !held[BudgetReview] {
			t.Errorf("member %q must list review under Held, got %v", it.Member.Ref, it.Allocation.Held)
		}
		if held[BudgetTime] || held[BudgetTokens] || held[BudgetWorkers] {
			t.Errorf("member %q wrongly held a budgeted dimension: %v", it.Member.Ref, it.Allocation.Held)
		}
		if it.Allocation.ReviewSlots != 0 {
			t.Errorf("held dimension must allocate 0, got %d", it.Allocation.ReviewSlots)
		}
	}
}

// TestWalkBudgetFloorNeverOverAllocates pins the honesty rung: floored even division
// means the sum of the per-member shares never exceeds a dimension's declared cap —
// the divide-down can under-allocate the remainder but never hand out more than the
// reservation.
func TestWalkBudgetFloorNeverOverAllocates(t *testing.T) {
	s := Super{
		Name:   "t",
		Title:  "t",
		Floor:  0,
		Budget: GenerationBudget{MaxMinutes: 100, TokenCeiling: 100, MaxWorkers: 100, ReviewSlots: 100},
		Members: []Member{
			{Kind: KindScorecard, Ref: "a"},
			{Kind: KindScorecard, Ref: "b"},
			{Kind: KindScorecard, Ref: "c"}, // 100/3 = 33, 33*3 = 99 <= 100
		},
	}
	rep := Walk(s, []MemberStatus{
		{Member: s.Members[0], Measured: true, Debt: 1},
		{Member: s.Members[1], Measured: true, Debt: 1},
		{Member: s.Members[2], Measured: true, Debt: 1},
	})
	n := len(rep.Worklist)
	for _, row := range rep.Budget {
		if !row.Budgeted {
			continue
		}
		if row.PerMember*n > row.Total {
			t.Errorf("dimension %q over-allocates: %d members * %d share = %d > cap %d",
				row.Dimension, n, row.PerMember, row.PerMember*n, row.Total)
		}
	}
}

// TestWalkBudgetEmptyWorklistZeroShares: a satisfied walk has no members to enter, so
// nothing is reserved — the four rows are still reported (the declared caps stay
// visible) but every per-member share is zero.
func TestWalkBudgetEmptyWorklistZeroShares(t *testing.T) {
	s := Super{
		Name:    "t",
		Title:   "t",
		Floor:   0,
		Budget:  GenerationBudget{MaxMinutes: 30, TokenCeiling: 200000, MaxWorkers: 2, ReviewSlots: 1},
		Members: []Member{{Kind: KindScorecard, Ref: "a"}},
	}
	rep := Walk(s, []MemberStatus{{Member: s.Members[0], Measured: true, Debt: 0}})
	if len(rep.Worklist) != 0 {
		t.Fatalf("a clean walk must have an empty worklist, got %d", len(rep.Worklist))
	}
	if len(rep.Budget) != 4 {
		t.Fatalf("budget rows must still be reported on a satisfied walk, got %d", len(rep.Budget))
	}
	for _, row := range rep.Budget {
		if row.PerMember != 0 {
			t.Errorf("dimension %q must reserve 0 with an empty worklist, got %d", row.Dimension, row.PerMember)
		}
		if row.Members != 0 {
			t.Errorf("dimension %q members = %d, want 0", row.Dimension, row.Members)
		}
	}
}

// TestRegistryBudgetsDeclared is the registry budget no-drift witness: every
// registered intent declares at least one budgeted dimension (no intent silently
// escapes the budget contract), any Stream it names is a known generation horizon,
// and a later-horizon (gen/second-next, gen/future) reservation MUST carry an expiry
// — the contract forbids an open-ended research/design budget.
func TestRegistryBudgetsDeclared(t *testing.T) {
	knownStreams := map[string]bool{
		"gen/now": true, "gen/next": true, "gen/second-next": true, "gen/future": true,
	}
	needsExpiry := map[string]bool{"gen/second-next": true, "gen/future": true}
	for _, s := range Registry() {
		b := s.Budget
		if b.MaxMinutes == 0 && b.TokenCeiling == 0 && b.MaxWorkers == 0 && b.ReviewSlots == 0 {
			t.Errorf("super loop %q declares no budgeted dimension — every intent must reserve at least one", s.Name)
		}
		if b.Stream != "" && !knownStreams[b.Stream] {
			t.Errorf("super loop %q names unknown generation stream %q", s.Name, b.Stream)
		}
		if needsExpiry[b.Stream] && strings.TrimSpace(b.Expiry) == "" {
			t.Errorf("super loop %q is stream %q and must carry an expiry (no open-ended later-horizon budget)", s.Name, b.Stream)
		}
	}
}

// TestWalkExpandedMemberDirectDenominator pins that when template members expand
// (e.g. KindLoopFleet:all or KindTrajectory:open into multiple statuses), rep.Members
// reflects the evaluated candidate denominator (Walked + Unmeasured + Containers) rather
// than remaining locked to the unexpanded template count len(s.Members).
func TestWalkExpandedMemberDirectDenominator(t *testing.T) {
	src := Member{Kind: KindLoopFleet, Ref: "all", Why: "fleet"}
	s := Super{
		Name:    "fleet-test",
		Title:   "fleet test",
		Floor:   0,
		Members: []Member{src}, // declared count = 1
	}
	statuses := []MemberStatus{
		{Member: Member{Kind: KindLoopFleet, Ref: "loop1"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoopFleet, Ref: "loop2"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoopFleet, Ref: "loop3"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoopFleet, Ref: "loop4"}, Measured: false, Detail: "no ledger"},
	}

	rep := Walk(s, statuses)
	if rep.DeclaredMembers != 1 {
		t.Errorf("DeclaredMembers: want 1 (unexpanded template count), got %d", rep.DeclaredMembers)
	}
	if rep.Members != 4 {
		t.Errorf("Members: want 4 (evaluated candidate denominator), got %d", rep.Members)
	}
	if rep.Walked != 3 {
		t.Errorf("Walked: want 3, got %d", rep.Walked)
	}
	if rep.Unmeasured != 1 {
		t.Errorf("Unmeasured: want 1, got %d", rep.Unmeasured)
	}
	if rep.Walked+rep.Unmeasured != rep.Members {
		t.Errorf("conservation broken: Walked (%d) + Unmeasured (%d) != Members (%d)",
			rep.Walked, rep.Unmeasured, rep.Members)
	}
	if rep.Rollup.LeafMembers != 4 {
		t.Errorf("Rollup.LeafMembers: want 4, got %d", rep.Rollup.LeafMembers)
	}
	if rep.Rollup.Walked != 3 || rep.Rollup.Unmeasured != 1 {
		t.Errorf("Rollup Walked/Unmeasured: want 3/1, got %d/%d", rep.Rollup.Walked, rep.Rollup.Unmeasured)
	}
}

// TestWalkRollupDenominatorAcrossHierarchy pins the hierarchical roll-up: when a parent
// intent walks sub-super-loops, SubwalkStatus preserves the sub-walk summary and Walk
// computes a true leaf-level denominator (Rollup.LeafMembers), rolled-up walked/unmeasured
// counts, and propagates unmeasured or dark failures from the subtree.
func TestWalkRollupDenominatorAcrossHierarchy(t *testing.T) {
	sub1Def := Super{
		Name:    "sub-one",
		Title:   "sub one",
		Floor:   0,
		Members: []Member{{Kind: KindScorecard, Ref: "c1"}, {Kind: KindScorecard, Ref: "c2"}, {Kind: KindScorecard, Ref: "c3"}},
	}
	sub1Rep := Walk(sub1Def, []MemberStatus{
		{Member: sub1Def.Members[0], Measured: true, Debt: 0},
		{Member: sub1Def.Members[1], Measured: true, Debt: 0},
		{Member: sub1Def.Members[2], Measured: false, Detail: "no baseline"},
	})
	if sub1Rep.Rollup.LeafMembers != 3 || sub1Rep.Rollup.Unmeasured != 1 {
		t.Fatalf("sub1 rollup want 3 leaves, 1 unmeasured, got %+v", sub1Rep.Rollup)
	}

	sub2Def := Super{
		Name:    "sub-two",
		Title:   "sub two",
		Floor:   0,
		Members: []Member{{Kind: KindLoop, Ref: "l1"}, {Kind: KindLoop, Ref: "l2"}},
	}
	sub2Rep := Walk(sub2Def, []MemberStatus{
		{Member: sub2Def.Members[0], Measured: true, Debt: 0},
		{Member: sub2Def.Members[1], Measured: true, Dark: true, Debt: 0},
	})
	if sub2Rep.Rollup.LeafMembers != 2 || sub2Rep.Rollup.Dark != 1 {
		t.Fatalf("sub2 rollup want 2 leaves, 1 dark, got %+v", sub2Rep.Rollup)
	}

	rootDef := Super{
		Name:  "root-intent",
		Title: "root intent",
		Floor: 0,
		Members: []Member{
			{Kind: KindSuperloop, Ref: "sub-one"},
			{Kind: KindSuperloop, Ref: "sub-two"},
			{Kind: KindScorecard, Ref: "direct-leaf"},
		},
	}

	rootStatuses := []MemberStatus{
		SubwalkStatus(rootDef.Members[0], sub1Rep),
		SubwalkStatus(rootDef.Members[1], sub2Rep),
		{Member: rootDef.Members[2], Measured: true, Debt: 0},
	}

	rootRep := Walk(rootDef, rootStatuses)

	// Direct member assertions:
	if rootRep.Members != 3 {
		t.Errorf("direct Members: want 3, got %d", rootRep.Members)
	}
	if rootRep.Walked != 3 || rootRep.Unmeasured != 0 {
		t.Errorf("direct Walked/Unmeasured: want 3/0, got %d/%d", rootRep.Walked, rootRep.Unmeasured)
	}

	// Rollup assertions:
	if rootRep.Rollup.Intents != 3 {
		t.Errorf("Rollup.Intents: want 3 (root + sub-one + sub-two), got %d", rootRep.Rollup.Intents)
	}
	// LeafMembers: 3 from sub1 + 2 from sub2 + 1 direct = 6 leaves
	if rootRep.Rollup.LeafMembers != 6 {
		t.Errorf("Rollup.LeafMembers: want 6, got %d", rootRep.Rollup.LeafMembers)
	}
	if rootRep.Rollup.Walked != 5 {
		t.Errorf("Rollup.Walked: want 5, got %d", rootRep.Rollup.Walked)
	}
	if rootRep.Rollup.Unmeasured != 1 {
		t.Errorf("Rollup.Unmeasured: want 1 (from sub1), got %d", rootRep.Rollup.Unmeasured)
	}
	if rootRep.Rollup.Dark != 1 {
		t.Errorf("Rollup.Dark: want 1 (from sub2), got %d", rootRep.Rollup.Dark)
	}
	if rootRep.Satisfied {
		t.Error("rootRep must NOT be satisfied when rollup has unmeasured/dark leaves")
	}
	if rootRep.Rollup.Satisfied {
		t.Error("rootRep.Rollup must NOT be satisfied")
	}
}

// TestWalkRollupDeduplicatesSharedSubSuperloopLeaves pins the once-only / non-double-counting
// invariant in the roll-up denominator: when two parent intents descend the same shared
// sub-super-loop, the root roll-up counts each distinct leaf surface exactly once.
func TestWalkRollupDeduplicatesSharedSubSuperloopLeaves(t *testing.T) {
	sharedDef := Super{
		Name:    "shared-sub",
		Title:   "shared sub",
		Floor:   0,
		Members: []Member{{Kind: KindLoop, Ref: "shared-loop1"}, {Kind: KindLoop, Ref: "shared-loop2"}},
	}
	sharedRep := Walk(sharedDef, []MemberStatus{
		{Member: sharedDef.Members[0], Measured: true, Debt: 0},
		{Member: sharedDef.Members[1], Measured: true, Debt: 0},
	})

	parent1Def := Super{
		Name:    "parent-one",
		Title:   "parent one",
		Floor:   0,
		Members: []Member{{Kind: KindSuperloop, Ref: "shared-sub"}, {Kind: KindScorecard, Ref: "card-one"}},
	}
	parent1Rep := Walk(parent1Def, []MemberStatus{
		SubwalkStatus(parent1Def.Members[0], sharedRep),
		{Member: parent1Def.Members[1], Measured: true, Debt: 0},
	})

	parent2Def := Super{
		Name:    "parent-two",
		Title:   "parent two",
		Floor:   0,
		Members: []Member{{Kind: KindSuperloop, Ref: "shared-sub"}, {Kind: KindScorecard, Ref: "card-two"}},
	}
	parent2Rep := Walk(parent2Def, []MemberStatus{
		SubwalkStatus(parent2Def.Members[0], sharedRep),
		{Member: parent2Def.Members[1], Measured: true, Debt: 0},
	})

	rootDef := Super{
		Name:    "grandparent",
		Title:   "grandparent",
		Floor:   0,
		Members: []Member{{Kind: KindSuperloop, Ref: "parent-one"}, {Kind: KindSuperloop, Ref: "parent-two"}},
	}
	rootRep := Walk(rootDef, []MemberStatus{
		SubwalkStatus(rootDef.Members[0], parent1Rep),
		SubwalkStatus(rootDef.Members[1], parent2Rep),
	})

	// Total distinct leaves are: shared-loop1, shared-loop2, card-one, card-two = 4.
	// Without deduplication it would be 2 (from p1) + 1 + 2 (from p2) + 1 = 6.
	if rootRep.Rollup.LeafMembers != 4 {
		t.Errorf("Rollup.LeafMembers: want 4 distinct leaves, got %d", rootRep.Rollup.LeafMembers)
	}
	if rootRep.Rollup.Walked != 4 {
		t.Errorf("Rollup.Walked: want 4, got %d", rootRep.Rollup.Walked)
	}
	if rootRep.Rollup.Intents != 4 {
		t.Errorf("Rollup.Intents: want 4 (grandparent, parent-one, parent-two, shared-sub), got %d", rootRep.Rollup.Intents)
	}
	if !rootRep.Satisfied || !rootRep.Rollup.Satisfied {
		t.Error("clean tree across all leaves must be satisfied")
	}
}

// TestWalkRollupSpinningAndOrphanedPropagation pins that spinning and orphaned loop verdicts
// in a sub-super-loop propagate through SubwalkStatus into the parent's Rollup.
func TestWalkRollupSpinningAndOrphanedPropagation(t *testing.T) {
	subDef := Super{
		Name:  "sub-faults",
		Title: "sub faults",
		Floor: 0,
		Members: []Member{
			{Kind: KindLoop, Ref: "loop-spin"},
			{Kind: KindLoop, Ref: "loop-orph"},
		},
	}
	subRep := Walk(subDef, []MemberStatus{
		{Member: subDef.Members[0], Measured: true, Progress: ProgressSpinning, ProgressReason: relay.ReasonNoProgress, Debt: 1},
		{Member: subDef.Members[1], Measured: true, FollowOn: FollowonOrphaned, FollowOnReason: relay.ReasonOrphanedFollowon, Debt: 1},
	})
	if subRep.Spinning != 1 || subRep.Orphaned != 1 {
		t.Fatalf("subRep want 1 spinning, 1 orphaned, got spinning=%d orphaned=%d", subRep.Spinning, subRep.Orphaned)
	}

	parentDef := Super{
		Name:    "parent",
		Title:   "parent",
		Floor:   0,
		Members: []Member{{Kind: KindSuperloop, Ref: "sub-faults"}},
	}
	parentRep := Walk(parentDef, []MemberStatus{
		SubwalkStatus(parentDef.Members[0], subRep),
	})

	if parentRep.Rollup.Spinning != 1 {
		t.Errorf("Rollup.Spinning: want 1, got %d", parentRep.Rollup.Spinning)
	}
	if parentRep.Rollup.Orphaned != 1 {
		t.Errorf("Rollup.Orphaned: want 1, got %d", parentRep.Rollup.Orphaned)
	}
	if parentRep.Satisfied {
		t.Error("parentRep must NOT be satisfied with spinning/orphaned leaves in rollup")
	}
}

// TestGraphLeafDenominatorObservable pins that Graph() computes the structural leaf
// denominator for each intent node and the root.
func TestGraphLeafDenominatorObservable(t *testing.T) {
	rep := Graph()
	if rep.TotalLeafDenominator <= 0 {
		t.Fatalf("TotalLeafDenominator must be > 0, got %d", rep.TotalLeafDenominator)
	}
	byName := map[string]IntentNode{}
	for _, n := range rep.Nodes {
		byName[n.Name] = n
	}
	sweep, ok := byName["sweep-surfaces"]
	if !ok {
		t.Fatal("sweep-surfaces not in graph nodes")
	}
	if sweep.LeafDenominator != 7 {
		t.Errorf("sweep-surfaces LeafDenominator: want 7, got %d", sweep.LeafDenominator)
	}
	iq, ok := byName["improve-quality"]
	if !ok {
		t.Fatal("improve-quality not in graph nodes")
	}
	// sweep-surfaces has 7 scorecards; improve-quality adds 4 direct scorecards + 1 garden = 12
	if iq.LeafDenominator != 12 {
		t.Errorf("improve-quality LeafDenominator: want 12, got %d", iq.LeafDenominator)
	}
}
