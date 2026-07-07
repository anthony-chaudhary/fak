package superloop

import (
	"strings"
	"testing"

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
