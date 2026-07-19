package superloop

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// TestFleetDebtIsTheThreeDimensionProduct pins the meta-walker's worst-first key
// (#4958): debt is the PRODUCT of liveness × progress × follow-on minus one, so the
// dimensions COMPOUND (a stale spinning loop is worse than either fault alone), a
// clean live leaf folds to 0, DARK's urgency stays on the flag (never double-counted
// into the product), and an unread/unmeasured axis multiplies by 1 — surface-only,
// never fabricated debt.
func TestFleetDebtIsTheThreeDimensionProduct(t *testing.T) {
	cases := []struct {
		name     string
		state    string
		dark     bool
		prog     MemberProgress
		followOn MemberFollowon
		want     int
	}{
		{name: "clean_live_leaf", state: "live", want: 0},
		{name: "stale_alone", state: "stale", want: 1},
		{name: "spinning_live", state: "live", prog: ProgressSpinning, want: 1},
		{name: "orphaned_live", state: "live", followOn: FollowonOrphaned, want: 1},
		{name: "stale_and_spinning_compound", state: "stale", prog: ProgressSpinning, want: 3},
		{name: "spinning_and_orphaned_compound", state: "live", prog: ProgressSpinning, followOn: FollowonOrphaned, want: 3},
		{name: "all_three_compound", state: "stale", prog: ProgressSpinning, followOn: FollowonOrphaned, want: 7},
		// DARK rides the Dark flag into tier 0 (the worst band); the product never
		// counts it again — the roster fold's standing convention (#4955).
		{name: "dark_urgency_rides_the_flag", state: "dark", dark: true, want: 0},
		{name: "dark_orphaned_still_compounds_followon", state: "dark", dark: true, followOn: FollowonOrphaned, want: 1},
		// Unread / unmeasured axes are neutral: never fabricated into debt.
		{name: "advancing_is_clean", state: "live", prog: ProgressAdvancing, want: 0},
		{name: "unmeasured_progress_neutral", state: "live", prog: ProgressUnmeasured, want: 0},
		{name: "idle_parked_neutral", state: "live", prog: ProgressIdleParked, want: 0},
		{name: "unknown_followon_neutral", state: "live", followOn: FollowonUnknown, want: 0},
		{name: "advancing_followon_neutral", state: "live", followOn: FollowonAdvancing, want: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FleetDebt(tc.state, tc.dark, tc.prog, tc.followOn); got != tc.want {
				t.Fatalf("FleetDebt(%q, dark=%v, %q, %q) = %d, want %d",
					tc.state, tc.dark, tc.prog, tc.followOn, got, tc.want)
			}
		})
	}
}

// TestFleetWalkSpinningOutranksCleanLiveLeaf is the #4958 DoD regression: through
// the ONE meta-walk (the tend-fleet intent over the fleet enumeration), a
// live-on-cadence zero-progress throughput loop OUT-RANKS a clean live leaf — the
// clean leaf produces no work item at all, the spinning loop leads the worklist with
// its product debt, and the fold refuses to read satisfied.
func TestFleetWalkSpinningOutranksCleanLiveLeaf(t *testing.T) {
	s, ok := Lookup("tend-fleet")
	if !ok {
		t.Fatal("tend-fleet not registered")
	}
	src := Member{Kind: KindLoopFleet, Ref: "all", Why: "the whole fleet"}
	sts := LoopFleetStatuses(src, []RosterLoop{
		{Kind: "cadence", State: "live"},
		{Kind: "loopmgr:issue-resolve-dispatch/claude/throughput", State: "live"},
	}, nil)
	for i := range sts {
		if sts[i].Member.Ref == "loopmgr:issue-resolve-dispatch/claude/throughput" {
			// The shell's product re-fold once the #4956 verdict lands SPINNING.
			sts[i].Progress, sts[i].ProgressReason = ProgressSpinning, relay.ReasonNoProgress
			sts[i].Debt = FleetDebt("live", false, sts[i].Progress, sts[i].FollowOn)
		}
	}
	// Walk the registered intent over just the enumeration slice (the reporting-family
	// child is exercised by its own tests; here the fleet ranking is the subject).
	fleetOnly := Super{Name: s.Name, Title: s.Title, Floor: s.Floor, Members: []Member{src}}
	rep := Walk(fleetOnly, sts)

	if rep.Satisfied {
		t.Fatal("a fleet with a SPINNING loop read satisfied — live-but-producing-nothing hid behind its cadence (#4958 regression)")
	}
	if len(rep.Worklist) != 1 {
		t.Fatalf("worklist = %d item(s), want exactly 1 (the spinning loop; the clean live leaf is nothing to enter): %+v",
			len(rep.Worklist), rep.Worklist)
	}
	head := rep.Worklist[0]
	if head.Member.Ref != "loopmgr:issue-resolve-dispatch/claude/throughput" || head.Debt != 1 {
		t.Fatalf("worst-first head = %q debt %d, want the spinning throughput loop at product debt 1", head.Member.Ref, head.Debt)
	}
	if rep.Finding != "superloop_spinning" {
		t.Fatalf("finding = %q, want superloop_spinning", rep.Finding)
	}
}

// TestWalkOrphanedMemberIsDebtAndBlocks pins the ORPHANED wiring (#4957 verdict into
// the #4958 fold): an orphaned member is work-eligible in the debt band ahead of
// every clean live leaf, blocks Satisfied with its own closed finding, and carries a
// chase/redirect action naming the closed relay token. An UNKNOWN follow-on read
// stays surface-only: no debt, no worklist row, satisfied.
func TestWalkOrphanedMemberIsDebtAndBlocks(t *testing.T) {
	s := Super{Name: "t", Members: []Member{
		{Kind: KindLoop, Ref: "cadence", Why: "clean live leaf"},
		{Kind: KindLoop, Ref: "emitter", Why: "emits follow-on work"},
	}}

	t.Run("orphaned blocks and leads", func(t *testing.T) {
		orphaned := MemberStatus{Member: s.Members[1], Measured: true, FollowOn: FollowonOrphaned}
		if !workEligible(orphaned) {
			t.Fatal("an ORPHANED member is not work-eligible")
		}
		if got := tier(orphaned); got != 1 {
			t.Fatalf("tier(orphaned, debt 0) = %d, want 1 (the debt band, ahead of clean live 3)", got)
		}
		rep := Walk(s, []MemberStatus{
			{Member: s.Members[0], Measured: true, Progress: ProgressAdvancing},
			orphaned,
		})
		if rep.Satisfied {
			t.Fatal("a walk with an ORPHANED member read satisfied — emitted-but-unowned work vanished (#4957)")
		}
		if rep.Orphaned != 1 {
			t.Fatalf("Orphaned = %d, want 1", rep.Orphaned)
		}
		if rep.Finding != "superloop_orphaned" {
			t.Fatalf("finding = %q, want superloop_orphaned", rep.Finding)
		}
		if !strings.Contains(rep.Reason, relay.ReasonOrphanedFollowon) {
			t.Fatalf("reason %q does not name the closed token %s", rep.Reason, relay.ReasonOrphanedFollowon)
		}
		if len(rep.Worklist) != 1 || rep.Worklist[0].Member.Ref != "emitter" {
			t.Fatalf("worklist should hold exactly the orphaned member, got %+v", rep.Worklist)
		}
		it := rep.Worklist[0]
		if it.FollowOn != FollowonOrphaned {
			t.Fatalf("worklist head FollowOn = %q, want %q (machine-readable next to the action)", it.FollowOn, FollowonOrphaned)
		}
		if !strings.Contains(it.Action, "chase/redirect") || !strings.Contains(it.Action, relay.ReasonOrphanedFollowon) {
			t.Fatalf("action %q is not a chase/redirect naming %s", it.Action, relay.ReasonOrphanedFollowon)
		}
		if !strings.HasPrefix(it.Detail, "ORPHANED-FOLLOWON — ") {
			t.Fatalf("detail %q does not lead with the ORPHANED-FOLLOWON marker", it.Detail)
		}
	})

	t.Run("unknown followon is surface-only", func(t *testing.T) {
		rep := Walk(s, []MemberStatus{
			{Member: s.Members[0], Measured: true},
			{Member: s.Members[1], Measured: true, FollowOn: FollowonUnknown},
		})
		if !rep.Satisfied || rep.Orphaned != 0 || len(rep.Worklist) != 0 {
			t.Fatalf("an UNKNOWN follow-on read must stay surface-only (satisfied, no debt): satisfied=%v orphaned=%d worklist=%d",
				rep.Satisfied, rep.Orphaned, len(rep.Worklist))
		}
		if rep.Statuses[1].FollowOn != FollowonUnknown {
			t.Fatalf("the unknown verdict must still be SURFACED on the status, got %q", rep.Statuses[1].FollowOn)
		}
	})
}

// TestTendFleetRegisteredAndReparented is the registry witness for the #4958 graph
// move: tend-fleet exists with the KindLoopFleet enumeration as a member, the
// reporting family (#4862) rides beneath it as exactly ONE KindSuperloop child, tend
// descends tend-fleet (root reachability), and the shipped graph is still sound —
// acyclic, fully reachable, no scorecard counted twice.
func TestTendFleetRegisteredAndReparented(t *testing.T) {
	s, ok := Lookup("tend-fleet")
	if !ok {
		t.Fatal("tend-fleet not registered")
	}
	var fleetAll bool
	var reportingChildren int
	for _, m := range s.Members {
		if m.Kind == KindLoopFleet && m.Ref == "all" {
			fleetAll = true
		}
		if m.Kind == KindSuperloop && m.Ref == "tend-reporting" {
			reportingChildren++
		}
		if m.Kind == KindScorecard {
			t.Errorf("tend-fleet holds scorecard %q — the meta-walker walks LOOPS; a scorecard here risks double-counting at the root", m.Ref)
		}
	}
	if !fleetAll {
		t.Error("tend-fleet must enumerate the whole roster via a KindLoopFleet ref \"all\" member (#4955), else a ledgered loop nobody hand-named escapes the meta-walk")
	}
	if reportingChildren != 1 {
		t.Errorf("tend-fleet descends tend-reporting %d time(s), want exactly ONE KindSuperloop child (#4862 re-parent)", reportingChildren)
	}

	g := Graph()
	if g.Verdict != "OK" {
		t.Fatalf("shipped registry graph is %q (%s) after the re-parent: cycle=%v orphans=%v double=%v dangling=%v",
			g.Verdict, g.Reason, g.Cycle, g.Orphans, g.DoubleCounted, g.Dangling)
	}
	var fleetNode, reportingNode bool
	for _, n := range g.Nodes {
		if n.Name == "tend-fleet" && n.Reachable {
			fleetNode = true
		}
		if n.Name == "tend-reporting" && n.Reachable {
			reportingNode = true
		}
	}
	if !fleetNode || !reportingNode {
		t.Fatalf("tend-fleet reachable=%v tend-reporting reachable=%v — both must stay reachable from the root after the re-parent", fleetNode, reportingNode)
	}
}

// TestOrphanedFollowonTokenIsClassified is the dos_check_reason conformance witness
// for the token the follow-on verdict emits (#4957) and the fleet walk ranks on
// (#4958): RELAY_ORPHANED_FOLLOWON must resolve against a declared dos.toml
// [reasons] table as a refusable OPERATOR_GATE, or a supervisor reading the
// ORPHANED finding would grade it UNCLASSIFIED prose. Reuses the spinning test's
// repo-toml helpers (same package).
func TestOrphanedFollowonTokenIsClassified(t *testing.T) {
	content := readRepoDosTomlForSpinning(t)
	header := "[reasons." + relay.ReasonOrphanedFollowon + "]"
	if !strings.Contains(content, header) {
		t.Fatalf("follow-on verdict emits token %q with no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED)", relay.ReasonOrphanedFollowon, header)
	}
	block := spinningDosBlock(content, header)
	if !strings.Contains(block, "refusal  = true") && !strings.Contains(block, "refusal = true") {
		t.Errorf("reason %q is declared but not marked refusal = true — a supervisor could not refuse on an orphaned loop", relay.ReasonOrphanedFollowon)
	}
	if !strings.Contains(block, `category = "OPERATOR_GATE"`) {
		t.Errorf("reason %q should be declared an OPERATOR_GATE like its RELAY_NO_PROGRESS sibling", relay.ReasonOrphanedFollowon)
	}
}

// TestFleetVerdictHasNoClaimedField extends the no-claimed reflective invariant
// (#4956's TestProgressVerdictHasNoClaimedField) over the #4958 surface: the
// re-parent and the product fold reuse the standing Member/MemberStatus machinery —
// no field anywhere on the walked types lets an agent SELF-REPORT a verdict.
func TestFleetVerdictHasNoClaimedField(t *testing.T) {
	banned := []string{"claimed", "claim", "selfreport", "self_report", "asserted", "trustme"}
	var check func(t *testing.T, tp reflect.Type, path string, seen map[reflect.Type]bool)
	check = func(t *testing.T, tp reflect.Type, path string, seen map[reflect.Type]bool) {
		for tp.Kind() == reflect.Ptr || tp.Kind() == reflect.Slice || tp.Kind() == reflect.Array || tp.Kind() == reflect.Map {
			tp = tp.Elem()
		}
		if tp.Kind() != reflect.Struct || seen[tp] {
			return
		}
		seen[tp] = true
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			lower := strings.ToLower(f.Name)
			for _, b := range banned {
				if strings.Contains(lower, b) {
					t.Errorf("%s.%s: field name suggests a self-reported claim (%q)", path, f.Name, b)
				}
			}
			check(t, f.Type, path+"."+f.Name, seen)
		}
	}
	seen := map[reflect.Type]bool{}
	check(t, reflect.TypeOf(Super{}), "Super", seen)
	check(t, reflect.TypeOf(WalkReport{}), "WalkReport", seen)
	check(t, reflect.TypeOf(RosterLoop{}), "RosterLoop", seen)
	check(t, reflect.TypeOf(FollowonRead{}), "FollowonRead", seen)
}
