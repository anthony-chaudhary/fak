package superloop

import (
	"strings"
	"testing"
)

// rosterFixture is the three-source union the builder must reconcile (#4955):
// folded loops from the cross-ledger fold, the loopmgr job registry, and the
// registered super loops — with one loop deliberately present in TWO sources
// (fold + registry) and one loop present in NONE of the hand-named refs.
func rosterFixture() ([]RosterLoop, []string, []Super, []RosterGap) {
	folded := []RosterLoop{
		{Kind: "cadence", State: "live"},
		{Kind: "loopmgr:garden-tick", State: "stale"},
		{Kind: "orphan-loop", State: "dark", Dark: true},
	}
	registryIDs := []string{"garden-tick", "registered-only"}
	supers := []Super{
		{
			Name: "improve-loops",
			Members: []Member{
				{Kind: KindLoop, Ref: "cadence", Why: "the cadence pulse"},
				{Kind: KindScorecard, Ref: "loopindex", Why: "not a loop ref"},
			},
		},
	}
	gaps := []RosterGap{{Ledger: "dojo", Path: "docs/dojo/history.jsonl", Reason: "absent"}}
	return folded, registryIDs, supers, gaps
}

// TestBuildRosterUnionsAndDedupes pins the DoD's first bullet: one canonical
// roster function returns the deduped union of folded loops + loopmgr registry +
// super loops, and every folded loop appears exactly once.
func TestBuildRosterUnionsAndDedupes(t *testing.T) {
	folded, registryIDs, supers, gaps := rosterFixture()
	r := BuildRoster(folded, registryIDs, supers, gaps)

	if r.Schema != RosterSchema {
		t.Errorf("schema = %q, want %q", r.Schema, RosterSchema)
	}

	seen := map[string]int{}
	for _, e := range r.Entries {
		seen[e.ID]++
	}
	// Every folded loop appears EXACTLY once — the dedupe-on-identity witness.
	for _, l := range folded {
		if seen[l.Kind] != 1 {
			t.Errorf("folded loop %q appears %d time(s), want exactly 1", l.Kind, seen[l.Kind])
		}
	}
	// The loop present in BOTH the fold and the loopmgr registry is ONE entry
	// carrying both sources — counted once, never twice.
	e, ok := rosterEntry(r, "loopmgr:garden-tick")
	if !ok {
		t.Fatalf("no entry for loopmgr:garden-tick; entries: %v", rosterIDs(r))
	}
	if !hasSource(e, RosterSourceFold) || !hasSource(e, RosterSourceLoopRegistry) {
		t.Errorf("loopmgr:garden-tick sources = %v, want fold + loop-registry", e.Sources)
	}
	if !e.Measured {
		t.Errorf("loopmgr:garden-tick has a folded ledger row, want Measured=true")
	}

	// A hand-named loop carries Named=true and both its sources.
	c, ok := rosterEntry(r, "cadence")
	if !ok {
		t.Fatalf("no entry for cadence")
	}
	if !c.Named {
		t.Errorf("cadence is hand-named by improve-loops, want Named=true")
	}
	if !hasSource(c, RosterSourceSuperloop) {
		t.Errorf("cadence sources = %v, want to include superloop", c.Sources)
	}

	// The registered super loop is itself a roster entry (it is supervised too).
	if _, ok := rosterEntry(r, "superloop:improve-loops"); !ok {
		t.Errorf("registered super loop missing from roster; entries: %v", rosterIDs(r))
	}

	// A scorecard member ref must NOT leak in as a loop.
	if _, ok := rosterEntry(r, "loopindex"); ok {
		t.Errorf("scorecard ref %q leaked into the loop roster", "loopindex")
	}

	// Entries are ordered (sorted by ID) so the roster is deterministic.
	for i := 1; i < len(r.Entries); i++ {
		if r.Entries[i-1].ID >= r.Entries[i].ID {
			t.Errorf("entries not strictly ordered: %q >= %q", r.Entries[i-1].ID, r.Entries[i].ID)
		}
	}
}

// TestBuildRosterKeepsUnnamedFoldedLoopVisible is the DoD's regression test: a
// loop present in loopfleet.Report but absent from EVERY hand-named registry ref
// still appears on the roster (previously it was invisible to the walk).
func TestBuildRosterKeepsUnnamedFoldedLoopVisible(t *testing.T) {
	folded, registryIDs, supers, gaps := rosterFixture()
	r := BuildRoster(folded, registryIDs, supers, gaps)

	e, ok := rosterEntry(r, "orphan-loop")
	if !ok {
		t.Fatalf("ledgered-and-folding loop %q is missing from the roster — the exact invisibility #4955 closes", "orphan-loop")
	}
	if e.Named {
		t.Errorf("orphan-loop is named by no registry ref, want Named=false")
	}
	if !e.Measured {
		t.Errorf("orphan-loop folds from a real ledger, want Measured=true")
	}
	if !e.Dark {
		t.Errorf("orphan-loop folded dark, want Dark carried through")
	}
	if r.Unnamed < 1 {
		t.Errorf("rollup Unnamed = %d, want >= 1 (orphan-loop)", r.Unnamed)
	}
}

// TestBuildRosterSurfacesGapsNotHealthyZeros pins the declared-vs-measured
// honesty: a registry loop with no foldable ledger reads UNMEASURED (never
// dropped, never a healthy zero), and a skipped ledger stays a KNOWN gap.
func TestBuildRosterSurfacesGapsNotHealthyZeros(t *testing.T) {
	folded, registryIDs, supers, gaps := rosterFixture()
	r := BuildRoster(folded, registryIDs, supers, gaps)

	e, ok := rosterEntry(r, "loopmgr:registered-only")
	if !ok {
		t.Fatalf("registry-declared loop with no ledger was dropped, want an UNMEASURED entry")
	}
	if e.Measured {
		t.Errorf("registered-only has no foldable ledger, want Measured=false")
	}
	if len(r.Gaps) != 1 || r.Gaps[0].Ledger != "dojo" {
		t.Errorf("gaps = %+v, want the one skipped dojo ledger surfaced", r.Gaps)
	}
	if r.Unmeasured < 1 {
		t.Errorf("rollup Unmeasured = %d, want >= 1", r.Unmeasured)
	}
	if r.Total != len(r.Entries) {
		t.Errorf("rollup Total = %d, want %d (len(entries))", r.Total, len(r.Entries))
	}
}

// TestLoopFleetStatusesEnumeratesEveryLedgeredLoop pins the KindLoopFleet
// expansion: ONE member with Ref="all" expands into one MemberStatus per
// ledgered loop (the KindTrajectory/"open" precedent, one level up), each
// carrying the concrete loop identity as its Ref.
func TestLoopFleetStatusesEnumeratesEveryLedgeredLoop(t *testing.T) {
	src := Member{Kind: KindLoopFleet, Ref: "all", Why: "the whole fleet"}
	folded := []RosterLoop{
		{Kind: "cadence", State: "live"},
		{Kind: "dispatch", State: "stale"},
		{Kind: "orphan-loop", State: "dark", Dark: true},
	}
	sts := LoopFleetStatuses(src, folded, nil)
	if len(sts) != len(folded) {
		t.Fatalf("statuses = %d, want one per ledgered loop (%d)", len(sts), len(folded))
	}
	byRef := map[string]MemberStatus{}
	for _, st := range sts {
		if st.Member.Kind != KindLoopFleet {
			t.Errorf("enumerated member kind = %q, want %q", st.Member.Kind, KindLoopFleet)
		}
		byRef[st.Member.Ref] = st
	}
	if st := byRef["dispatch"]; !st.Measured || st.Debt != 1 {
		t.Errorf("stale loop: measured=%v debt=%d, want measured with 1 unit of debt", st.Measured, st.Debt)
	}
	if st := byRef["orphan-loop"]; !st.Dark {
		t.Errorf("dark loop must carry Dark=true through enumeration")
	}
	if st := byRef["cadence"]; !st.Measured || st.Debt != 0 || st.Dark {
		t.Errorf("live loop: %+v, want measured, clean, live", st)
	}
}

// TestLoopFleetStatusesSurfacesGapsAsUnmeasured pins the nuance clause: an
// absent/unreadable ledger surfaces as a KNOWN gap (UNMEASURED), which blocks
// Satisfied — never dropped, never counted as a healthy zero.
func TestLoopFleetStatusesSurfacesGapsAsUnmeasured(t *testing.T) {
	src := Member{Kind: KindLoopFleet, Ref: "all", Why: "the whole fleet"}
	folded := []RosterLoop{{Kind: "cadence", State: "live"}}
	gaps := []RosterGap{{Ledger: "dojo", Path: "docs/dojo/history.jsonl", Reason: "absent"}}

	sts := LoopFleetStatuses(src, folded, gaps)
	if len(sts) != 2 {
		t.Fatalf("statuses = %d, want 2 (one folded loop + one surfaced gap)", len(sts))
	}
	var gap *MemberStatus
	for i := range sts {
		if !sts[i].Measured {
			gap = &sts[i]
		}
	}
	if gap == nil {
		t.Fatalf("skipped ledger vanished — want an UNMEASURED status, got %+v", sts)
	}
	if !strings.Contains(gap.Detail, "absent") {
		t.Errorf("gap detail %q should carry the skip reason", gap.Detail)
	}

	// An unmeasured gap must block Satisfied at the walk fold.
	rep := Walk(Super{Name: "fleet-test", Members: []Member{src}}, sts)
	if rep.Satisfied {
		t.Errorf("walk with an unfoldable ledger reads Satisfied — a known gap must block it")
	}
	if rep.Unmeasured != 1 {
		t.Errorf("walk Unmeasured = %d, want 1", rep.Unmeasured)
	}
}

// TestLoopFleetStatusesSelectsOneAndNeverReadsMissingAsClean: a non-"all" Ref
// selects one loop by identity; a roster loop with no foldable ledger reads
// UNMEASURED, not clean, and an empty fleet is UNMEASURED too.
func TestLoopFleetStatusesSelectsOneAndNeverReadsMissingAsClean(t *testing.T) {
	src := Member{Kind: KindLoopFleet, Ref: "cadence", Why: "one loop"}
	folded := []RosterLoop{
		{Kind: "cadence", State: "live"},
		{Kind: "dispatch", State: "stale"},
	}
	sts := LoopFleetStatuses(src, folded, nil)
	if len(sts) != 1 || sts[0].Member.Ref != "cadence" {
		t.Fatalf("select-one: got %+v, want exactly the cadence status", sts)
	}

	missing := LoopFleetStatuses(Member{Kind: KindLoopFleet, Ref: "no-such-loop"}, folded, nil)
	if len(missing) != 1 || missing[0].Measured {
		t.Fatalf("a roster loop with no foldable ledger must read UNMEASURED, got %+v", missing)
	}

	empty := LoopFleetStatuses(Member{Kind: KindLoopFleet, Ref: "all"}, nil, nil)
	if len(empty) != 1 || empty[0].Measured {
		t.Fatalf("an empty fleet must surface as UNMEASURED, got %+v", empty)
	}
}

// TestClassifyWorkLoopFleet pins the classifyWork case the issue scopes in: each
// enumerated loop carries the gardening/throughput axis by its own ref, so the
// mix fold (#4956's gate input) sees the fleet's real balance.
func TestClassifyWorkLoopFleet(t *testing.T) {
	drain := Member{Kind: KindLoopFleet, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput"}
	if got := classifyWork(drain); got != WorkThroughput {
		t.Errorf("classifyWork(loopfleet drain ref) = %q, want %q", got, WorkThroughput)
	}
	neutral := Member{Kind: KindLoopFleet, Ref: "cadence"}
	if got := classifyWork(neutral); got != WorkNeutral {
		t.Errorf("classifyWork(loopfleet cadence ref) = %q, want %q", got, WorkNeutral)
	}
}

// --- small test helpers ---

func rosterEntry(r Roster, id string) (RosterEntry, bool) {
	for _, e := range r.Entries {
		if e.ID == id {
			return e, true
		}
	}
	return RosterEntry{}, false
}

func rosterIDs(r Roster) []string {
	out := make([]string, 0, len(r.Entries))
	for _, e := range r.Entries {
		out = append(out, e.ID)
	}
	return out
}

func hasSource(e RosterEntry, src string) bool {
	for _, s := range e.Sources {
		if s == src {
			return true
		}
	}
	return false
}
