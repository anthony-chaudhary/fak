package fleetmon

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// treeByRoot indexes a result's classified trees by root PID for assertions.
func treeByRoot(r DeadOwnerResult) map[int]DeadOwnerTree {
	m := map[int]DeadOwnerTree{}
	for _, t := range r.Trees {
		m[t.RootPID] = t
	}
	return m
}

func candidateSet(r DeadOwnerResult) map[int]bool {
	m := map[int]bool{}
	for _, c := range r.Candidates {
		m[c.RootPID] = true
	}
	return m
}

// TestDeadOwnerFlagsDeadTreeSparesLive is the core acceptance: a fak-tagged tree
// whose run-id maps to a DEAD owner is flagged; a sibling whose run-id is LIVE is
// spared — even though BOTH are busy (a live `go test` descendant), which is what
// the CPU/idle/age heuristics structurally miss.
func TestDeadOwnerFlagsDeadTreeSparesLive(t *testing.T) {
	procs := []procguard.Proc{
		// Dead-owner tree: launcher (ppid 1) already exited, run-id r-dead is dead.
		proc(100, 1, "fak", "fak c --run-id r-dead --lane fleetmon", 600, ""),
		proc(101, 100, "go", "go test ./...", 300, ""), // busy descendant
		// Live-owner tree: identical shape, but run-id r-live is still leased.
		proc(200, 1, "fak", "fak c --run-id r-live --lane other", 600, ""),
		proc(201, 200, "go", "go test ./...", 300, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead, "r-live": OwnerLive},
	})
	cands := candidateSet(res)
	if !cands[100] {
		t.Errorf("dead-owner busy tree (pid 100) must be a reap candidate; trees=%+v", res.Trees)
	}
	if cands[200] {
		t.Error("live-owner tree (pid 200) must be spared, not a candidate")
	}
	byRoot := treeByRoot(res)
	if byRoot[100].Descendants != 1 {
		t.Errorf("dead tree should carry its busy descendant count (1), got %d", byRoot[100].Descendants)
	}
	if byRoot[200].Owner != OwnerLive || byRoot[200].Candidate {
		t.Errorf("live tree row wrong: %+v", byRoot[200])
	}
}

// TestDeadOwnerAbsentLeaseIsCandidate: an owner looked up and found ABSENT (the
// crashed-owner case — no lease/registry row at all) is a reap candidate, exactly
// like an expired one.
func TestDeadOwnerAbsentLeaseIsCandidate(t *testing.T) {
	procs := []procguard.Proc{
		proc(100, 1, "fak", "fak c --run-id gone", 600, ""),
		proc(101, 100, "go", "go test ./...", 300, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"gone": OwnerAbsent},
	})
	if !candidateSet(res)[100] {
		t.Fatalf("an absent-lease owner must flag its tree; got %+v", res.Trees)
	}
}

// TestDeadOwnerSparesProtectedName: a protected OS name is reported but NEVER a
// candidate, even with a dead owner (the no-false-reap contract).
func TestDeadOwnerSparesProtectedName(t *testing.T) {
	procs := []procguard.Proc{
		// A protected OS name carrying a fak marker + dead owner — must be spared.
		proc(4, 1, "services", "services fak c --run-id r-dead", 600, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead},
	})
	if candidateSet(res)[4] {
		t.Fatal("a protected OS name must never be a reap candidate")
	}
	if row := treeByRoot(res)[4]; !row.Protected {
		t.Fatalf("protected OS name should be reported protected, got %+v", row)
	}
}

// TestDeadOwnerSparesAttendedTerminal: a tree whose parent is a LIVE interactive
// terminal is reported but never reaped — a human may be attending it.
func TestDeadOwnerSparesAttendedTerminal(t *testing.T) {
	procs := []procguard.Proc{
		proc(50, 1, "WindowsTerminal", "WindowsTerminal.exe", 9000, ""), // live attended parent
		proc(100, 50, "fak", "fak c --run-id r-dead", 600, ""),
		proc(101, 100, "go", "go test ./...", 300, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead},
	})
	if candidateSet(res)[100] {
		t.Fatal("a tree under a LIVE attended terminal must never be a reap candidate")
	}
	if row := treeByRoot(res)[100]; !row.Attended {
		t.Fatalf("attended-terminal tree should be reported attended, got %+v", row)
	}
}

// TestDeadOwnerDetachedLauncherStillReaped: the same interactive-shell name as a
// parent does NOT spare the tree when that parent has EXITED (absent from the
// snapshot) — the detached-orphan case TaskStop leaves behind. Only a LIVE
// attended parent spares.
func TestDeadOwnerDetachedLauncherStillReaped(t *testing.T) {
	procs := []procguard.Proc{
		// ppid 999 (a pwsh launcher) already exited — not in the snapshot.
		proc(100, 999, "fak", "fak c --run-id r-dead", 600, ""),
		proc(101, 100, "go", "go test ./...", 300, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead},
	})
	if !candidateSet(res)[100] {
		t.Fatalf("a detached dead-owner tree whose launcher exited must still be reapable; got %+v", res.Trees)
	}
	if treeByRoot(res)[100].Attended {
		t.Error("a dead (absent) parent must not count as an attended terminal")
	}
}

// TestDeadOwnerFailsClosedOnUnknownOwner: a root with no run-id tag, or a key the
// caller did not resolve, is Unknown and SPARED — never guessed into a kill.
func TestDeadOwnerFailsClosedOnUnknownOwner(t *testing.T) {
	procs := []procguard.Proc{
		proc(100, 1, "fak", "fak c --lane fleetmon", 600, ""), // key "fleetmon" absent from map
		proc(200, 1, "fak", "fak c", 600, ""),                 // no key flag at all
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{}, // nothing resolved
	})
	if res.CandidateCount != 0 {
		t.Fatalf("unknown-owner trees must fail closed to spared; got candidates %+v", res.Candidates)
	}
	for _, tr := range res.Trees {
		if tr.Owner != OwnerUnknown {
			t.Errorf("pid %d should classify Unknown, got %s", tr.RootPID, tr.Owner)
		}
	}
}

// TestDeadOwnerSubtreeProtectedDemotes: a dead-owner tree that still holds a
// persistent MCP server is demoted to spared — a tree-kill would take the MCP
// server down with it.
func TestDeadOwnerSubtreeProtectedDemotes(t *testing.T) {
	procs := []procguard.Proc{
		proc(300, 1, "fak", "fak c --run-id r-dead", 600, ""),
		proc(301, 300, "python", "python -m dos_mcp.server", 3000, ""), // protected descendant
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead},
	})
	if candidateSet(res)[300] {
		t.Fatal("a dead-owner tree holding an MCP server must be demoted, not reaped")
	}
	if row := treeByRoot(res)[300]; !row.Protected {
		t.Fatalf("subtree-protected tree should be reported protected, got %+v", row)
	}
}

// TestDeadOwnerPayloadCarriesClassAndCount: the JSON payload carries the new
// candidate class + count and the report-first OK bit, so the control pane folds
// it.
func TestDeadOwnerPayloadCarriesClassAndCount(t *testing.T) {
	procs := []procguard.Proc{
		proc(100, 1, "fak", "fak c --run-id r-dead", 600, ""),
		proc(101, 100, "go", "go test ./...", 300, ""),
	}
	res := EvaluateDeadOwnerReaper(DeadOwnerInput{
		Procs:       procs,
		OwnerStates: map[string]OwnerState{"r-dead": OwnerDead},
	})
	if res.Schema != DeadOwnerSchema {
		t.Fatalf("schema not stamped: %q", res.Schema)
	}
	if res.CandidateCount != 1 {
		t.Fatalf("candidate_count should be 1, got %d", res.CandidateCount)
	}
	if res.OK {
		t.Fatal("OK must be false when a dead-owner tree is live (report-first ACTION)")
	}
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	js := string(b)
	for _, want := range []string{
		`"schema":"fak-fleet-deadowner/1"`,
		`"candidate_count":1`,
		`"ok":false`,
		`"candidate":true`,
		`"owner_state":"dead"`,
	} {
		if !strings.Contains(js, want) {
			t.Errorf("payload missing %s\n%s", want, js)
		}
	}
	if res.NextAction == "" {
		t.Error("next action should never be empty when a candidate exists")
	}
}
