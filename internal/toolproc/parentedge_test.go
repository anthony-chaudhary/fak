package toolproc

import (
	"encoding/json"
	"strings"
	"testing"
)

// subtreeIDs collects the CallIDs of a Subtree walk in the returned order —
// the order is load-bearing (it must be the table's own deterministic order).
func subtreeIDs(tab Table, root string) []string {
	var ids []string
	for _, p := range tab.Subtree(root) {
		ids = append(ids, p.CallID)
	}
	return ids
}

// TestFold_ParentCallEdge_SubtreeWalk is the substrate proof (#4332): a spawn
// row's parent_call_id folds onto Proc.ParentCallID, and Table.Subtree walks
// those edges to yield a call's transitive descendants (the run tree #2361 will
// indent and the lineage kill #2357 must account for) — root-excluded, in the
// table's own deterministic order, with unrelated top-level calls left out.
func TestFold_ParentCallEdge_SubtreeWalk(t *testing.T) {
	// Spawn order == table order. Tree:
	//   t-root ─┬─ t-child-a ── t-grand
	//           └─ t-child-b
	//   t-solo (unrelated top-level)
	events := []Event{
		{Kind: EvSpawn, CallID: "t-root", Tool: "agent", AtMS: 1_000},
		{Kind: EvSpawn, CallID: "t-child-a", Tool: "bash", AtMS: 2_000, ParentCallID: "t-root"},
		{Kind: EvSpawn, CallID: "t-child-b", Tool: "bash", AtMS: 3_000, ParentCallID: "t-root"},
		{Kind: EvSpawn, CallID: "t-grand", Tool: "grep", AtMS: 4_000, ParentCallID: "t-child-a"},
		{Kind: EvSpawn, CallID: "t-solo", Tool: "agent", AtMS: 5_000},
	}
	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}

	// The edge folds onto the row; a top-level spawn stays edge-free.
	if got := findProc(t, tab, "t-child-a").ParentCallID; got != "t-root" {
		t.Errorf("t-child-a parent: want t-root, got %q", got)
	}
	if got := findProc(t, tab, "t-grand").ParentCallID; got != "t-child-a" {
		t.Errorf("t-grand parent: want t-child-a, got %q", got)
	}
	if got := findProc(t, tab, "t-root").ParentCallID; got != "" {
		t.Errorf("t-root is top-level, want empty parent, got %q", got)
	}

	// Subtree(root) = transitive descendants, root-excluded, table order.
	if got := subtreeIDs(tab, "t-root"); strings.Join(got, ",") != "t-child-a,t-child-b,t-grand" {
		t.Errorf("Subtree(t-root): want [t-child-a t-child-b t-grand], got %v", got)
	}
	// A mid-tree call yields only its own descendants.
	if got := subtreeIDs(tab, "t-child-a"); strings.Join(got, ",") != "t-grand" {
		t.Errorf("Subtree(t-child-a): want [t-grand], got %v", got)
	}
	// A leaf, an unrelated top-level, and an unknown id all yield nothing.
	if got := tab.Subtree("t-grand"); len(got) != 0 {
		t.Errorf("Subtree(t-grand leaf): want empty, got %v", got)
	}
	if got := tab.Subtree("t-solo"); len(got) != 0 {
		t.Errorf("Subtree(t-solo unrelated): want empty, got %v", got)
	}
	if got := tab.Subtree("t-nonexistent"); len(got) != 0 {
		t.Errorf("Subtree(unknown): want empty, got %v", got)
	}
	if got := tab.Subtree(""); got != nil {
		t.Errorf("Subtree(empty id): want nil, got %v", got)
	}
}

// TestFold_ParentEdge_BackwardCompatEmpty pins the additive contract (#4332):
// a journal that declares no parent edge folds exactly as before — every Proc
// is top-level and the parent_call_id key is omitted from the wire entirely, so
// existing journals stay byte-identical and every Subtree walk is empty.
func TestFold_ParentEdge_BackwardCompatEmpty(t *testing.T) {
	events, now, cfg := Sample()
	tab, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	for _, p := range tab.Procs {
		if p.ParentCallID != "" {
			t.Errorf("%s: pre-edge journal must fold top-level, got parent %q", p.CallID, p.ParentCallID)
		}
		if got := tab.Subtree(p.CallID); len(got) != 0 {
			t.Errorf("%s: flat journal has no subtree, got %v", p.CallID, got)
		}
	}
	// omitempty keeps the wire byte-compatible: the key never appears.
	blob, err := json.Marshal(tab)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(blob), "parent_call_id") {
		t.Error("a parent-free table must not emit parent_call_id (omitempty broken)")
	}
}

// TestParentCallID_ValidatesSpawnOnlyAndNoSelfParent fails closed on the two
// impossible parent edges: a parent_call_id on a non-spawn event (the edge is
// admission-only, like Monitor) and a spawn that names itself as parent (a
// self-cycle). A valid distinct edge round-trips through the JSONL journal.
func TestParentCallID_ValidatesSpawnOnlyAndNoSelfParent(t *testing.T) {
	if err := ValidateEvent(Event{Kind: EvPulse, CallID: "c1", AtMS: 1_000, ParentCallID: "p1"}); err == nil ||
		!strings.Contains(err.Error(), "spawn-only") {
		t.Errorf("parent_call_id on a pulse must refuse spawn-only, got %v", err)
	}
	if err := ValidateEvent(Event{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000, ParentCallID: "c1"}); err == nil ||
		!strings.Contains(err.Error(), "itself") {
		t.Errorf("self-parent spawn must refuse, got %v", err)
	}
	if err := ValidateEvent(Event{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000, ParentCallID: "root"}); err != nil {
		t.Errorf("a distinct parent edge is valid, got %v", err)
	}

	// The edge survives the JSONL boundary: parse then fold.
	journal := strings.Join([]string{
		`{"kind":"spawn","call_id":"root","tool":"agent","at_unix_ms":1000}`,
		`{"kind":"spawn","call_id":"kid","tool":"bash","at_unix_ms":2000,"parent_call_id":"root"}`,
	}, "\n")
	events, err := ParseEvents(strings.NewReader(journal))
	if err != nil {
		t.Fatalf("ParseEvents: %v", err)
	}
	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if got := findProc(t, tab, "kid").ParentCallID; got != "root" {
		t.Errorf("parent_call_id must round-trip through the journal, got %q", got)
	}
}
