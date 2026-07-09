package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestSessionInventory drives the pure fold (buildSessionInventory) over a durable C1 set
// plus a folded-in C2 fleet set: it proves the deterministic rollup (count, per-state
// tally, warm/cold cache posture from the resume oracle), the dedupe (a local C1 row
// supersedes the node's own C2 ref), and the stable (host,id) ordering.
func TestSessionInventory(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	local := []session.Descriptor{
		{ID: "sess-a", Host: "node-1", PCBState: "RUNNING", CreatedAt: now.Add(-2 * time.Minute), UpdatedAt: now.Add(-10 * time.Second), LastSeen: now.Add(-10 * time.Second)},
		{ID: "sess-b", Host: "node-1", PCBState: "PAUSED", CreatedAt: now.Add(-1 * time.Hour), UpdatedAt: now.Add(-30 * time.Minute), LastSeen: now.Add(-30 * time.Minute)},
	}
	fleet := []leaseref.SessionDescriptor{
		// A peer node's session — appears only via C2.
		{ID: "sess-c", Host: "node-2", PCBState: "RUNNING", UpdatedAt: now.Add(-5 * time.Second).Unix()},
		// The node's own session, also pushed to C2 — must dedupe against the C1 row.
		{ID: "sess-a", Host: "node-1", PCBState: "STOPPED", UpdatedAt: now.Unix()},
	}

	inv := buildSessionInventory(local, fleet, now, 5*time.Minute, true)

	if inv.Count != 3 {
		t.Fatalf("count = %d, want 3 (2 local + 1 fresh peer, sess-a deduped)", inv.Count)
	}
	if inv.ByState["RUNNING"] != 2 || inv.ByState["PAUSED"] != 1 {
		t.Fatalf("by_state = %v, want RUNNING:2 PAUSED:1", inv.ByState)
	}
	if inv.ByState["STOPPED"] != 0 {
		t.Fatalf("sess-a should keep its local RUNNING row, not the C2 STOPPED one: %v", inv.ByState)
	}
	// liveness_class reads #750's vocabulary off the heartbeat: sess-a heartbeat 10s ago (a
	// running row well within the 5m stale window) is live; sess-b is PAUSED, so idle.
	byID := map[string]sessionInventoryRow{}
	for _, r := range inv.Sessions {
		byID[r.ID] = r
	}
	if byID["sess-a"].LivenessClass != "live" {
		t.Fatalf("sess-a liveness = %q, want live", byID["sess-a"].LivenessClass)
	}
	if byID["sess-b"].LivenessClass != "idle" {
		t.Fatalf("sess-b liveness = %q, want idle (paused)", byID["sess-b"].LivenessClass)
	}
	if byID["sess-c"].LivenessClass != "live" {
		t.Fatalf("sess-c (fresh peer) liveness = %q, want live", byID["sess-c"].LivenessClass)
	}
	// sess-a idle 10s and sess-c idle 5s are within the 5m TTL → warm; sess-b idle 30m → cold.
	if inv.Warm != 2 || inv.Cold != 1 {
		t.Fatalf("warm/cold = %d/%d, want 2/1", inv.Warm, inv.Cold)
	}
	// Stable ordering: node-1 rows before node-2, id-sorted within a host.
	gotIDs := []string{inv.Sessions[0].ID, inv.Sessions[1].ID, inv.Sessions[2].ID}
	wantIDs := []string{"sess-a", "sess-b", "sess-c"}
	for i := range wantIDs {
		if gotIDs[i] != wantIDs[i] {
			t.Fatalf("order = %v, want %v", gotIDs, wantIDs)
		}
	}
	if inv.Sessions[0].Source != "local" || inv.Sessions[2].Source != "fleet" {
		t.Fatalf("source tagging wrong: %q / %q", inv.Sessions[0].Source, inv.Sessions[2].Source)
	}
	if got := sessionInventorySummary(inv); !strings.Contains(got, "3 session(s) [fleet]") || !strings.Contains(got, "2 RUNNING, 1 PAUSED") || !strings.Contains(got, "2 warm, 1 cold") {
		t.Fatalf("summary = %q", got)
	}
}

// TestSessionLsFleet proves the fleet scope tag and that a non-fleet fold ignores the C2
// descriptors entirely (so `--durable` alone never leaks a peer's session into the view).
func TestSessionLsFleet(t *testing.T) {
	now := time.Unix(2_000_000, 0)
	local := []session.Descriptor{
		{ID: "local-1", Host: "me", PCBState: "RUNNING", CreatedAt: now.Add(-time.Minute), UpdatedAt: now},
	}
	fleet := []leaseref.SessionDescriptor{
		{ID: "peer-1", Host: "peer", PCBState: "RUNNING", UpdatedAt: now.Unix()},
	}

	// Fleet on: the peer row folds in and the summary is scoped [fleet].
	inv := buildSessionInventory(local, fleet, now, 5*time.Minute, true)
	if inv.Count != 2 {
		t.Fatalf("fleet count = %d, want 2", inv.Count)
	}
	if !strings.Contains(sessionInventorySummary(inv), "[fleet]") {
		t.Fatalf("fleet summary missing scope tag: %q", sessionInventorySummary(inv))
	}

	// Fleet off (durable only): the C2 set is not folded in even when supplied.
	durableOnly := buildSessionInventory(local, fleet, now, 5*time.Minute, false)
	if durableOnly.Count != 1 {
		t.Fatalf("durable-only count = %d, want 1 (no peer)", durableOnly.Count)
	}
	if strings.Contains(sessionInventorySummary(durableOnly), "[fleet]") {
		t.Fatalf("durable-only summary must not be scoped [fleet]: %q", sessionInventorySummary(durableOnly))
	}
}

// TestSessionInventoryStalled proves the liveness oracle's STALLED verdict (criterion 3):
// a RUNNING descriptor whose heartbeat lapsed past the stale window is classified stalled
// and rolls up as a distinct STALLED bucket in the count summary — not counted as RUNNING.
func TestSessionInventoryStalled(t *testing.T) {
	now := time.Unix(3_000_000, 0)
	local := []session.Descriptor{
		// Claims RUNNING, but its last heartbeat was 20 minutes ago — wedged, not progressing.
		{ID: "wedged", Host: "n", PCBState: "RUNNING", CreatedAt: now.Add(-time.Hour), LastSeen: now.Add(-20 * time.Minute)},
		// A healthy RUNNING peer, heartbeat 5s ago.
		{ID: "healthy", Host: "n", PCBState: "RUNNING", CreatedAt: now.Add(-time.Hour), LastSeen: now.Add(-5 * time.Second)},
	}

	inv := buildSessionInventory(local, nil, now, 5*time.Minute, false)

	byID := map[string]sessionInventoryRow{}
	for _, r := range inv.Sessions {
		byID[r.ID] = r
	}
	if byID["wedged"].LivenessClass != "stalled" {
		t.Fatalf("wedged liveness = %q, want stalled", byID["wedged"].LivenessClass)
	}
	if byID["healthy"].LivenessClass != "live" {
		t.Fatalf("healthy liveness = %q, want live", byID["healthy"].LivenessClass)
	}
	// The summary separates the wedged one out: 1 RUNNING, 1 STALLED (never 2 RUNNING).
	if inv.ByState["RUNNING"] != 1 || inv.ByState["STALLED"] != 1 {
		t.Fatalf("by_state = %v, want RUNNING:1 STALLED:1", inv.ByState)
	}
	if got := sessionInventorySummary(inv); !strings.Contains(got, "1 RUNNING, 1 STALLED") {
		t.Fatalf("summary = %q, want it to separate RUNNING from STALLED", got)
	}
}

// TestSessionLsDurable drives the real `fak session ls --durable` wire end to end against a
// file-backed registry with NO gateway running — proving the durable read survives a
// process with no live serve (the point of the surface). It asserts the count summary
// headline and a deterministic row.
func TestSessionLsDurable(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/session-registry.json"
	reg := session.NewRegistry(session.NewFileStore(path))
	now := time.Now()
	if _, err := reg.Register("sess-durable", "node-x", session.State{TraceID: "sess-durable", Run: session.Running}, 0, now); err != nil {
		t.Fatalf("register: %v", err)
	}
	if _, err := reg.Register("sess-paused", "node-x", session.State{TraceID: "sess-paused", Run: session.Paused}, 0, now); err != nil {
		t.Fatalf("register: %v", err)
	}

	var stdout, stderr bytes.Buffer
	// A durable read needs no gateway: point --registry at the file and read it offline.
	rc := runSession(&stdout, &stderr, []string{"ls", "--durable", "--registry", path})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr=%q", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "2 session(s):") || !strings.Contains(out, "1 RUNNING, 1 PAUSED") {
		t.Fatalf("count summary wrong:\n%s", out)
	}
	if !strings.Contains(out, "sess-durable") || !strings.Contains(out, "RUNNING") {
		t.Fatalf("durable row missing:\n%s", out)
	}

	// --json emits the same numbers for a machine.
	stdout.Reset()
	stderr.Reset()
	rc = runSession(&stdout, &stderr, []string{"ls", "--durable", "--registry", path, "--json"})
	if rc != 0 {
		t.Fatalf("json rc = %d, stderr=%q", rc, stderr.String())
	}
	if j := stdout.String(); !strings.Contains(j, `"count": 2`) || !strings.Contains(j, `"pcb_state": "RUNNING"`) {
		t.Fatalf("json wrong:\n%s", j)
	}
}
