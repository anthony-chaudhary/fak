package main

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// TestWipLiveSessionsIndexesAgentUUID is the direct #5343 witness at the smallest git seam:
// wipLiveSessions must build a live set a wip checkpoint's stamped Claude UUID can actually
// hit. A LIVE guard-session descriptor carrying AgentUUID contributes that UUID to the set; an
// EXPIRED descriptor's UUID does NOT (a dead owner is not live); and no phantom empty key is
// created. This is the join that was inert before — the volatile agent-claude-<pid> descriptor
// ID never matched the stable UUID a checkpoint stamps.
func TestWipLiveSessionsIndexesAgentUUID(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	store := leaseref.NewInDir(dir)
	now := time.Now()

	const liveUUID = "1e21323a-b92d-4b43-a495-1e0c1d46f3ef"
	const deadUUID = "dead0000-0000-0000-0000-000000000000"

	// A LIVE guard session (fresh UpdatedAt, long TTL) carrying the stable Claude UUID.
	if _, err := store.PublishSession(ctx, leaseref.SessionDescriptor{
		ID: "agent-claude-10800-17b88dff", Host: "h1", PCBState: "RUNNING",
		UpdatedAt: now.Unix(), TTLSecs: 1800, AgentUUID: liveUUID,
	}); err != nil {
		t.Fatalf("PublishSession live: %v", err)
	}
	// An EXPIRED guard session (heartbeat lapsed) carrying a different UUID — must NOT be live.
	if _, err := store.PublishSession(ctx, leaseref.SessionDescriptor{
		ID: "agent-claude-20900-deadbeef", Host: "h2", PCBState: "RUNNING",
		UpdatedAt: now.Unix() - 7200, TTLSecs: 60, AgentUUID: deadUUID,
	}); err != nil {
		t.Fatalf("PublishSession expired: %v", err)
	}

	live, err := wipLiveSessions(ctx, dir)
	if err != nil {
		t.Fatalf("wipLiveSessions: %v", err)
	}
	if !live[liveUUID] {
		t.Fatalf("live session UUID %q must be in the live set (%v)", liveUUID, live)
	}
	if live[deadUUID] {
		t.Fatalf("expired session UUID %q must NOT be live", deadUUID)
	}
	if live[""] {
		t.Fatalf("empty key must never be live (no phantom join)")
	}
	if live["unknown-uuid-never-published"] {
		t.Fatalf("an unjoinable UUID must not be live")
	}
}

// TestWipReconcileLiveCheckpointResolvesLive drives the full reconcile join end to end: a wip
// checkpoint stamped with the STABLE Claude UUID whose LIVE guard session carries that UUID as
// AgentUUID classifies OwnerLive -> SKIP (kept, its recovery snapshot is never a delete
// candidate). A second checkpoint stamped a UUID that NO live descriptor carries is unjoinable
// -> not live -> never SKIP-as-live: the fail-safe that a live owner's checkpoint and a
// stranger's are told apart, which #5340 keys its deletes on.
func TestWipReconcileLiveCheckpointResolvesLive(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)

	const liveUUID = "1e21323a-b92d-4b43-a495-1e0c1d46f3ef"
	const orphanUUID = "0fee1dea-dead-0000-0000-000000000000"

	// Checkpoint #1: a real working-tree delta stamped with the live session's UUID.
	if err := os.WriteFile(file, []byte("base line\nedit A\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, liveUUID, true, 1000); err != nil || r.Object == "" {
		t.Fatalf("checkpoint liveUUID: %v (%+v)", err, r)
	}
	// Checkpoint #2: a distinct delta stamped with an orphan UUID no descriptor will carry.
	if err := os.WriteFile(file, []byte("base line\nedit A\nedit B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r, err := wipCheckpoint(ctx, dir, orphanUUID, true, 2000); err != nil || r.Object == "" {
		t.Fatalf("checkpoint orphanUUID: %v (%+v)", err, r)
	}

	// Publish the LIVE guard session (volatile agent-claude id) carrying the stable UUID.
	if _, err := leaseref.NewInDir(dir).PublishSession(ctx, leaseref.SessionDescriptor{
		ID: "agent-claude-10800-17b88dff", Host: "h1", PCBState: "RUNNING",
		UpdatedAt: time.Now().Unix(), TTLSecs: 1800, AgentUUID: liveUUID,
	}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}

	res, err := wipReconcile(ctx, dir)
	if err != nil {
		t.Fatalf("wipReconcile: %v", err)
	}
	got := map[string]wiprecon.Action{}
	for _, d := range res.Decisions {
		got[d.Session] = d.Action
	}
	if got[liveUUID] != wiprecon.ActSkip {
		t.Fatalf("live session checkpoint action = %q, want SKIP (OwnerLive); decisions=%+v", got[liveUUID], res.Decisions)
	}
	if got[orphanUUID] == wiprecon.ActSkip {
		t.Fatalf("unjoinable checkpoint must NOT be SKIP-as-live (fail-safe); got SKIP for %q; decisions=%+v", orphanUUID, res.Decisions)
	}
	if _, ok := got[orphanUUID]; !ok {
		t.Fatalf("orphan checkpoint missing from decisions=%+v", res.Decisions)
	}
}
