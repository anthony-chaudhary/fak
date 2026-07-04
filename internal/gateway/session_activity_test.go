package gateway

import (
	"strconv"
	"testing"
	"time"
)

// TestSessionActivitySnapshotSequence drives one trace through the exact lifecycle the
// agents pane reads (#2627): an open served turn (in-flight age), the turn returning, two
// admitted calls (last-tool + spawn-count + idle age), then a re-opened turn (in-flight
// suppresses idle). It is the gateway unit witness the issue requires: last-tool /
// spawn-count / inflight / idle asserted from a synthesized adjudication sequence.
func TestSessionActivitySnapshotSequence(t *testing.T) {
	a := newSessionActivity()
	base := time.Unix(1_000_000, 0)

	if _, ok := a.snapshot("t1", base); ok {
		t.Fatal("empty registry must report no record for an unseen trace")
	}

	// A served model turn opens: the trace is in-flight.
	a.beginTurn("t1", base)
	v, ok := a.snapshot("t1", base.Add(3*time.Second))
	if !ok {
		t.Fatal("an open turn must produce a record")
	}
	if v.InflightSeconds != 3 || v.IdleSeconds != 0 {
		t.Fatalf("in-flight age want inflight=3 idle=0, got inflight=%d idle=%d", v.InflightSeconds, v.IdleSeconds)
	}

	// The turn returns, then two calls are admitted — the second is a subagent-spawn shape.
	a.endTurn("t1")
	a.stampProposed("t1", "Read", base.Add(10*time.Second))
	a.stampProposed("t1", "Task", base.Add(10*time.Second))
	v, ok = a.snapshot("t1", base.Add(25*time.Second))
	if !ok {
		t.Fatal("record must persist after adjudication")
	}
	if v.LastTool != "Task" {
		t.Errorf("last_tool want the last admitted tool Task, got %q", v.LastTool)
	}
	if v.SpawnCount != 1 {
		t.Errorf("spawn_count want 1 (only Task is a spawn shape), got %d", v.SpawnCount)
	}
	if v.InflightSeconds != 0 {
		t.Errorf("no request open after endTurn, want inflight=0, got %d", v.InflightSeconds)
	}
	if v.IdleSeconds != 15 {
		t.Errorf("idle age = now - last adjudication, want 15, got %d", v.IdleSeconds)
	}

	// A new turn opens: in-flight is shown again and idle is suppressed (never both).
	a.beginTurn("t1", base.Add(30*time.Second))
	v, _ = a.snapshot("t1", base.Add(34*time.Second))
	if v.InflightSeconds != 4 || v.IdleSeconds != 0 {
		t.Errorf("re-opened turn want inflight=4 idle=0, got inflight=%d idle=%d", v.InflightSeconds, v.IdleSeconds)
	}
	// last_tool/spawn survive across the new turn (they describe the last admitted call).
	if v.LastTool != "Task" || v.SpawnCount != 1 {
		t.Errorf("activity must persist across a new turn, got tool=%q spawn=%d", v.LastTool, v.SpawnCount)
	}
}

// TestSessionActivityClampsNegative proves a backwards clock never renders a negative age
// (the projection clamps at 0 and simply omits the clause).
func TestSessionActivityClampsNegative(t *testing.T) {
	a := newSessionActivity()
	base := time.Unix(2_000_000, 0)
	a.stampProposed("t", "Edit", base)
	v, ok := a.snapshot("t", base.Add(-5*time.Second))
	if !ok {
		t.Fatal("record must exist")
	}
	if v.IdleSeconds != 0 || v.InflightSeconds != 0 {
		t.Errorf("a skewed clock must clamp to 0, got inflight=%d idle=%d", v.InflightSeconds, v.IdleSeconds)
	}
}

// TestSessionActivityNilSafe proves every method is a no-op on a nil registry (a bare
// Server that never went through New) — the calls on the proxy hot path must never panic.
func TestSessionActivityNilSafe(t *testing.T) {
	var a *sessionActivity
	base := time.Unix(3_000_000, 0)
	a.beginTurn("t", base)
	a.endTurn("t")
	a.stampProposed("t", "Read", base)
	a.retain(map[string]struct{}{"t": {}})
	if _, ok := a.snapshot("t", base); ok {
		t.Error("nil registry must report no record")
	}
}

// TestSessionActivityRetainFoldsStopped proves the read-path lifecycle: a trace absent
// from the live set (stopped or vanished) is dropped, and a live trace survives.
func TestSessionActivityRetainFoldsStopped(t *testing.T) {
	a := newSessionActivity()
	base := time.Unix(4_000_000, 0)
	a.stampProposed("live", "Read", base)
	a.stampProposed("gone", "Read", base)
	a.retain(map[string]struct{}{"live": {}})
	if _, ok := a.snapshot("live", base); !ok {
		t.Error("a live trace must be retained")
	}
	if _, ok := a.snapshot("gone", base); ok {
		t.Error("a trace absent from the live set must be folded")
	}
}

// TestSessionActivityCapEvictsColdest proves the write-path bound: the map never exceeds
// the cap, the coldest (least-recently-seen) record is the eviction victim, and an open
// in-flight trace is treated as fresh so it survives a flood of newer stamps.
func TestSessionActivityCapEvictsColdest(t *testing.T) {
	a := newSessionActivity()
	base := time.Unix(5_000_000, 0)

	// One trace is in-flight from the start — it must never be evicted.
	a.beginTurn("hot", base)
	// Fill exactly to the cap with progressively newer records ("cold" is the oldest).
	a.stampProposed("cold", "Read", base.Add(1*time.Second))
	for i := 0; i < sessionActivityCap-2; i++ {
		a.stampProposed(traceName(i), "Read", base.Add(time.Duration(100+i)*time.Second))
	}
	if len(a.rec) != sessionActivityCap {
		t.Fatalf("registry must be full at the cap, got %d want %d", len(a.rec), sessionActivityCap)
	}
	// One more distinct trace forces an eviction of the coldest ("cold").
	a.stampProposed("newest", "Read", base.Add(10_000*time.Second))
	if len(a.rec) != sessionActivityCap {
		t.Fatalf("registry must stay bounded at the cap, got %d", len(a.rec))
	}
	if _, ok := a.snapshot("cold", base); ok {
		t.Error("the coldest record must be the eviction victim")
	}
	if _, ok := a.snapshot("hot", base); !ok {
		t.Error("an in-flight trace must survive eviction (it is seen 'now')")
	}
	if _, ok := a.snapshot("newest", base); !ok {
		t.Error("the newest write must be present")
	}
}

// TestSubagentSpawnToolShapes pins the spawn-shape detector the spawn_count leans on.
func TestSubagentSpawnToolShapes(t *testing.T) {
	for _, tool := range []string{"Task", "spawn_agent", "dispatch", "SubAgent"} {
		if !subagentSpawnTool(tool) {
			t.Errorf("%q should read as a subagent-spawn shape", tool)
		}
	}
	for _, tool := range []string{"Read", "Edit", "Bash", "Grep"} {
		if subagentSpawnTool(tool) {
			t.Errorf("%q must NOT read as a subagent-spawn shape", tool)
		}
	}
}

// traceName is a tiny deterministic id generator for the cap test — distinct per index so
// each stamp creates a fresh record.
func traceName(i int) string {
	return "t-" + strconv.Itoa(i)
}
