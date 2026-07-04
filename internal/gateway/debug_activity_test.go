package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// activityWireKeys are the four /debug/vars session-row fields the activity cell adds
// (#2627). A pre-activity row must carry none of them so the wire shape is byte-identical
// to before the change.
var activityWireKeys = []string{"last_tool", "spawn_count", "inflight_seconds", "idle_seconds"}

// TestDebugSessionsZeroActivityWireUnchanged is the golden witness: a live session with no
// recorded activity projects a row that omits every new field, so the /debug/vars wire
// shape for a pre-activity session is unchanged. Proven for both a wired-but-empty
// registry and a nil one (a bare Server), since both must fail safe to "omit".
func TestDebugSessionsZeroActivityWireUnchanged(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{{TraceID: "main", Run: "running", Budget: SessionBudget{TurnsLeft: 5}}}
	}
	now := time.Unix(2_000_000, 0)

	assertNoActivityKeys := func(t *testing.T, label string) {
		t.Helper()
		rows := srv.debugSessions(context.Background(), now)
		if len(rows) != 1 {
			t.Fatalf("%s: want 1 live row, got %d", label, len(rows))
		}
		b, err := json.Marshal(rows[0])
		if err != nil {
			t.Fatalf("%s: marshal: %v", label, err)
		}
		for _, k := range activityWireKeys {
			if strings.Contains(string(b), k) {
				t.Errorf("%s: zero-activity row must omit %q: %s", label, k, b)
			}
		}
	}

	assertNoActivityKeys(t, "empty registry")
	srv.activity = nil // a bare Server that never wired a registry must also omit
	assertNoActivityKeys(t, "nil registry")
}

// TestDebugSessionsProjectsActivity proves the projection joins the per-trace record onto
// the matching session row: last_tool and idle age appear once a call is admitted, the
// stopped session is dropped, and its record is folded from the registry (bounded lifecycle
// on the read path).
func TestDebugSessionsProjectsActivity(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{TraceID: "main", Run: "running"},
			{TraceID: "child", Run: "running", ParentTrace: "main", Generation: 1},
			{TraceID: "done", Run: "stopped"},
		}
	}
	now := time.Unix(3_000_000, 0)

	// main admitted an Edit 20s ago; child holds an open turn and has spawned once.
	srv.activity.stampProposed("main", "Edit", now.Add(-20*time.Second))
	srv.activity.stampProposed("child", "Task", now.Add(-40*time.Second))
	srv.activity.beginTurn("child", now.Add(-6*time.Second))
	// A record for the stopped session must be folded by retain on the read below.
	srv.activity.stampProposed("done", "Read", now.Add(-1*time.Second))

	rows := srv.debugSessions(context.Background(), now)
	if len(rows) != 2 {
		t.Fatalf("stopped session must be dropped: want 2 rows, got %d", len(rows))
	}
	byTrace := map[string]debugSessionVars{}
	for _, r := range rows {
		byTrace[r.TraceID] = r
	}

	main := byTrace["main"]
	if main.LastTool != "Edit" {
		t.Errorf("main last_tool want Edit, got %q", main.LastTool)
	}
	if main.IdleSeconds != 20 || main.InflightSeconds != 0 {
		t.Errorf("main want idle=20 inflight=0, got idle=%d inflight=%d", main.IdleSeconds, main.InflightSeconds)
	}

	child := byTrace["child"]
	if child.LastTool != "Task" || child.SpawnCount != 1 {
		t.Errorf("child want tool=Task spawn=1, got tool=%q spawn=%d", child.LastTool, child.SpawnCount)
	}
	if child.InflightSeconds != 6 || child.IdleSeconds != 0 {
		t.Errorf("child holds an open turn: want inflight=6 idle=0, got inflight=%d idle=%d", child.InflightSeconds, child.IdleSeconds)
	}

	// The stopped session's record was folded — a later stamp+read must not resurrect it.
	if _, ok := srv.activity.snapshot("done", now); ok {
		t.Error("stopped session's activity record must be folded by retain")
	}
}
