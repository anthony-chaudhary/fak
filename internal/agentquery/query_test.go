package agentquery

import (
	"testing"
	"time"
)

func TestUnionLivePrecedenceAndOrder(t *testing.T) {
	now := time.Unix(100, 0).UTC()
	a, b := int64(100), int64(300)
	live := []Row{{AgentID: "same", Liveness: "LIVE", Source: "live", ElapsedMS: &a}, {AgentID: "long", Liveness: "LIVE", Source: "live", ElapsedMS: &b}}
	history := []Row{{AgentID: "same", Liveness: "CLOSED", Source: "history", ElapsedMS: &b}, {AgentID: "old", Liveness: "CLOSED", Source: "history"}}
	got := Union(live, history, "union", false, 10, now)
	if got.Metadata.Deduplicated != 1 || len(got.Rows) != 3 {
		t.Fatalf("got=%+v", got)
	}
	if got.Rows[0].AgentID != "long" || got.Rows[1].Source != "live" {
		t.Fatalf("rows=%+v", got.Rows)
	}
	if active := Union(live, history, "union", true, 10, now); len(active.Rows) != 2 {
		t.Fatalf("active=%+v", active.Rows)
	}
}
