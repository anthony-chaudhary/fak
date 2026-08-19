package agentquery

import (
	"fmt"
	"testing"
	"time"
)

func TestAcceptanceTwelveAgentsAcrossFleetDimensions(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	lanes := []string{"cmd", "docs", "gateway"}
	hosts := []string{"h1", "h2"}
	epochs := []string{"boot-a", "boot-b"}
	root, parent := "root-1", "parent-1"
	history := make([]Row, 0, 12)
	states := []string{"LIVE", "CLOSED", "STALE", "CRASHED"}
	for i := 0; i < 12; i++ {
		id := fmtID(i)
		lane := lanes[i%3]
		host := hosts[i%2]
		epoch := epochs[i%2]
		started := now.Add(-time.Duration(i+1) * time.Hour).Format(time.RFC3339)
		elapsed := int64((i + 1) * 1000)
		state := states[i%4]
		r := Row{AgentID: id, LogicalSessionID: id, Lane: &lane, Host: &host, ExecutionEpoch: &epoch, State: state, Liveness: state, StartedAt: &started, ElapsedMS: &elapsed, Source: "history"}
		if i == 1 {
			r.RootID = &root
			r.ParentID = &parent
		}
		if state == "STALE" {
			r.Stale = true
		}
		history = append(history, r)
	}
	liveElapsed := int64(99_000)
	live := []Row{{AgentID: "agent-00", LogicalSessionID: "agent-00", State: "LIVE", Liveness: "LIVE", ElapsedMS: &liveElapsed, Source: "live"}}
	got := Union(live, history, "union", false, 100, now)
	if len(got.Rows) != 12 || got.Metadata.Deduplicated != 1 {
		t.Fatalf("rows=%d metadata=%+v", len(got.Rows), got.Metadata)
	}
	laneSet, hostSet, stateSet, epochSet := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	lineage, stale := false, false
	for _, r := range got.Rows {
		if r.Lane != nil {
			laneSet[*r.Lane] = true
		}
		if r.Host != nil {
			hostSet[*r.Host] = true
		}
		stateSet[r.State] = true
		if r.ExecutionEpoch != nil {
			epochSet[*r.ExecutionEpoch] = true
		}
		lineage = lineage || (r.RootID != nil && r.ParentID != nil)
		stale = stale || r.Stale
	}
	if len(laneSet) != 3 || len(hostSet) != 2 || len(stateSet) != 4 || len(epochSet) != 2 || !lineage || !stale {
		t.Fatalf("lanes=%v hosts=%v states=%v epochs=%v lineage=%v stale=%v", laneSet, hostSet, stateSet, epochSet, lineage, stale)
	}
}
func fmtID(i int) string { return fmt.Sprintf("agent-%02d", i) }
