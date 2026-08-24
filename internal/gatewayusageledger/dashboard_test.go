package gatewayusageledger

import (
	"testing"
	"time"
)

func TestDashboardEventRowAndFoldAreBounded(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	for _, event := range []string{"lightweight_open", "rich_ready", "rich_unavailable"} {
		row, err := DashboardEventRow(event, now)
		if err != nil {
			t.Fatal(err)
		}
		if row.Kind != "dashboard_"+event || row.SessionType != "serve" || row.Context != "dashboard" || row.SessionID != "" {
			t.Fatalf("event row leaked or drifted: %+v", row)
		}
	}
	if _, err := DashboardEventRow("/?dashboard=rich&uid=secret", now); err == nil {
		t.Fatal("unbounded event accepted")
	}
	rows := []Row{}
	for _, event := range []string{"lightweight_open", "lightweight_open", "rich_ready", "rich_unavailable"} {
		row, _ := DashboardEventRow(event, now)
		rows = append(rows, row)
	}
	old, _ := DashboardEventRow("lightweight_open", now.Add(-8*24*time.Hour))
	rows = append(rows, old)
	got := FoldDashboardAdoption(rows, now.Add(-7*24*time.Hour))
	if got.Counts["lightweight_open"] != 2 || got.Counts["rich_ready"] != 1 || got.Counts["rich_unavailable"] != 1 {
		t.Fatalf("fold = %+v", got)
	}
}
