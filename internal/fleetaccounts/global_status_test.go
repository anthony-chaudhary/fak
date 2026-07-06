package fleetaccounts

import (
	"strings"
	"testing"
	"time"
)

func TestGlobalStatusExcludesStaleSnapshotsByDefault(t *testing.T) {
	now := time.Date(2026, 7, 6, 15, 0, 0, 0, time.UTC)
	fresh := statusSnapshotForTest("node-a", now.Add(-5*time.Minute), statusAccountForTest("node-a", "groq-a", "groq", 1, "ready", 1, 1, 0, 0))
	stale := statusSnapshotForTest("node-b", now.Add(-2*time.Hour), statusAccountForTest("node-b", "groq-b", "groq", 1, "ready", 2, 2, 0, 0))

	rep := BuildGlobalStatusReport([]StatusSnapshot{fresh, stale}, GlobalStatusOptions{
		Filter:      StatusFilter{Provider: "groq", Tier: 1},
		GroupBy:     []string{"provider", "tier"},
		FreshWithin: 30 * time.Minute,
		Now:         now,
	})

	if rep.Schema != GlobalStatusReportSchema {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Totals.FreeSlots != 1 || rep.Totals.TotalSlots != 1 {
		t.Fatalf("fresh totals = %+v, want only node-a capacity", rep.Totals)
	}
	if rep.StaleTotals.FreeSlots != 2 || rep.StaleTotals.TotalSlots != 2 {
		t.Fatalf("stale totals = %+v, want node-b capacity separate", rep.StaleTotals)
	}
	if len(rep.Nodes) != 2 || !rep.Nodes[0].Fresh || rep.Nodes[1].Fresh {
		t.Fatalf("nodes freshness/order = %+v, want fresh node then stale node", rep.Nodes)
	}
	ru := findStatusRollup(StatusReport{Rollups: rep.Rollups}, "provider=groq tier=t1")
	if ru == nil || ru.FreeSlots != 1 || ru.TotalSlots != 1 {
		t.Fatalf("fresh rollup = %+v, want one fresh slot", rep.Rollups)
	}
	if len(rep.Warnings) == 0 || !strings.Contains(strings.Join(rep.Warnings, "\n"), "stale node node-b excluded") {
		t.Fatalf("warnings = %+v, want stale exclusion", rep.Warnings)
	}
	rendered := RenderGlobalStatusReport(rep, true)
	if !strings.Contains(rendered, "fleet account status (global provider=groq tier=t1)") ||
		!strings.Contains(rendered, "stale excluded: 2 free / 2 total") ||
		!strings.Contains(rendered, "node-b") {
		t.Fatalf("render missing global/stale/node detail:\n%s", rendered)
	}
}

func TestGlobalStatusIncludeStaleAndNodeGrouping(t *testing.T) {
	now := time.Date(2026, 7, 6, 15, 0, 0, 0, time.UTC)
	fresh := statusSnapshotForTest("node-a", now.Add(-5*time.Minute), statusAccountForTest("node-a", "groq-a", "groq", 1, "ready", 1, 1, 0, 0))
	stale := statusSnapshotForTest("node-b", now.Add(-2*time.Hour), statusAccountForTest("node-b", "groq-b", "groq", 1, "usage", 2, 0, 0, 2))

	rep := BuildGlobalStatusReport([]StatusSnapshot{fresh, stale}, GlobalStatusOptions{
		Filter:       StatusFilter{Provider: "groq"},
		GroupBy:      []string{"node", "provider"},
		FreshWithin:  30 * time.Minute,
		IncludeStale: true,
		Now:          now,
	})

	if rep.Totals.TotalSlots != 3 || rep.Totals.FreeSlots != 1 || rep.Totals.BlockedSlots != 2 {
		t.Fatalf("include-stale totals = %+v, want fresh+stale capacity", rep.Totals)
	}
	if len(rep.Rollups) != 2 {
		t.Fatalf("rollups = %+v, want node-level rollups", rep.Rollups)
	}
	if rep.Rollups[0].Dimensions["node"] == "" || rep.Rollups[1].Dimensions["node"] == "" {
		t.Fatalf("node dimension missing from rollups: %+v", rep.Rollups)
	}
	if !strings.Contains(strings.Join(rep.Warnings, "\n"), "stale node node-b included by request") {
		t.Fatalf("warnings = %+v, want stale included warning", rep.Warnings)
	}
}

func statusSnapshotForTest(node string, generatedAt time.Time, accounts ...StatusAccount) StatusSnapshot {
	rep := StatusReport{
		Schema:      StatusReportSchema,
		Node:        node,
		GeneratedAt: generatedAt.UTC().Format(time.RFC3339),
		Accounts:    accounts,
	}
	return StatusSnapshot{Node: node, Report: rep}
}

func statusAccountForTest(node, tag, provider string, tier int, state string, cap, free, leased, blocked int) StatusAccount {
	return StatusAccount{
		Node:            node,
		Account:         "opencode-" + tag,
		Tag:             tag,
		Product:         "opencode",
		Provider:        provider,
		ModelTier:       intp(tier),
		Model:           provider + "/model",
		Kind:            string(KindWorker),
		State:           state,
		Pool:            "dir:opencode-" + tag,
		CapacityCounted: true,
		SessionCap:      cap,
		FreeSlots:       free,
		LeasedSlots:     leased,
		BlockedSlots:    blocked,
		Reason:          "fixture",
	}
}
