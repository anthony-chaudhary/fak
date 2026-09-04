package dogfoodissues

import (
	"testing"
	"time"
)

// BenchmarkDogfoodIssues exercises action item extraction and receipt building.
func BenchmarkDogfoodIssues(b *testing.B) {
	report := map[string]any{
		"schema":  "fak.recent-feature-dogfood.v1",
		"ok":      true,
		"out_dir": ".fak/recent-feature-dogfood/bench",
		"probes": []any{
			map[string]any{
				"key": "code-slop-scorecard",
				"ok":  true,
				"payload": map[string]any{
					"schema":      "fleet-code-slop-scorecard/1",
					"ok":          false,
					"verdict":     "ACTION",
					"finding":     "code_slop",
					"corpus":      map[string]any{"score": 71.5, "grade": "C", "slop_debt": 12},
					"next_action": "retire slop-debt worst-first; re-run to prove the drop",
				},
			},
			map[string]any{
				"key": "dogfood-coverage-scorecard",
				"ok":  true,
				"payload": map[string]any{
					"schema":       "dogfood-coverage/1",
					"coverage":     88.9,
					"grade":        "B",
					"dogfood_debt": 0,
					"audit_rows":   0,
					"worst_first":  []any{"audit_journal_evidence"},
				},
			},
		},
	}

	res := Result{
		Report: "bench_report.json",
		Planned: []PlanRow{
			{Action: "create", Key: "k1"},
			{Action: "update", Key: "k2"},
		},
		Synced: []SyncRow{
			{Key: "k1", Action: "create", OK: true},
			{Key: "k2", Action: "update", OK: true},
		},
		Skipped: []SkippedRow{{Key: "k3", Reason: "vague"}},
	}
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items := ExtractActionItems(report, "bench_report.json")
		if len(items) == 0 {
			b.Fatal("unexpected empty action items")
		}
		rec := BuildReceipt(res, ReceiptModeLive, now)
		if rec.Actions == 0 {
			b.Fatal("unexpected empty receipt")
		}
	}
}
