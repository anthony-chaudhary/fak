package dosdecision

import (
	"fmt"
	"testing"
)

func benchmarkRefuseRow(lane, reason string) Row {
	return Row{
		"kind":          KindArbiterRefuse,
		"resolver_kind": "HUMAN",
		"lane":          lane,
		"reason_token":  "",
		"reason_text":   reason,
		"age_seconds":   164000.0,
		"source_path":   ".dos/lane-journal.jsonl",
		"evidence":      []any{"journal seq #1733"},
	}
}

func generateBenchmarkRows(count int, refuseRatio float64) []Row {
	rows := make([]Row, count)
	for i := 0; i < count; i++ {
		lane := fmt.Sprintf("lane-%03d", i%15)
		if float64(i)/float64(count) < refuseRatio {
			reason := fmt.Sprintf(
				"lane '%s' cannot share live lane 'blocker-%03d': exact-glob overlap: claimed identical glob",
				lane, (i+1)%15,
			)
			rows[i] = benchmarkRefuseRow(lane, reason)
		} else {
			switch i % 3 {
			case 0:
				rows[i] = Row{"kind": "LIVENESS", "lane": lane, "reason_text": "SPINNING past budget"}
			case 1:
				rows[i] = Row{"kind": "HOST_QUEUE_ITEM", "key": fmt.Sprintf("item-%d", i), "action": "OPEN_ISSUE"}
			default:
				rows[i] = Row{"kind": "WEDGE", "lane": lane, "reason_text": "lane wedged"}
			}
		}
	}
	return rows
}

func BenchmarkLaneKey(b *testing.B) {
	cases := []struct {
		name  string
		input string
	}{
		{name: "Bare", input: "devcmd"},
		{name: "DecoratedCluster", input: "a/b/apply cluster (AFR, ALO)"},
		{name: "ForwardSlashPath", input: "internal/dosdecision/revalidate"},
		{name: "BackslashPath", input: "internal\\dosdecision\\revalidate"},
		{name: "WhitespaceAndCase", input: "   GATEWAY_ROUTER   "},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = LaneKey(tc.input)
			}
		})
	}
}

func BenchmarkBlockingLanes(b *testing.B) {
	cases := []struct {
		name string
		row  Row
	}{
		{
			name: "OwnLaneHolder",
			row: benchmarkRefuseRow(
				"hooks",
				"lane 'hooks' is already held by a live loop — pick a different --lane or wait.",
			),
		},
		{
			name: "ExactGlobOverlap",
			row: benchmarkRefuseRow(
				"cmd",
				"lane 'cmd' cannot share live lane 'devcmd': exact-glob overlap: "+
					"identical glob claimed by both lanes (2: cmd/fak/**) — same write region, hard collision regardless of ratio.",
			),
		},
		{
			name: "ExclusiveGlobal",
			row: benchmarkRefuseRow(
				"coordinateoperator",
				"an exclusive lane is live (lane='global', kind='global', loop='20260811-0900'); "+
					"it touches the whole portfolio — wait for it to finish.",
			),
		},
		{
			name: "MultiLaneProse",
			row: benchmarkRefuseRow(
				"worker",
				"lane 'worker' conflicts with lane 'cache', lane 'gateway', and live lane 'indexer'",
			),
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				lanes := BlockingLanes(tc.row)
				if len(lanes) == 0 {
					b.Fatal("unexpected empty blocking lanes")
				}
			}
		})
	}
}

func BenchmarkNeedsLiveSet(b *testing.B) {
	nonRefusal50 := generateBenchmarkRows(50, 0.0)
	earlyHit50 := generateBenchmarkRows(50, 1.0)
	lateHit50 := append(generateBenchmarkRows(49, 0.0), benchmarkRefuseRow("cmd", "lane 'cmd' is already held"))

	resolved50 := make([]Row, 50)
	for i := 0; i < 50; i++ {
		r := benchmarkRefuseRow(fmt.Sprintf("lane-%d", i), "lane held")
		r["resolved"] = true
		resolved50[i] = r
	}

	b.Run("AllNonRefusal_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if NeedsLiveSet(nonRefusal50) {
				b.Fatal("expected false")
			}
		}
	})

	b.Run("EarlyHit_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !NeedsLiveSet(earlyHit50) {
				b.Fatal("expected true")
			}
		}
	})

	b.Run("LateHit_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !NeedsLiveSet(lateHit50) {
				b.Fatal("expected true")
			}
		}
	})

	b.Run("ResolvedOnly_50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if NeedsLiveSet(resolved50) {
				b.Fatal("expected false")
			}
		}
	})
}

func BenchmarkLiveSetHolds(b *testing.B) {
	lanes5 := []string{"cmd", "devcmd", "gateway", "hooks", "engine"}
	set5 := LiveSet{Lanes: lanes5, Known: true}

	lanes25 := make([]string, 25)
	for i := 0; i < 25; i++ {
		lanes25[i] = fmt.Sprintf("subsystem/lane-%02d", i)
	}
	set25 := LiveSet{Lanes: lanes25, Known: true}

	b.Run("Small5_Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !set5.holds("gateway") {
				b.Fatal("expected hit")
			}
		}
	})

	b.Run("Small5_Miss", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if set5.holds("missing-lane") {
				b.Fatal("expected miss")
			}
		}
	})

	b.Run("Medium25_Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !set25.holds("lane-12") {
				b.Fatal("expected hit")
			}
		}
	})

	b.Run("Medium25_Miss", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if set25.holds("nonexistent") {
				b.Fatal("expected miss")
			}
		}
	})
}

func BenchmarkRevalidate(b *testing.B) {
	liveNone := LiveSet{Lanes: []string{}, Known: true}
	liveAll := LiveSet{
		Lanes: []string{
			"lane-000", "lane-001", "lane-002", "lane-003", "lane-004",
			"lane-005", "lane-006", "lane-007", "lane-008", "lane-009",
			"lane-010", "lane-011", "lane-012", "lane-013", "lane-014",
			"blocker-000", "blocker-001", "blocker-002", "blocker-003", "blocker-004",
			"blocker-005", "blocker-006", "blocker-007", "blocker-008", "blocker-009",
			"blocker-010", "blocker-011", "blocker-012", "blocker-013", "blocker-014",
		},
		Known: true,
	}
	liveUnknown := LiveSet{Known: false}

	rows10AllRefuse := generateBenchmarkRows(10, 1.0)
	rows50Mixed := generateBenchmarkRows(50, 0.5)
	rows200AllRefuse := generateBenchmarkRows(200, 1.0)
	rows100Mixed := generateBenchmarkRows(100, 0.5)

	b.Run("Batch10_AllActive", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Revalidate(rows10AllRefuse, liveAll)
			if len(res.Active) != 10 || res.Cleared != 0 {
				b.Fatalf("unexpected result: active=%d cleared=%d", len(res.Active), res.Cleared)
			}
		}
	})

	b.Run("Batch10_AllCleared", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Revalidate(rows10AllRefuse, liveNone)
			if len(res.Superseded) != 10 || res.Cleared != 10 {
				b.Fatalf("unexpected result: superseded=%d cleared=%d", len(res.Superseded), res.Cleared)
			}
		}
	})

	b.Run("Batch50_MixedWorkload", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Revalidate(rows50Mixed, liveNone)
			if res.Cleared == 0 {
				b.Fatal("expected cleared rows in mixed workload")
			}
		}
	})

	b.Run("Batch200_BacklogSweep", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Revalidate(rows200AllRefuse, liveNone)
			if res.Cleared != 200 {
				b.Fatalf("expected 200 cleared, got %d", res.Cleared)
			}
		}
	})

	b.Run("Batch100_UnknownLiveSet_FailClosed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := Revalidate(rows100Mixed, liveUnknown)
			if res.Cleared != 0 || len(res.Active) != 100 {
				b.Fatalf("unexpected unk live set clear: cleared=%d active=%d", res.Cleared, len(res.Active))
			}
		}
	})
}

func BenchmarkRevalidateParallel(b *testing.B) {
	rows := generateBenchmarkRows(50, 0.5)
	live := LiveSet{Lanes: []string{"blocker-001", "blocker-005"}, Known: true}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res := Revalidate(rows, live)
			if len(res.Active)+len(res.Superseded) != 50 {
				b.Fatalf("partition lost rows: active=%d superseded=%d", len(res.Active), len(res.Superseded))
			}
		}
	})
}
