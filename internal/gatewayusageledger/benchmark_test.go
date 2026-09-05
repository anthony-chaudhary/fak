package gatewayusageledger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var (
	benchSinkRow       Row
	benchSinkRows      []Row
	benchSinkTrend     Trend
	benchSinkShare     SelfHostedShare
	benchSinkEconomics *CompactionEconomics
	benchSinkReport    CompactionReport
	benchSinkRewrites  RewriteReport
	benchSinkCutResult CutResult
	benchSinkKey       string
	benchSinkBool      bool
	benchSinkInt       int
)

func benchSampleCounters(i int) Counters {
	return Counters{
		Submits:                   int64(10 + i),
		VDSOHits:                  int64(4 + i/2),
		EngineCalls:               int64(6 + i/2),
		Denies:                    int64(i % 3),
		Admitted:                  int64(9 + i),
		ObservedTurns:             uint64(5 + (i % 60)),
		InputTokens:               uint64(1000 + i*50),
		OutputTokens:              uint64(200 + i*20),
		CachedPromptTokens:        uint64(400 + i*30),
		CachedTurns:               uint64(3 + i%10),
		CacheCreationTokens:       uint64(100 + (i%5)*200),
		SelfHostedTurns:           uint64(2 + i%5),
		SelfHostedInputTokens:     uint64(400 + i*20),
		SelfHostedOutputTokens:    uint64(80 + i*10),
		VendorTurns:               uint64(3 + i%5),
		VendorInputTokens:         uint64(600 + i*30),
		VendorOutputTokens:        uint64(120 + i*10),
		CompactionFired:           uint64(i % 4),
		CompactionBailed:          uint64((i + 1) % 5),
		CompactionShedTokens:      uint64((i % 4) * 1500),
		CompactionCacheReadTokens: uint64((i % 4) * 1000),
		CompactionBailReasons: map[string]uint64{
			"under_budget":  uint64((i + 1) % 3),
			"too_few_turns": uint64(i % 2),
		},
		UpstreamErrorKinds: map[string]uint64{
			"rate_limited": uint64(i % 2),
		},
		ByReason: map[string]uint64{
			"POLICY_BLOCK": uint64(i % 3),
		},
	}
}

func benchSampleProvenance(i int) *Provenance {
	budgets := []int{0, 48000, 96000}
	profiles := []string{"interactive", "headless"}
	return &Provenance{
		AssumeSessionTurns:   50,
		CompactHistoryBudget: budgets[i%len(budgets)],
		ExposeProfile:        profiles[i%len(profiles)],
		BuildRevision:        "r123+gabcdef",
	}
}

func benchMakeRows(n int, sessionCount int) []Row {
	baseTime := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	rows := make([]Row, n)
	for i := 0; i < n; i++ {
		sid := fmt.Sprintf("session-%03d", i%sessionCount)
		now := baseTime.Add(time.Duration(i) * 10 * time.Minute)
		rows[i] = NewRow(
			"exit",
			"serve",
			"http",
			sid,
			time.Duration(10+i)*time.Second,
			benchSampleProvenance(i),
			benchSampleCounters(i),
			now,
		)
	}
	return rows
}

func benchMakeJSONL(rows []Row) string {
	var buf strings.Builder
	for _, r := range rows {
		b, _ := json.Marshal(r)
		buf.Write(b)
		buf.WriteByte('\n')
	}
	return buf.String()
}

func BenchmarkNewRow(b *testing.B) {
	c := benchSampleCounters(42)
	prov := benchSampleProvenance(1)
	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row := NewRow("exit", "serve", "http", "bench-session-1", 45*time.Second, prov, c, now)
		benchSinkRow = row
	}
}

func BenchmarkComputeRowKey(b *testing.B) {
	c := benchSampleCounters(42)
	nowMs := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC).UnixMilli()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := computeRowKey(Schema, "bench-session-1", 1234, nowMs, c)
		benchSinkKey = key
	}
}

func BenchmarkCompactionEconomicsOf(b *testing.B) {
	b.Run("ActiveCompaction", func(b *testing.B) {
		c := benchSampleCounters(3)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			econ := CompactionEconomicsOf(c)
			benchSinkEconomics = econ
		}
	})
	b.Run("QuietSession", func(b *testing.B) {
		var c Counters
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			econ := CompactionEconomicsOf(c)
			benchSinkEconomics = econ
		}
	})
}

func BenchmarkAppend(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "usage.jsonl")
	row := NewRow("exit", "serve", "http", "bench-sess", 10*time.Second, nil, benchSampleCounters(1), time.Now())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Append(path, row); err != nil {
			b.Fatalf("Append failed: %v", err)
		}
	}
}

func BenchmarkParseLedger(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, 10)
			content := benchMakeJSONL(rows)

			b.SetBytes(int64(len(content)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				parsed := ParseLedger(content)
				if len(parsed) != count {
					b.Fatalf("got %d rows, want %d", len(parsed), count)
				}
				benchSinkRows = parsed
			}
		})
	}
}

func BenchmarkReadLedgerFile(b *testing.B) {
	for _, count := range []int{10, 100, 500} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			dir := b.TempDir()
			path := filepath.Join(dir, "ledger.jsonl")
			rows := benchMakeRows(count, 10)
			content := benchMakeJSONL(rows)
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				b.Fatalf("WriteFile: %v", err)
			}

			b.SetBytes(int64(len(content)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				got := ReadLedgerFile(path)
				if len(got) != count {
					b.Fatalf("got %d rows, want %d", len(got), count)
				}
				benchSinkRows = got
			}
		})
	}
}

func BenchmarkDedupeByKey(b *testing.B) {
	for _, count := range []int{100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, count/2)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				deduped, dropped := DedupeByKey(rows)
				benchSinkRows = deduped
				benchSinkInt = dropped
			}
		})
	}
}

func BenchmarkFoldTrend(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				trend, ok := FoldTrend(rows)
				if !ok {
					b.Fatal("FoldTrend returned ok=false")
				}
				benchSinkTrend = trend
				benchSinkBool = ok
			}
		})
	}
}

func BenchmarkFoldSelfHostedShare(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				share := FoldSelfHostedShare(rows)
				benchSinkShare = share
			}
		})
	}
}

func BenchmarkFoldCompaction(b *testing.B) {
	for _, count := range []int{100, 1000} {
		b.Run(fmt.Sprintf("WholeWindow_%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, 20)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := FoldCompaction(rows, "2026-09-01T00:00:00Z")
				benchSinkReport = rep
			}
		})
		b.Run(fmt.Sprintf("ByPeriod_Day_%d_rows", count), func(b *testing.B) {
			rows := benchMakeRows(count, 20)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := FoldCompactionByPeriod(rows, "2026-09-01T00:00:00Z", "day")
				benchSinkReport = rep
			}
		})
	}
}

func BenchmarkRankPrefixRewrites(b *testing.B) {
	for _, tc := range []struct {
		sessions int
		rows     int
	}{
		{sessions: 10, rows: 100},
		{sessions: 50, rows: 1000},
	} {
		b.Run(fmt.Sprintf("%d_sessions_%d_rows", tc.sessions, tc.rows), func(b *testing.B) {
			rows := benchMakeRows(tc.rows, tc.sessions)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := RankPrefixRewrites(rows, 3.75)
				benchSinkRewrites = rep
			}
		})
	}
}

func BenchmarkCut(b *testing.B) {
	for _, count := range []int{100, 500} {
		b.Run(fmt.Sprintf("DryRun_%d_rows_keep_%d", count, count/5), func(b *testing.B) {
			dir := b.TempDir()
			path := filepath.Join(dir, "usage.jsonl")
			rows := benchMakeRows(count, 20)
			content := benchMakeJSONL(rows)
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				b.Fatalf("WriteFile: %v", err)
			}
			now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
			keep := count / 5

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := Cut(path, keep, true, now)
				if err != nil {
					b.Fatalf("Cut dry-run: %v", err)
				}
				benchSinkCutResult = res
			}
		})
	}
}

func BenchmarkFoldDashboardAdoption(b *testing.B) {
	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	events := []string{"lightweight_open", "rich_ready", "rich_unavailable"}
	rows := make([]Row, 300)
	for i := 0; i < 300; i++ {
		r, _ := DashboardEventRow(events[i%len(events)], now.Add(time.Duration(-i)*time.Hour))
		rows[i] = r
	}
	since := now.Add(-7 * 24 * time.Hour)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adoption := FoldDashboardAdoption(rows, since)
		benchSinkInt = len(adoption.Counts)
	}
}
