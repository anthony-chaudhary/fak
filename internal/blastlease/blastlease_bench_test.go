package blastlease_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func generateSyntheticJSONL(records int) []byte {
	var buf strings.Builder
	for i := 0; i < records; i++ {
		line := fmt.Sprintf(`{"lane": "lane-%04d", "tree_globs": ["internal/pkg%d/**", "cmd/pkg%d/**"]}`+"\n", i, i, i)
		buf.WriteString(line)
	}
	return []byte(buf.String())
}

func BenchmarkRead(b *testing.B) {
	cases := []struct {
		name    string
		records int
	}{
		{name: "10_records", records: 10},
		{name: "100_records", records: 100},
		{name: "1000_records", records: 1000},
	}

	dir := b.TempDir()
	for _, tc := range cases {
		data := generateSyntheticJSONL(tc.records)
		path := filepath.Join(dir, fmt.Sprintf("leases_%s.jsonl", tc.name))
		if err := os.WriteFile(path, data, 0o644); err != nil {
			b.Fatalf("WriteFile failed: %v", err)
		}

		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				leases, err := blastlease.Read(path)
				if err != nil {
					b.Fatalf("Read failed: %v", err)
				}
				if len(leases) != tc.records {
					b.Fatalf("got %d leases, want %d", len(leases), tc.records)
				}
			}
			b.StopTimer()
			nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
			nsPerRecord := nsPerOp / float64(tc.records)
			recordsPerSec := float64(tc.records*b.N) / b.Elapsed().Seconds()
			b.ReportMetric(recordsPerSec, "records/s")
			b.ReportMetric(nsPerRecord, "ns/record")
		})
	}
}

func BenchmarkLive(b *testing.B) {
	if _, err := exec.LookPath("git"); err != nil {
		b.Skip("git not on PATH")
	}
	dir := b.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "test"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			b.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	t0 := time.Now()

	// leaseCount is the fixture size: Live must return every lease acquired
	// below, so the len() checks assert against the acquired count itself.
	const leaseCount = 20
	for i := 0; i < leaseCount; i++ {
		rec := leaseref.Record{
			ID:          fmt.Sprintf("lane-%02d", i),
			TreeGlobs:   []string{fmt.Sprintf("internal/pkg%d/**", i), fmt.Sprintf("cmd/pkg%d/**", i)},
			Holder:      fmt.Sprintf("agent-%d", i),
			AcquiredAt:  t0.Unix(),
			TTLSeconds:  3600,
			Description: fmt.Sprintf("bench lease %d", i),
		}
		if _, err := store.Acquire(ctx, rec); err != nil {
			b.Fatalf("store.Acquire failed: %v", err)
		}
	}

	now := t0.Add(10 * time.Second)
	initial, err := blastlease.Live(dir, now)
	if err != nil {
		b.Fatalf("Live failed: %v", err)
	}
	if len(initial) != leaseCount {
		b.Fatalf("Live returned %d leases, want %d", len(initial), leaseCount)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leases, err := blastlease.Live(dir, now)
		if err != nil {
			b.Fatalf("Live failed: %v", err)
		}
		if len(leases) != leaseCount {
			b.Fatalf("Live returned %d leases, want %d", len(leases), leaseCount)
		}
	}
}

func TestAllocationBudget(t *testing.T) {
	dir := t.TempDir()
	const records = 10
	path := filepath.Join(dir, "leases_10.jsonl")
	data := generateSyntheticJSONL(records)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	allocs := testing.AllocsPerRun(100, func() {
		leases, err := blastlease.Read(path)
		if err != nil {
			t.Fatalf("Read failed: %v", err)
		}
		if len(leases) != records {
			t.Fatalf("got %d leases, want %d", len(leases), records)
		}
	})

	t.Logf("Read %d records allocations per run: %.1f", records, allocs)
	const maxBudget = 40.0
	if allocs > maxBudget {
		t.Fatalf("allocations per run %.1f exceeds budget %.1f", allocs, maxBudget)
	}
}

func TestReadThroughputLinear(t *testing.T) {
	dir := t.TempDir()
	path10 := filepath.Join(dir, "leases_10.jsonl")
	path100 := filepath.Join(dir, "leases_100.jsonl")
	path1000 := filepath.Join(dir, "leases_1000.jsonl")

	if err := os.WriteFile(path10, generateSyntheticJSONL(10), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if err := os.WriteFile(path100, generateSyntheticJSONL(100), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if err := os.WriteFile(path1000, generateSyntheticJSONL(1000), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	allocs10 := testing.AllocsPerRun(50, func() {
		if _, err := blastlease.Read(path10); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	})
	allocs100 := testing.AllocsPerRun(50, func() {
		if _, err := blastlease.Read(path100); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	})
	allocs1000 := testing.AllocsPerRun(50, func() {
		if _, err := blastlease.Read(path1000); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	})

	t.Logf("allocs: 10 records=%.1f, 100 records=%.1f, 1000 records=%.1f", allocs10, allocs100, allocs1000)

	// Verify linear allocation scaling: each record should cost <= 3 allocs
	rate100 := (allocs100 - allocs10) / 90.0
	rate1000 := (allocs1000 - allocs100) / 900.0
	if rate100 > 3.0 || rate1000 > 3.0 {
		t.Fatalf("allocations per record scale non-linearly: rate100=%.2f rate1000=%.2f", rate100, rate1000)
	}

	// Warm up
	for i := 0; i < 5; i++ {
		_, _ = blastlease.Read(path100)
		_, _ = blastlease.Read(path1000)
	}

	// Verify runtime throughput scaling: one Read of 1000 records (10x workload)
	// should take ~10x one Read of 100 records, well below maxLinearRatio.
	//
	// Sampling envelope: a single back-to-back timing of each size is
	// load-sensitive - under `go test ./...` on a shared runner (GOMAXPROCS=2,
	// -p=2) preemption, a GC cycle, or a neighbour package can land inside just
	// one of the two windows and inflate the ratio with no code change (CI
	// observed dur100=2.72ms but dur1000=164ms, a 60x ratio, while the package
	// alone measures ~6-18x). Two measures cancel that noise without moving the
	// ceiling:
	//   - Equal exposure: each window reads the same number of records (10x as
	//     many Reads of the 100-record file), so both windows are equally long
	//     and equally likely to be preempted; the ratio is taken per Read.
	//   - Fastest of N: the sizes are timed in interleaved trials, each after a
	//     GC, and the fastest trial of each size is compared. Host contention
	//     only ever ADDS time, so the minimum is the least-contended estimate.
	// A genuinely super-linear parser (O(n^2) => ~100x per Read for 10x records)
	// still exceeds the ceiling on its best trial.
	const (
		throughputTrials = 21
		reads1000        = 2
		reads100         = reads1000 * 10 // same record volume as the 1000-record window
		maxLinearRatio   = 30.0
	)
	best100 := time.Duration(math.MaxInt64)
	best1000 := time.Duration(math.MaxInt64)
	for trial := 0; trial < throughputTrials; trial++ {
		best100 = min(best100, timeReads(t, path100, reads100))
		best1000 = min(best1000, timeReads(t, path1000, reads1000))
	}

	perRead100 := best100 / reads100
	perRead1000 := best1000 / reads1000
	ratio := float64(perRead1000) / float64(perRead100)
	t.Logf("runtime ratio per Read 1000/100 records (10x workload, fastest of %d trials): %.2fx (read100=%v, read1000=%v)",
		throughputTrials, ratio, perRead100, perRead1000)
	if ratio > maxLinearRatio {
		t.Fatalf("throughput scales super-linearly: 10x workload took %.2fx time per Read on the fastest of %d trials (ceiling %.0fx)",
			ratio, throughputTrials, maxLinearRatio)
	}
}

// timeReads returns the wall-clock duration of n consecutive blastlease.Read
// calls on path. It collects garbage first so a GC cycle owed by earlier
// allocations is not charged to this window.
func timeReads(t *testing.T, path string, n int) time.Duration {
	t.Helper()
	runtime.GC()
	start := time.Now()
	for i := 0; i < n; i++ {
		if _, err := blastlease.Read(path); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	}
	return time.Since(start)
}
