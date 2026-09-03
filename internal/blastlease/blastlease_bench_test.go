package blastlease_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

	for i := 0; i < 20; i++ {
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
	if len(initial) != 20 {
		b.Fatalf("Live returned %d leases, want 20", len(initial))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leases, err := blastlease.Live(dir, now)
		if err != nil {
			b.Fatalf("Live failed: %v", err)
		}
		if len(leases) != 20 {
			b.Fatalf("Live returned %d leases, want 20", len(leases))
		}
	}
}

func TestAllocationBudget(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "leases_10.jsonl")
	data := generateSyntheticJSONL(10)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	allocs := testing.AllocsPerRun(100, func() {
		leases, err := blastlease.Read(path)
		if err != nil {
			t.Fatalf("Read failed: %v", err)
		}
		if len(leases) != 10 {
			t.Fatalf("got %d leases, want 10", len(leases))
		}
	})

	t.Logf("Read 10 records allocations per run: %.1f", allocs)
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

	// Verify runtime throughput scaling: 1000 records (10x) should scale ~10x, well below 30x
	const iters = 50
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := blastlease.Read(path100); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	}
	dur100 := time.Since(t0)

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := blastlease.Read(path1000); err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	}
	dur1000 := time.Since(t1)

	ratio := float64(dur1000) / float64(dur100)
	t.Logf("runtime ratio 1000/100 records (10x workload): %.2fx (dur100=%v, dur1000=%v)", ratio, dur100, dur1000)
	if ratio > 30.0 {
		t.Fatalf("throughput scales super-linearly: 10x workload took %.2fx time", ratio)
	}
}
