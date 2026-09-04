package fleetreap

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// BenchmarkFleetReap measures throughput and allocations for scanning and calculating
// footprint metrics across realistic fleet session artifacts in a b.N loop.
func BenchmarkFleetReap(b *testing.B) {
	dir := b.TempDir()
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < 40; i++ {
		name := fmt.Sprintf("session-%03d-pid-%d.jsonl", i, 1000+i)
		body := strings.Repeat("m", (i+1)*64)
		mod := now.Add(-time.Duration(i) * 15 * time.Minute)
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			b.Fatal(err)
		}
		if err := os.Chtimes(path, mod, mod); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp, err := MeasureFootprint(dir, "*.jsonl", now)
		if err != nil {
			b.Fatalf("failed to measure footprint: %v", err)
		}
		if fp.Files != 40 {
			b.Fatalf("expected 40 files, got %d", fp.Files)
		}
	}
}

// BenchmarkReapByAgeCount measures pruning throughput across aging fleet session files.
func BenchmarkReapByAgeCount(b *testing.B) {
	dir := b.TempDir()
	now := time.Unix(1_700_000_000, 0)
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		for j := 0; j < 16; j++ {
			p := filepath.Join(dir, fmt.Sprintf("entry-%02d.jsonl", j))
			_ = os.WriteFile(p, []byte("payload"), 0o600)
			mod := now.Add(-time.Duration(j) * time.Hour)
			_ = os.Chtimes(p, mod, mod)
		}
		b.StartTimer()

		res, err := ReapByAgeCount(dir, "*.jsonl", 6*time.Hour, 4, now)
		if err != nil {
			b.Fatalf("reap by age count failed: %v", err)
		}
		if res.Removed == 0 {
			b.Fatal("expected files to be removed during benchmark iteration")
		}
	}
}

// BenchmarkReapByDeadOwner measures process-liveness filtering and dead-owner cleanup.
func BenchmarkReapByDeadOwner(b *testing.B) {
	dir := b.TempDir()
	now := time.Unix(1_700_000_000, 0)
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		for j := 0; j < 16; j++ {
			p := filepath.Join(dir, fmt.Sprintf("task-pid-%d.tmp", 2000+j))
			_ = os.WriteFile(p, []byte("active"), 0o600)
		}
		b.StartTimer()

		res, err := ReapByDeadOwner(dir, "*.tmp", func(pid int) bool { return pid%2 == 0 }, now)
		if err != nil {
			b.Fatalf("reap by dead owner failed: %v", err)
		}
		if res.Removed == 0 {
			b.Fatal("expected dead owner files to be removed")
		}
	}
}

// BenchmarkOwnerPID measures basename parsing and PID extraction throughput.
func BenchmarkOwnerPID(b *testing.B) {
	names := []string{
		"session-worker-pid-12345.jsonl",
		"fleet-task-8888.tmp",
		"artifact-without-pid.log",
		"multi-pid-123-and-456.tmp",
		"agent-777.trace",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, name := range names {
			_, _ = ownerPID(name)
		}
	}
}

// TestBenchmarkFleetReapSanity verifies that the benchmark routine executes and measures correctly.
func TestBenchmarkFleetReapSanity(t *testing.T) {
	dir := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < 4; i++ {
		name := fmt.Sprintf("session-%03d-pid-%d.jsonl", i, 1000+i)
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("test"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	fp, err := MeasureFootprint(dir, "*.jsonl", now)
	if err != nil {
		t.Fatalf("unexpected error measuring footprint: %v", err)
	}
	if fp.Files != 4 {
		t.Fatalf("expected 4 files, got %d", fp.Files)
	}
}
