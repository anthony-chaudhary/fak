package stallpage

import (
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

var (
	benchKeySink     string
	benchPageSink    Page
	benchAdvisedSink bool
	benchResultSink  PublishResult
)

func BenchmarkDedupKey(b *testing.B) {
	advice := stallscan.RebootAdvice{
		Axis:      "handle_high_water",
		Process:   "WindowsTerminal.exe",
		PID:       1234,
		Count:     31500,
		Threshold: 30000,
	}

	b.Run("Canonical", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchKeySink = DedupKey(advice)
		}
	})

	dirty := stallscan.RebootAdvice{
		Axis:    "  HANDLE_HIGH_WATER  ",
		Process: " windowsterminal.EXE \t",
		PID:     9999,
	}

	b.Run("NormalizeWhitespace", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchKeySink = DedupKey(dirty)
		}
	})
}

func BenchmarkFromAdvice(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	advised := stallscan.RebootAdvice{
		Advised:   true,
		Axis:      "handle_high_water",
		Process:   "WindowsTerminal.exe",
		PID:       42,
		Count:     31_000,
		Threshold: 30_000,
		Reason:    "reboot the host before it freezes",
	}
	unadvised := stallscan.RebootAdvice{Advised: false}

	b.Run("AdvisedCrossing", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPageSink, benchAdvisedSink = FromAdvice(advised, now)
		}
	})

	b.Run("UnadvisedSilent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPageSink, benchAdvisedSink = FromAdvice(unadvised, now)
		}
	})
}

func BenchmarkFromSample(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	below := pageSample(10_000, 500)
	handleCrossing := pageSample(31_000, 100)
	threadCrossing := pageSample(10_000, 2_500)

	b.Run("BelowThreshold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPageSink, benchAdvisedSink = FromSample(below, now)
		}
	})

	b.Run("HandleCrossing", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPageSink, benchAdvisedSink = FromSample(handleCrossing, now)
		}
	})

	b.Run("ThreadCrossing", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPageSink, benchAdvisedSink = FromSample(threadCrossing, now)
		}
	})
}

func BenchmarkPruneState(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	keep := 12 * time.Hour

	b.Run("SmallMap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := pageState{
				Schema: stateSchema,
				Last: map[string]int64{
					"handle_high_water\x00proc_a": now.Add(-1 * time.Hour).UnixNano(),
					"thread_high_water\x00proc_b": now.Add(-24 * time.Hour).UnixNano(),
					"handle_high_water\x00proc_c": now.Add(-15 * time.Hour).UnixNano(),
				},
			}
			pruneState(&st, now, keep)
		}
	})

	b.Run("FiftyEntries", func(b *testing.B) {
		baseMap := make(map[string]int64, 50)
		for j := 0; j < 50; j++ {
			k := "axis" + strconv.Itoa(j%2) + "\x00proc" + strconv.Itoa(j)
			offset := time.Duration(j) * time.Hour
			baseMap[k] = now.Add(-offset).UnixNano()
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := pageState{
				Schema: stateSchema,
				Last:   make(map[string]int64, len(baseMap)),
			}
			for k, v := range baseMap {
				st.Last[k] = v
			}
			pruneState(&st, now, keep)
		}
	})
}

func BenchmarkPublish(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	below := pageSample(10_000, 500)
	crossing := pageSample(31_000, 100)

	b.Run("BelowLineNoOp", func(b *testing.B) {
		dir := b.TempDir()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := Publish(dir, below, now, DefaultDedupWindow)
			if err != nil {
				b.Fatalf("Publish: %v", err)
			}
			benchResultSink = res
		}
	})

	b.Run("DedupHit", func(b *testing.B) {
		dir := b.TempDir()
		first, err := Publish(dir, crossing, now, DefaultDedupWindow)
		if err != nil || !first.Published {
			b.Fatalf("seed Publish: res=%+v, err=%v", first, err)
		}

		repeatTime := now.Add(time.Minute)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := Publish(dir, crossing, repeatTime, DefaultDedupWindow)
			if err != nil {
				b.Fatalf("Publish: %v", err)
			}
			benchResultSink = res
		}
	})

	b.Run("FreshCrossing", func(b *testing.B) {
		baseDir := b.TempDir()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir := filepath.Join(baseDir, strconv.Itoa(i))
			res, err := Publish(dir, crossing, now, DefaultDedupWindow)
			if err != nil {
				b.Fatalf("Publish: %v", err)
			}
			benchResultSink = res
		}
	})
}

func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkFromSample)
	if res.N <= 0 {
		t.Fatalf("expected positive iterations, got %d", res.N)
	}
}
