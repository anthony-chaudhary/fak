package gardenbudget

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

var (
	benchResultSink Result
	benchCursorSink Cursor
	benchDurSink    time.Duration
	benchIntSink    int
)

var benchPhases = []Phase{
	"leases",
	"lock-files",
	"intents",
	"growth-logs",
	"sentinel-fold",
}

func BenchmarkExecute_Unbounded(b *testing.B) {
	cur := Cursor{Stage: "act", Ticks: 1}
	opt := Options{}
	noopRun := func(p Phase) error { return nil }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Execute(benchPhases, cur, opt, noopRun)
	}
}

func BenchmarkExecute_WithCheckpoint(b *testing.B) {
	cur := Cursor{Stage: "act", Ticks: 1}
	opt := Options{
		Checkpoint: func(c Cursor) error {
			benchCursorSink = c
			return nil
		},
	}
	noopRun := func(p Phase) error { return nil }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Execute(benchPhases, cur, opt, noopRun)
	}
}

func BenchmarkExecute_ResumedSuffix(b *testing.B) {
	cur := Cursor{Stage: "act", Next: "intents", Ticks: 1}
	opt := Options{}
	noopRun := func(p Phase) error { return nil }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Execute(benchPhases, cur, opt, noopRun)
	}
}

func BenchmarkExecute_BudgetExhausted(b *testing.B) {
	start := time.Unix(1_700_000_000, 0)
	now := start.Add(10 * time.Second)
	clk := func() time.Time { return now }
	opt := Options{
		Budget: 1 * time.Second,
		Start:  start,
		Now:    clk,
	}
	cur := Cursor{Stage: "act", Ticks: 1}
	noopRun := func(p Phase) error { return nil }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Execute(benchPhases, cur, opt, noopRun)
	}
}

func BenchmarkExecute_WithSaveCursorCheckpoint(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "cursor.json")
	cur := Cursor{Stage: "act", Ticks: 1}
	opt := Options{
		Checkpoint: func(c Cursor) error {
			return SaveCursor(path, c)
		},
	}
	noopRun := func(p Phase) error { return nil }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Execute(benchPhases, cur, opt, noopRun)
	}
}

func BenchmarkStamp(b *testing.B) {
	base := Cursor{
		Schema:      CursorSchema,
		Stage:       "collect",
		Next:        "leases",
		Ticks:       5,
		UpdatedUnix: 1_700_000_000,
		Payload:     json.RawMessage(`{"members":12,"counts":{"stale":2}}`),
	}
	now := time.Unix(1_700_000_100, 0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCursorSink = Stamp(base, "act", now)
	}
}

func BenchmarkRemaining(b *testing.B) {
	start := time.Unix(1_700_000_000, 0)
	nowVal := start.Add(2500 * time.Millisecond)
	nowFn := func() time.Time { return nowVal }
	budget := 10 * time.Second

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDurSink = Remaining(budget, start, nowFn)
	}
}

func BenchmarkSaveCursor(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "cursor.json")
	c := Cursor{
		Schema:      CursorSchema,
		Stage:       "act",
		Next:        "growth-logs",
		Ticks:       4,
		UpdatedUnix: 1_700_000_000,
		Payload:     json.RawMessage(`{"counts":{"leases":3,"locks":1}}`),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveCursor(path, c); err != nil {
			b.Fatalf("SaveCursor: %v", err)
		}
	}
}

func BenchmarkLoadCursor(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "cursor.json")
	c := Cursor{
		Schema:      CursorSchema,
		Stage:       "act",
		Next:        "growth-logs",
		Ticks:       4,
		UpdatedUnix: 1_700_000_000,
		Payload:     json.RawMessage(`{"counts":{"leases":3,"locks":1}}`),
	}
	if err := SaveCursor(path, c); err != nil {
		b.Fatalf("setup SaveCursor: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cur, err := LoadCursor(path)
		if err != nil {
			b.Fatalf("LoadCursor: %v", err)
		}
		benchCursorSink = cur
	}
}

func BenchmarkLoadCursor_NonExistent(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "absent.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cur, err := LoadCursor(path)
		if err != nil {
			b.Fatalf("LoadCursor absent: %v", err)
		}
		benchCursorSink = cur
	}
}

func BenchmarkSaveAndLoadCursor_RoundTrip(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "cursor.json")
	c := Cursor{
		Schema:      CursorSchema,
		Stage:       "act",
		Next:        "growth-logs",
		Ticks:       4,
		UpdatedUnix: 1_700_000_000,
		Payload:     json.RawMessage(`{"counts":{"leases":3,"locks":1}}`),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveCursor(path, c); err != nil {
			b.Fatalf("SaveCursor: %v", err)
		}
		cur, err := LoadCursor(path)
		if err != nil {
			b.Fatalf("LoadCursor: %v", err)
		}
		benchCursorSink = cur
	}
}

func BenchmarkResult_Errors(b *testing.B) {
	res := Result{
		Ran: []PhaseRun{
			{Phase: "leases", Millis: 1},
			{Phase: "lock-files", Millis: 2, Err: "locked"},
			{Phase: "intents", Millis: 1},
			{Phase: "growth-logs", Millis: 3, Err: "disk full"},
			{Phase: "sentinel-fold", Millis: 1},
		},
	}
	var n int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n += res.Errors()
	}
	benchIntSink = n
}
