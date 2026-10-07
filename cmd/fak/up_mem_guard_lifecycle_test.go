package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

// newSampledMemGuard builds a governor driven only through sample(): the timer interval
// is an hour so the background goroutine never samples during the test.
func newSampledMemGuard(t *testing.T, limit uint64, rss *uint64, now *time.Time, mu *sync.Mutex, order *[]string) *memGuardGovernor {
	t.Helper()
	g := newMemGuardGovernor(limit, time.Hour, 10*time.Second,
		func() uint64 { mu.Lock(); defer mu.Unlock(); return *rss },
		func() { mu.Lock(); *order = append(*order, "shutdown"); mu.Unlock() },
		func(string, ...any) {})
	if g == nil {
		t.Fatal("newMemGuardGovernor returned nil for a positive limit")
	}
	g.now = func() time.Time { mu.Lock(); defer mu.Unlock(); return *now }
	t.Cleanup(g.close)
	return g
}

// fak-test:runtime fast est=5ms lane=default
func TestMemGuardOnStopRunsOnceBeforeShutdown(t *testing.T) {
	var (
		mu     sync.Mutex
		order  []string
		events []memGuardStopEvent
		rss    = uint64(100)
		now    = time.Unix(1000, 0)
	)
	g := newSampledMemGuard(t, 50, &rss, &now, &mu, &order)
	g.onStop = func(ev memGuardStopEvent) {
		mu.Lock()
		order = append(order, "onStop")
		events = append(events, ev)
		mu.Unlock()
	}

	g.sample()
	if g.didFire() {
		t.Fatal("didFire after the first over-ceiling sample, want false (sustain window)")
	}
	mu.Lock()
	now = now.Add(5 * time.Second)
	rss = 120
	mu.Unlock()
	g.sample()
	mu.Lock()
	now = now.Add(6 * time.Second)
	rss = 110
	tripAt := now
	mu.Unlock()
	g.sample()
	if !g.didFire() {
		t.Fatal("didFire = false after a sustained breach")
	}
	// Further samples after the stop must not fire the hook or shutdown again.
	mu.Lock()
	now = now.Add(time.Minute)
	mu.Unlock()
	g.sample()
	g.sample()

	mu.Lock()
	defer mu.Unlock()
	if len(order) != 2 || order[0] != "onStop" || order[1] != "shutdown" {
		t.Fatalf("call order = %v, want [onStop shutdown] exactly once", order)
	}
	if len(events) != 1 {
		t.Fatalf("onStop ran %d times, want 1", len(events))
	}
	ev := events[0]
	if ev.RSS != 110 || ev.Peak != 120 || ev.Ceiling != 50 || !ev.At.Equal(tripAt) {
		t.Fatalf("stop event = %+v, want rss=110 peak=120 ceiling=50 at=%v", ev, tripAt)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestMemGuardDidFireFalseWithoutBreach(t *testing.T) {
	var nilGuard *memGuardGovernor
	if nilGuard.didFire() {
		t.Fatal("nil governor didFire = true")
	}
	var (
		mu    sync.Mutex
		order []string
		rss   = uint64(10)
		now   = time.Unix(1000, 0)
	)
	g := newSampledMemGuard(t, 50, &rss, &now, &mu, &order)
	fired := 0
	g.onStop = func(memGuardStopEvent) { fired++ }
	for i := 0; i < 3; i++ {
		g.sample()
		mu.Lock()
		now = now.Add(time.Minute)
		mu.Unlock()
	}
	g.close()
	mu.Lock()
	rss = 1000
	mu.Unlock()
	g.sample()
	g.sample()
	if g.didFire() || fired != 0 || len(order) != 0 {
		t.Fatalf("closed/under-ceiling guard fired=%v hook=%d order=%v, want no stop", g.didFire(), fired, order)
	}

	var s *turnkeyServer
	if s.memGuardFired() {
		t.Fatal("nil server memGuardFired = true")
	}
	if (&turnkeyServer{}).memGuardFired() {
		t.Fatal("server with no guard memGuardFired = true")
	}
}

func readLifecycleLines(t *testing.T, path string) []map[string]json.RawMessage {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []map[string]json.RawMessage
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("lifecycle line %q is not JSON: %v", sc.Text(), err)
		}
		out = append(out, m)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// fak-test:runtime fast est=5ms lane=default
func TestAppendUpLifecycleRecord(t *testing.T) {
	if upLifecycleSchema != "fak.up.lifecycle.v1" || upLifecycleReasonMaxRSS != "max_rss_guard" {
		t.Fatalf("schema/reason = %q/%q, want fak.up.lifecycle.v1/max_rss_guard", upLifecycleSchema, upLifecycleReasonMaxRSS)
	}
	dir := filepath.Join(t.TempDir(), "nested", "up")
	path := filepath.Join(dir, "lifecycle.jsonl")
	at := time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC)
	for i, rss := range []uint64{19 << 30, 20 << 30} {
		err := appendUpLifecycleRecord(path, upLifecycleRecord{
			Schema: upLifecycleSchema, At: at.Add(time.Duration(i) * time.Minute),
			Reason: upLifecycleReasonMaxRSS, RSS: rss, Ceiling: 16 << 30, PID: 4242,
		})
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
	}
	if runtime.GOOS != "windows" {
		di, err := os.Stat(dir)
		if err != nil {
			t.Fatal(err)
		}
		if perm := di.Mode().Perm(); perm != 0o700 {
			t.Fatalf("lifecycle dir perm = %o, want 700", perm)
		}
		fi, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if perm := fi.Mode().Perm(); perm != 0o600 {
			t.Fatalf("lifecycle file perm = %o, want 600", perm)
		}
	}
	lines := readLifecycleLines(t, path)
	if len(lines) != 2 {
		t.Fatalf("lifecycle lines = %d, want 2 (append, not truncate)", len(lines))
	}
	for i, m := range lines {
		for _, k := range []string{"schema", "at", "reason", "rss", "ceiling", "pid"} {
			if _, ok := m[k]; !ok {
				t.Fatalf("line %d missing %q: %v", i, k, m)
			}
		}
		var rec upLifecycleRecord
		raw, _ := json.Marshal(m)
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Schema != upLifecycleSchema || rec.Reason != upLifecycleReasonMaxRSS || rec.Ceiling != 16<<30 || rec.PID != 4242 {
			t.Fatalf("line %d = %+v", i, rec)
		}
		if !rec.At.Equal(at.Add(time.Duration(i) * time.Minute)) {
			t.Fatalf("line %d at = %v, want %v", i, rec.At, at.Add(time.Duration(i)*time.Minute))
		}
	}
	if string(lines[0]["rss"]) != "20401094656" || string(lines[1]["rss"]) != "21474836480" {
		t.Fatalf("rss values = %s,%s, want insertion order 19GiB,20GiB", lines[0]["rss"], lines[1]["rss"])
	}

	// An empty schema is defaulted to the versioned schema.
	path2 := filepath.Join(t.TempDir(), "x", "lifecycle.jsonl")
	if err := appendUpLifecycleRecord(path2, upLifecycleRecord{Reason: upLifecycleReasonMaxRSS}); err != nil {
		t.Fatal(err)
	}
	var schema string
	_ = json.Unmarshal(readLifecycleLines(t, path2)[0]["schema"], &schema)
	if schema != upLifecycleSchema {
		t.Fatalf("defaulted schema = %q, want %q", schema, upLifecycleSchema)
	}
}

// The guard-stop hook writes through the path seam, never under the real home.
// fak-test:runtime fast est=5ms lane=default
func TestRecordMemGuardStopUsesPathSeam(t *testing.T) {
	path := filepath.Join(t.TempDir(), "up", "lifecycle.jsonl")
	orig := upLifecycleLogPath
	t.Cleanup(func() { upLifecycleLogPath = orig })
	upLifecycleLogPath = func() (string, error) { return path, nil }

	at := time.Date(2026, 10, 6, 13, 0, 0, 0, time.UTC)
	recordMemGuardStop(memGuardStopEvent{At: at, RSS: 30, Peak: 40, Ceiling: 20}, func(string, ...any) {
		t.Fatal("recordMemGuardStop logged a failure on a writable seam path")
	})
	lines := readLifecycleLines(t, path)
	if len(lines) != 1 {
		t.Fatalf("lifecycle lines = %d, want 1", len(lines))
	}
	var rec upLifecycleRecord
	raw, _ := json.Marshal(lines[0])
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatal(err)
	}
	if rec.Schema != upLifecycleSchema || rec.Reason != upLifecycleReasonMaxRSS || rec.RSS != 30 || rec.Ceiling != 20 || rec.PID != os.Getpid() || !rec.At.Equal(at) {
		t.Fatalf("record = %+v", rec)
	}

	// A failing seam is logged, not fatal.
	upLifecycleLogPath = func() (string, error) { return "", errors.New("no home") }
	logged := 0
	recordMemGuardStop(memGuardStopEvent{At: at}, func(string, ...any) { logged++ })
	if logged != 1 {
		t.Fatalf("seam failure logged %d times, want 1", logged)
	}
}
