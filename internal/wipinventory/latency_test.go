package wipinventory

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestProtectionLatencyMathAndBudgetRatio(t *testing.T) {
	now := time.Unix(1700003600, 0).UTC()
	budget := time.Hour // 3600s

	// 1. Empty samples -> zero latencies, 100% ratio, PASS.
	repEmpty := ComputeProtectionLatency(now, "/fak/repo", budget, nil)
	if repEmpty.TotalSourcePaths != 0 || repEmpty.P50Seconds != 0 || repEmpty.P95Seconds != 0 || repEmpty.MaxSeconds != 0 {
		t.Fatalf("unexpected empty metrics: %#v", repEmpty)
	}
	if repEmpty.ProtectedWithinBudgetRatio != 1.0 || repEmpty.SLOVerdict != "PASS" {
		t.Fatalf("unexpected empty verdict: ratio=%v verdict=%s", repEmpty.ProtectedWithinBudgetRatio, repEmpty.SLOVerdict)
	}

	// 2. 20-element nearest-rank fixture (10s to 200s, steps of 10s).
	var samples []PathLatencySample
	for i := 1; i <= 20; i++ {
		sec := float64(i * 10)
		tSeen := now.Add(-time.Duration(sec) * time.Second)
		samples = append(samples, PathLatencySample{
			Path:           "pkg/file" + string(rune('a'+i-1)) + ".go",
			FirstSeen:      &tSeen,
			ProtectedAt:    &now,
			LatencySeconds: Float64Ptr(sec),
			Outcome:        "checkpointed",
			Surface:        "checkpoint",
			ClockKnown:     true,
		})
	}

	// Nearest-rank math:
	// N = 20
	// P50: ceil(0.50 * 20) = 10 -> index 9 -> 100s
	// P95: ceil(0.95 * 20) = 19 -> index 18 -> 190s
	// Max: 200s
	rep20 := ComputeProtectionLatency(now, "/fak/repo", 150*time.Second, samples)
	if rep20.P50Seconds != 100.0 {
		t.Errorf("P50Seconds = %v, want 100.0", rep20.P50Seconds)
	}
	if rep20.P95Seconds != 190.0 {
		t.Errorf("P95Seconds = %v, want 190.0", rep20.P95Seconds)
	}
	if rep20.MaxSeconds != 200.0 {
		t.Errorf("MaxSeconds = %v, want 200.0", rep20.MaxSeconds)
	}
	// Under budget 150s: elements 10..150s (15 elements) are <= 150s.
	// 15 / 20 = 0.75
	if rep20.ProtectedWithinBudgetRatio != 0.75 {
		t.Errorf("ProtectedWithinBudgetRatio = %v, want 0.75", rep20.ProtectedWithinBudgetRatio)
	}
	// Ratio 0.75 < 0.95 -> VIOLATION
	if rep20.SLOVerdict != "VIOLATION" {
		t.Errorf("SLOVerdict = %s, want VIOLATION", rep20.SLOVerdict)
	}
}

func TestFixtureStreams(t *testing.T) {
	now := time.Unix(1700005000, 0).UTC()
	budget := time.Hour // 3600s

	t.Run("protected stream", func(t *testing.T) {
		tSeen := now.Add(-120 * time.Second)
		tProt := now.Add(-60 * time.Second)
		samples := []PathLatencySample{
			{
				Path:           "internal/auth/auth.go",
				FirstSeen:      &tSeen,
				ProtectedAt:    &tProt,
				LatencySeconds: Float64Ptr(60.0),
				Outcome:        "checkpointed",
				Surface:        "checkpoint",
				ClockKnown:     true,
			},
			{
				Path:           "internal/gateway/gw.go",
				FirstSeen:      &tProt,
				ProtectedAt:    &now,
				LatencySeconds: Float64Ptr(60.0),
				Outcome:        "worker_isolated",
				Surface:        "detached_worker",
				ClockKnown:     true,
			},
		}
		rep := ComputeProtectionLatency(now, "/fak/repo", budget, samples)
		if rep.TotalSourcePaths != 2 {
			t.Fatalf("TotalSourcePaths = %d, want 2", rep.TotalSourcePaths)
		}
		if rep.ProtectedWithinBudgetRatio != 1.0 {
			t.Fatalf("ProtectedWithinBudgetRatio = %v, want 1.0", rep.ProtectedWithinBudgetRatio)
		}
		if rep.SLOVerdict != "PASS" {
			t.Fatalf("SLOVerdict = %s, want PASS", rep.SLOVerdict)
		}
		if rep.StaleRefusalCount != 0 {
			t.Fatalf("StaleRefusalCount = %d, want 0", rep.StaleRefusalCount)
		}
	})

	t.Run("stale stream", func(t *testing.T) {
		tSeen := now.Add(-7200 * time.Second) // 2 hours old
		samples := []PathLatencySample{
			{
				Path:           "internal/unprotected/stale.go",
				FirstSeen:      &tSeen,
				ProtectedAt:    nil,
				LatencySeconds: Float64Ptr(7200.0),
				Outcome:        "unprotected",
				Surface:        "shared_trunk",
				ClockKnown:     true,
			},
		}
		rep := ComputeProtectionLatency(now, "/fak/repo", budget, samples)
		if rep.StaleRefusalCount != 1 {
			t.Errorf("StaleRefusalCount = %d, want 1", rep.StaleRefusalCount)
		}
		if rep.SLOVerdict != "VIOLATION" {
			t.Errorf("SLOVerdict = %s, want VIOLATION", rep.SLOVerdict)
		}
		if rep.ProtectedWithinBudgetRatio != 0.0 {
			t.Errorf("ProtectedWithinBudgetRatio = %v, want 0.0", rep.ProtectedWithinBudgetRatio)
		}
	})

	t.Run("mixed stream", func(t *testing.T) {
		tSeen1 := now.Add(-100 * time.Second)
		tProt1 := now.Add(-50 * time.Second)
		tSeen2 := now.Add(-300 * time.Second) // fresh unprotected (under budget)
		samples := []PathLatencySample{
			{
				Path:           "cmd/fak/main.go",
				FirstSeen:      &tSeen1,
				ProtectedAt:    &tProt1,
				LatencySeconds: Float64Ptr(50.0),
				Outcome:        "landed",
				Surface:        "shared_trunk",
				ClockKnown:     true,
			},
			{
				Path:           "cmd/fak/wip.go",
				FirstSeen:      &tSeen2,
				ProtectedAt:    nil,
				LatencySeconds: Float64Ptr(300.0),
				Outcome:        "unprotected",
				Surface:        "shared_trunk",
				ClockKnown:     true,
			},
		}
		rep := ComputeProtectionLatency(now, "/fak/repo", budget, samples)
		if rep.TotalSourcePaths != 2 {
			t.Fatalf("TotalSourcePaths = %d, want 2", rep.TotalSourcePaths)
		}
		if rep.StaleRefusalCount != 0 {
			t.Errorf("StaleRefusalCount = %d, want 0", rep.StaleRefusalCount)
		}
		// 1 protected within budget out of 2 = 0.5
		if rep.ProtectedWithinBudgetRatio != 0.5 {
			t.Errorf("ProtectedWithinBudgetRatio = %v, want 0.5", rep.ProtectedWithinBudgetRatio)
		}
		// Unprotected exists under budget -> NEEDS_ATTENTION or VIOLATION (since 0.5 < 0.95, VIOLATION)
		if rep.SLOVerdict != "VIOLATION" {
			t.Errorf("SLOVerdict = %s, want VIOLATION", rep.SLOVerdict)
		}
	})

	t.Run("missing clock never inferred zero", func(t *testing.T) {
		tSeen := now.Add(-100 * time.Second)
		tProt := now.Add(-20 * time.Second)
		samples := []PathLatencySample{
			{
				Path:           "known.go",
				FirstSeen:      &tSeen,
				ProtectedAt:    &tProt,
				LatencySeconds: Float64Ptr(80.0),
				Outcome:        "checkpointed",
				Surface:        "checkpoint",
				ClockKnown:     true,
			},
			{
				Path:           "unknown_clock.go",
				FirstSeen:      nil,
				ProtectedAt:    nil,
				LatencySeconds: nil,
				Outcome:        "unprotected",
				Surface:        "shared_trunk",
				ClockKnown:     false,
			},
		}
		rep := ComputeProtectionLatency(now, "/fak/repo", budget, samples)
		if rep.UnknownClockCount != 1 {
			t.Errorf("UnknownClockCount = %d, want 1", rep.UnknownClockCount)
		}
		// Latencies slice must only contain 80.0, NOT 0.0!
		if rep.P50Seconds != 80.0 || rep.P95Seconds != 80.0 || rep.MaxSeconds != 80.0 {
			t.Errorf("unexpected latencies: p50=%v p95=%v max=%v, want 80.0", rep.P50Seconds, rep.P95Seconds, rep.MaxSeconds)
		}
	})
}

type latencyFakeRunner struct {
	statusOut   string
	ignoredOut  string
	forRefsOut  string
	diffTreeOut string
	worktreeOut string
	logOut      string
	statusErr   error
	forRefsErr  error
	diffTreeErr error
	worktreeErr error
	logErr      error
}

func (r *latencyFakeRunner) Run(dir string, args ...string) ([]byte, error) {
	if len(args) == 0 {
		return nil, errors.New("empty args")
	}
	switch args[0] {
	case "ls-files":
		return []byte(r.ignoredOut), nil
	case "status":
		if r.statusErr != nil {
			return nil, r.statusErr
		}
		return []byte(r.statusOut), nil
	case "for-each-ref":
		if r.forRefsErr != nil {
			return nil, r.forRefsErr
		}
		return []byte(r.forRefsOut), nil
	case "diff-tree":
		if r.diffTreeErr != nil {
			return nil, r.diffTreeErr
		}
		return []byte(r.diffTreeOut), nil
	case "worktree":
		if r.worktreeErr != nil {
			return nil, r.worktreeErr
		}
		return []byte(r.worktreeOut), nil
	case "log":
		if r.logErr != nil {
			return nil, r.logErr
		}
		return []byte(r.logOut), nil
	default:
		return nil, errors.New("unexpected command: " + strings.Join(args, " "))
	}
}

type mockFileInfo struct {
	modTime time.Time
}

func (m mockFileInfo) Name() string       { return "mock" }
func (m mockFileInfo) Size() int64        { return 100 }
func (m mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m mockFileInfo) ModTime() time.Time { return m.modTime }
func (m mockFileInfo) IsDir() bool        { return false }
func (m mockFileInfo) Sys() any           { return nil }

func TestMeasureProtectionLatencyEndToEnd(t *testing.T) {
	now := time.Unix(1700005000, 0).UTC()
	budget := time.Hour

	// Status stream includes:
	// - tracked.go (checkpointed)
	// - fresh.go (unprotected, age 60s)
	// - stale.go (unprotected, age 7200s)
	// - deleted.go (marked D, must be excluded)
	// - vendor/dep.go (vendor, must be excluded)
	// - generated.pb.go (generated, must be excluded)
	// - ignored.go (ignored via ls-files, must be excluded)
	// - missing_clock.go (stat returns error)
	statusOut := strings.Join([]string{
		" M tracked.go",
		"?? fresh.go",
		"?? stale.go",
		" D deleted.go",
		"?? vendor/dep.go",
		"?? generated.pb.go",
		"?? ignored.go",
		"?? missing_clock.go",
	}, "\x00") + "\x00"

	ignoredOut := "ignored.go\x00vendor/\x00"

	// refs/fak/wip/session-1
	forRefsOut := "refs/fak/wip/session-1\x00sha123\x001700004950\n"
	diffTreeOut := "M\ttracked.go\n"

	// Git log includes landed.go
	logOut := "commit sha999\x001700004900\x001700004850\nA\tlanded.go\n"

	runner := &latencyFakeRunner{
		statusOut:   statusOut,
		ignoredOut:  ignoredOut,
		forRefsOut:  forRefsOut,
		diffTreeOut: diffTreeOut,
		logOut:      logOut,
		worktreeOut: "worktree /repo\nHEAD abc\nbranch refs/heads/main\n",
	}

	statMap := map[string]time.Time{
		"tracked.go": now.Add(-100 * time.Second),  // modified 1700004900, checkpointed at 1700004950 -> latency = 50s
		"fresh.go":   now.Add(-60 * time.Second),   // age = 60s
		"stale.go":   now.Add(-7200 * time.Second), // age = 7200s
	}

	statFn := func(p string) (os.FileInfo, error) {
		base := filepath.ToSlash(p)
		for k, mtime := range statMap {
			if strings.HasSuffix(base, k) {
				return mockFileInfo{modTime: mtime}, nil
			}
		}
		return nil, errors.New("file stat failed")
	}

	rep, err := MeasureProtectionLatency(context.Background(), "/fak/repo", LatencyOptions{
		Now:    now,
		Budget: budget,
		Runner: runner,
		Stat:   statFn,
	})
	if err != nil {
		t.Fatalf("MeasureProtectionLatency failed: %v", err)
	}

	if rep.Schema != ProtectionLatencySchema {
		t.Errorf("rep.Schema = %s, want %s", rep.Schema, ProtectionLatencySchema)
	}

	// Active source paths should be:
	// 1. fresh.go (unprotected)
	// 2. landed.go (landed)
	// 3. missing_clock.go (unprotected, unknown clock)
	// 4. stale.go (unprotected, stale)
	// 5. tracked.go (checkpointed)
	// Total = 5 (deleted, vendor, generated, ignored must NOT be in denominator!)
	if rep.TotalSourcePaths != 5 {
		t.Errorf("TotalSourcePaths = %d, want 5 (samples: %v)", rep.TotalSourcePaths, rep.PathSamples)
	}

	if rep.UnknownClockCount != 1 {
		t.Errorf("UnknownClockCount = %d, want 1", rep.UnknownClockCount)
	}

	if rep.StaleRefusalCount != 1 {
		t.Errorf("StaleRefusalCount = %d, want 1", rep.StaleRefusalCount)
	}

	if rep.Outcomes["checkpointed"] != 1 {
		t.Errorf("Outcomes[checkpointed] = %d, want 1", rep.Outcomes["checkpointed"])
	}
	if rep.Outcomes["landed"] != 1 {
		t.Errorf("Outcomes[landed] = %d, want 1", rep.Outcomes["landed"])
	}
	if rep.Outcomes["unprotected"] != 3 {
		t.Errorf("Outcomes[unprotected] = %d, want 3", rep.Outcomes["unprotected"])
	}

	if rep.SLOVerdict != "VIOLATION" {
		t.Errorf("SLOVerdict = %s, want VIOLATION", rep.SLOVerdict)
	}
}
