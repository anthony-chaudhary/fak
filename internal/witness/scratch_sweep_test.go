// Package witness in-package tests for the exec-witness scratch sweep.
//
// Impl-Context: SESS-WORKER-T01-IMPL
// Test-Context: SESS-TEST-T01
// Separation-Verdict: SEPARATED
package witness

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// scratchTestMkdir creates a named directory under parent and returns its path.
func scratchTestMkdir(t *testing.T, parent, name string) string {
	t.Helper()
	dir := filepath.Join(parent, name)
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	return dir
}

// scratchTestAge backdates a directory's mtime by age.
func scratchTestAge(t *testing.T, path string, age time.Duration) {
	t.Helper()
	ts := time.Now().Add(-age)
	if err := os.Chtimes(path, ts, ts); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
}

// scratchTestName builds an owner-named scratch directory name.
func scratchTestName(pid int, start int64) string {
	return execWitnessScratchPrefix + strconv.Itoa(pid) + "-" + strconv.FormatInt(start, 36) + "-rnd"
}

func TestExecWitnessScratchClassifyOwner(t *testing.T) {
	const (
		deadPID = 2147483647
		livePID = 4242
	)
	liveStart := time.Now().UnixMilli()
	alive := func(pid int) bool { return pid != deadPID }
	matching := func(int) (time.Time, bool) { return time.UnixMilli(liveStart), true }
	unknown := func(int) (time.Time, bool) { return time.Time{}, false }
	reused := func(int) (time.Time, bool) { return time.UnixMilli(liveStart + 99), true }

	cases := []struct {
		name         string
		dir          string
		start        func(int) (time.Time, bool)
		wantEligible bool
		wantPID      int
		wantReason   string
	}{
		{"dead owner eligible", scratchTestName(deadPID, liveStart), matching, true, deadPID, "owner_exited"},
		{"live owner matches kept", scratchTestName(livePID, liveStart), matching, false, livePID, "owner_process_live"},
		{"live pid reused eligible", scratchTestName(livePID, liveStart), reused, true, livePID, "owner_pid_reused"},
		{"zero start kept", scratchTestName(livePID, 0), matching, false, livePID, "owner_process_live"},
		{"unknown start kept", scratchTestName(livePID, liveStart), unknown, false, livePID, "owner_process_live"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), tc.dir)
			eligible, pid, reason := classifyExecWitnessScratch(path, time.Now(), legacyExecWitnessMaxAge, alive, tc.start)
			if eligible != tc.wantEligible || pid != tc.wantPID || reason != tc.wantReason {
				t.Fatalf("classify=%v pid=%d reason=%q, want %v %d %q",
					eligible, pid, reason, tc.wantEligible, tc.wantPID, tc.wantReason)
			}
		})
	}
}

func TestExecWitnessScratchClassifyLegacy(t *testing.T) {
	alive := func(int) bool { return true }
	start := func(int) (time.Time, bool) { return time.Time{}, false }
	parent := t.TempDir()

	young := scratchTestMkdir(t, parent, execWitnessScratchPrefix+"123456")
	stale := scratchTestMkdir(t, parent, execWitnessScratchPrefix+"654321")
	scratchTestAge(t, stale, 3*time.Hour)

	now := time.Now()
	if eligible, _, reason := classifyExecWitnessScratch(young, now, legacyExecWitnessMaxAge, alive, start); eligible || reason != "legacy_too_young" {
		t.Fatalf("young legacy: eligible=%v reason=%q, want kept legacy_too_young", eligible, reason)
	}
	if eligible, _, reason := classifyExecWitnessScratch(stale, now, legacyExecWitnessMaxAge, alive, start); !eligible || reason != "legacy_stale" {
		t.Fatalf("stale legacy: eligible=%v reason=%q, want eligible legacy_stale", eligible, reason)
	}
}

func TestExecWitnessScratchPatternRoundTrips(t *testing.T) {
	pattern := execWitnessPattern()
	if !strings.HasPrefix(pattern, execWitnessScratchPrefix) || !strings.HasSuffix(pattern, "-*") {
		t.Fatalf("pattern %q lacks prefix or suffix", pattern)
	}
	name := strings.Replace(pattern, "*", "abc", 1)
	owner, ok := parseExecWitnessOwner(name)
	if !ok {
		t.Fatalf("parseExecWitnessOwner(%q) = !ok, want ok", name)
	}
	if owner.PID != os.Getpid() {
		t.Fatalf("parsed pid=%d, want %d", owner.PID, os.Getpid())
	}
	encoded := strconv.FormatInt(owner.Start, 36)
	if !strings.Contains(name, "-"+encoded+"-") {
		t.Fatalf("parsed start=%d (base36 %q) not re-encodable in %q", owner.Start, encoded, name)
	}
}

func TestExecWitnessScratchParseRejectsMalformed(t *testing.T) {
	bad := []string{
		execWitnessScratchPrefix + "2015920908",
		execWitnessScratchPrefix + "x-0-1",
		execWitnessScratchPrefix + "0-0-1",
		execWitnessScratchPrefix + "12-0-",
		execWitnessScratchPrefix + "12-!!-1",
		".fak-other-12-0-1",
	}
	for _, name := range bad {
		if _, ok := parseExecWitnessOwner(name); ok {
			t.Errorf("parseExecWitnessOwner(%q) = ok, want !ok", name)
		}
	}
	if _, ok := parseExecWitnessOwner(execWitnessScratchPrefix + "1234-1a-zzz"); !ok {
		t.Errorf("parseExecWitnessOwner rejected a well-formed owner name")
	}
}

func TestExecWitnessScratchDryRunRemovesNothing(t *testing.T) {
	parent := t.TempDir()
	const deadPID = 2147483647
	dead := scratchTestMkdir(t, parent, scratchTestName(deadPID, 12345))
	other := scratchTestMkdir(t, parent, ".fak-other-1")

	alive := func(pid int) bool { return pid != deadPID }
	start := func(int) (time.Time, bool) { return time.Time{}, false }

	wouldReap, reaped := sweepExecWitnessScratch(parent, time.Now(), false, 0, legacyExecWitnessMaxAge, alive, start)
	if wouldReap != 1 || reaped != 0 {
		t.Fatalf("dry-run: wouldReap=%d reaped=%d, want 1 0", wouldReap, reaped)
	}
	if _, err := os.Stat(dead); err != nil {
		t.Fatalf("dry-run removed eligible dir: %v", err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatalf("sweep touched a non-prefix dir: %v", err)
	}
}

func TestExecWitnessScratchLimitCapsReaping(t *testing.T) {
	parent := t.TempDir()
	const deadPID = 2147483647
	dirs := []string{
		scratchTestMkdir(t, parent, scratchTestName(deadPID, 111)),
		scratchTestMkdir(t, parent, scratchTestName(deadPID, 222)),
		scratchTestMkdir(t, parent, scratchTestName(deadPID, 333)),
	}
	alive := func(pid int) bool { return pid != deadPID }
	start := func(int) (time.Time, bool) { return time.Time{}, false }

	wouldReap, reaped := sweepExecWitnessScratch(parent, time.Now(), true, 2, legacyExecWitnessMaxAge, alive, start)
	if wouldReap != 3 || reaped != 2 {
		t.Fatalf("limit=2: wouldReap=%d reaped=%d, want 3 2", wouldReap, reaped)
	}
	remaining := 0
	for _, d := range dirs {
		if _, err := os.Stat(d); err == nil {
			remaining++
		}
	}
	if remaining != 1 {
		t.Fatalf("remaining=%d, want 1", remaining)
	}
}

// TestExecWitnessScratchKilledOwnerCollectedLiveKept is the DoD end-to-end case:
// a killed-owner scratch dir beside a temp repo root is collected; a live-pid
// one is kept.
func TestExecWitnessScratchKilledOwnerCollectedLiveKept(t *testing.T) {
	parent := t.TempDir()
	livePID := os.Getpid()
	const deadPID = 2147483647
	liveStart := time.Now().UnixMilli()

	dead := scratchTestMkdir(t, parent, scratchTestName(deadPID, 0))
	live := scratchTestMkdir(t, parent, scratchTestName(livePID, liveStart))

	alive := func(pid int) bool { return pid != deadPID }
	start := func(int) (time.Time, bool) { return time.UnixMilli(liveStart), true }

	wouldReap, reaped := sweepExecWitnessScratch(parent, time.Now(), true, 0, legacyExecWitnessMaxAge, alive, start)
	if wouldReap < 1 || reaped < 1 {
		t.Fatalf("wouldReap=%d reaped=%d, want >=1 each", wouldReap, reaped)
	}
	if _, err := os.Stat(dead); !os.IsNotExist(err) {
		t.Fatalf("killed-owner dir still present: err=%v", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("live-owner dir was removed: %v", err)
	}
}
