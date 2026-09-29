package workerworktree

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// These adversarial cases are authored from the #2857 SPEC. They drive the
// owner-aware sweep through its injected liveness probes only; they never read
// the real process table and never register a git worktree.

// TestExecWitnessSweepDeadOwnerReapedLiveKept is SPEC (public) item 1: the one
// owner-aware sweep collects the fak-exec-witness-* family too. An owner-named
// witness checkout whose creator is gone is removed under Apply; one whose
// creator is alive is kept.
func TestExecWitnessSweepDeadOwnerReapedLiveKept(t *testing.T) {
	parent := t.TempDir()
	start := time.Date(2026, 9, 29, 8, 0, 0, 0, time.UTC)
	token := strconv.FormatInt(start.UnixMilli(), 36)

	const deadPID, livePID = 4242, 4343
	dead := filepath.Join(parent, execWitnessCandidatePrefix+strconv.Itoa(deadPID)+"-"+token+"-1")
	live := filepath.Join(parent, execWitnessCandidatePrefix+strconv.Itoa(livePID)+"-"+token+"-1")
	for _, dir := range []string{dead, live} {
		if err := os.Mkdir(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	opts := CandidateSweepOptions{
		Now:          time.Now(),
		Apply:        true,
		ProcessAlive: func(pid int) bool { return pid == livePID },
		ProcessStart: func(pid int) (time.Time, bool) {
			if pid == livePID {
				return start, true
			}
			return time.Time{}, false
		},
	}
	report := SweepTopologyCandidates(parent, parent, noGit, opts)
	if report.Reaped != 1 || report.WouldReap != 1 || len(report.Failures) != 0 {
		t.Fatalf("sweep did not reap exactly the dead-owner witness checkout: %+v", report)
	}
	if len(report.Candidates) != 2 {
		t.Fatalf("want both witness checkouts classified, got %d: %+v", len(report.Candidates), report.Candidates)
	}
	if _, err := os.Stat(dead); !os.IsNotExist(err) {
		t.Fatalf("dead-owner witness checkout survived apply: %v", err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("live-owner witness checkout was removed: %v", err)
	}
	reasons := map[string]string{}
	for _, item := range report.Candidates {
		reasons[item.Path] = item.Reason
	}
	if reasons[live] != "owner_process_live" {
		t.Fatalf("live witness reason = %q; want owner_process_live", reasons[live])
	}
	if reasons[dead] != "owner_exited" {
		t.Fatalf("dead witness reason = %q; want owner_exited", reasons[dead])
	}
}

// TestParseTopologyCandidateOwnerCoversBothFamilies is SPEC (public) item 2:
// the owner parser reads BOTH the land-verify and the exec-witness prefixes, and
// rejects a bare/legacy name.
func TestParseTopologyCandidateOwnerCoversBothFamilies(t *testing.T) {
	start := int64(1234567890123)
	token := strconv.FormatInt(start, 36)
	for _, prefix := range []string{topologyCandidatePrefix, execWitnessCandidatePrefix} {
		name := prefix + "31337-" + token + "-9"
		owner, ok := parseTopologyCandidateOwner(name)
		if !ok {
			t.Fatalf("%q (prefix %q) did not parse", name, prefix)
		}
		if owner.PID != 31337 || owner.Start != start {
			t.Fatalf("%q parsed owner = %+v; want PID 31337 Start %d", name, owner, start)
		}
	}
	for _, bad := range []string{
		execWitnessCandidatePrefix + token,                  // bare legacy witness suffix
		topologyCandidatePrefix + token,                     // bare legacy land-verify suffix
		execWitnessCandidatePrefix + "31337-" + token,       // missing random chunk
		execWitnessCandidatePrefix + "31337-" + token + "-", // empty random chunk
		execWitnessCandidatePrefix + "0-" + token + "-1",    // non-positive pid
		execWitnessCandidatePrefix + "-" + token + "-1",     // missing pid
		"fak-other-prefix-31337-" + token + "-1",            // foreign family
	} {
		if owner, ok := parseTopologyCandidateOwner(bad); ok {
			t.Fatalf("%q parsed as owner-named (%+v); want rejection", bad, owner)
		}
	}
}

// TestOwnerNamedScratchPatternExecWitnessRoundTrip is SPEC (public) item 3: the
// witness pattern carries the family prefix, and filling its MkdirTemp "*" slot
// yields a name the owner parser round-trips.
func TestOwnerNamedScratchPatternExecWitnessRoundTrip(t *testing.T) {
	pattern := OwnerNamedScratchPattern(execWitnessCandidatePrefix)
	if !strings.HasPrefix(pattern, execWitnessCandidatePrefix) {
		t.Fatalf("pattern %q does not carry prefix %q", pattern, execWitnessCandidatePrefix)
	}
	if !strings.Contains(pattern, "*") {
		t.Fatalf("pattern %q has no MkdirTemp random slot", pattern)
	}
	filled := strings.Replace(pattern, "*", "abc", 1)
	owner, ok := parseTopologyCandidateOwner(filled)
	if !ok {
		t.Fatalf("filled pattern %q did not round-trip", filled)
	}
	if owner.PID != os.Getpid() {
		t.Fatalf("filled pattern owner PID = %d; want this process %d", owner.PID, os.Getpid())
	}
}
