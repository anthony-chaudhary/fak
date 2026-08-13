package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// decodeAutoCheckpoint runs `fak wip autocheckpoint --json` and returns the parsed result.
func decodeAutoCheckpoint(t *testing.T, args ...string) wipAutoCheckpointResult {
	t.Helper()
	var out, errout bytes.Buffer
	code := runWipAutoCheckpoint(&out, &errout, args)
	var got wipAutoCheckpointResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode %q (code=%d stderr=%s): %v", out.String(), code, errout.String(), err)
	}
	return got
}

// TestWipAutoCheckpointDeadlineIsCountableNotSilent pins the Stop-hook boundary contract:
// an unmeetable deadline must degrade to a SKIP, not to a hang and not to a blocked exit.
//
// Why this matters at all: the capture seeds a fresh temp index with `read-tree HEAD`, so it
// carries no stat cache and `add -A` re-hashes the whole worktree. Measured on the reference
// box (2026-08-05, n=5, 12,227 tracked files / 44.5 MB): mean 1.33s and ~64,757 page faults
// per checkpoint, once per turn per session. Before the deadline this ran on
// context.Background() -- unbounded -- at exactly the boundary where the host is most likely
// to be stalling, and the Stop hook discards its output, so a slow host produced an invisible
// open-ended stall.
func TestWipAutoCheckpointDeadlineIsCountableNotSilent(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := decodeAutoCheckpoint(t,
		"-C", dir, "--session", "deadlinesess", "--reason", "stop", "--json",
		"--timeout", "1ns", // unmeetable by construction: no git process starts this fast
	)
	if got.Captured {
		t.Fatalf("1ns deadline still captured: %+v", got)
	}
	// Spelled apart from capture-error on purpose. A deadline says the HOST was too slow to
	// walk the tree; capture-error says the capture is broken. Collapsing them would read
	// host pressure as a fak defect -- and would make the skip uncountable, which is the
	// same fail-open-SILENTLY defect the host-churn gate had.
	if got.Skipped != "capture-timeout" {
		t.Fatalf("want skipped=capture-timeout, got %q (error=%q)", got.Skipped, got.Error)
	}
	if !strings.Contains(got.Error, "exceeded") {
		t.Fatalf("timeout error should name the budget, got %q", got.Error)
	}
}

// TestWipAutoCheckpointDeadlineFailsOpen holds the posture the Stop hook depends on: the
// boundary is never blocked by a slow capture. runGuardStopHook calls this with its output
// discarded and its exit ignored, so a non-zero here would be silently swallowed today --
// but --strict callers must still be able to see it.
func TestWipAutoCheckpointDeadlineFailsOpen(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	base := []string{"-C", dir, "--session", "opensess", "--reason", "stop", "--timeout", "1ns"}

	var out, errout bytes.Buffer
	if code := runWipAutoCheckpoint(&out, &errout, base); code != 0 {
		t.Fatalf("best-effort timeout must exit 0, got %d (%s)", code, errout.String())
	}
	if !strings.Contains(out.String(), "capture-timeout") {
		t.Fatalf("skip should be legible on stdout, got %q", out.String())
	}

	out.Reset()
	errout.Reset()
	if code := runWipAutoCheckpoint(&out, &errout, append(append([]string{}, base...), "--strict")); code != 1 {
		t.Fatalf("--strict timeout must exit 1, got %d (%s)", code, errout.String())
	}
}

// TestWipAutoCheckpointDefaultDeadlineCapturesNormally guards against a too-tight default: the
// 30s budget is ~22x the measured 1.33s mean, so an ordinary capture must be untouched by it.
// If a future change tightens the default toward the mean, this fails rather than silently
// turning the WIP safety net off.
func TestWipAutoCheckpointDefaultDeadlineCapturesNormally(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := decodeAutoCheckpoint(t, "-C", dir, "--session", "defaultsess", "--reason", "stop", "--json")
	if !got.Captured {
		t.Fatalf("default deadline blocked an ordinary capture: %+v", got)
	}
}

// TestWipAutoCheckpointZeroTimeoutDisablesDeadline keeps the pre-deadline behaviour reachable
// for a caller that genuinely wants an unbounded capture.
func TestWipAutoCheckpointZeroTimeoutDisablesDeadline(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := decodeAutoCheckpoint(t,
		"-C", dir, "--session", "unboundedsess", "--reason", "stop", "--json", "--timeout", "0")
	if !got.Captured {
		t.Fatalf("--timeout 0 should disable the deadline, got %+v", got)
	}
}
