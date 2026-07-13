package main

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestRunBoundedGHCmdKillsWedgedGH proves the #3470 bound: a gh subprocess that never
// returns is killed at the deadline and reported as a FAILED run carrying a timeout note,
// instead of hanging the Stop hook that spawns it. It reuses the blocking TestHelperProcess
// seam (which Syncs then sleeps 10 minutes) as a stand-in for a wedged gh, and fails loudly
// via an outer watchdog if the bound is not enforced.
func TestRunBoundedGHCmdKillsWedgedGH(t *testing.T) {
	const bound = 150 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")

	type res struct {
		out, errb string
		ok        bool
	}
	done := make(chan res, 1)
	start := time.Now()
	go func() {
		out, errb, ok := runBoundedGHCmd(ctx, cmd, bound)
		done <- res{out, errb, ok}
	}()

	select {
	case r := <-done:
		if r.ok {
			t.Fatalf("want ok=false on deadline kill, got ok=true (out=%q err=%q)", r.out, r.errb)
		}
		if !strings.Contains(r.errb, "timed out") {
			t.Fatalf("want a timeout note in stderr, got %q", r.errb)
		}
		if elapsed := time.Since(start); elapsed > 10*time.Second {
			t.Fatalf("deadline enforcement too slow (%s) — the bound is not tight", elapsed)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("runBoundedGHCmd never returned after the deadline — the gh subprocess is unbounded (#3470 regression)")
	}
}

// TestRunBoundedGHCmdReportsSuccess is the non-deadline control: a command that exits 0
// well within the bound returns ok=true with no timeout note, so the classifier does not
// paint an ordinary success as a timeout. `-test.run=^$` matches no tests, so the re-exec'd
// test binary exits 0 promptly.
func TestRunBoundedGHCmdReportsSuccess(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^$")

	_, errb, ok := runBoundedGHCmd(ctx, cmd, 30*time.Second)
	if !ok {
		t.Fatalf("fast command should report ok=true, got ok=false (err=%q)", errb)
	}
	if strings.Contains(errb, "timed out") {
		t.Fatalf("no timeout note expected on a fast success, got %q", errb)
	}
}
