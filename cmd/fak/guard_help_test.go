package main

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The `fak guard -h` anti-wall ratchet — mirrors help_test.go's ratchet for
// the top-level dispatcher, pointed at the front-door verb's own flag set.
// `cmdGuard` calls os.Exit(0) on -h (flag.ExitOnError's built-in help
// short-circuit), so exercising the real flag registration means re-exec'ing
// this test binary as a subprocess — the same pattern TestLoopRunHelper and
// TestClaudeMacFakSSHHelperProcess already use in this package.

// TestGuardHelpHelperProcess is the re-exec target: with GUARD_HELP_HELPER=1
// it calls the real cmdGuard with the argv from GUARD_HELP_ARGS, so its -h
// path runs (and exits) against the actual, live flag set — not a copy that
// could drift from it. Run bare (the normal `go test` pass), it instead
// asserts guardArgvHasAll's pure contract in-process, so this is never a
// silent no-op test.
func TestGuardHelpHelperProcess(t *testing.T) {
	if os.Getenv("GUARD_HELP_HELPER") != "1" {
		if !guardArgvHasAll([]string{"-h", "-all"}) {
			t.Fatal("guardArgvHasAll([-h -all]) = false, want true")
		}
		if guardArgvHasAll([]string{"-h"}) {
			t.Fatal("guardArgvHasAll([-h]) = true, want false")
		}
		return
	}
	cmdGuard(strings.Fields(os.Getenv("GUARD_HELP_ARGS")))
}

// runGuardHelp re-execs the test binary as `fak guard <args>` and captures
// combined stdout+stderr (Usage writes to os.Stderr).
func runGuardHelp(t *testing.T, args string) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestGuardHelpHelperProcess")
	cmd.Env = append(os.Environ(), "GUARD_HELP_HELPER=1", "GUARD_HELP_ARGS="+args)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	_ = cmd.Run()
	return out.String()
}

// TestGuardHelpOverviewStaysCompact is the anti-wall ratchet: `fak guard -h`
// used to dump all 64 flags alphabetically with paragraph-length internal
// narrative. It must now fit on one screen and stay free of issue-number
// citations and internal-subsystem jargon — that detail is real, just one
// `fak guard -h -all` away.
func TestGuardHelpOverviewStaysCompact(t *testing.T) {
	out := runGuardHelp(t, "-h")
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) > 20 {
		t.Fatalf("`fak guard -h` is %d lines; the budget is 20 — trim guardCommonFlags, don't grow the wall back:\n%s", len(lines), out)
	}
	if !strings.Contains(out, "-all") {
		t.Errorf("curated overview must point to the `-h -all` escape hatch; got:\n%s", out)
	}
	for _, tok := range []string{"#1", "#2", "#5", "#7", "#9"} {
		if strings.Contains(out, tok) {
			t.Errorf("curated overview leaks an issue-number citation %q, the jargon this pass exists to hide:\n%s", tok, out)
		}
	}
}

// TestGuardCommonFlagsAreLive pins every curated overview entry to the real,
// live flag set (via `-h -all`, the unabridged fs.PrintDefaults() wall), so
// the curated list can never advertise a flag `fak guard` no longer accepts.
func TestGuardCommonFlagsAreLive(t *testing.T) {
	full := runGuardHelp(t, "-h -all")
	if n := strings.Count(full, "\n  -"); n < 50 {
		t.Fatalf("`fak guard -h -all` only lists %d flags; expected the full ~64-flag reference:\n%s", n, full)
	}
	for _, f := range guardCommonFlags {
		if !strings.Contains(full, "\n  -"+f.name+" ") && !strings.Contains(full, "\n  -"+f.name+"\n") {
			t.Errorf("curated overview advertises --%s but the live flag reference has no such flag", f.name)
		}
	}
}
