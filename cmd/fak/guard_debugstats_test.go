package main

import "testing"

// TestGuardDebugStatsToSharedStderr pins the per-turn economy-line routing decision: the
// `fak-turn …` line must NOT stream to the shared terminal stderr when an attended
// interactive agent (Claude Code) owns this terminal with a full-screen TUI — there a
// per-turn stderr write corrupts the agent's UI, and the economy belongs in the `fak info`
// pane + exit summary. An explicit --debug-stats opt-in still streams; a headless/piped run
// keeps streaming (no TUI to corrupt); --debug-stats=false / --quiet silence everything.
func TestGuardDebugStatsToSharedStderr(t *testing.T) {
	cases := []struct {
		name                               string
		debugStats, quiet, userSet         bool
		stdinInteractive, childInteractive bool
		want                               bool
	}{
		// Silenced regardless of everything else.
		{"off", false, false, false, true, true, false},
		{"quiet wins over on", true, true, false, false, false, false},
		{"quiet wins over explicit", true, true, true, false, false, false},

		// The headline fix: default-on + attended interactive child sharing the terminal
		// auto-suppresses the shared-stderr stream (the corruption case).
		{"default interactive attended -> suppress", true, false, false, true, true, false},

		// Explicit --debug-stats is a knowing opt-in: stream even over an interactive child.
		{"explicit interactive attended -> stream", true, false, true, true, true, true},

		// No full-screen TUI to corrupt -> keep the stream.
		{"default headless stdin -> stream", true, false, false, false, true, true},
		{"default non-interactive child (-p) -> stream", true, false, false, true, false, true},
		{"default fully headless -> stream", true, false, false, false, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardDebugStatsToSharedStderr(tc.debugStats, tc.quiet, tc.userSet, tc.stdinInteractive, tc.childInteractive)
			if got != tc.want {
				t.Errorf("guardDebugStatsToSharedStderr(debug=%v quiet=%v userSet=%v stdin=%v child=%v) = %v, want %v",
					tc.debugStats, tc.quiet, tc.userSet, tc.stdinInteractive, tc.childInteractive, got, tc.want)
			}
		})
	}
}
