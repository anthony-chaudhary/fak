package main

import (
	"strings"
	"testing"
)

// Issue #5482 — the Apple Terminal companion overlay window outlived its process. The old
// spawn appended a bare `; exit` and the comment claimed the window would then "close itself
// ... with the default close-on-clean-exit profile". Apple Terminal's stock Basic profile ships
// `When the shell exits: Don't close the window`, so there is no such default to lean on: the
// overlay process exited on time (the #2340 --max-idle backstop), the shell exited, and the
// window stayed parked on `[Process completed]` — one dead window per guarded session,
// accumulating monotonically (~100 observed across two days, ~84% of the app's windows).
//
// These tests pin the SPAWN PLAN (a pure string/argv builder), which is the only part of this
// that is checkable without a Mac: that the plan emits an EXPLICIT close, that the close is
// keyed to the window the overlay owns (never a re-resolved `front window`), that it is gated
// on a clean exit so a crash stays readable, that it survives the AppleScript/shell quoting
// with a hostile argument, and that no non-Apple split host is touched by any of it. The
// end-to-end behavior (does Terminal actually close that window, and without a "terminate
// running processes?" prompt) is NOT witnessed here and needs a Mac operator.

// appleScriptUnquote is the inverse of appleScriptQuote: it recovers the text an AppleScript
// string literal actually denotes, so a test can assert on the SHELL LINE Terminal types into
// the companion window rather than on its escaped spelling.
func appleScriptUnquote(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			i++
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

// appleTerminalDoScriptLine extracts the `do script "..."` payload from an Apple Terminal spawn
// argv and returns the shell line it denotes. It fails if any `"` inside the literal is not
// backslash-escaped: an unescaped quote would END the AppleScript string early and hand the
// remainder to the AppleScript parser as CODE — the injection seam an overlay argument carrying
// a quote could otherwise open.
func appleTerminalDoScriptLine(t *testing.T, spawn []string) string {
	t.Helper()
	const prefix = `do script "`
	for _, arg := range spawn {
		if !strings.HasPrefix(arg, prefix) {
			continue
		}
		if !strings.HasSuffix(arg, `"`) || len(arg) <= len(prefix) {
			t.Fatalf("`do script` argument is not a closed string literal: %s", arg)
		}
		lit := arg[len(prefix) : len(arg)-1]
		for i := 0; i < len(lit); i++ {
			switch lit[i] {
			case '\\':
				i++ // an escape consumes the next byte, whatever it is
			case '"':
				t.Fatalf("unescaped double quote at byte %d ends the AppleScript literal early: %s", i, lit)
			}
		}
		return appleScriptUnquote(lit)
	}
	t.Fatalf("no `do script` argument in spawn: %v", spawn)
	return ""
}

func appleTerminalPlan(t *testing.T, env map[string]string, overlay []string) guardSplitPlan {
	t.Helper()
	env["TERM_PROGRAM"] = "Apple_Terminal"
	plan, err := buildGuardSplitPlan("darwin", envFunc(env), lookPathOK, "fak", "bottom", overlay)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Host != "terminal-app" {
		t.Fatalf("host = %q, want terminal-app", plan.Host)
	}
	return plan
}

// TestGuardSplitAppleTerminalClosesItsOwnWindow is the #5482 witness: the spawn must carry an
// EXPLICIT close of the companion window, instead of assuming a profile setting the operator
// never chose.
func TestGuardSplitAppleTerminalClosesItsOwnWindow(t *testing.T) {
	plan := appleTerminalPlan(t, map[string]string{}, guardOverlayArgs())
	line := appleTerminalDoScriptLine(t, plan.Spawn)
	// Echo both under -v: the spawn argv and the shell line Terminal types are the whole
	// reviewable artifact of a fix whose end-to-end behavior needs a Mac to see.
	t.Logf("spawn argv:\n%q", plan.Spawn)
	t.Logf("typed shell line:\n%s", line)

	// The overlay command still runs FIRST and unchanged — the closer is a tail, so a shell that
	// cannot parse the tail (tcsh) still runs the dashboard.
	const overlay = "fak info --gateway-url http://127.0.0.1:5000 --interval 2s --max-idle 5m0s"
	if !strings.HasPrefix(line, overlay+";") {
		t.Fatalf("the overlay command must lead the typed line unchanged, got:\n%s", line)
	}
	// An explicit close, addressed to Terminal, of a window selected by the overlay's own tty.
	for _, want := range []string{
		"osascript",
		`tell application "Terminal" to close (every window whose tty of tab 1 is`,
		"$(tty)",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("typed line is missing %q — the window is not explicitly closed:\n%s", want, line)
		}
	}
	// The shell still exits either way, so an opted-out / failed close degrades to the old
	// `[Process completed]` window rather than to a live shell nobody reaps.
	if !strings.Contains(line, "exit") {
		t.Fatalf("the typed line must still exit the shell:\n%s", line)
	}
}

// TestGuardSplitAppleTerminalCloserCannotMisTarget pins the correctness trap the ticket
// calls out: the spawn script resolves `front window` (to re-front the AGENT window right after
// `do script` steals focus), but the CLOSER runs minutes-to-hours later, when the front window
// is whatever the operator has since focused. Closing that would destroy a live terminal — far
// worse than leaking a dead one. The close must therefore key on the overlay's own tty and must
// never re-resolve a positional window.
func TestGuardSplitAppleTerminalCloserCannotMisTarget(t *testing.T) {
	line := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{}, guardOverlayArgs()).Spawn)
	for _, forbidden := range []string{"front window", "window 1", "first window", "current window"} {
		if strings.Contains(line, forbidden) {
			t.Fatalf("the closer must not select a window positionally (%q) — it would close whatever the operator focused:\n%s", forbidden, line)
		}
	}
	if !strings.Contains(line, "whose tty of tab 1 is") {
		t.Fatalf("the closer must select the window by the overlay's own tty:\n%s", line)
	}
}

// TestGuardSplitAppleTerminalCloseGatedOnCleanExit pins the second correctness trap: if the overlay dies
// with a real error, auto-closing the window destroys the only copy of the diagnostic. The close
// must sit behind the overlay's exit status, and the status must be captured IMMEDIATELY after
// the overlay (any command in between clobbers `$?`).
func TestGuardSplitAppleTerminalCloseGatedOnCleanExit(t *testing.T) {
	const overlay = "fak info --gateway-url http://127.0.0.1:5000 --interval 2s --max-idle 5m0s"
	line := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{}, guardOverlayArgs()).Spawn)
	tail := strings.TrimPrefix(line, overlay+"; ")
	if tail == line {
		t.Fatalf("expected the closer to be appended after %q, got:\n%s", overlay, line)
	}
	if !strings.HasPrefix(tail, "__fak_rc=$?;") {
		t.Fatalf("the overlay's exit status must be captured immediately after it (before anything can clobber $?), got:\n%s", tail)
	}
	gate := strings.Index(tail, `if [ "$__fak_rc" = 0 ]; then`)
	closeAt := strings.Index(tail, "close (every window")
	if gate < 0 || closeAt < 0 || closeAt < gate {
		t.Fatalf("the close must run only on a clean overlay exit (gate=%d close=%d):\n%s", gate, closeAt, tail)
	}
	if fi := strings.Index(tail, "fi"); fi < 0 || fi < closeAt {
		t.Fatalf("the clean-exit conditional must enclose the close:\n%s", tail)
	}
}

// TestGuardSplitAppleTerminalSurvivesHostileOverlayArgument pins the quoting: the overlay argv goes
// through shellJoin (for the shell that will re-parse it) and then appleScriptQuote (for the
// AppleScript literal that carries it). An argument containing a double quote and a backslash
// must come out the far end byte-identical, with no unescaped quote able to end the literal.
func TestGuardSplitAppleTerminalSurvivesHostileOverlayArgument(t *testing.T) {
	nasty := `http://127.0.0.1:5000/x"; do shell script "boom`
	overlay := []string{"info", "--gateway-url", nasty, "--interval", `2s\ `}
	plan := appleTerminalPlan(t, map[string]string{}, overlay)
	line := appleTerminalDoScriptLine(t, plan.Spawn) // fails on an unescaped quote in the literal

	want := shellJoin(append([]string{"fak"}, overlay...))
	if !strings.HasPrefix(line, want+";") {
		t.Fatalf("the shell line must carry the overlay argv verbatim.\n got: %s\nwant prefix: %s", line, want+";")
	}
	// The hostile text must be inert: it survives only inside the shell-quoted overlay argument,
	// never as a second AppleScript statement.
	if strings.Contains(strings.TrimPrefix(line, want), "do shell script") {
		t.Fatalf("hostile argument text escaped into the appended closer:\n%s", line)
	}
}

// TestGuardSplitCloseOnExitOptOut pins the escape hatch (#5482 item 3): an operator who wants
// the dead window for post-mortem reading can restore the pre-fix behavior, and the default must
// actually differ from it (otherwise the knob — and the fix — would be vacuous).
func TestGuardSplitCloseOnExitOptOut(t *testing.T) {
	def := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{}, guardOverlayArgs()).Spawn)
	off := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{"FAK_SPLIT_CLOSE_ON_EXIT": "0"}, guardOverlayArgs()).Spawn)

	const legacy = "fak info --gateway-url http://127.0.0.1:5000 --interval 2s --max-idle 5m0s; exit"
	if off != legacy {
		t.Fatalf("opting out must reproduce the pre-#5482 lingering-window line.\n got: %s\nwant: %s", off, legacy)
	}
	if def == off {
		t.Fatalf("close-on-exit must be the DEFAULT — the default line is identical to the opted-out one:\n%s", def)
	}
	for _, off := range []string{"false", "off", "no", "0", " OFF "} {
		got := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{"FAK_SPLIT_CLOSE_ON_EXIT": off}, guardOverlayArgs()).Spawn)
		if got != legacy {
			t.Fatalf("FAK_SPLIT_CLOSE_ON_EXIT=%q must opt out, got:\n%s", off, got)
		}
	}
	// Anything else (including an unset var and a stray value) keeps the default: close.
	for _, on := range []string{"", "1", "true", "yes", "banana"} {
		got := appleTerminalDoScriptLine(t, appleTerminalPlan(t, map[string]string{"FAK_SPLIT_CLOSE_ON_EXIT": on}, guardOverlayArgs()).Spawn)
		if !strings.Contains(got, "close (every window") {
			t.Fatalf("FAK_SPLIT_CLOSE_ON_EXIT=%q must keep the default close, got:\n%s", on, got)
		}
	}
}

// TestGuardSplitNonAppleHostsUnaffectedByCloser is the no-regression pin: #5482 is an Apple
// Terminal-only defect (tmux/wt panes and iTerm2 sessions die WITH their process), so every
// other split host's argv must stay byte-identical and carry none of the closer.
func TestGuardSplitNonAppleHostsUnaffectedByCloser(t *testing.T) {
	for _, tc := range []struct {
		name string
		goos string
		env  map[string]string
		want []string
	}{
		{
			"tmux", "linux", map[string]string{"TMUX": "/tmp/tmux-1/default,1,0"},
			[]string{"tmux", "split-window", "-v", "-d", "-l", "20%", "--", "fak", "info", "--gateway-url", "http://127.0.0.1:5000", "--interval", "2s", "--max-idle", "5m0s"},
		},
		{
			"wt", "windows", map[string]string{"WT_SESSION": "abc-123"},
			[]string{"wt", "-w", "0", "split-pane", "-H", "-s", "0.2", "fak", "info", "--gateway-url", "http://127.0.0.1:5000", "--interval", "2s", "--max-idle", "5m0s", ";", "move-focus", "up"},
		},
		{
			"iterm2", "darwin", map[string]string{"TERM_PROGRAM": "iTerm.app"},
			[]string{"osascript", "-e", `tell application "iTerm2" to tell current session of current window to split horizontally with same profile command "fak info --gateway-url http://127.0.0.1:5000 --interval 2s --max-idle 5m0s"`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The opt-out knob must be inert here too: these hosts never grew the behavior.
			for _, closeOnExit := range []string{"", "0"} {
				env := map[string]string{"FAK_SPLIT_CLOSE_ON_EXIT": closeOnExit}
				for k, v := range tc.env {
					env[k] = v
				}
				plan, err := buildGuardSplitPlan(tc.goos, envFunc(env), lookPathOK, "fak", "bottom", guardOverlayArgs())
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if strings.Join(plan.Spawn, "\x00") != strings.Join(tc.want, "\x00") {
					t.Fatalf("spawn drifted (FAK_SPLIT_CLOSE_ON_EXIT=%q):\n got: %v\nwant: %v", closeOnExit, plan.Spawn, tc.want)
				}
				for _, forbidden := range []string{"__fak_rc", "close (every window", "$(tty)"} {
					if strings.Contains(strings.Join(plan.Spawn, " "), forbidden) {
						t.Fatalf("%s host must not carry the Apple Terminal closer (%q): %v", tc.name, forbidden, plan.Spawn)
					}
				}
			}
		})
	}
}
