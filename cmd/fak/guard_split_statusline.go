package main

import (
	"fmt"
	"strings"
)

// Issue #12304: `fak guard --split` / `fak opencode` on a single-pane Mac terminal with NO
// splittable multiplexer host previously went fully invisible — the operator got no live
// feedback at all, because the whole point of --split is the overlay pane and there was
// nowhere to put it. This module supplies the two wired pieces of the degraded-mode fallback:
// the kill switch (guardSplitBannerEnabled) that lets an operator opt back into the silent
// pane, and the best-effort macOS milestone notification (guardSplitMilestoneNotify) delivered
// through Notification Center. The guard/opencode launcher wires them at the call site;
// nothing here does real I/O in a render function, so every behavior below is deterministic
// and covered without touching the host.

// guardSplitBannerEnabled resolves the kill switch for the inline banner. Default ON: the
// whole issue is that the no-host degraded path had no feedback, so opting IN by default is
// the fix. FAK_SPLIT_BANNER=0|off|false|no restores the silent behavior for an operator who
// really wants a clean pane. Read through injected getenv so the knob stays inside the pure
// helper (the same shape guardSplitAppleCloseOnExit uses).
func guardSplitBannerEnabled(getenv func(string) string) bool {
	switch strings.TrimSpace(strings.ToLower(getenv("FAK_SPLIT_BANNER"))) {
	case "0", "off", "false", "no":
		return false
	}
	return true
}

// guardSplitMilestoneNotify raises an OS notification for a major milestone (session
// established, N calls adjudicated, the floor refusing at a notable rate). macOS only: the
// existing toast path is rwToast's osascript `display notification`. goos/lookPath/run are
// injected so a test can pin the platform, prove osascript presence, and capture the exact
// argv without spawning anything.
//
// It returns true IFF a notification was actually attempted. Everything else — a non-darwin
// host, or macOS without osascript — is a quiet false: this module deliberately does NOT
// invent Linux/Windows notification mechanisms, because an unproven platform guess is worse
// than an honest no-op.
func guardSplitMilestoneNotify(goos string, lookPath func(string) (string, error), run func(name string, args ...string) error, title, message string) bool {
	if goos != "darwin" {
		return false
	}
	osa, err := lookPath("osascript")
	if err != nil || strings.TrimSpace(osa) == "" {
		return false
	}
	// appleScriptQuote escapes the two characters AppleScript's double-quoted string literals
	// treat specially, so a hostile title/message (a tool name with a quote, a path with a
	// backslash) cannot close the literal and open a second AppleScript statement.
	script := fmt.Sprintf(`display notification "%s" with title "%s"`, appleScriptQuote(message), appleScriptQuote(title))
	_ = run(osa, "-e", script)
	return true
}
