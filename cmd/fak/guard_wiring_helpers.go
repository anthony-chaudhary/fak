package main

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// abortChildWiring aborts a launch whose child wiring could not be completed. By the time the
// hook/extension installers run the gateway is already up, so every one of these failures has
// to end the same way: tear the gateway's context down FIRST, then say which setup died, then
// exit. Routing them through one abort is what keeps a half-wired child from ever being
// spawned against a gateway that outlives the failure. what names the setup in the message
// ("Claude Stop hook setup", "Pi extension setup", ...) and code is its exit status.
func abortChildWiring(cancel context.CancelFunc, what string, err error, code int) {
	cancel()
	fmt.Fprintf(os.Stderr, "fak guard: %s failed: %v\n", what, err)
	os.Exit(code)
}

// guardSharedHookSettingsPath resolves the ONE `--settings` file every guard hook installer
// converges on. The installers run in a fixed order (PreCompact, Stop, toolproc, SessionStart):
// the first one enabled creates the file and injects `--settings`, and each later one
// read-modify-writes that same file instead of writing a second.
//
// Convergence therefore has to be resolved from what the earlier installers CREATED — which is
// what their install records report — not from what they were OFFERED. SessionStart used to be
// handed toolproc's INPUT path, so with PreCompact and Stop both off it saw an empty path even
// though toolproc had just created a file: it wrote a SECOND settings file and appended a second
// `--settings`. Claude resolves `--settings` last-wins, so the fold in appendClaudeSettingsArg
// (#5510) strips the earlier occurrence and refuses its `hooks` key with SETTINGS_HOOKS_DROPPED
// — the argv still carries exactly one `--settings`, but the child starts with toolproc's three
// observation hooks missing (#5526).
//
// Records are consulted newest-installer-first: an installer that MERGED records the file it
// merged into, so the latest non-empty SettingsPath always names the converged file.
func guardSharedHookSettingsPath(toolproc guardToolprocInstall, stop guardStopHookInstall, preCompact guardPreCompactInstall) string {
	for _, path := range []string{toolproc.SettingsPath, stop.SettingsPath, preCompact.SettingsPath} {
		if strings.TrimSpace(path) != "" {
			return path
		}
	}
	return ""
}
