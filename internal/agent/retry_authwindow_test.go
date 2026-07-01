package agent

import (
	"testing"
	"time"
)

// retry_authwindow_test.go — the #1834 witness for the reactive-path backstop: a headless
// `fak accounts launch` has no interactive `claude` process rewriting .credentials.json, so
// a re-login/refresh landing on disk after a 401 realistically takes longer than the OLD
// defaultAuthRefreshWindow of 3s to show up — every headless 401 was timing this window out
// by construction, surfacing a raw upstream_unauthorized instead of self-healing. This test
// pins the fixed floor directly: it FAILS against the pre-#1834 3s default and PASSES at the
// widened 10s default, so the constant can never quietly regress back below a usable window.

// TestAuthRefreshWindow_DefaultMeetsHeadlessFloor proves the unconfigured default is wide
// enough to plausibly observe a headless re-login landing (#1834's raised backstop), not the
// pre-fix 3s value that a headless launch could never realistically clear.
func TestAuthRefreshWindow_DefaultMeetsHeadlessFloor(t *testing.T) {
	t.Setenv("FAK_AUTH_REFRESH_WINDOW", "") // exercise the unconfigured default path
	const headlessFloor = 10 * time.Second
	if got := authRefreshWindow(); got < headlessFloor {
		t.Fatalf("authRefreshWindow() default = %s, want >= %s (the pre-#1834 3s default left every headless launch's 401 wait unable to observe a realistic re-login)", got, headlessFloor)
	}
}

// TestAuthRefreshWindow_EnvOverrideStillClampsToCeiling proves the operator escape hatch
// (FAK_AUTH_REFRESH_WINDOW) introduced alongside the default bump still honors the existing
// maxAuthRefreshWindow ceiling rather than letting a fat-fingered huge value wedge a turn.
func TestAuthRefreshWindow_EnvOverrideStillClampsToCeiling(t *testing.T) {
	t.Setenv("FAK_AUTH_REFRESH_WINDOW", "1h")
	if got := authRefreshWindow(); got != maxAuthRefreshWindow {
		t.Fatalf("authRefreshWindow() with FAK_AUTH_REFRESH_WINDOW=1h = %s, want clamped to ceiling %s", got, maxAuthRefreshWindow)
	}
}
