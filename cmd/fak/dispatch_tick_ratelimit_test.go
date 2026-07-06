package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// writeRateLimitWitnessFixture seeds one .witness sidecar (and its sibling .backend) so
// dispatchPreflightRateLimit can be exercised against a controlled runs dir. ageMinutes>0
// back-dates the sidecar mtime so the window filter can be tested.
func writeRateLimitWitnessFixture(t *testing.T, runsDir, name, backend, claim, reason string, ageMinutes int) {
	t.Helper()
	stem := filepath.Join(runsDir, name)
	doc := `{"issue":1,"log":"` + name + `.log","sha":null,"claim":"` + claim + `","verdict":null,"witness":null`
	if reason != "" {
		doc += `,"reason":"` + reason + `"`
	}
	doc += `}`
	wf := stem + dispatchtick.WitnessSidecarSuffix
	if err := os.WriteFile(wf, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	if backend != "" {
		if err := os.WriteFile(stem+".backend", []byte(backend), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if ageMinutes > 0 {
		ts := time.Now().Add(-time.Duration(ageMinutes) * time.Minute)
		if err := os.Chtimes(wf, ts, ts); err != nil {
			t.Fatal(err)
		}
	}
}

// TestDispatchPreflightRateLimitDisambiguatesAndScopes proves the load-bearing
// disambiguation: only GENUINE concurrency 429s (reason=rate_limit) on THIS backend within
// the window are counted. Fake 429s -- a weekly/usage cap, a model cap, a login wall -- and
// other backends' and aged exits are excluded, so a weekly-cap burst never freezes the
// fleet when the other providers are healthy.
func TestDispatchPreflightRateLimitDisambiguatesAndScopes(t *testing.T) {
	t.Setenv("FAK_RATELIMIT_WINDOW", "15m")
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// 3 genuine claude concurrency 429s within the window -> counted.
	writeRateLimitWitnessFixture(t, runsDir, "resolve-11-20260706-120000", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 0)
	writeRateLimitWitnessFixture(t, runsDir, "resolve-12-20260706-120100", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 0)
	writeRateLimitWitnessFixture(t, runsDir, "resolve-13-20260706-120200", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 0)
	// FAKE 429s that MUST be excluded (the disambiguation): weekly cap, model cap, login wall.
	writeRateLimitWitnessFixture(t, runsDir, "resolve-14-20260706-120300", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitUsageCap, 0)
	writeRateLimitWitnessFixture(t, runsDir, "resolve-15-20260706-120400", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitModelUnknown, 0)
	writeRateLimitWitnessFixture(t, runsDir, "resolve-16-20260706-120500", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitAuthWall, 0)
	// A witnessed, diff-witnessed commit (not a no-commit) -> not counted.
	writeRateLimitWitnessFixture(t, runsDir, "resolve-19-20260706-120700", "claude", dispatchtick.ClaimWitnessed, "", 0)
	// A different backend's 429 -> excluded when scoped to claude.
	writeRateLimitWitnessFixture(t, runsDir, "resolve-17-20260706-120600", "opencode", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 0)
	// An aged claude 429 (30m old) -> outside the 15m window.
	writeRateLimitWitnessFixture(t, runsDir, "resolve-18-20260706-100000", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 30)

	got := dispatchPreflightRateLimit(root, "claude")
	if got.Recent != 3 {
		t.Fatalf("claude rate_limit count = %d, want 3 (fake 429s, other backends, and aged exits excluded)", got.Recent)
	}
	if got.Threshold != dispatchtick.DefaultRateLimitMin429 {
		t.Fatalf("threshold = %d, want default %d", got.Threshold, dispatchtick.DefaultRateLimitMin429)
	}
	// The opencode backend sees only its own single rate_limit exit.
	if oc := dispatchPreflightRateLimit(root, "opencode"); oc.Recent != 1 {
		t.Fatalf("opencode rate_limit count = %d, want 1", oc.Recent)
	}
}

// TestDispatchPreflightRateLimitWindowOffDisables proves the kill switch: an operator can
// turn the term off entirely, yielding the zero-value no-op fold.
func TestDispatchPreflightRateLimitWindowOffDisables(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeRateLimitWitnessFixture(t, runsDir, "resolve-21-20260706-120000", "claude", dispatchtick.ClaimNoCommit, dispatchtick.NoCommitRateLimit, 0)
	t.Setenv("FAK_RATELIMIT_WINDOW", "off")
	got := dispatchPreflightRateLimit(root, "claude")
	if got.Recent != 0 || got.Window != 0 {
		t.Fatalf("window=off must disable the term: got Recent=%d Window=%v, want 0/0", got.Recent, got.Window)
	}
}
