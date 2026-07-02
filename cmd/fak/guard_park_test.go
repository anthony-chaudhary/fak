package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// guard_park_test.go — the #2260 STALE_CRED park's witnesses. Every test drives the park
// with an injected clock/sleep so nothing here waits wall-clock time, and the credential
// file is a real temp file rotated mid-park — the exact recovery the fleet needs (a human
// runs `claude` once, the file gets a live token, the parked spawn proceeds).

// writeParkCred writes a minimal .credentials.json with the given expiresAt (unix millis;
// 0 = no expiry recorded) and returns its path.
func writeParkCred(t *testing.T, dir string, expiresAt int64) string {
	t.Helper()
	path := filepath.Join(dir, ".credentials.json")
	body := fmt.Sprintf(`{"claudeAiOauth":{"accessToken":"tok-park-test","expiresAt":%d}}`, expiresAt)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// parkClock is the injected time source: sleep advances the clock and runs an optional
// per-sleep hook (the test's chance to rotate the credential file mid-park).
type parkClock struct {
	t       time.Time
	sleeps  int
	onSleep func(n int)
}

func (c *parkClock) now() time.Time { return c.t }

func (c *parkClock) sleep(d time.Duration) {
	c.t = c.t.Add(d)
	c.sleeps++
	if c.onSleep != nil {
		c.onSleep(c.sleeps)
	}
}

// The recovery the park exists for: an expired credential rotates to a live one a few
// polls in, and the park reports Recovered with the honest elapsed wait — instead of the
// pre-#2260 behavior (a refusal inside 30s against a human-paced re-login).
func TestGuardParkForRelogin_RecoversOnRotation(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 2, 8, 0, 0, 0, time.UTC)
	cred := writeParkCred(t, dir, base.Add(-time.Hour).UnixMilli()) // expired an hour ago

	clk := &parkClock{t: base}
	clk.onSleep = func(n int) {
		if n == 3 { // the re-login lands during the third poll interval
			writeParkCred(t, dir, clk.t.Add(time.Hour).UnixMilli())
		}
	}
	var out bytes.Buffer
	poll := 2 * time.Minute
	res := guardParkForRelogin(cred, 24*time.Hour, poll, clk.now, clk.sleep, &out)
	if !res.Attempted || !res.Recovered {
		t.Fatalf("park = %+v, want Attempted+Recovered", res)
	}
	if want := 3 * poll; res.Elapsed != want {
		t.Fatalf("Elapsed = %s, want %s (3 polls)", res.Elapsed, want)
	}
	if !strings.Contains(out.String(), "parked — STALE_CRED") {
		t.Fatalf("park line missing from output:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "re-login landed after 6m0s") {
		t.Fatalf("recovery line missing/wrong:\n%s", out.String())
	}
}

// Budget exhaustion is a LOGGED give-up: the park runs the whole ceiling, reports
// Attempted-but-not-Recovered with the elapsed time, and says so on stderr — never a
// silent stop, mirroring the deny-all Stop-hook give-up discipline.
func TestGuardParkForRelogin_BudgetExhaustsLoudly(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 2, 8, 0, 0, 0, time.UTC)
	cred := writeParkCred(t, dir, base.Add(-time.Hour).UnixMilli()) // expired, never rotates

	clk := &parkClock{t: base}
	var out bytes.Buffer
	budget, poll := 10*time.Minute, 3*time.Minute
	res := guardParkForRelogin(cred, budget, poll, clk.now, clk.sleep, &out)
	if !res.Attempted || res.Recovered {
		t.Fatalf("park = %+v, want Attempted && !Recovered", res)
	}
	// 4 polls of 3m pass the 10m deadline; the loop then observes exhaustion.
	if res.Elapsed < budget {
		t.Fatalf("Elapsed = %s, want >= budget %s", res.Elapsed, budget)
	}
	if !strings.Contains(out.String(), "park gave up") {
		t.Fatalf("give-up line missing:\n%s", out.String())
	}
}

// FAK_GUARD_PARK_BUDGET=0 (or an empty credPath) keeps today's immediate fail-loud
// behavior byte-for-byte: the park never runs and never prints.
func TestGuardParkForRelogin_DisabledIsSilentNoop(t *testing.T) {
	var out bytes.Buffer
	clk := &parkClock{t: time.Now()}
	if res := guardParkForRelogin("some/cred.json", 0, time.Minute, clk.now, clk.sleep, &out); res.Attempted {
		t.Fatalf("budget 0: park ran: %+v", res)
	}
	if res := guardParkForRelogin("", time.Hour, time.Minute, clk.now, clk.sleep, &out); res.Attempted {
		t.Fatalf("empty credPath: park ran: %+v", res)
	}
	if out.Len() != 0 || clk.sleeps != 0 {
		t.Fatalf("disabled park produced output (%q) or slept (%d)", out.String(), clk.sleeps)
	}
}

// A credential with NO expiry recorded (expiresAt <= 0) is never-expiring by Claude Code's
// own convention — the park must treat it as live on the first poll, not wait a day on it.
func TestGuardParkForRelogin_NeverExpiringTokenIsLive(t *testing.T) {
	dir := t.TempDir()
	cred := writeParkCred(t, dir, 0)
	clk := &parkClock{t: time.Date(2026, 7, 2, 8, 0, 0, 0, time.UTC)}
	res := guardParkForRelogin(cred, time.Hour, time.Minute, clk.now, clk.sleep, nil)
	if !res.Recovered || clk.sleeps != 1 {
		t.Fatalf("park = %+v after %d sleeps, want Recovered on the first poll", res, clk.sleeps)
	}
}

// The env knobs parse and clamp: budget to [0, 24h] (0 = off), poll to [1s, 1h]; garbage
// falls back to the defaults.
func TestGuardParkKnobs(t *testing.T) {
	cases := []struct {
		env, val string
		want     time.Duration
		resolve  func() time.Duration
	}{
		{"FAK_GUARD_PARK_BUDGET", "", defaultGuardParkBudget, guardParkBudget},
		{"FAK_GUARD_PARK_BUDGET", "2h", 2 * time.Hour, guardParkBudget},
		{"FAK_GUARD_PARK_BUDGET", "0", 0, guardParkBudget},
		{"FAK_GUARD_PARK_BUDGET", "48h", maxGuardParkBudget, guardParkBudget},
		{"FAK_GUARD_PARK_BUDGET", "bogus", defaultGuardParkBudget, guardParkBudget},
		{"FAK_GUARD_PARK_POLL", "", defaultGuardParkPoll, guardParkPoll},
		{"FAK_GUARD_PARK_POLL", "30s", 30 * time.Second, guardParkPoll},
		{"FAK_GUARD_PARK_POLL", "5ms", minGuardParkPoll, guardParkPoll},
		{"FAK_GUARD_PARK_POLL", "3h", maxGuardParkPoll, guardParkPoll},
		{"FAK_GUARD_PARK_POLL", "bogus", defaultGuardParkPoll, guardParkPoll},
	}
	for _, c := range cases {
		t.Setenv(c.env, c.val)
		if got := c.resolve(); got != c.want {
			t.Errorf("%s=%q resolved %s, want %s", c.env, c.val, got, c.want)
		}
		os.Unsetenv(c.env)
	}
}
