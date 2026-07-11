package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func cdMustTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return ts
}

func TestLaunchKindToCooldownKind(t *testing.T) {
	cases := []struct {
		in   launchModelUnavailKind
		want accounts.CooldownKind
		ok   bool
	}{
		{launchModelUsageLimit, accounts.CooldownUsageLimit, true},
		{launchModelRateLimit, accounts.CooldownRateLimit, true},
		{launchModelUnknown, "", false},
		{launchModelAvailable, "", false},
	}
	for _, c := range cases {
		got, ok := launchKindToCooldownKind(c.in)
		if ok != c.ok || got != c.want {
			t.Fatalf("launchKindToCooldownKind(%v) = (%q,%v), want (%q,%v)", c.in, got, ok, c.want, c.ok)
		}
	}
}

// Reset-string parsing now lives in the single source accounts.ParseReset (internal/accounts);
// its behavior is guarded by TestParseReset* in that package.

func TestCooldownReasonFromStderrPicksLimitLine(t *testing.T) {
	stderr := "starting claude...\nresolving model\nError: weekly limit reached for this account; resets at 15:00\ndone\n"
	got := cooldownReasonFromStderr(stderr)
	if got == "" {
		t.Fatal("expected a reason line from a usage-limit stderr")
	}
	if want := "Error: weekly limit reached for this account; resets at 15:00"; got != want {
		t.Fatalf("reason line: got %q want %q", got, want)
	}
}

func TestCooldownReasonFromStderrEmptyWhenNoSignal(t *testing.T) {
	if got := cooldownReasonFromStderr("just an ordinary crash\nstack trace\n"); got != "" {
		t.Fatalf("no-signal stderr should yield empty reason, got %q", got)
	}
}

func TestApplyCooldownToHeadroomWallsCooledBucket(t *testing.T) {
	now := cdMustTime(t, "2026-07-07T12:00:00Z")
	cd, err := accounts.LoadCooldownStore(t.TempDir() + "/cd.json")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cd.Cool("uuid:acct-walled", accounts.CooldownUsageLimit, "weekly", now, now.Add(time.Hour))

	// Roster read said this bucket was offerable (a stale/lagging positive).
	hr := accounts.RotationHeadroom{
		"uuid:acct-walled": 1.5,
		"uuid:acct-free":   1.9,
	}
	out := applyCooldownToHeadroom(hr, cd, now)
	if out["uuid:acct-walled"] != -1 {
		t.Fatalf("cooled bucket must be walled (-1), got %v", out["uuid:acct-walled"])
	}
	if out["uuid:acct-free"] != 1.9 {
		t.Fatalf("uncooled bucket must be untouched, got %v", out["uuid:acct-free"])
	}
}

func TestApplyCooldownToHeadroomMaterializesNilForCooledOnlyAccount(t *testing.T) {
	now := cdMustTime(t, "2026-07-07T12:00:00Z")
	cd, err := accounts.LoadCooldownStore(t.TempDir() + "/cd.json")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cd.Cool("uuid:acct-x", accounts.CooldownRateLimit, "429", now, time.Time{})

	// No live roster signal at all (nil map) — a cooled account still registers.
	out := applyCooldownToHeadroom(nil, cd, now)
	if out == nil {
		t.Fatal("expected a materialized map for a cooled-only account")
	}
	if out["uuid:acct-x"] != -1 {
		t.Fatalf("cooled-only account must be walled, got %v", out["uuid:acct-x"])
	}
}

func TestApplyCooldownToHeadroomExpiredIsNoop(t *testing.T) {
	now := cdMustTime(t, "2026-07-07T12:00:00Z")
	cd, err := accounts.LoadCooldownStore(t.TempDir() + "/cd.json")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	// Cooled a rate-limit at base; by now+10m the 5m window has elapsed.
	cd.Cool("uuid:acct-x", accounts.CooldownRateLimit, "429", now.Add(-10*time.Minute), time.Time{})
	hr := accounts.RotationHeadroom{"uuid:acct-x": 1.5}
	out := applyCooldownToHeadroom(hr, cd, now)
	if out["uuid:acct-x"] != 1.5 {
		t.Fatalf("expired cooldown must not wall the bucket, got %v", out["uuid:acct-x"])
	}
}
