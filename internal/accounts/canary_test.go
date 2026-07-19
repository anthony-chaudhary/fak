package accounts

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// TestCooldownExitGatedOnCanaryRoundTrip is the #3389 core contract: on a
// canary-armed store, a cooled account does NOT auto-restore when its window
// elapses — it sits in probation, still cooled, until one successful canary
// round-trip witnesses the recovery. A failed round-trip keeps it cooled. The
// probe is never run while the window still holds.
func TestCooldownExitGatedOnCanaryRoundTrip(t *testing.T) {
	base := time.Date(2026, 7, 18, 8, 0, 0, 0, time.UTC)
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	calls := 0
	probeErr := errors.New("round-trip failed: still limited")
	s.RequireCanaryExit(func(account string) error {
		calls++
		if account != "acct-canary" {
			t.Fatalf("canary probed %q, want acct-canary", account)
		}
		return probeErr
	})
	s.Cool("acct-canary", CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))

	// Inside the window: cooled as always, and CanaryExit must NOT probe yet.
	if _, ok := s.CooledDown("acct-canary", base.Add(30*time.Minute)); !ok {
		t.Fatalf("account must be cooled inside the window")
	}
	if exited, err := s.CanaryExit("acct-canary", base.Add(30*time.Minute)); exited || err != nil {
		t.Fatalf("CanaryExit inside the window = (%v, %v), want (false, nil)", exited, err)
	}
	if calls != 0 {
		t.Fatalf("canary ran %d times inside the window, want 0 (never probe while the timer holds)", calls)
	}

	// Window elapsed: timer elapse alone must NOT re-admit (the #3389 gate).
	after := base.Add(2 * time.Hour)
	e, ok := s.CooledDown("acct-canary", after)
	if !ok {
		t.Fatalf("gated store re-admitted the account on timer elapse alone; want probation hold until a canary passes")
	}
	if !e.Probation {
		t.Fatalf("elapsed-but-held entry must be marked Probation, got %+v", e)
	}

	// Failed canary keeps it cooled.
	if exited, err := s.CanaryExit("acct-canary", after); exited || err == nil {
		t.Fatalf("CanaryExit with failing probe = (%v, %v), want (false, non-nil)", exited, err)
	}
	if calls != 1 {
		t.Fatalf("canary ran %d times, want exactly 1", calls)
	}
	if _, ok := s.CooledDown("acct-canary", after); !ok {
		t.Fatalf("a failed canary must keep the account cooled")
	}

	// Successful canary round-trip re-admits it.
	probeErr = nil
	if exited, err := s.CanaryExit("acct-canary", after); !exited || err != nil {
		t.Fatalf("CanaryExit with passing probe = (%v, %v), want (true, nil)", exited, err)
	}
	if calls != 2 {
		t.Fatalf("canary ran %d times, want exactly 2", calls)
	}
	if _, ok := s.CooledDown("acct-canary", after); ok {
		t.Fatalf("account must be servable after a successful canary round-trip")
	}
	if s.Degraded() {
		t.Fatalf("fleet must leave degraded mode once the canary-witnessed recovery clears the last entry")
	}
}

// TestCanaryUnarmedStoreKeepsTimerOnlyExit pins backward compatibility: a store
// that never armed the gate behaves exactly as before #3389 — the account
// auto-restores when the window elapses, Prune drops the elapsed entry, and
// CanaryExit realizes the timer-only exit without any probe to run.
func TestCanaryUnarmedStoreKeepsTimerOnlyExit(t *testing.T) {
	base := time.Date(2026, 7, 18, 8, 0, 0, 0, time.UTC)
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-legacy", CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	after := base.Add(2 * time.Hour)
	if _, ok := s.CooledDown("acct-legacy", after); ok {
		t.Fatalf("unarmed store must keep timer-only recovery (elapsed => not cooled)")
	}
	if exited, err := s.CanaryExit("acct-legacy", after); !exited || err != nil {
		t.Fatalf("CanaryExit on an unarmed store = (%v, %v), want (true, nil)", exited, err)
	}
	if len(s.entries) != 0 {
		t.Fatalf("unarmed CanaryExit must drop the elapsed entry, %d left", len(s.entries))
	}
}

// TestPruneRetainsProbationEntries pins the persistence half of the gate: Prune
// must not drop an elapsed entry on an armed store — dropping it would silently
// re-admit the account on the timer alone the moment the file is rewritten.
func TestPruneRetainsProbationEntries(t *testing.T) {
	base := time.Date(2026, 7, 18, 8, 0, 0, 0, time.UTC)
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.RequireCanaryExit(func(string) error { return errors.New("no") })
	s.Cool("acct-hold", CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	after := base.Add(2 * time.Hour)
	if n := s.Prune(after); n != 0 {
		t.Fatalf("Prune removed %d probation entries, want 0", n)
	}
	if _, ok := s.CooledDown("acct-hold", after); !ok {
		t.Fatalf("probation entry must survive Prune until a canary passes")
	}
	// Disarming restores timer-only pruning.
	s.RequireCanaryExit(nil)
	if n := s.Prune(after); n != 1 {
		t.Fatalf("Prune on the disarmed store removed %d, want 1", n)
	}
}

// TestCanaryExitUntrackedAccountSkipsProbe pins the no-op edges: an account the
// store never cooled (and the empty key) is already free, so CanaryExit reports
// true without spending a probe on it.
func TestCanaryExitUntrackedAccountSkipsProbe(t *testing.T) {
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	calls := 0
	s.RequireCanaryExit(func(string) error { calls++; return nil })
	now := time.Date(2026, 7, 18, 8, 0, 0, 0, time.UTC)
	for _, account := range []string{"never-cooled", ""} {
		if exited, err := s.CanaryExit(account, now); !exited || err != nil {
			t.Fatalf("CanaryExit(%q) = (%v, %v), want (true, nil)", account, exited, err)
		}
	}
	if calls != 0 {
		t.Fatalf("canary ran %d times for untracked accounts, want 0", calls)
	}
}

// TestServeAtWalksPastProbationSeat proves the gate reaches the pool: with the
// window elapsed, a cooldown-armed ServeAt still walks PAST the probation seat
// (an unarmed store would land on it — the timer-only baseline), and one
// successful canary round-trip makes the same walk land on the seat again.
func TestServeAtWalksPastProbationSeat(t *testing.T) {
	r := cooldownServeFixture()
	base := time.Date(2026, 7, 18, 8, 0, 0, 0, time.UTC)
	after := base.Add(2 * time.Hour)
	sink := UUIDBucketKey("u-sink")

	unarmed := &CooldownStore{entries: map[string]CooldownEntry{}}
	unarmed.Cool(sink, CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	if h, _, _, err := r.ServeAt("gone", unarmed, after); err != nil || h.Name != "sink" {
		t.Fatalf("unarmed baseline ServeAt = %q,%v, want sink (timer-only exit re-admits on elapse)", h.Name, err)
	}

	armed := &CooldownStore{entries: map[string]CooldownEntry{}}
	probeErr := errors.New("still limited")
	armed.RequireCanaryExit(func(string) error { return probeErr })
	armed.Cool(sink, CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	h, chain, _, err := r.ServeAt("gone", armed, after)
	if err != nil {
		t.Fatalf("ServeAt: %v", err)
	}
	if h.Name != "anchor-seat" || strings.Join(chain, ",") != "gone,sink" {
		t.Fatalf("armed ServeAt = %q via %v, want anchor-seat via [gone sink] (probation seat walked past)", h.Name, chain)
	}

	probeErr = nil
	if exited, err := armed.CanaryExit(sink, after); !exited || err != nil {
		t.Fatalf("CanaryExit = (%v, %v), want (true, nil)", exited, err)
	}
	if h, _, _, err := r.ServeAt("gone", armed, after); err != nil || h.Name != "sink" {
		t.Fatalf("post-canary ServeAt = %q,%v, want sink (witnessed recovery re-admits the seat)", h.Name, err)
	}
}
