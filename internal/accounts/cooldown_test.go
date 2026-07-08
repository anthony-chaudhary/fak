package accounts

import (
	"path/filepath"
	"testing"
	"time"
)

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return ts
}

func TestCooldownActiveBeforeResetAbsentAfter(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-1", CooldownUsageLimit, "weekly limit", base, time.Time{})

	// Default window is 1h: active at +59m, gone at +61m.
	if _, ok := s.CooledDown("acct-1", base.Add(59*time.Minute)); !ok {
		t.Fatal("expected acct-1 cooled at +59m")
	}
	if _, ok := s.CooledDown("acct-1", base.Add(61*time.Minute)); ok {
		t.Fatal("expected acct-1 NOT cooled at +61m (window elapsed)")
	}
}

func TestCooldownRateLimitShortWindow(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-r", CooldownRateLimit, "429", base, time.Time{})
	if _, ok := s.CooledDown("acct-r", base.Add(4*time.Minute)); !ok {
		t.Fatal("rate-limit should still hold at +4m")
	}
	if _, ok := s.CooledDown("acct-r", base.Add(6*time.Minute)); ok {
		t.Fatal("rate-limit should clear by +6m")
	}
}

func TestCooldownExplicitResetHonored(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	reset := base.Add(3 * time.Hour)
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	e := s.Cool("acct-x", CooldownUsageLimit, "resets at 15:00", base, reset)
	if !e.ResetAt.Equal(reset.UTC()) {
		t.Fatalf("explicit reset not honored: got %s want %s", e.ResetAt, reset.UTC())
	}
	if _, ok := s.CooledDown("acct-x", base.Add(2*time.Hour)); !ok {
		t.Fatal("should hold to explicit 3h reset")
	}
}

func TestCoolNeverShortensLongerWindow(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	// Weekly usage cap: long window.
	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(6*time.Hour))
	// A later transient 429 must not shrink it to 5 minutes.
	e := s.Cool("acct-1", CooldownRateLimit, "429", base.Add(time.Minute), time.Time{})
	if e.ResetAt.Before(base.Add(6 * time.Hour)) {
		t.Fatalf("transient 429 shortened the usage-cap window: reset=%s", e.ResetAt)
	}
}

func TestCooldownClearAndPrune(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("a", CooldownUsageLimit, "", base, base.Add(time.Hour))
	s.Cool("b", CooldownUsageLimit, "", base, base.Add(time.Hour))

	if !s.Clear("a") {
		t.Fatal("Clear(a) should report an entry was removed")
	}
	if s.Clear("a") {
		t.Fatal("second Clear(a) should report nothing removed")
	}
	// b elapsed → pruned.
	if n := s.Prune(base.Add(2 * time.Hour)); n != 1 {
		t.Fatalf("Prune should drop 1 elapsed entry, dropped %d", n)
	}
	if len(s.entries) != 0 {
		t.Fatalf("store should be empty, has %d", len(s.entries))
	}
}

func TestCooldownEmptyAccountNeverCooled(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	if _, ok := s.CooledDown("", base); ok {
		t.Fatal("empty account must never be cooled")
	}
}

func TestCooldownRoundTripsThroughDisk(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	path := filepath.Join(t.TempDir(), "nested", "cooldown.json")

	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	s.Cool("acct-1", CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	reloaded, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	e, ok := reloaded.CooledDown("acct-1", base.Add(30*time.Minute))
	if !ok {
		t.Fatal("reloaded store lost the active cooldown")
	}
	if e.Reason != "weekly limit" || e.Kind != CooldownUsageLimit {
		t.Fatalf("reloaded entry corrupted: %+v", e)
	}
}

func TestLoadMissingFileIsEmptyNotError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.json")
	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("missing file must not error: %v", err)
	}
	if len(s.Active(mustTime(t, "2026-07-06T12:00:00Z"))) != 0 {
		t.Fatal("missing file should yield empty store")
	}
}

// TestParseResetAbsolute: an explicit RFC3339 reset in the message is parsed to that instant
// (UTC). ParseReset is the single source both cooldown writers share, so this pins its contract
// in one place instead of once per writer.
func TestParseResetAbsolute(t *testing.T) {
	got := ParseReset("usage limit reached; resets at 2026-07-07T15:00:00Z, try later")
	want := mustTime(t, "2026-07-07T15:00:00Z")
	if !got.Equal(want) {
		t.Fatalf("ParseReset absolute: got %s want %s", got, want)
	}
	if got.Location() != time.UTC {
		t.Fatalf("ParseReset must normalize to UTC, got %s", got.Location())
	}
}

// TestParseResetVagueOrAbsentIsZero: a date-less "in 42 minutes" phrasing and a message with no
// reset language both yield the zero time — the caller then falls back to the kind's default
// window rather than a mis-parsed wall-clock.
func TestParseResetVagueOrAbsentIsZero(t *testing.T) {
	if got := ParseReset("session limit; resets in 42 minutes"); !got.IsZero() {
		t.Fatalf("vague reset should be zero, got %s", got)
	}
	if got := ParseReset("no limit language here"); !got.IsZero() {
		t.Fatalf("absent reset should be zero, got %s", got)
	}
	// A bare wall-clock without a date must NOT be guessed as absolute.
	if got := ParseReset("weekly limit reached; resets at 15:00"); !got.IsZero() {
		t.Fatalf("date-less wall-clock should be zero, got %s", got)
	}
}
