package accounts

import (
	"testing"
	"time"
)

// cooldown_weekly_test.go covers the WEEKLY floor (#5890): the cooldown writers must not
// hold an unannounced weekly cap for DefaultCooldownWindow, which is sized for the rolling
// 5-hour cap. It lives beside cooldown_test.go rather than inside it because it pins a
// different question — not "what window did the upstream announce?" (ResolveReset, already
// covered there) but "how long should the seat be held when nothing was announced?".

func weeklyMustTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return ts.UTC()
}

// TestIsWeeklyLimitPhrases: the weekly classifier is phrase-anchored. It must catch the
// forms Claude and the gateway actually emit, and must NOT promote an ordinary rolling-cap
// line to the longer weekly floor.
func TestIsWeeklyLimitPhrases(t *testing.T) {
	weekly := []string{
		"You've hit your weekly limit. Your limit will reset at Jun 2, 9am",
		"Claude usage limit reached. Your weekly limit will reset on Jun 2",
		"You've hit your Opus weekly limit",
		"kind=weekly_limit announced_wait=1h7m0s",
		"You've hit your limit for the week",
		"7-day limit reached",
	}
	for _, msg := range weekly {
		if !IsWeeklyLimit(msg) {
			t.Errorf("IsWeeklyLimit(%q) = false, want true", msg)
		}
	}
	notWeekly := []string{
		"Claude usage limit reached. Your limit will reset at 5pm",
		"You've hit your 5-hour limit",
		"You've hit your session limit; resets 6:10am",
		"Error 429: too many requests",
		"rehomed_seat",
		"", // an empty reason must never take the weekly floor
	}
	for _, msg := range notWeekly {
		if IsWeeklyLimit(msg) {
			t.Errorf("IsWeeklyLimit(%q) = true, want false", msg)
		}
	}
}

// TestResolveCooldownResetWeeklyFloor is the defect in one assertion: a weekly cap whose
// message announces NO window used to fall to DefaultCooldownWindow (1h), so the seat was
// re-admitted ~168 times before its real reset. It now takes the weekly floor instead.
func TestResolveCooldownResetWeeklyFloor(t *testing.T) {
	now := weeklyMustTime(t, "2026-07-06T12:00:00Z")

	// The exact message the repo's own TestResolveResetNoWindowIsZero pins to zero: a
	// weekly cap whose only reset is a bare, deliberately-untrusted wall-clock.
	msg := "weekly limit reached; resets at 15:00"
	if got := ResolveReset(msg, now); !got.IsZero() {
		t.Fatalf("ResolveReset must still refuse the ambiguous wall-clock, got %s", got)
	}
	got := ResolveCooldownReset(msg, now)
	if want := now.Add(WeeklyLimitWindow).UTC(); !got.Equal(want) {
		t.Fatalf("weekly floor: got %s want %s", got, want)
	}
	if WeeklyLimitWindow <= DefaultCooldownWindow {
		t.Fatalf("the weekly floor must exceed the rolling default (%s vs %s)",
			WeeklyLimitWindow, DefaultCooldownWindow)
	}
}

// TestResolveCooldownResetAnnouncedWindowWins: the floor is a FALLBACK, never an override.
// An announced window — absolute or relative — still decides, so #2610's announced_wait
// path is untouched and a weekly cap announcing 1h7m is held 1h7m, not 6h.
func TestResolveCooldownResetAnnouncedWindowWins(t *testing.T) {
	now := weeklyMustTime(t, "2026-07-06T12:00:00Z")

	rel := ResolveCooldownReset("kind=weekly_limit announced_wait≈1h7m", now)
	if want := now.Add(67 * time.Minute).UTC(); !rel.Equal(want) {
		t.Fatalf("announced relative wait: got %s want %s", rel, want)
	}
	abs := ResolveCooldownReset("weekly limit reached; resets at 2026-07-07T15:00:00Z", now)
	if want := weeklyMustTime(t, "2026-07-07T15:00:00Z"); !abs.Equal(want) {
		t.Fatalf("announced absolute reset: got %s want %s", abs, want)
	}
}

// TestResolveCooldownResetNonWeeklyKeepsDefault: a rolling 5-hour cap (and anything else
// with no announced window) still resolves to zero, so the caller falls back to
// DefaultCooldownWindow. The floor must not silently lengthen every usage cooldown.
func TestResolveCooldownResetNonWeeklyKeepsDefault(t *testing.T) {
	now := weeklyMustTime(t, "2026-07-06T12:00:00Z")
	for _, msg := range []string{
		"Claude usage limit reached. Your limit will reset at 5pm (America/Los_Angeles).",
		"You've hit your 5-hour limit",
		"rehomed_seat",
	} {
		if got := ResolveCooldownReset(msg, now); !got.IsZero() {
			t.Fatalf("ResolveCooldownReset(%q) should be zero, got %s", msg, got)
		}
	}
}

// TestCoolUnannouncedWeeklyHoldsPastTheRollingDefault is the behavioral half: the seat is
// still cooled at the moment the old 1-hour default would have re-offered it.
func TestCoolUnannouncedWeeklyHoldsPastTheRollingDefault(t *testing.T) {
	now := weeklyMustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}

	reset := ResolveCooldownReset("You've hit your weekly limit for Opus", now)
	s.Cool("acct-weekly", CooldownUsageLimit, "weekly limit", now, reset)

	if _, ok := s.CooledDown("acct-weekly", now.Add(DefaultCooldownWindow)); !ok {
		t.Fatal("seat re-offered at the 1-hour rolling default — the weekly floor was dropped")
	}
	if _, ok := s.CooledDown("acct-weekly", now.Add(WeeklyLimitWindow)); ok {
		t.Fatal("the weekly floor must still expire on its own — no permanent hold")
	}
}
