package signals

import (
	"testing"
	"time"
)

// The banner grammar cases mirror tools/fleet_session_signals.py's documented
// behaviors: tz-parenthesized windows, sentence-final periods, daily+weekly in one
// banner, and the week-hint prefix classification.

func TestLimitResetsDailyOnly(t *testing.T) {
	w := LimitResets("You've hit your usage limit · resets 12:10am (America/Los_Angeles)")
	if w.Daily != "12:10am (America/Los_Angeles)" {
		t.Fatalf("daily = %q", w.Daily)
	}
	if w.Weekly != "" {
		t.Fatalf("weekly = %q, want empty", w.Weekly)
	}
}

func TestLimitResetsDailyAndWeeklySentenceFinalPeriods(t *testing.T) {
	text := "You've hit your usage limit. Your limit resets 6am (America/Los_Angeles). " +
		"Your weekly limit resets Jul 8 at 10am (America/Los_Angeles)."
	w := LimitResets(text)
	if w.Daily != "6am (America/Los_Angeles)" {
		t.Fatalf("daily = %q", w.Daily)
	}
	if w.Weekly != "Jul 8 at 10am (America/Los_Angeles)" {
		t.Fatalf("weekly = %q", w.Weekly)
	}
}

func TestLimitResetPrefersDailyFallsBackWeekly(t *testing.T) {
	if got := LimitReset("weekly limit · resets Jul 8, 10am"); got != "Jul 8, 10am" {
		t.Fatalf("weekly-only banner: %q", got)
	}
	both := "limit resets 6pm. weekly limit resets Jul 8 at 10am."
	if got := LimitReset(both); got != "6pm" {
		t.Fatalf("short window should win: %q", got)
	}
	if got := WeeklyReset(both); got != "Jul 8 at 10am" {
		t.Fatalf("weekly: %q", got)
	}
}

func TestLimitResetNoBanner(t *testing.T) {
	if got := LimitReset("all tests pass, shipping now"); got != "" {
		t.Fatalf("no banner should yield empty, got %q", got)
	}
}

func TestResetPassedFutureAndPast(t *testing.T) {
	// Anchor: 2026-06-30 10:00 UTC == 03:00 Pacific (UTC-7). Banner: "resets 6am".
	anchor := time.Date(2026, 6, 30, 10, 0, 0, 0, time.UTC)
	when := "6am (America/Los_Angeles)" // next 6am Pacific = 13:00 UTC same day

	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) // 05:00 Pacific — still capped
	passed, ok := ResetPassed(when, now, anchor)
	if !ok || passed {
		t.Fatalf("before reset: passed=%v ok=%v", passed, ok)
	}
	now = time.Date(2026, 6, 30, 13, 0, 1, 0, time.UTC) // just past 6am Pacific
	passed, ok = ResetPassed(when, now, anchor)
	if !ok || !passed {
		t.Fatalf("after reset: passed=%v ok=%v", passed, ok)
	}
}

func TestResetPassedNextOccurrenceRollsToTomorrow(t *testing.T) {
	// Anchor at 07:00 Pacific; "resets 6am" names TOMORROW's 6am, so a now at today's
	// 8am Pacific has NOT passed it.
	anchor := time.Date(2026, 6, 30, 14, 0, 0, 0, time.UTC) // 07:00 Pacific
	now := time.Date(2026, 6, 30, 15, 0, 0, 0, time.UTC)    // 08:00 Pacific
	passed, ok := ResetPassed("6am (America/Los_Angeles)", now, anchor)
	if !ok || passed {
		t.Fatalf("reset names tomorrow: passed=%v ok=%v", passed, ok)
	}
	// A day later it has.
	now = time.Date(2026, 7, 1, 13, 30, 0, 0, time.UTC)
	passed, ok = ResetPassed("6am (America/Los_Angeles)", now, anchor)
	if !ok || !passed {
		t.Fatalf("next day: passed=%v ok=%v", passed, ok)
	}
}

func TestResetPassedMinutesAndPM(t *testing.T) {
	anchor := time.Date(2026, 6, 30, 10, 0, 0, 0, time.UTC) // 03:00 Pacific
	// 7:10pm Pacific = 02:10 UTC next day.
	now := time.Date(2026, 7, 1, 2, 9, 0, 0, time.UTC)
	passed, ok := ResetPassed("7:10pm (America/Los_Angeles)", now, anchor)
	if !ok || passed {
		t.Fatalf("one minute early: passed=%v ok=%v", passed, ok)
	}
	now = time.Date(2026, 7, 1, 2, 10, 0, 0, time.UTC)
	if passed, _ = ResetPassed("7:10pm (America/Los_Angeles)", now, anchor); !passed {
		t.Fatal("at the minute: want passed")
	}
}

func TestResetPassedUnparseable(t *testing.T) {
	if _, ok := ResetPassed("Jul 8, 2026", time.Now().UTC(), time.Time{}); ok {
		t.Fatal("date-only window should be unparseable (conservative not-yet-passed)")
	}
}

func TestResetPassedZeroAnchorUsesNow(t *testing.T) {
	// With anchor==now the named time is always at/after now, so never passed.
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	passed, ok := ResetPassed("6am (America/Los_Angeles)", now, time.Time{})
	if !ok || passed {
		t.Fatalf("anchor=now: passed=%v ok=%v", passed, ok)
	}
}

func TestAuthTaxonomy(t *testing.T) {
	cases := []struct {
		text, kind string
		login      bool
	}{
		{"API Error: 401 {\"type\":\"error\"} please run /login", "auth", true},
		{"OAuth token has expired · Please run /login", "auth", true},
		{"Your credit balance is too low to access the Anthropic API", "credit", false},
		{"Your organization has disabled Claude subscription access. Use an Anthropic API key instead.", "access", false},
	}
	for _, c := range cases {
		if !IsAuthError(c.text) {
			t.Fatalf("IsAuthError(%q) = false", c.text)
		}
		if got := AuthBlockKind(c.text); got != c.kind {
			t.Fatalf("AuthBlockKind(%q) = %q, want %q", c.text, got, c.kind)
		}
		if got := NeedsLoginPrompt(c.text); got != c.login {
			t.Fatalf("NeedsLoginPrompt(%q) = %v, want %v", c.text, got, c.login)
		}
	}
}

func TestIsAPIErrorExcludesAuth(t *testing.T) {
	if !IsAPIError("API Error: 529 overloaded_error") {
		t.Fatal("529 overload should be an API error")
	}
	if IsAPIError("API Error: 401 authentication_error") {
		t.Fatal("auth wall must not classify as transient API error")
	}
	if IsAPIError("everything went fine") {
		t.Fatal("clean text is not an API error")
	}
}

func TestHTTPStatus(t *testing.T) {
	if got := HTTPStatus("API Error: 529 overloaded"); got != "529" {
		t.Fatalf("got %q", got)
	}
	if got := HTTPStatus("session limit · resets 6pm"); got != "" {
		t.Fatalf("plain limit banner carries no status, got %q", got)
	}
}

// TerminalFailure precedence follows remediation cost: AUTH > LIMIT > API_ERR. The
// empty-input rung is the error-channel discipline: no error record, no failure —
// never an inference from prose.
func TestTerminalFailurePrecedenceAndEmpty(t *testing.T) {
	kind, detail := TerminalFailure("")
	if kind != "" || detail != "" {
		t.Fatalf("empty err text: (%q,%q)", kind, detail)
	}
	kind, detail = TerminalFailure("API Error: 401 please run /login — also you've hit your usage limit · resets 6pm and a 529")
	if kind != KindAuth || detail != "auth/login required" {
		t.Fatalf("auth outranks all: (%q,%q)", kind, detail)
	}
	kind, detail = TerminalFailure("You've hit your usage limit · resets 6pm (America/Los_Angeles)\nAPI Error: 529")
	if kind != KindLimit || detail != "6pm (America/Los_Angeles)" {
		t.Fatalf("limit outranks api-err: (%q,%q)", kind, detail)
	}
	kind, _ = TerminalFailure("API Error: 529 {\"error\":{\"type\":\"overloaded_error\"}}")
	if kind != KindAPIErr {
		t.Fatalf("529: %q", kind)
	}
	kind, _ = TerminalFailure("just some final prose about nothing")
	if kind != "" {
		t.Fatalf("non-failure text: %q", kind)
	}
}
