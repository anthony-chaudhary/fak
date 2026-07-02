package sessionsignals

import (
	"testing"
	"time"
)

// The load-bearing facts these pin (ported from the Python fleet_session_signals users):
//   - a throttle banner's daily AND weekly windows both parse, tz suffix preserved,
//     even when each ends in a sentence-final period;
//   - ResetPassed resolves the banner's wall-clock time against the anchor (the
//     transcript's own last timestamp), not the file clock;
//   - TerminalFailure keys off error text only, with AUTH > LIMIT > API_ERR precedence
//     and ("","") for empty text — no error record means no failure bucket.

func TestLimitResetsDailyAndWeekly(t *testing.T) {
	text := "You've hit your session limit · resets 6am (America/Los_Angeles). " +
		"You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles)."
	w := LimitResets(text)
	if w.Daily != "6am (America/Los_Angeles)" {
		t.Fatalf("daily = %q", w.Daily)
	}
	if w.Weekly != "Jul 3, 2026 at 9am (America/Los_Angeles)" {
		t.Fatalf("weekly = %q", w.Weekly)
	}
	if got := LimitReset(text); got != w.Daily {
		t.Fatalf("LimitReset = %q, want the daily window %q", got, w.Daily)
	}
}

func TestLimitResetWeeklyOnlyStillBlocks(t *testing.T) {
	text := "You've hit your weekly limit · resets Jul 3 at 9am"
	if got := LimitReset(text); got == "" {
		t.Fatal("weekly-only banner must still read as throttled (non-empty reset)")
	}
	if WeeklyReset(text) == "" {
		t.Fatal("WeeklyReset should carry the weekly window")
	}
}

func TestLimitResetAbsent(t *testing.T) {
	if got := LimitReset("all done, shipped and green"); got != "" {
		t.Fatalf("clean text should carry no reset, got %q", got)
	}
}

// 2026-06-23T18:00Z == 11:00 PDT — the fixture time the Python sweep tests used.
var now1100PDT = time.Date(2026, 6, 23, 18, 0, 0, 0, time.UTC)

func TestResetPassed(t *testing.T) {
	anchor := time.Date(2026, 6, 23, 13, 0, 0, 0, time.UTC) // 6:00 PDT
	passed, ok := ResetPassed("6am (America/Los_Angeles)", now1100PDT, anchor)
	if !ok || !passed {
		t.Fatalf("6am anchored 06:00 PDT should have passed by 11:00 PDT (passed=%v ok=%v)", passed, ok)
	}
	anchor2 := time.Date(2026, 6, 23, 16, 0, 0, 0, time.UTC) // 9:00 PDT
	passed, ok = ResetPassed("11pm (America/Los_Angeles)", now1100PDT, anchor2)
	if !ok || passed {
		t.Fatalf("11pm should still be in the future at 11:00 PDT (passed=%v ok=%v)", passed, ok)
	}
}

func TestResetPassedMinutesAndZeroAnchor(t *testing.T) {
	// 7:10am with the anchor omitted (zero time): anchored on now, the next 7:10am is
	// tomorrow — not passed.
	passed, ok := ResetPassed("7:10am (America/Los_Angeles)", now1100PDT, time.Time{})
	if !ok || passed {
		t.Fatalf("7:10am anchored at 11:00 PDT is tomorrow (passed=%v ok=%v)", passed, ok)
	}
}

func TestResetPassedUnparseable(t *testing.T) {
	if _, ok := ResetPassed("sometime later", now1100PDT, time.Time{}); ok {
		t.Fatal("unparseable reset must report ok=false (caller treats as not-yet-passed)")
	}
}

func TestHTTPStatus(t *testing.T) {
	if got := HTTPStatus("API Error: 529 Overloaded"); got != "529" {
		t.Fatalf("HTTPStatus = %q, want 529", got)
	}
	if got := HTTPStatus("session limit; resets 6pm"); got != "" {
		t.Fatalf("no-code banner should yield empty, got %q", got)
	}
}

func TestAuthTaxonomy(t *testing.T) {
	cases := []struct {
		text, kind string
		login      bool
	}{
		{"Not logged in · Please run /login", "auth", true},
		{"OAuth token has expired", "auth", true},
		{"credit balance is too low", "credit", false},
		{"organization has disabled Claude subscription access", "access", false},
	}
	for _, c := range cases {
		if got := AuthBlockKind(c.text); got != c.kind {
			t.Errorf("AuthBlockKind(%q) = %q, want %q", c.text, got, c.kind)
		}
		if got := NeedsLoginPrompt(c.text); got != c.login {
			t.Errorf("NeedsLoginPrompt(%q) = %v, want %v", c.text, got, c.login)
		}
	}
	if !IsAuthError("please run /login") {
		t.Error("IsAuthError should match the login prompt")
	}
}

func TestIsAPIErrorExcludesAuth(t *testing.T) {
	if !IsAPIError("API Error: Overloaded (529) server-side issue") {
		t.Fatal("529 overload is an API error")
	}
	// An auth wall that also names an HTTP status must classify as auth, not transient.
	if IsAPIError("API Error: 401 authentication_error") {
		t.Fatal("auth outranks: a 401 wall is not a retry-now API error")
	}
}

func TestTerminalFailurePrecedenceAndEmpty(t *testing.T) {
	if k, d := TerminalFailure(""); k != "" || d != "" {
		t.Fatalf("empty error text must yield no bucket, got (%q,%q)", k, d)
	}
	if k, d := TerminalFailure("Not logged in · Please run /login"); k != FailureAuth || d != "auth/login required" {
		t.Fatalf("auth = (%q,%q)", k, d)
	}
	if k, d := TerminalFailure("You've hit your session limit · resets 6am (America/Los_Angeles)"); k != FailureLimit ||
		d != "6am (America/Los_Angeles)" {
		t.Fatalf("limit = (%q,%q)", k, d)
	}
	if k, _ := TerminalFailure("API Error: Server is temporarily limiting requests (not your usage limit) · Rate limited"); k != FailureAPIErr {
		t.Fatalf("transient 529-style wall = %q, want API_ERR", k)
	}
	// Prose that merely mentions an API error but carries a limit banner: LIMIT wins
	// (precedence by remediation cost).
	if k, _ := TerminalFailure("API Error · You've hit your session limit · resets 6am"); k != FailureLimit {
		t.Fatalf("limit outranks transient, got %q", k)
	}
}
