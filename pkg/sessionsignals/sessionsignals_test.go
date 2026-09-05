package sessionsignals

import (
	"testing"
	"time"
)

func TestLimitResetsDailyAndWeekly(t *testing.T) {
	text := "You've hit your session limit · resets 6am (America/Los_Angeles). " +
		"You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles)."
	w := LimitResets(text)
	if w.Daily != "6am (America/Los_Angeles)" {
		t.Fatalf("daily = %q, want %q", w.Daily, "6am (America/Los_Angeles)")
	}
	if w.Weekly != "Jul 3, 2026 at 9am (America/Los_Angeles)" {
		t.Fatalf("weekly = %q, want %q", w.Weekly, "Jul 3, 2026 at 9am (America/Los_Angeles)")
	}
	if got := LimitReset(text); got != w.Daily {
		t.Fatalf("LimitReset = %q, want the daily window %q", got, w.Daily)
	}
	if got := WeeklyReset(text); got != w.Weekly {
		t.Fatalf("WeeklyReset = %q, want %q", got, w.Weekly)
	}
}

func TestLimitResetWeeklyOnly(t *testing.T) {
	text := "You've hit your weekly limit · resets Jul 3 at 9am"
	if got := LimitReset(text); got == "" {
		t.Fatal("weekly-only banner must report non-empty LimitReset")
	}
	if got := WeeklyReset(text); got != "Jul 3 at 9am" {
		t.Fatalf("WeeklyReset = %q, want %q", got, "Jul 3 at 9am")
	}
	if !IsLimitError(text) {
		t.Fatal("weekly limit text must be recognized as limit error")
	}
}

func TestLimitResetAbsent(t *testing.T) {
	text := "all operational tests passed cleanly"
	if got := LimitReset(text); got != "" {
		t.Fatalf("clean text should yield empty reset, got %q", got)
	}
	if IsLimitError(text) {
		t.Fatal("clean text should not be a limit error")
	}
}

func TestBareLimits(t *testing.T) {
	cases := []struct {
		text    string
		isLimit bool
	}{
		{"You've reached your Fable 5 limit. Run /usage-credits to continue", true},
		{"session limit reached", true},
		{"weekly limit hit", true},
		{"usage limit exceeded", true},
		{"API Error: Server is temporarily limiting requests (not your usage limit) · Rate limited", false},
	}
	for _, tc := range cases {
		if got := IsLimitError(tc.text); got != tc.isLimit {
			t.Errorf("IsLimitError(%q) = %v, want %v", tc.text, got, tc.isLimit)
		}
	}
}

func TestResetPassedAndResetPassedAt(t *testing.T) {
	anchor := time.Date(2026, 6, 23, 13, 0, 0, 0, time.UTC) // 6:00 PDT
	now := time.Date(2026, 6, 23, 18, 0, 0, 0, time.UTC)    // 11:00 PDT

	passed, ok := ResetPassed("6am (America/Los_Angeles)", now, anchor)
	if !ok || !passed {
		t.Fatalf("6am anchor should have passed by 11:00 PDT (passed=%v, ok=%v)", passed, ok)
	}

	futureAnchor := time.Date(2026, 6, 23, 16, 0, 0, 0, time.UTC) // 9:00 PDT
	passed, ok = ResetPassedAt("11pm (America/Los_Angeles)", now, futureAnchor)
	if !ok || passed {
		t.Fatalf("11pm anchor should not have passed by 11:00 PDT (passed=%v, ok=%v)", passed, ok)
	}

	// Unparseable format
	if _, ok := ResetPassed("some invalid timestamp", now, anchor); ok {
		t.Fatal("unparseable reset should return ok=false")
	}
}

func TestAuthDetectionAndTaxonomy(t *testing.T) {
	cases := []struct {
		text       string
		isAuth     bool
		kind       string
		reason     string
		needsLogin bool
	}{
		{
			text:       "Not logged in · Please run /login",
			isAuth:     true,
			kind:       "auth",
			reason:     "auth/login required",
			needsLogin: true,
		},
		{
			text:       "OAuth token has expired",
			isAuth:     true,
			kind:       "auth",
			reason:     "auth/login required",
			needsLogin: true,
		},
		{
			text:       "credit balance is too low",
			isAuth:     true,
			kind:       "credit",
			reason:     "credit balance too low",
			needsLogin: false,
		},
		{
			text:       "organization has disabled Claude subscription access",
			isAuth:     true,
			kind:       "access",
			reason:     "Claude subscription access disabled",
			needsLogin: false,
		},
		{
			text:       "API Error: 401 Unauthorized",
			isAuth:     true,
			kind:       "auth",
			reason:     "auth/login required",
			needsLogin: true,
		},
		{
			text:       "clean response from server",
			isAuth:     false,
			kind:       "auth",
			reason:     "auth/login required",
			needsLogin: false,
		},
	}

	for _, tc := range cases {
		if got := IsAuthError(tc.text); got != tc.isAuth {
			t.Errorf("IsAuthError(%q) = %v, want %v", tc.text, got, tc.isAuth)
		}
		if got := AuthBlockKind(tc.text); got != tc.kind {
			t.Errorf("AuthBlockKind(%q) = %q, want %q", tc.text, got, tc.kind)
		}
		if got := AuthBlockReason(tc.text); got != tc.reason {
			t.Errorf("AuthBlockReason(%q) = %q, want %q", tc.text, got, tc.reason)
		}
		if got := NeedsLoginPrompt(tc.text); got != tc.needsLogin {
			t.Errorf("NeedsLoginPrompt(%q) = %v, want %v", tc.text, got, tc.needsLogin)
		}
	}
}

func TestAPIErrorsAndOperatorStop(t *testing.T) {
	if !IsAPIError("API Error: Overloaded (529) server-side issue") {
		t.Fatal("529 overload should be API error")
	}
	if !IsAPIError("Rate limit reached for gpt-5-codex") {
		t.Fatal("Codex rate limit should be API error")
	}
	if !IsAPIError("Request timed out.") {
		t.Fatal("Bare request timeout should be API error for IsAPIError")
	}
	if IsAPIErrorWithoutBareTimeout("Request timed out.") {
		t.Fatal("Bare request timeout should not match IsAPIErrorWithoutBareTimeout")
	}

	// Auth error takes precedence over API error
	if IsAPIError("API Error: 401 authentication_error") {
		t.Fatal("401 auth error must not be classified as transient API error")
	}

	// Operator stop is excluded from API error
	opStopText := "API Error: 409 session f017ca75 is stopped (operator control); request refused: BUDGET_CONTEXT_EXHAUSTED"
	if IsAPIError(opStopText) {
		t.Fatal("operator stop must not be classified as transient API error")
	}
	if !IsOperatorStop(opStopText) {
		t.Fatal("operator stop text must match IsOperatorStop")
	}

	if got := HTTPStatus("API Error: 529 Overloaded"); got != "529" {
		t.Fatalf("HTTPStatus = %q, want 529", got)
	}
	if got := HTTPStatus("clean text with no code"); got != "" {
		t.Fatalf("HTTPStatus for clean text = %q, want empty", got)
	}
}

func TestUnknownModel(t *testing.T) {
	valid := []string{
		`error: model "fable" is not available for this account`,
		`invalid model: fable`,
		`model_not_found: gpt-5`,
		`your account does not have access to model claude-opus-4-8`,
		`unsupported model claude-x`,
		`your plan is not entitled to use the model`,
	}
	for _, text := range valid {
		if !UnknownModel(text) {
			t.Errorf("UnknownModel(%q) = false, want true", text)
		}
	}

	invalid := []string{
		"clean output",
		"Not logged in",
		"API Error: 500 Internal Server Error",
	}
	for _, text := range invalid {
		if UnknownModel(text) {
			t.Errorf("UnknownModel(%q) = true, want false", text)
		}
	}
}

func TestTerminalFailure(t *testing.T) {
	if k, d := TerminalFailure(""); k != "" || d != "" {
		t.Fatalf("empty text should yield empty category, got (%q, %q)", k, d)
	}

	// Auth classification
	if k, d := TerminalFailure("Not logged in · Please run /login"); k != FailureAuth || d != "auth/login required" {
		t.Fatalf("TerminalFailure auth = (%q, %q)", k, d)
	}

	// Limit classification with detail
	if k, d := TerminalFailure("You've hit your session limit · resets 6am (America/Los_Angeles)"); k != FailureLimit || d != "6am (America/Los_Angeles)" {
		t.Fatalf("TerminalFailure limit = (%q, %q)", k, d)
	}

	// Bare limit classification
	if k, d := TerminalFailure("You've reached your Fable 5 limit. Run /usage-credits to continue"); k != FailureLimit || d != "" {
		t.Fatalf("TerminalFailure bare limit = (%q, %q)", k, d)
	}

	// API error classification
	if k, d := TerminalFailure("API Error: 529 Overloaded"); k != FailureAPIErr || d != "" {
		t.Fatalf("TerminalFailure API error = (%q, %q)", k, d)
	}

	// Precedence: Auth outranks Limit
	authAndLimit := "You've hit your session limit · resets 6am (America/Los_Angeles) · Not logged in · Please run /login"
	if k, _ := TerminalFailure(authAndLimit); k != FailureAuth {
		t.Fatalf("Auth must outrank Limit, got %q", k)
	}

	// Precedence: Limit outranks API error
	limitAndAPI := "API Error: Overloaded · You've hit your session limit · resets 6am (America/Los_Angeles)"
	if k, _ := TerminalFailure(limitAndAPI); k != FailureLimit {
		t.Fatalf("Limit must outrank API error, got %q", k)
	}

	// Options
	opts := TerminalFailureOptions{
		IncludeBareLimit:          false,
		IncludeBareRequestTimeout: false,
	}
	if k, _ := TerminalFailureWithOptions("You've reached your Fable 5 limit.", opts); k != "" {
		t.Fatalf("With IncludeBareLimit=false, bare limit should not classify as limit, got %q", k)
	}
	if k, _ := TerminalFailureWithOptions("Request timed out.", opts); k != "" {
		t.Fatalf("With IncludeBareRequestTimeout=false, bare timeout should not classify, got %q", k)
	}
}
