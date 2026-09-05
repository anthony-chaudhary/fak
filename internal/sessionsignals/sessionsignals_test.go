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

// UnknownModel names an unknown/invalid/unentitled MODEL refusal — a startup class a
// fallback to a DIFFERENT model can fix, distinct from a usage cap (IsLimitError) and
// an auth wall (IsAuthError). The "model" dimension must be named, so a generic service
// blip is not misread as a model refusal.
func TestUnknownModel(t *testing.T) {
	yes := []string{
		`error: model "fable" is not available for this account`,
		`invalid model: fable`,
		`model_not_found: fable`,
		`your account does not have access to model claude-opus-4-8`,
		`unsupported model claude-x`,
		`your plan is not entitled to use the model`,
	}
	for _, s := range yes {
		if !UnknownModel(s) {
			t.Errorf("UnknownModel(%q) = false, want true", s)
		}
	}
	no := []string{
		`Not logged in. Please run /login`,              // auth wall — a model switch cannot fix
		`You've hit your weekly limit`,                  // usage cap — no model dimension
		`network unavailable while contacting provider`, // generic blip, "model" not named
		``,
	}
	for _, s := range no {
		if UnknownModel(s) {
			t.Errorf("UnknownModel(%q) = true, want false", s)
		}
	}
	// A model refusal must not be misclassified as an auth wall or a usage cap.
	if IsAuthError(`invalid model: fable`) || IsLimitError(`invalid model: fable`) {
		t.Fatalf("model refusal leaked into auth/limit classification")
	}
}

func TestLimitResetAbsent(t *testing.T) {
	if got := LimitReset("all done, shipped and green"); got != "" {
		t.Fatalf("clean text should carry no reset, got %q", got)
	}
}

func TestBareFableLimitStillClassifiesAsLimit(t *testing.T) {
	text := "You've reached your Fable 5 limit. Run /usage-credits to continue or switch models with /model."
	if got := LimitReset(text); got != "" {
		t.Fatalf("bare Fable limit should not invent a reset, got %q", got)
	}
	if !IsLimitError(text) {
		t.Fatal("bare Fable limit must still classify as a limit")
	}
	if k, d := TerminalFailure(text); k != FailureLimit || d != "" {
		t.Fatalf("TerminalFailure = (%q,%q), want LIMIT with no reset detail", k, d)
	}
}

func TestBareUsageLimitStillClassifiesAsLimit(t *testing.T) {
	if !IsLimitError("usage limit reached") {
		t.Fatal("bare usage limit must classify as a limit")
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
	// The bare retry banner a dying session leaves as its terminal record (#2368) —
	// no "API Error" prefix, just the timeout text.
	if !IsAPIError("Request timed out.") {
		t.Fatal("a bare request-timeout banner is a transient API error")
	}
}

// A 409 operator-stop / BUDGET_CONTEXT_EXHAUSTED refusal rides in on the guard's
// "API Error:" prefix but is a session-state wall a raw resume cannot clear (#3457):
// it must classify as a distinct terminal operator-stop, never a transient API error.
func TestIsAPIErrorExcludesOperatorStop(t *testing.T) {
	cases := []struct {
		text                 string
		apiErr, operatorStop bool
	}{
		{"API Error: 409 session f017ca75 is stopped (operator control); request refused: BUDGET_CONTEXT_EXHAUSTED", false, true},
		{"API Error: 409 session f8d84269 is stopped (operator stop); restart_fresh_session", false, true},
		{"API Error: 500 Internal Server Error", true, false},
	}
	for _, c := range cases {
		if got := IsAPIError(c.text); got != c.apiErr {
			t.Errorf("IsAPIError(%q) = %v, want %v", c.text, got, c.apiErr)
		}
		if got := IsOperatorStop(c.text); got != c.operatorStop {
			t.Errorf("IsOperatorStop(%q) = %v, want %v", c.text, got, c.operatorStop)
		}
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

// A Codex/OpenAI-wire per-minute throttle names the 429 family WITHOUT a literal "429" in
// the log tail ("Too Many Requests", "Rate limit reached for <model> ... per min", a
// `rate_limit_exceeded` code). Each must read as a transient API error so the witness
// classifier grades it rate_limit (the reason the concurrency-backoff term counts and
// Layer-2 downgrade re-dispatches), never UNKNOWN — matching resume.ClassifyLimitText, which
// already treats the same phrasings as LimitRate.
func TestCodexThrottleIsTransientAPIError(t *testing.T) {
	for _, text := range []string{
		"Rate limit reached for gpt-5-codex in organization org-abc on tokens per min (TPM): Limit 30000, Used 30000. Please try again in 2s.",
		"stream error: Too Many Requests; retrying 1/5",
		"stream error: unexpected status 429 Too Many Requests; retrying 2/5",
		`{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}`,
		"upstream rate-limited the request",
	} {
		if !IsAPIError(text) {
			t.Errorf("IsAPIError(%q) = false, want true (a 429-family throttle is transient)", text)
		}
		if k, _ := TerminalFailure(text); k != FailureAPIErr {
			t.Errorf("TerminalFailure(%q) = %q, want %q", text, k, FailureAPIErr)
		}
	}
}

// Precedence guard: widening the API-error set to catch a "rate limit" throttle must NOT
// swallow a genuine Codex/subscription USAGE CAP. A cap names session/weekly/usage (a
// different remediation: seat reset or model downgrade, not concurrency backoff), so the
// LIMIT arm must still win over the new API_ERR arm.
func TestUsageCapStillOutranksThrottleWidening(t *testing.T) {
	for _, text := range []string{
		"You've hit your usage limit. Visit https://chatgpt.com/codex to add more.",
		"You've hit your session limit · resets 6am (America/Los_Angeles)",
		"You've hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles)",
	} {
		if !IsLimitError(text) {
			t.Errorf("IsLimitError(%q) = false, want true (a usage/session/weekly cap)", text)
		}
		if k, _ := TerminalFailure(text); k != FailureLimit {
			t.Errorf("TerminalFailure(%q) = %q, want %q (cap outranks throttle)", text, k, FailureLimit)
		}
	}
}

func BenchmarkExtractReset(b *testing.B) {
	cases := []string{
		"You've hit your session limit · resets 6am (America/Los_Angeles). You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles).",
		"You've hit your weekly limit · resets Jul 3 at 9am",
		"clean text should carry no reset",
		"You've reached your Fable 5 limit. Run /usage-credits to continue.",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, text := range cases {
			_ = LimitReset(text)
			_ = LimitResets(text)
		}
	}
}

func BenchmarkIsAPIError(b *testing.B) {
	samples := []string{
		"API Error: Overloaded (529) server-side issue",
		"API Error: 401 authentication_error",
		"Request timed out.",
		"stream error: Too Many Requests; retrying 1/5",
		"API Error: 409 session stopped (operator control)",
		"Rate limit reached for gpt-5-codex in organization org-abc on tokens per min (TPM): Limit 30000, Used 30000. Please try again in 2s.",
		"clean operational log message with no error keywords",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			_ = IsAPIError(s)
			_ = IsAPIErrorWithoutBareTimeout(s)
		}
	}
}

func BenchmarkClassifyHTTPStatus(b *testing.B) {
	samples := []string{
		"API Error: 529 Overloaded",
		"HTTP 401 Unauthorized",
		"session limit; resets 6pm",
		"error code 500 internal error",
		"clean output without status codes",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			_ = HTTPStatus(s)
		}
	}
}


