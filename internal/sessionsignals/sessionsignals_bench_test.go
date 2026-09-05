package sessionsignals

import (
	"testing"
	"time"
)

var (
	sinkKind   string
	sinkDetail string
	sinkBool   bool
	sinkResets Resets
	sinkStatus string
)

// BenchmarkClassify measures end-to-end failure mode classification across representative
// production terminal error texts.
func BenchmarkClassify(b *testing.B) {
	samples := []string{
		"You've hit your session limit · resets 6am (America/Los_Angeles). You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles).",
		"Not logged in · Please run /login",
		"OAuth token has expired",
		"API Error: 529 Overloaded (server-side issue)",
		"Rate limit reached for gpt-5-codex in organization org-abc on tokens per min (TPM): Limit 30000, Used 30000. Please try again in 2s.",
		"You've reached your Fable 5 limit. Run /usage-credits to continue or switch models with /model.",
		"API Error: 409 session f017ca75 is stopped (operator control); request refused: BUDGET_CONTEXT_EXHAUSTED",
		"Request timed out.",
		"all done, shipped and green",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkKind, sinkDetail = TerminalFailure(s)
		}
	}
}

// BenchmarkTerminalFailure measures classification latency for individual failure modes.
func BenchmarkTerminalFailure(b *testing.B) {
	cases := []struct {
		name string
		text string
	}{
		{"AuthLogin", "Not logged in · Please run /login"},
		{"AuthCredit", "credit balance is too low"},
		{"AuthAccess", "organization has disabled Claude subscription access"},
		{"LimitDailyWeekly", "You've hit your session limit · resets 6am (America/Los_Angeles). You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles)."},
		{"LimitBareFable", "You've reached your Fable 5 limit. Run /usage-credits to continue"},
		{"APIErr529", "API Error: Overloaded (529) server-side issue"},
		{"APIErrCodexThrottle", "Rate limit reached for gpt-5-codex in organization org-abc on tokens per min (TPM): Limit 30000"},
		{"OperatorStop", "API Error: 409 session f017ca75 is stopped (operator control); request refused: BUDGET_CONTEXT_EXHAUSTED"},
		{"CleanNone", "task finished with status 0"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tc := range cases {
			sinkKind, sinkDetail = TerminalFailure(tc.text)
		}
	}
}

// BenchmarkResetPassed measures evaluation of usage limit reset windows against anchor timestamps.
func BenchmarkResetPassed(b *testing.B) {
	anchor := time.Date(2026, 6, 23, 13, 0, 0, 0, time.UTC)
	now := time.Date(2026, 6, 23, 18, 0, 0, 0, time.UTC)
	cases := []struct {
		when   string
		anchor time.Time
	}{
		{"6am (America/Los_Angeles)", anchor},
		{"11pm (America/Los_Angeles)", anchor},
		{"7:10am (America/Los_Angeles)", time.Time{}},
		{"9am (UTC)", anchor},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tc := range cases {
			p, ok := ResetPassed(tc.when, now, tc.anchor)
			sinkBool = p && ok
		}
	}
}

// BenchmarkLimitResets measures parsing of daily and weekly reset windows from throttle banners.
func BenchmarkLimitResets(b *testing.B) {
	text := "You've hit your session limit · resets 6am (America/Los_Angeles). You've also hit your weekly limit · resets Jul 3, 2026 at 9am (America/Los_Angeles)."

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkResets = LimitResets(text)
	}
}

// BenchmarkIsLimitError measures detection of usage, weekly, and bare quota limits.
func BenchmarkIsLimitError(b *testing.B) {
	samples := []string{
		"You've hit your session limit · resets 6am",
		"You've reached your Fable 5 limit. Run /usage-credits to continue",
		"usage limit reached",
		"API Error: Server is temporarily limiting requests (not your usage limit)",
		"stream error: connection reset by peer",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkBool = IsLimitError(s)
		}
	}
}

// BenchmarkIsAPIError measures detection of transient transport/server errors and rate limits.
func BenchmarkIsAPIError(b *testing.B) {
	samples := []string{
		"API Error: Overloaded (529) server-side issue",
		"Request timed out.",
		"Rate limit reached for gpt-5-codex",
		"API Error: 401 authentication_error",
		"API Error: 409 session f017ca75 is stopped (operator control); request refused: BUDGET_CONTEXT_EXHAUSTED",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkBool = IsAPIError(s)
		}
	}
}

// BenchmarkIsAuthError measures detection of authentication and credential failure walls.
func BenchmarkIsAuthError(b *testing.B) {
	samples := []string{
		"Not logged in · Please run /login",
		"OAuth token has expired",
		"credit balance is too low",
		"organization has disabled Claude subscription access",
		"API Error: 529 Overloaded",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkBool = IsAuthError(s)
		}
	}
}

// BenchmarkHTTPStatus measures extracting HTTP status codes from error banners.
func BenchmarkHTTPStatus(b *testing.B) {
	samples := []string{
		"API Error: 529 Overloaded",
		"HTTP 401 Unauthorized",
		"session limit; resets 6pm",
		"server error 503 Service Unavailable",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkStatus = HTTPStatus(s)
		}
	}
}

// BenchmarkUnknownModel measures detection of model not found and unentitled model refusals.
func BenchmarkUnknownModel(b *testing.B) {
	samples := []string{
		`error: model "fable" is not available for this account`,
		`your account does not have access to model claude-opus-4-8`,
		`Not logged in. Please run /login`,
		`network unavailable while contacting provider`,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range samples {
			sinkBool = UnknownModel(s)
		}
	}
}
