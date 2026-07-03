// Package signals keeps the older internal/resume signal API while delegating the
// shared terminal-turn taxonomy to internal/sessionsignals.
package signals

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// ResetWindows are the usage-limit reset windows one throttle banner carries. Claude's
// banner can name a short (hourly/daily) window AND a weekly one in the same message.
type ResetWindows struct {
	// Daily is the short (hourly/daily) reset window, raw string, tz suffix preserved.
	Daily string
	// Weekly is the weekly-cap reset window, raw string, tz suffix preserved.
	Weekly string
}

// LimitResets extracts all usage-limit reset windows from a throttle banner.
func LimitResets(text string) ResetWindows {
	w := sessionsignals.LimitResets(text)
	return ResetWindows{Daily: w.Daily, Weekly: w.Weekly}
}

// LimitReset is the primary reset window: daily when present, else weekly.
func LimitReset(text string) string { return sessionsignals.LimitReset(text) }

// WeeklyReset is just the weekly reset window, or "".
func WeeklyReset(text string) string { return sessionsignals.WeeklyReset(text) }

// ResetPassed reports whether a usage-limit reset window has already elapsed.
func ResetPassed(when string, nowUTC, anchorUTC time.Time) (passed, ok bool) {
	return sessionsignals.ResetPassedAt(when, nowUTC, anchorUTC)
}

// HTTPStatus returns the first HTTP/transport status code named in an error banner.
func HTTPStatus(text string) string { return sessionsignals.HTTPStatus(text) }

// IsAuthError reports whether text names a login/credit/access wall.
func IsAuthError(text string) bool { return sessionsignals.IsAuthError(text) }

// IsAPIError reports whether text names a transient transport/server error.
func IsAPIError(text string) bool { return sessionsignals.IsAPIErrorWithoutBareTimeout(text) }

// AuthBlockKind classifies an auth-family blocker: "credit" | "access" | "auth".
func AuthBlockKind(text string) string { return sessionsignals.AuthBlockKind(text) }

// AuthBlockReason is the human reason matching AuthBlockKind.
func AuthBlockReason(text string) string { return sessionsignals.AuthBlockReason(text) }

// NeedsLoginPrompt is true only for blockers a human login/credential refresh can fix.
func NeedsLoginPrompt(text string) bool { return sessionsignals.NeedsLoginPrompt(text) }

// The closed failure taxonomy, ordered by recovery-remediation cost.
const (
	KindAuth   = sessionsignals.FailureAuth
	KindLimit  = sessionsignals.FailureLimit
	KindAPIErr = sessionsignals.FailureAPIErr
)

// TerminalFailure classifies a session's TERMINAL ERROR text using the older
// internal/resume/signals contract: only reset-bearing limit banners classify as LIMIT,
// and a bare request-timeout banner is not an API_ERR.
func TerminalFailure(errText string) (kind, detail string) {
	return sessionsignals.TerminalFailureWithOptions(errText, sessionsignals.TerminalFailureOptions{
		IncludeBareLimit:          false,
		IncludeBareRequestTimeout: false,
	})
}
