// Package sessionsignals is the shared, closed vocabulary of TERMINAL-TURN signals the
// fleet session/resume tools classify a Claude Code transcript's last turn with: the
// usage-limit banner (and its reset window), the auth/login/credit/access walls, and the
// transient transport/server errors. It is the Go port of tools/fleet_session_signals.py —
// the single source of truth the Python resume tools shared so the sweep, the stopped-
// session classifier, and the resume watchdogs could never disagree about what state a
// session is in. Porting it as ONE leaf preserves that property on the Go side.
//
// Everything here is a pure string/time computation: no clock (ResetPassed takes its
// times), no I/O, no process. Tier-1 foundation leaf — stdlib-only, imports nothing
// internal, off the hot path.
//
// The one behavioral fence, inherited from the Python: a failure classification keys off
// the session's ERROR channel (the injected isApiErrorMessage / error record), NEVER the
// assistant prose. TerminalFailure documents that contract; callers must feed it error-
// record text only.
package sessionsignals

import (
	"regexp"
	"strings"
	"time"
)

// limitRE captures the `<when>` of a "limit … resets <when>" throttle banner. `<when>`
// can itself contain a parenthesized timezone, e.g. "12:10am (America/Los_Angeles)":
// capture the time and an optional trailing "(...)" group as a unit, then stop before
// banner junk. The terminator accepts a sentence-final period (". " / end) so both the
// daily and the weekly window of a Claude throttle banner parse even when each ends in ".".
var limitRE = regexp.MustCompile(`(?i)limit\s*[·:|.\-]?\s*resets?\s+([^()"` + "\n" + `]+?(?:\([^()` + "\n" + `]*\))?)\s*(?:["` + "\n" + `<]|$|\.(?:\s|$))`)

var authRE = regexp.MustCompile(`(?i)Login interrupted|please run /login|authentication_error|` +
	`invalid x-api-key|invalid authentication credentials|` +
	`API Error:\s*401|HTTP\s*401|401\s+(?:authentication required|unauthorized)|` +
	`OAuth token has expired|credit balance is too low|` +
	`organization has disabled Claude subscription access|` +
	`Use an Anthropic API key instead`)

var accessWallRE = regexp.MustCompile(`(?i)organization has disabled Claude subscription access|` +
	`Claude subscription access .*disabled|` +
	`Use an Anthropic API key instead|` +
	`ask your admin to enable access`)

var loginRequiredRE = regexp.MustCompile(`(?i)Login interrupted|please run /login|authentication_error|` +
	`invalid x-api-key|invalid authentication credentials|` +
	`API Error:\s*401|HTTP\s*401|401\s+(?:authentication required|unauthorized)|` +
	`OAuth token has expired|Not logged in`)

// apiErrRE matches transport/server errors only. Auth and quota signals are checked
// separately (IsAPIError subtracts IsAuthError, exactly as the Python did).
var apiErrRE = regexp.MustCompile(`(?i)isApiErrorMessage|API Error|overloaded_error|\boverloaded\b|` +
	`\b429\b|\b529\b|\b503\b|fetch failed|ECONNRESET|ETIMEDOUT|` +
	`socket hang up|Internal Server Error|service unavailable|` +
	`connection error|network error`)

// Resets is the set of usage-limit reset windows one throttle banner carries. Claude's
// banner can name a short (hourly/daily) window AND a weekly one in the same message;
// either field is empty when that window is absent.
type Resets struct {
	Daily  string `json:"daily,omitempty"`
	Weekly string `json:"weekly,omitempty"`
}

// LimitResets extracts every "limit … resets <when>" window from a throttle banner. Each
// occurrence is classified by the ~24 chars preceding it: a "week" hint makes it the
// weekly window, otherwise it is the short window. Raw reset strings are returned with
// any timezone suffix preserved.
func LimitResets(text string) Resets {
	var out Resets
	for _, m := range limitRE.FindAllStringSubmatchIndex(text, -1) {
		when := strings.TrimSpace(text[m[2]:m[3]])
		start := m[0] - 24
		if start < 0 {
			start = 0
		}
		prefix := strings.ToLower(text[start:m[0]])
		if strings.Contains(prefix, "week") {
			if out.Weekly == "" {
				out.Weekly = when
			}
		} else if out.Daily == "" {
			out.Daily = when
		}
	}
	return out
}

// LimitReset is the primary (blocking) reset window: the short (hourly/daily) window when
// present, else the weekly one — a weekly cap still blocks the account, so the absence of
// a short window must not read as "not throttled". Empty when the text carries no reset.
func LimitReset(text string) string {
	w := LimitResets(text)
	if w.Daily != "" {
		return w.Daily
	}
	return w.Weekly
}

// WeeklyReset is just the weekly reset window, or "".
func WeeklyReset(text string) string { return LimitResets(text).Weekly }

var resetTimeRE = regexp.MustCompile(`(?i)(\d{1,2})(?::(\d{2}))?\s*([ap])m\b`)

// tzOffsetHours maps the IANA zones the fleet's banners actually name to a fixed UTC
// offset. The fleet's banners only ever name the US zones; a small explicit table avoids
// a tzdata dependency. PDT (DST, Mar–Nov) is UTC-7; the fleet runs year-round on Pacific.
var tzOffsetHours = map[string]int{
	"america/los_angeles": -7, "america/denver": -6, "america/chicago": -5,
	"america/new_york": -4, "utc": 0,
}

var tzParenRE = regexp.MustCompile(`\(([^)]+)\)`)

// resetTZOffset resolves the "(America/Los_Angeles)" suffix of a reset string to a fixed
// offset, defaulting to Pacific — the only zone the banners use.
func resetTZOffset(when string) int {
	if m := tzParenRE.FindStringSubmatch(when); m != nil {
		if off, ok := tzOffsetHours[strings.ToLower(strings.TrimSpace(m[1]))]; ok {
			return off
		}
	}
	return -7
}

// ResetPassed reports whether a usage-limit reset window has already elapsed.
//
// when is a raw reset string from LimitReset, e.g. "6am (America/Los_Angeles)" or
// "7:10am (America/Los_Angeles)". A reset banner names the NEXT occurrence of that
// wall-clock time, so the window is resolved against the banner's own time (anchorUTC —
// the transcript's last timestamp, when known; pass the zero time to anchor on nowUTC)
// and compared to nowUTC.
//
// Returns (passed, ok): ok is false when the reset string is unparseable — the caller
// should treat that conservatively as not-yet-passed, exactly as the Python None did.
// Pure and injectable, so it unit-tests without a clock.
func ResetPassed(when string, nowUTC, anchorUTC time.Time) (passed, ok bool) {
	m := resetTimeRE.FindStringSubmatch(when)
	if m == nil {
		return false, false
	}
	hour := atoiSafe(m[1]) % 12
	if strings.EqualFold(m[3], "p") {
		hour += 12
	}
	minute := 0
	if m[2] != "" {
		minute = atoiSafe(m[2])
	}
	tz := time.FixedZone("banner", resetTZOffset(when)*3600)
	if nowUTC.IsZero() {
		nowUTC = time.Now().UTC()
	}
	anchor := anchorUTC
	if anchor.IsZero() {
		anchor = nowUTC
	}
	// The reset is the FIRST occurrence of (hour:minute) in tz at/after the anchor.
	aLocal := anchor.In(tz)
	reset := time.Date(aLocal.Year(), aLocal.Month(), aLocal.Day(), hour, minute, 0, 0, tz)
	if reset.Before(aLocal) {
		reset = reset.AddDate(0, 0, 1)
	}
	return !nowUTC.Before(reset.UTC()), true
}

// atoiSafe parses a digits-only string already vetted by a regexp; malformed input (which
// the pattern cannot produce) yields 0 rather than an error path.
func atoiSafe(s string) int {
	n := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0
		}
		n = n*10 + int(r-'0')
	}
	return n
}

// httpStatusRE captures the HTTP/transport status codes that show up in a terminal error
// banner, so "what status did this session last report?" is answerable directly instead
// of eyeballing the prose.
var httpStatusRE = regexp.MustCompile(`\b(401|403|429|500|502|503|529)\b`)

// HTTPStatus is the first HTTP/transport status code named in an error banner, or "".
// A plain "session limit; resets 6pm" banner carries no code at all — that returns "".
func HTTPStatus(text string) string {
	if m := httpStatusRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

// IsAuthError reports whether text names an auth/login/credit/access wall.
func IsAuthError(text string) bool { return authRE.MatchString(text) }

// IsAPIError reports whether text names a transient transport/server error that is NOT
// also an auth wall (auth outranks: a 401 is never a retry-now signal).
func IsAPIError(text string) bool {
	return apiErrRE.MatchString(text) && !IsAuthError(text)
}

// AuthBlockKind classifies an auth wall's text: "credit" (balance too low), "access"
// (org disabled subscription access), else "auth" (login/credential refresh).
func AuthBlockKind(text string) string {
	if strings.Contains(strings.ToLower(text), "credit balance is too low") {
		return "credit"
	}
	if accessWallRE.MatchString(text) {
		return "access"
	}
	return "auth"
}

// AuthBlockReason is the human reason matching AuthBlockKind.
func AuthBlockReason(text string) string {
	switch AuthBlockKind(text) {
	case "credit":
		return "credit balance too low"
	case "access":
		return "Claude subscription access disabled"
	default:
		return "auth/login required"
	}
}

// NeedsLoginPrompt is true only for blockers a human login/credential refresh can
// plausibly fix — never for a credit or org-access wall, which a `/login` cannot clear.
func NeedsLoginPrompt(text string) bool {
	return AuthBlockKind(text) == "auth" && loginRequiredRE.MatchString(text)
}

// The closed failure taxonomy, ordered by recovery-remediation cost: AUTH (needs a human
// /login) outranks LIMIT (wait for the named reset) outranks API_ERR (transient, retry).
const (
	FailureAuth   = "AUTH"
	FailureLimit  = "LIMIT"
	FailureAPIErr = "API_ERR"
)

// TerminalFailure classifies a session's TERMINAL ERROR text into its failure mode — the
// single source of truth shared by the sweep classifier and the resume watchdogs, so they
// can never disagree about what state a session is in.
//
// Keyed off the ERROR record ONLY (the injected isApiErrorMessage / error turn), NEVER
// the assistant prose: a session that merely *discusses* an auth wall, a 529, or a usage
// limit in its final message (e.g. a worker editing the resume tooling itself) is NOT in
// that failure state. Precedence follows remediation cost, so the most expensive-to-
// recover wall is never masked by a cheaper one.
//
// Returns (kind, detail): kind is one of the Failure* tokens or ""; detail is the auth
// reason for AUTH, the reset window for LIMIT, else "". Empty/blank errText (no error
// record at all) yields ("", "") — no error record means no failure bucket, never an
// inference from prose.
func TerminalFailure(errText string) (kind, detail string) {
	t := strings.TrimSpace(errText)
	if t == "" {
		return "", ""
	}
	if NeedsLoginPrompt(t) || IsAuthError(t) {
		return FailureAuth, AuthBlockReason(t)
	}
	if when := LimitReset(t); when != "" {
		return FailureLimit, when
	}
	if IsAPIError(t) {
		return FailureAPIErr, ""
	}
	return "", ""
}
