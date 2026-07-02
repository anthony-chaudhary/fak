// Package signals is the Go port of tools/fleet_session_signals.py — the shared
// transcript signal patterns every fleet session/account tool classifies with. One
// place owns the limit-banner grammar, the auth/access wall taxonomy, the transient
// API-error family, and the reset-window time math, so the sweep, the stopped-session
// classifier, and the resume watchdog can never disagree about what state a session
// is in (the same single-source-of-truth discipline the Python module enforced).
//
// Pure text + injected clock: no I/O, no ambient time read (ResetPassed takes its
// now/anchor), stdlib-only. A sub-leaf of internal/resume, like rehome.
package signals

import (
	"regexp"
	"strings"
	"time"
)

// limitRE parses "limit … resets <when>" from a Claude throttle banner. `<when>` can
// itself contain a parenthesized timezone, e.g. "12:10am (America/Los_Angeles)": the
// time and an optional trailing "(...)" group are captured as a unit, then the match
// stops before banner junk. The terminator accepts a sentence-final period (". " /
// end) so both the daily and the weekly window parse even when each ends in ".".
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
// separately (IsAPIError subtracts the auth family).
var apiErrRE = regexp.MustCompile(`(?i)isApiErrorMessage|API Error|overloaded_error|\boverloaded\b|` +
	`\b429\b|\b529\b|\b503\b|fetch failed|ECONNRESET|ETIMEDOUT|` +
	`socket hang up|Internal Server Error|service unavailable|` +
	`connection error|network error`)

// httpStatusRE captures HTTP/transport status codes that show up in a terminal error
// banner, so "what status did this session last report?" is answerable directly.
var httpStatusRE = regexp.MustCompile(`\b(401|403|429|500|502|503|529)\b`)

// ResetWindows are the usage-limit reset windows one throttle banner carries. Claude's
// banner can name a short (hourly/daily) window AND a weekly one in the same message.
type ResetWindows struct {
	// Daily is the short (hourly/daily) reset window, raw string, tz suffix preserved.
	Daily string
	// Weekly is the weekly-cap reset window, raw string, tz suffix preserved.
	Weekly string
}

// LimitResets extracts all usage-limit reset windows from a throttle banner. Each
// "limit ... resets <when>" occurrence is classified by the ~24 chars preceding it: a
// "week" hint makes it the weekly window, otherwise it is the short window. First
// occurrence of each wins. Zero-value ResetWindows when no reset is present.
func LimitResets(text string) ResetWindows {
	var out ResetWindows
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

// LimitReset is the primary (blocking) reset window: the short (hourly/daily) window
// when present, else the weekly one — a weekly cap still blocks the account, so the
// absence of a short window must not read as "not throttled". "" when none.
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
var tzParenRE = regexp.MustCompile(`\(([^)]+)\)`)

// tzOffsetHours maps the IANA zones the banners actually name onto fixed UTC offset
// hours. The fleet's banners only ever name the US Pacific zone; a small explicit
// table avoids a tzdata dependency, exactly as the Python module chose. PDT (DST,
// Mar–Nov) is UTC-7; the fleet runs year-round on Pacific.
var tzOffsetHours = map[string]int{
	"america/los_angeles": -7, "america/denver": -6, "america/chicago": -5,
	"america/new_york": -4, "utc": 0,
}

// resetTZOffset resolves the parenthesized zone in a reset string, defaulting to
// Pacific — the only zone the banners use.
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
// `when` is a raw reset string from LimitReset, e.g. "6am (America/Los_Angeles)". A
// reset banner names the NEXT occurrence of that wall-clock time, so the window is
// resolved against the banner's own time (anchorUTC — the transcript's last
// timestamp, when known; pass the zero time to anchor on nowUTC) and the verdict is
// whether nowUTC is at/after it. ok is false when the string is unparseable — the
// caller should treat that conservatively as not-yet-passed.
//
// Pure and injectable so it unit-tests without a clock; production passes
// time.Now().UTC(). This is the primitive that makes a reset-cleared session
// distinguishable from a still-capped one.
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
	tz := time.FixedZone("reset", resetTZOffset(when)*3600)
	if anchorUTC.IsZero() {
		anchorUTC = nowUTC
	}
	// The reset is the FIRST occurrence of (hour:minute) in tz at/after the anchor.
	aLocal := anchorUTC.In(tz)
	resetLocal := time.Date(aLocal.Year(), aLocal.Month(), aLocal.Day(), hour, minute, 0, 0, tz)
	if resetLocal.Before(aLocal) {
		resetLocal = resetLocal.Add(24 * time.Hour)
	}
	return !nowUTC.Before(resetLocal), true
}

// atoiSafe parses digits already matched by \d{1,2} — never fails on that input.
func atoiSafe(s string) int {
	n := 0
	for _, r := range s {
		n = n*10 + int(r-'0')
	}
	return n
}

// HTTPStatus returns the first HTTP/transport status code named in an error banner
// (401/403/429/500/502/503/529), or "". The literal "last reported status" the
// disposition taxonomy otherwise folds away. Pure string scan.
func HTTPStatus(text string) string {
	if m := httpStatusRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

// IsAuthError reports whether text names a login/credit/access wall.
func IsAuthError(text string) bool { return authRE.MatchString(text) }

// IsAPIError reports whether text names a transient transport/server error that is
// NOT also an auth wall (auth outranks: a 401 is not a retryable API error).
func IsAPIError(text string) bool {
	return apiErrRE.MatchString(text) && !IsAuthError(text)
}

// AuthBlockKind classifies an auth-family blocker: "credit" | "access" | "auth".
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
// plausibly fix (kind "auth" and a login-required phrasing — never credit/access).
func NeedsLoginPrompt(text string) bool {
	return AuthBlockKind(text) == "auth" && loginRequiredRE.MatchString(text)
}

// The closed failure taxonomy, ordered by recovery-remediation cost: AUTH (needs a
// human /login) outranks LIMIT (wait for the named reset) outranks APIErr (transient,
// retry now), so the most expensive-to-recover wall is never masked by a cheaper one.
const (
	// KindAuth: a login/credit/access wall — a re-resume on the same account can't fix it.
	KindAuth = "AUTH"
	// KindLimit: a usage cap with a named reset window — resumable after the reset.
	KindLimit = "LIMIT"
	// KindAPIErr: a transient transport/server error — resumable now.
	KindAPIErr = "API_ERR"
)

// TerminalFailure classifies a session's TERMINAL ERROR text into its failure mode —
// the single source of truth the sweep and the resume watchdogs share.
//
// Keyed off the ERROR record ONLY (the injected isApiErrorMessage / error turn),
// NEVER the assistant prose: a session that merely *discusses* an auth wall, a 529,
// or a usage limit in its final message is NOT in that failure state. Returns
// (kind, detail): kind is one of KindAuth/KindLimit/KindAPIErr or ""; detail is the
// auth reason for AUTH, the reset window for LIMIT, else "". Empty errText (no error
// record at all) yields ("", "") — no error record means no failure bucket, never an
// inference from prose.
func TerminalFailure(errText string) (kind, detail string) {
	t := strings.TrimSpace(errText)
	if t == "" {
		return "", ""
	}
	if NeedsLoginPrompt(t) || IsAuthError(t) {
		return KindAuth, AuthBlockReason(t)
	}
	if when := LimitReset(t); when != "" {
		return KindLimit, when
	}
	if IsAPIError(t) {
		return KindAPIErr, ""
	}
	return "", ""
}
