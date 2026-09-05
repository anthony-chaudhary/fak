// Package sessionsignals defines the closed vocabulary of terminal-turn transcript signals
// (usage limits, auth/credit walls, and transient errors) used to classify session state.
// Calculations are pure string and time operations evaluated against the error channel.
package sessionsignals

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

// limitRE extracts the reset window from a "limit ... resets <when>" throttle banner.
var limitRE = regexp.MustCompile(`(?i)limit\s*[·:|.\-]?\s*resets?\s+([^()"` + "\n" + `]+?(?:\([^()` + "\n" + `]*\))?)\s*(?:["` + "\n" + `<]|$|\.(?:\s|$))`)

var bareLimitRE = regexp.MustCompile(`(?i)\b(?:session|weekly|usage|fable\s+\d+)\s+limit\b|/usage-credits`)
var negatedBareLimitRE = regexp.MustCompile(`(?i)\bnot\s+(?:your\s+)?(?:session|weekly|usage|fable\s+\d+)\s+limit\b`)

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

// apiErrPattern captures transient transport errors, server failures, and codex rate limits.
const apiErrPattern = `(?i)isApiErrorMessage|API Error|overloaded_error|\boverloaded\b|` +
	`\b429\b|\b529\b|\b503\b|too many requests|rate[ _-]?limit(?:ed|_exceeded|\s+reached)|` +
	`fetch failed|ECONNRESET|ETIMEDOUT|` +
	`socket hang up|Internal Server Error|service unavailable|` +
	`connection error|network error`

// apiErrRE matches transport and server errors, excluding auth and quota walls.
var apiErrRE = regexp.MustCompile(apiErrPattern + `|request timed out`)
var apiErrWithoutBareTimeoutRE = regexp.MustCompile(apiErrPattern)

// operatorStopPattern matches terminal operator-stop and context-exhaustion refusals.
const operatorStopPattern = `(?i)BUDGET_CONTEXT_EXHAUSTED|` +
	`\b409\b.*operator\s+(?:control|stop)|` +
	`restart_fresh_session`

var operatorStopRE = regexp.MustCompile(operatorStopPattern)

// Resets holds parsed daily and weekly usage-limit reset windows from a throttle banner.
type Resets struct {
	Daily  string `json:"daily,omitempty"`
	Weekly string `json:"weekly,omitempty"`
}

// LimitResets extracts daily and weekly reset windows from a throttle banner string.
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

// LimitReset returns the primary blocking reset window (daily if present, else weekly).
func LimitReset(text string) string {
	w := LimitResets(text)
	if w.Daily != "" {
		return w.Daily
	}
	return w.Weekly
}

// IsLimitError reports whether text contains a provider quota or session cap banner.
func IsLimitError(text string) bool {
	return LimitReset(text) != "" || hasBareLimitSignal(text)
}

func hasBareLimitSignal(text string) bool {
	return bareLimitRE.MatchString(text) && !negatedBareLimitRE.MatchString(text)
}

// WeeklyReset returns the weekly reset window from a throttle banner, or empty.
func WeeklyReset(text string) string { return LimitResets(text).Weekly }

var resetTimeRE = regexp.MustCompile(`(?i)(\d{1,2})(?::(\d{2}))?\s*([ap])m\b`)

// tzOffsetHours maps US timezones found in banners to fixed UTC offsets.
var tzOffsetHours = map[string]int{
	"america/los_angeles": -7, "america/denver": -6, "america/chicago": -5,
	"america/new_york": -4, "utc": 0,
}

var tzParenRE = regexp.MustCompile(`\(([^)]+)\)`)

// resetTZOffset resolves the timezone suffix of a reset string, defaulting to Pacific.
func resetTZOffset(when string) int {
	if m := tzParenRE.FindStringSubmatch(when); m != nil {
		if off, ok := tzOffsetHours[strings.ToLower(strings.TrimSpace(m[1]))]; ok {
			return off
		}
	}
	return -7
}

// ResetPassed reports whether a usage-limit reset window has elapsed relative to nowUTC
// and anchorUTC. Returns ok=false when the reset string cannot be parsed.
func ResetPassed(when string, nowUTC, anchorUTC time.Time) (passed, ok bool) {
	if nowUTC.IsZero() {
		nowUTC = time.Now().UTC()
	}
	return ResetPassedAt(when, nowUTC, anchorUTC)
}

// ResetPassedAt evaluates a reset window against injected nowUTC and anchorUTC timestamps.
func ResetPassedAt(when string, nowUTC, anchorUTC time.Time) (passed, ok bool) {
	m := resetTimeRE.FindStringSubmatch(when)
	if m == nil {
		return false, false
	}
	hour, _ := strconv.Atoi(m[1])
	hour = hour % 12
	if strings.EqualFold(m[3], "p") {
		hour += 12
	}
	minute := 0
	if m[2] != "" {
		minute, _ = strconv.Atoi(m[2])
	}
	tz := time.FixedZone("banner", resetTZOffset(when)*3600)
	anchor := anchorUTC
	if anchor.IsZero() {
		anchor = nowUTC
	}
	// The reset is the first occurrence of (hour:minute) in tz at or after the anchor.
	aLocal := anchor.In(tz)
	reset := time.Date(aLocal.Year(), aLocal.Month(), aLocal.Day(), hour, minute, 0, 0, tz)
	if reset.Before(aLocal) {
		reset = reset.AddDate(0, 0, 1)
	}
	return !nowUTC.Before(reset.UTC()), true
}

// httpStatusRE captures HTTP status codes present in terminal error messages.
var httpStatusRE = regexp.MustCompile(`\b(401|403|429|500|502|503|529)\b`)

// HTTPStatus returns the first HTTP status code named in error text, or empty.
func HTTPStatus(text string) string {
	if m := httpStatusRE.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

// IsAuthError reports whether text names an auth, login, credit, or access wall.
func IsAuthError(text string) bool { return authRE.MatchString(text) }

// IsOperatorStop reports whether text names a terminal operator-stop refusal.
func IsOperatorStop(text string) bool { return operatorStopRE.MatchString(text) }

// IsAPIError reports whether text names a transient transport or server error.
func IsAPIError(text string) bool {
	return isAPIError(text, apiErrRE)
}

// IsAPIErrorWithoutBareTimeout reports transient transport errors while excluding bare request timeouts.
func IsAPIErrorWithoutBareTimeout(text string) bool {
	return isAPIError(text, apiErrWithoutBareTimeoutRE)
}

func isAPIError(text string, re *regexp.Regexp) bool {
	return re.MatchString(text) && !IsAuthError(text) && !IsOperatorStop(text)
}

// unknownModelRE matches model-not-found or unentitled model refusal messages.
var unknownModelRE = regexp.MustCompile(`(?i)unknown model|invalid model|unsupported model|model_not_found|` +
	`no access to model|does not have access to model|not entitled to (?:use )?(?:the )?model|` +
	`model[^` + "\n" + `]{0,40}(?:not available|unavailable|not found|does not exist|not entitled)`)

// UnknownModel reports whether text names an unknown or unentitled model refusal.
func UnknownModel(text string) bool { return unknownModelRE.MatchString(text) }

// AuthBlockKind classifies an auth wall into categories: credit, access, or general auth.
func AuthBlockKind(text string) string {
	if strings.Contains(strings.ToLower(text), "credit balance is too low") {
		return "credit"
	}
	if accessWallRE.MatchString(text) {
		return "access"
	}
	return "auth"
}

// AuthBlockReason returns a human-readable explanation string matching the given AuthBlockKind.
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

// NeedsLoginPrompt reports whether an error requires an interactive human login prompt.
func NeedsLoginPrompt(text string) bool {
	return AuthBlockKind(text) == "auth" && loginRequiredRE.MatchString(text)
}

// Failure category identifiers ordered by remediation cost: AUTH outranks LIMIT outranks API_ERR.
const (
	FailureAuth   = "AUTH"
	FailureLimit  = "LIMIT"
	FailureAPIErr = "API_ERR"
)

// TerminalFailureOptions configures compatibility flags for TerminalFailure evaluation.
type TerminalFailureOptions struct {
	IncludeBareLimit          bool
	IncludeBareRequestTimeout bool
}

// TerminalFailure classifies session error text into a failure category (AUTH, LIMIT, API_ERR).
// Evaluates error record text only (never assistant prose) and returns (kind, detail).
func TerminalFailure(errText string) (kind, detail string) {
	return TerminalFailureWithOptions(errText, TerminalFailureOptions{
		IncludeBareLimit:          true,
		IncludeBareRequestTimeout: true,
	})
}

// TerminalFailureWithOptions classifies error text using explicit compatibility switches.
func TerminalFailureWithOptions(errText string, opts TerminalFailureOptions) (kind, detail string) {
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
	if opts.IncludeBareLimit && hasBareLimitSignal(t) {
		return FailureLimit, ""
	}
	apiErr := apiErrWithoutBareTimeoutRE
	if opts.IncludeBareRequestTimeout {
		apiErr = apiErrRE
	}
	if isAPIError(t, apiErr) {
		return FailureAPIErr, ""
	}
	return "", ""
}
